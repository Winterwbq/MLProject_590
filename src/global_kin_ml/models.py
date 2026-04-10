from __future__ import annotations

import copy
from dataclasses import dataclass
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    model_key: str
    model_family: str
    feature_set: str
    latent_k: int | None
    hyperparameters: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        payload = {
            "model_key": self.model_key,
            "model_family": self.model_family,
            "feature_set": self.feature_set,
            "latent_k": self.latent_k,
        }
        for key, value in self.hyperparameters.items():
            payload[f"hp_{key}"] = value
        return payload


class ExperimentModel:
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        validation_score_callback=None,
    ) -> "ExperimentModel":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DirectSklearnModel(ExperimentModel):
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, x_train, y_train, x_val=None, y_val=None, validation_score_callback=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.estimator.fit(x_train, y_train)
        return self

    def predict(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.estimator.predict(x)


class DirectMLPModel(ExperimentModel):
    def __init__(
        self,
        hidden_width: int,
        hidden_layers: int,
        dropout: float,
        weight_decay: float,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 250,
        patience: int = 25,
    ) -> None:
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.network: torch.nn.Module | None = None
        self.device = torch.device("cpu")

    def _build_network(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for _ in range(self.hidden_layers):
            layers.append(torch.nn.Linear(last_dim, self.hidden_width))
            layers.append(torch.nn.ReLU())
            if self.dropout > 0.0:
                layers.append(torch.nn.Dropout(self.dropout))
            last_dim = self.hidden_width
        layers.append(torch.nn.Linear(last_dim, output_dim))
        return torch.nn.Sequential(*layers)

    def fit(self, x_train, y_train, x_val=None, y_val=None, validation_score_callback=None):
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        x_train_scaled = self.x_scaler.fit_transform(x_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        x_val_scaled = self.x_scaler.transform(x_val) if x_val is not None else None
        y_val_scaled = self.y_scaler.transform(y_val) if y_val is not None else None

        self.network = self._build_network(x_train.shape[1], y_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        train_x = torch.tensor(x_train_scaled, dtype=torch.float32, device=self.device)
        train_y = torch.tensor(y_train_scaled, dtype=torch.float32, device=self.device)

        if x_val_scaled is not None and y_val_scaled is not None:
            val_x = torch.tensor(x_val_scaled, dtype=torch.float32, device=self.device)
            val_y = torch.tensor(y_val_scaled, dtype=torch.float32, device=self.device)
        else:
            val_x = None
            val_y = None

        best_state = copy.deepcopy(self.network.state_dict())
        best_score = float("inf")
        patience_left = self.patience

        for _epoch in range(self.max_epochs):
            permutation = np.random.permutation(train_x.shape[0])
            self.network.train()
            for start in range(0, len(permutation), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_x = train_x[batch_indices]
                batch_y = train_y[batch_indices]
                optimizer.zero_grad()
                predictions = self.network(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                if val_x is not None and val_y is not None:
                    val_predictions = self.network(val_x)
                    if validation_score_callback is not None:
                        unscaled_predictions = self.y_scaler.inverse_transform(
                            val_predictions.cpu().numpy()
                        )
                        score = float(validation_score_callback(unscaled_predictions))
                    else:
                        val_loss = loss_fn(val_predictions, val_y).item()
                        score = float(val_loss)
                else:
                    train_predictions = self.network(train_x)
                    score = float(loss_fn(train_predictions, train_y).item())
            if score < best_score - 1e-8:
                best_score = score
                best_state = copy.deepcopy(self.network.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.network.load_state_dict(best_state)
        return self

    def predict(self, x):
        if self.network is None:
            raise RuntimeError("Model must be fit before prediction.")
        x_scaled = self.x_scaler.transform(x)
        tensor_x = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(tensor_x).cpu().numpy()
        return self.y_scaler.inverse_transform(predictions)


class LatentPCAModel(ExperimentModel):
    def __init__(self, base_model: ExperimentModel, latent_k: int) -> None:
        self.base_model = base_model
        self.latent_k = latent_k
        self.pca_: PCA | None = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, validation_score_callback=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.pca_ = PCA(n_components=self.latent_k, random_state=RANDOM_STATE, svd_solver="full")
            latent_train = self.pca_.fit_transform(y_train)
            latent_val = self.pca_.transform(y_val) if y_val is not None else None
        callback = None
        if validation_score_callback is not None and latent_val is not None:
            def callback(latent_predictions: np.ndarray) -> float:
                reconstructed = self.pca_.inverse_transform(latent_predictions)
                return float(validation_score_callback(reconstructed))

        self.base_model.fit(
            x_train,
            latent_train,
            x_val=x_val,
            y_val=latent_val,
            validation_score_callback=callback,
        )
        return self

    def predict(self, x):
        if self.pca_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        latent_predictions = self.base_model.predict(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.pca_.inverse_transform(latent_predictions)


class TwoStageSparseTreeModel(ExperimentModel):
    def __init__(
        self,
        tree_family: str,
        n_estimators: int,
        max_depth: int | None,
        min_samples_leaf: int,
    ) -> None:
        classifier_cls = ExtraTreesClassifier if tree_family == "extra_trees" else RandomForestClassifier
        regressor_cls = ExtraTreesRegressor if tree_family == "extra_trees" else RandomForestRegressor
        self.classifier = classifier_cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.regressor = regressor_cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.zero_log_values_: np.ndarray | None = None
        self.always_active_mask_: np.ndarray | None = None
        self.variable_activity_indices_: np.ndarray | None = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, validation_score_callback=None):
        y_train = np.asarray(y_train, dtype=float)
        self.zero_log_values_ = np.min(y_train, axis=0)
        active_targets = y_train > (self.zero_log_values_[None, :] + 1e-12)
        self.always_active_mask_ = np.all(active_targets, axis=0)
        self.variable_activity_indices_ = np.where(~self.always_active_mask_)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.regressor.fit(x_train, y_train)
            if self.variable_activity_indices_.size > 0:
                self.classifier.fit(x_train, active_targets[:, self.variable_activity_indices_])
        return self

    def predict(self, x):
        if (
            self.zero_log_values_ is None
            or self.always_active_mask_ is None
            or self.variable_activity_indices_ is None
        ):
            raise RuntimeError("Model must be fit before prediction.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predicted_log = np.asarray(self.regressor.predict(x), dtype=float)
        predicted_active = np.ones_like(predicted_log, dtype=bool)
        if self.variable_activity_indices_.size > 0:
            variable_pred = np.asarray(self.classifier.predict(x), dtype=bool)
            predicted_active[:, self.variable_activity_indices_] = variable_pred
        return np.where(predicted_active, predicted_log, self.zero_log_values_[None, :])


def build_model(config: ModelConfig) -> ExperimentModel:
    family = config.model_family
    params = config.hyperparameters
    if family == "ridge":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=float(params["alpha"]))),
            ]
        )
        model: ExperimentModel = DirectSklearnModel(estimator)
    elif family == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = DirectSklearnModel(estimator)
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = DirectSklearnModel(estimator)
    elif family == "two_stage_extra_trees":
        model = TwoStageSparseTreeModel(
            tree_family="extra_trees",
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
        )
    elif family == "two_stage_random_forest":
        model = TwoStageSparseTreeModel(
            tree_family="random_forest",
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
        )
    elif family == "mlp":
        model = DirectMLPModel(
            hidden_width=int(params["hidden_width"]),
            hidden_layers=int(params["hidden_layers"]),
            dropout=float(params["dropout"]),
            weight_decay=float(params["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported model family: {family}")

    if config.latent_k is not None:
        return LatentPCAModel(base_model=model, latent_k=config.latent_k)
    return model


def build_model_configs(feature_sets: list[str]) -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    for feature_set in feature_sets:
        for alpha in (0.1, 1.0, 10.0, 100.0):
            configs.append(
                ModelConfig(
                    model_key=f"direct__ridge__{feature_set}__alpha_{alpha}",
                    model_family="ridge",
                    feature_set=feature_set,
                    latent_k=None,
                    hyperparameters={"alpha": alpha},
                )
            )
        for tree_family in ("random_forest", "extra_trees"):
            for n_estimators in (300, 600):
                for max_depth in (None, 12, 20):
                    for min_samples_leaf in (1, 2, 4):
                        depth_label = "none" if max_depth is None else str(max_depth)
                        configs.append(
                            ModelConfig(
                                model_key=(
                                    f"direct__{tree_family}__{feature_set}"
                                    f"__n_{n_estimators}__d_{depth_label}__leaf_{min_samples_leaf}"
                                ),
                                model_family=tree_family,
                                feature_set=feature_set,
                                latent_k=None,
                                hyperparameters={
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth,
                                    "min_samples_leaf": min_samples_leaf,
                                },
                            )
                        )
        for hidden_width in (128, 256):
            for hidden_layers in (2, 3):
                for dropout in (0.0, 0.1):
                    for weight_decay in (0.0, 1e-5, 1e-4):
                        configs.append(
                            ModelConfig(
                                model_key=(
                                    f"direct__mlp__{feature_set}__w_{hidden_width}"
                                    f"__layers_{hidden_layers}__drop_{dropout}__wd_{weight_decay}"
                                ),
                                model_family="mlp",
                                feature_set=feature_set,
                                latent_k=None,
                                hyperparameters={
                                    "hidden_width": hidden_width,
                                    "hidden_layers": hidden_layers,
                                    "dropout": dropout,
                                    "weight_decay": weight_decay,
                                },
                            )
                        )

        for latent_k in (2, 3, 4, 6, 8, 10, 12):
            for alpha in (0.1, 1.0, 10.0, 100.0):
                configs.append(
                    ModelConfig(
                        model_key=f"latent__ridge__{feature_set}__k_{latent_k}__alpha_{alpha}",
                        model_family="ridge",
                        feature_set=feature_set,
                        latent_k=latent_k,
                        hyperparameters={"alpha": alpha},
                    )
                )
            for tree_family in ("random_forest", "extra_trees"):
                for n_estimators in (300, 600):
                    for max_depth in (None, 12, 20):
                        for min_samples_leaf in (1, 2, 4):
                            depth_label = "none" if max_depth is None else str(max_depth)
                            configs.append(
                                ModelConfig(
                                    model_key=(
                                        f"latent__{tree_family}__{feature_set}__k_{latent_k}"
                                        f"__n_{n_estimators}__d_{depth_label}__leaf_{min_samples_leaf}"
                                    ),
                                    model_family=tree_family,
                                    feature_set=feature_set,
                                    latent_k=latent_k,
                                    hyperparameters={
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "min_samples_leaf": min_samples_leaf,
                                    },
                                )
                            )
            for hidden_width in (128, 256):
                for hidden_layers in (2, 3):
                    for dropout in (0.0, 0.1):
                        for weight_decay in (0.0, 1e-5, 1e-4):
                            configs.append(
                                ModelConfig(
                                    model_key=(
                                        f"latent__mlp__{feature_set}__k_{latent_k}__w_{hidden_width}"
                                        f"__layers_{hidden_layers}__drop_{dropout}__wd_{weight_decay}"
                                    ),
                                    model_family="mlp",
                                    feature_set=feature_set,
                                    latent_k=latent_k,
                                    hyperparameters={
                                        "hidden_width": hidden_width,
                                        "hidden_layers": hidden_layers,
                                        "dropout": dropout,
                                        "weight_decay": weight_decay,
                                    },
                                )
                            )

    return configs
