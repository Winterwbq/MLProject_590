from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_ID_COLUMNS = ["global_case_id", "density_group_id", "local_case_id"]
CASE_METADATA_CANDIDATES = [
    "global_case_id",
    "source_file_id",
    "source_file",
    "power_group_id",
    "power_label",
    "power_mj",
    "file_case_id",
    "density_group_id",
    "density_group_in_file_id",
    "local_case_id",
]
INPUT_PREFIX = "input_"
TARGET_PREFIX = "rate_const_"


@dataclass
class ValidationFold:
    fold_id: int
    seed: int
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class ExperimentSplits:
    test_indices: np.ndarray
    trainval_indices: np.ndarray
    validation_folds: list[ValidationFold]


def get_case_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CASE_METADATA_CANDIDATES if column in frame.columns]


class TargetTransformer:
    def __init__(self, target_columns: list[str]) -> None:
        self.target_columns = target_columns
        self.epsilons_: np.ndarray | None = None

    def fit(self, target_frame: pd.DataFrame) -> "TargetTransformer":
        matrix = target_frame[self.target_columns].to_numpy(dtype=float)
        epsilons = []
        for column_index in range(matrix.shape[1]):
            positives = matrix[:, column_index][matrix[:, column_index] > 0.0]
            min_positive = positives.min() if positives.size else 0.0
            epsilon = max(min_positive / 10.0, 1e-30)
            epsilons.append(epsilon)
        self.epsilons_ = np.array(epsilons, dtype=float)
        return self

    def transform(self, target_frame: pd.DataFrame) -> pd.DataFrame:
        if self.epsilons_ is None:
            raise RuntimeError("TargetTransformer must be fit before transform.")
        matrix = target_frame[self.target_columns].to_numpy(dtype=float)
        transformed = np.log10(matrix + self.epsilons_)
        return pd.DataFrame(transformed, columns=self.target_columns, index=target_frame.index)

    def inverse_transform_array(self, values: np.ndarray) -> np.ndarray:
        if self.epsilons_ is None:
            raise RuntimeError("TargetTransformer must be fit before inverse_transform.")
        restored = np.power(10.0, values) - self.epsilons_
        return np.clip(restored, a_min=0.0, a_max=None)

    def epsilon_frame(self) -> pd.DataFrame:
        if self.epsilons_ is None:
            raise RuntimeError("TargetTransformer must be fit before epsilon_frame.")
        return pd.DataFrame(
            {
                "target_column": self.target_columns,
                "epsilon": self.epsilons_,
            }
        )


class FullNonconstantFeatureTransformer:
    name = "full_nonconstant_plus_log_en"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "FullNonconstantFeatureTransformer":
        input_columns = [column for column in input_frame.columns if column.startswith(INPUT_PREFIX)]
        self.selected_columns_ = [
            column for column in input_columns if input_frame[column].nunique(dropna=False) > 1
        ]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before transform.")
        transformed = pd.DataFrame(index=input_frame.index)
        transformed["log10_e_over_n"] = np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float))
        for column in self.selected_columns_:
            transformed[column] = input_frame[column].to_numpy(dtype=float)
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.selected_columns_) + 1,
            "component_count": 0,
            "selected_columns": ",".join(self.selected_columns_),
        }


class FullNonconstantPowerFeatureTransformer:
    name = "full_nonconstant_plus_log_en_log_power"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "FullNonconstantPowerFeatureTransformer":
        input_columns = [column for column in input_frame.columns if column.startswith(INPUT_PREFIX)]
        self.selected_columns_ = [
            column for column in input_columns if input_frame[column].nunique(dropna=False) > 1
        ]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before transform.")
        transformed = pd.DataFrame(index=input_frame.index)
        transformed["log10_power_mj"] = np.log10(input_frame["power_mj"].to_numpy(dtype=float))
        transformed["log10_e_over_n"] = np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float))
        for column in self.selected_columns_:
            transformed[column] = input_frame[column].to_numpy(dtype=float)
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.selected_columns_) + 2,
            "component_count": 0,
            "selected_columns": ",".join(self.selected_columns_),
        }


class AllInputsFeatureTransformer:
    name = "all_inputs_plus_log_en"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "AllInputsFeatureTransformer":
        self.selected_columns_ = [
            column for column in input_frame.columns if column.startswith(INPUT_PREFIX)
        ]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before transform.")
        transformed = pd.DataFrame(index=input_frame.index)
        transformed["log10_e_over_n"] = np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float))
        for column in self.selected_columns_:
            transformed[column] = input_frame[column].to_numpy(dtype=float)
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.selected_columns_) + 1,
            "component_count": 0,
            "selected_columns": ",".join(self.selected_columns_),
        }


class AllInputsPowerFeatureTransformer:
    name = "all_inputs_plus_log_en_log_power"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "AllInputsPowerFeatureTransformer":
        self.selected_columns_ = [
            column for column in input_frame.columns if column.startswith(INPUT_PREFIX)
        ]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before transform.")
        transformed = pd.DataFrame(index=input_frame.index)
        transformed["log10_power_mj"] = np.log10(input_frame["power_mj"].to_numpy(dtype=float))
        transformed["log10_e_over_n"] = np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float))
        for column in self.selected_columns_:
            transformed[column] = input_frame[column].to_numpy(dtype=float)
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.selected_columns_) + 2,
            "component_count": 0,
            "selected_columns": ",".join(self.selected_columns_),
        }


class CompositionPCAFeatureTransformer:
    name = "composition_pca_plus_log_en"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.component_names_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "CompositionPCAFeatureTransformer":
        input_columns = [column for column in input_frame.columns if column.startswith(INPUT_PREFIX)]
        self.selected_columns_ = [
            column for column in input_columns if input_frame[column].nunique(dropna=False) > 1
        ]
        matrix = input_frame[self.selected_columns_].to_numpy(dtype=float)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_pca = PCA(random_state=42, svd_solver="full")
            full_pca.fit(scaled)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        component_count = int(np.searchsorted(cumulative, 0.99) + 1)
        component_count = max(1, component_count)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.pca_ = PCA(n_components=component_count, random_state=42, svd_solver="full")
            self.pca_.fit(scaled)
        self.component_names_ = [f"composition_pc_{index:02d}" for index in range(1, component_count + 1)]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self.selected_columns_ is None
            or self.scaler_ is None
            or self.pca_ is None
            or self.component_names_ is None
        ):
            raise RuntimeError("Feature transformer must be fit before transform.")
        matrix = input_frame[self.selected_columns_].to_numpy(dtype=float)
        scaled = self.scaler_.transform(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            components = self.pca_.transform(scaled)
        transformed = pd.DataFrame(components, columns=self.component_names_, index=input_frame.index)
        transformed.insert(0, "log10_e_over_n", np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float)))
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None or self.component_names_ is None or self.pca_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.component_names_) + 1,
            "component_count": len(self.component_names_),
            "selected_columns": ",".join(self.selected_columns_),
            "explained_variance_99pct": float(np.sum(self.pca_.explained_variance_ratio_)),
        }


class CompositionPCAPowerFeatureTransformer:
    name = "composition_pca_plus_log_en_log_power"

    def __init__(self) -> None:
        self.selected_columns_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.component_names_: list[str] | None = None

    def fit(self, input_frame: pd.DataFrame) -> "CompositionPCAPowerFeatureTransformer":
        input_columns = [column for column in input_frame.columns if column.startswith(INPUT_PREFIX)]
        self.selected_columns_ = [
            column for column in input_columns if input_frame[column].nunique(dropna=False) > 1
        ]
        matrix = input_frame[self.selected_columns_].to_numpy(dtype=float)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_pca = PCA(random_state=42, svd_solver="full")
            full_pca.fit(scaled)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        component_count = int(np.searchsorted(cumulative, 0.99) + 1)
        component_count = max(1, component_count)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.pca_ = PCA(n_components=component_count, random_state=42, svd_solver="full")
            self.pca_.fit(scaled)
        self.component_names_ = [f"composition_pc_{index:02d}" for index in range(1, component_count + 1)]
        return self

    def transform(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self.selected_columns_ is None
            or self.scaler_ is None
            or self.pca_ is None
            or self.component_names_ is None
        ):
            raise RuntimeError("Feature transformer must be fit before transform.")
        matrix = input_frame[self.selected_columns_].to_numpy(dtype=float)
        scaled = self.scaler_.transform(matrix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            components = self.pca_.transform(scaled)
        transformed = pd.DataFrame(components, columns=self.component_names_, index=input_frame.index)
        transformed.insert(0, "log10_e_over_n", np.log10(input_frame["e_over_n_v_cm2"].to_numpy(dtype=float)))
        transformed.insert(0, "log10_power_mj", np.log10(input_frame["power_mj"].to_numpy(dtype=float)))
        return transformed

    def metadata(self) -> dict[str, object]:
        if self.selected_columns_ is None or self.component_names_ is None or self.pca_ is None:
            raise RuntimeError("Feature transformer must be fit before metadata.")
        return {
            "feature_set": self.name,
            "feature_count": len(self.component_names_) + 2,
            "component_count": len(self.component_names_),
            "selected_columns": ",".join(self.selected_columns_),
            "explained_variance_99pct": float(np.sum(self.pca_.explained_variance_ratio_)),
        }


def build_feature_transformer(feature_set_name: str):
    if feature_set_name == AllInputsPowerFeatureTransformer.name:
        return AllInputsPowerFeatureTransformer()
    if feature_set_name == AllInputsFeatureTransformer.name:
        return AllInputsFeatureTransformer()
    if feature_set_name == FullNonconstantPowerFeatureTransformer.name:
        return FullNonconstantPowerFeatureTransformer()
    if feature_set_name == FullNonconstantFeatureTransformer.name:
        return FullNonconstantFeatureTransformer()
    if feature_set_name == CompositionPCAPowerFeatureTransformer.name:
        return CompositionPCAPowerFeatureTransformer()
    if feature_set_name == CompositionPCAFeatureTransformer.name:
        return CompositionPCAFeatureTransformer()
    raise ValueError(f"Unsupported feature set: {feature_set_name}")


def create_random_case_splits(
    total_cases: int,
    test_seed: int = 42,
    val_seeds: tuple[int, ...] = (7, 21, 42, 84, 126),
) -> ExperimentSplits:
    all_indices = np.arange(total_cases)
    trainval_indices, test_indices = train_test_split(
        all_indices, test_size=0.15, random_state=test_seed, shuffle=True
    )
    folds = []
    for fold_id, seed in enumerate(val_seeds, start=1):
        fold_train, fold_val = train_test_split(
            trainval_indices, test_size=0.20, random_state=seed, shuffle=True
        )
        folds.append(
            ValidationFold(
                fold_id=fold_id,
                seed=seed,
                train_indices=np.sort(fold_train),
                val_indices=np.sort(fold_val),
            )
        )
    return ExperimentSplits(
        test_indices=np.sort(test_indices),
        trainval_indices=np.sort(trainval_indices),
        validation_folds=folds,
    )


def create_group_holdout_splits(
    groups: pd.Series,
    holdout_group_values: list[object],
    val_seeds: tuple[int, ...] = (7, 21, 42, 84, 126),
) -> ExperimentSplits:
    group_array = groups.to_numpy()
    holdout_mask = np.isin(group_array, np.array(holdout_group_values, dtype=object))
    test_indices = np.where(holdout_mask)[0]
    trainval_indices = np.where(~holdout_mask)[0]
    folds = []
    for fold_id, seed in enumerate(val_seeds, start=1):
        fold_train, fold_val = train_test_split(
            trainval_indices,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
        )
        folds.append(
            ValidationFold(
                fold_id=fold_id,
                seed=seed,
                train_indices=np.sort(fold_train),
                val_indices=np.sort(fold_val),
            )
        )
    return ExperimentSplits(
        test_indices=np.sort(test_indices),
        trainval_indices=np.sort(trainval_indices),
        validation_folds=folds,
    )


def build_split_assignment_frame(inputs: pd.DataFrame, splits: ExperimentSplits) -> pd.DataFrame:
    assignment = pd.DataFrame(index=inputs.index)
    metadata_columns = get_case_metadata_columns(inputs)
    assignment[metadata_columns] = inputs[metadata_columns]
    assignment["locked_split"] = "trainval"
    assignment.loc[splits.test_indices, "locked_split"] = "test"
    for fold in splits.validation_folds:
        assignment[f"fold_{fold.fold_id}_role"] = ""
        assignment.loc[fold.train_indices, f"fold_{fold.fold_id}_role"] = "train"
        assignment.loc[fold.val_indices, f"fold_{fold.fold_id}_role"] = "validation"
        assignment.loc[splits.test_indices, f"fold_{fold.fold_id}_role"] = "test"
    return assignment
