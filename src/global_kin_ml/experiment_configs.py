from __future__ import annotations

from .models import ModelConfig


def _tree_params() -> dict[str, int]:
    return {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1}


def _mlp_params() -> dict[str, float | int]:
    return {
        "hidden_width": 256,
        "hidden_layers": 3,
        "dropout": 0.0,
        "weight_decay": 1e-5,
    }


def _multitask_common_mlp_params() -> dict[str, float | int]:
    return {
        "hidden_layers": 3,
        "dropout": 0.0,
        "weight_decay": 1e-5,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "max_epochs": 120,
        "patience": 15,
    }


def build_rate_const_curated_configs() -> list[ModelConfig]:
    tree_params = _tree_params()
    return [
        ModelConfig(
            model_key="direct__ridge__full_nonconstant_plus_log_en_log_power__alpha_10.0",
            model_family="ridge",
            feature_set="full_nonconstant_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"alpha": 10.0},
        ),
        ModelConfig(
            model_key="direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=_mlp_params(),
        ),
        ModelConfig(
            model_key="latent__ridge__composition_pca_plus_log_en_log_power__k_8__alpha_10.0",
            model_family="ridge",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters={"alpha": 10.0},
        ),
        ModelConfig(
            model_key="latent__random_forest__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1",
            model_family="random_forest",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="latent__mlp__composition_pca_plus_log_en_log_power__k_8__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="mlp",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters=_mlp_params(),
        ),
    ]


def build_super_rate_curated_configs() -> list[ModelConfig]:
    tree_params = _tree_params()
    return [
        ModelConfig(
            model_key="direct__ridge__full_nonconstant_plus_log_en_log_power__alpha_10.0",
            model_family="ridge",
            feature_set="full_nonconstant_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"alpha": 10.0},
        ),
        ModelConfig(
            model_key="direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="two_stage__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="two_stage_extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="two_stage__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="two_stage_extra_trees",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=dict(tree_params),
        ),
        ModelConfig(
            model_key="direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters=_mlp_params(),
        ),
        ModelConfig(
            model_key="latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters=dict(tree_params),
        ),
    ]


def build_multitask_configs() -> list[ModelConfig]:
    common = _multitask_common_mlp_params()
    return [
        ModelConfig(
            model_key="joint__single_head_mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_single_head_mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, **common},
        ),
        ModelConfig(
            model_key="joint__single_head_mlp__composition_pca_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_single_head_mlp",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, **common},
        ),
        ModelConfig(
            model_key="joint__two_head_mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_two_head_mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={
                "hidden_width": 256,
                "rate_loss_weight": 0.5,
                "super_loss_weight": 0.5,
                **common,
            },
        ),
        ModelConfig(
            model_key="joint__two_head_mlp__composition_pca_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_two_head_mlp",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={
                "hidden_width": 256,
                "rate_loss_weight": 0.5,
                "super_loss_weight": 0.5,
                **common,
            },
        ),
    ]
