"""
FMGAD 调参：固定项与搜索空间（按数据集名精确映射 + 未知数据集 Discovery）

与 run_tune_refined.py、tune_hyperparams.py 共用，保证两脚本口径一致。

机制划分（Gemini）：
- 全局无效、在前序消融中已否定的模块：任意数据集（含未知）一律在 fixed 中锁死，不进网格。
- 因图而异的机制：已知数据集用消融结论写死或收窄；未知数据集在搜索空间中全面放开，由数据投票。
"""

# 已在代表性数据上标定过的数据集（精确策略）
SOCIAL_LIGHT = frozenset({"weibo", "reddit"})
STRUCTURED_KNOWN = frozenset({"disney", "books", "enron"})
KNOWN_DATASETS = SOCIAL_LIGHT | STRUCTURED_KNOWN


def _norm(dataset: str) -> str:
    return dataset.strip().lower()


def is_known_dataset(dataset: str) -> bool:
    """是否为已标定的五数据集之一。"""
    return _norm(dataset) in KNOWN_DATASETS


def get_fixed_overrides(dataset: str) -> dict:
    """
    固定超参：全局无效机制一律 False；已知数据集按消融/best 精确设定；
    未知数据集（Discovery）只锁无效模块，不写死 weight / flow_t_sampling / use_virtual_neighbors。
    """
    d = _norm(dataset)
    base = {
        "use_multi_scale_residual": False,
        "use_adaptive_residual_scale": False,
        "use_multi_score_fusion": False,
        "use_score_smoothing": True,
    }
    if d in SOCIAL_LIGHT:
        base["flow_t_sampling"] = "uniform"
        base["weight"] = 0.0
        base["use_virtual_neighbors"] = d == "reddit"
        return base

    if d in STRUCTURED_KNOWN:
        base["flow_t_sampling"] = "logit_normal"
        base["use_virtual_neighbors"] = d in ("books", "enron")
        return base

    # --- Discovery：全新 / 未知数据集 ---
    # 仅保留上对「机制废料」的锁死（见模块 docstring）；结构性开关交给网格探索。
    return base


def get_refined_search_space(dataset: str) -> dict:
    """精细调参搜索空间（与 run_tune_refined 一致）。"""
    d = _norm(dataset)
    if d in SOCIAL_LIGHT:
        return {
            "ae_dropout": [0.1, 0.2, 0.3, 0.4],
            "ae_lr": [0.003, 0.005, 0.01, 0.02],
            "ae_alpha": [0.6, 0.8, 1.0],
            "residual_scale": [5.0, 10.0, 20.0],
            "sample_steps": [50, 100, 150],
        }

    if d in STRUCTURED_KNOWN:
        return {
            "ae_dropout": [0.1, 0.2, 0.3, 0.4],
            "ae_lr": [0.003, 0.005, 0.01, 0.02],
            "ae_alpha": [0.6, 0.8, 1.0],
            "residual_scale": [5.0, 10.0, 20.0],
            "sample_steps": [50, 100, 150],
            "weight": [0.5, 1.0, 1.5],
            "proto_alpha": [0.001, 0.005, 0.01, 0.05],
        }

    # --- Discovery：未知数据集，结构性机制全面进网格（weight 含 0 表示可不需要原型引导）---
    return {
        "ae_dropout": [0.1, 0.2, 0.3],
        "ae_lr": [0.005, 0.01],
        "ae_alpha": [0.6, 0.8, 1.0],
        "residual_scale": [5.0, 10.0, 20.0],
        "sample_steps": [50, 100, 150],
        "weight": [0.0, 0.5, 1.0, 1.5],
        "flow_t_sampling": ["uniform", "logit_normal"],
        "use_virtual_neighbors": [True, False],
        "proto_alpha": [0.001, 0.01],
    }


def get_reduced_search_space(dataset: str) -> dict:
    """tune_hyperparams --reduced 用的小网格，结构同 get_refined_search_space。"""
    d = _norm(dataset)
    if d in SOCIAL_LIGHT:
        return {
            "ae_dropout": [0.2, 0.3],
            "ae_lr": [0.005, 0.01],
            "ae_alpha": [0.8],
            "residual_scale": [10.0],
            "sample_steps": [50, 100],
        }

    if d in STRUCTURED_KNOWN:
        return {
            "ae_dropout": [0.2, 0.3],
            "ae_lr": [0.005, 0.01],
            "ae_alpha": [0.8],
            "residual_scale": [10.0],
            "sample_steps": [50, 100],
            "weight": [0.5, 1.0],
            "proto_alpha": [0.001, 0.01],
        }

    return {
        "ae_dropout": [0.2, 0.3],
        "ae_lr": [0.005, 0.01],
        "ae_alpha": [0.8, 1.0],
        "residual_scale": [10.0, 20.0],
        "sample_steps": [50, 100],
        "weight": [0.0, 0.5, 1.0, 1.5],
        "flow_t_sampling": ["uniform", "logit_normal"],
        "use_virtual_neighbors": [True, False],
        "proto_alpha": [0.001, 0.01],
    }
