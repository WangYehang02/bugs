import os
import sys
import time
import yaml
import argparse
import json

from res_flow_gad import ResFlowGAD


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "weibo.yaml"))
    parser.add_argument("--result-file", type=str, default=None, help="Optional: write metrics JSON to this file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_trial", type=int, default=None, help="Number of trials (default from config or 3)")
    return parser.parse_args()


def _set_seed(seed: int):
    """固定随机种子，使结果可复现"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = get_arguments()
    _set_seed(args.seed)
    print("Random seed:", args.seed, flush=True)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    dset = cfg["dataset"]

    # 和 v3 一样：如果配置里给了 ae_alpha=0.0，自动回退到一个安全值，避免 NaN
    ae_alpha_cfg = cfg.get("ae_alpha", 0.8)
    if ae_alpha_cfg == 0.0:
        ae_alpha_cfg = 0.9

    # 优化版：多尺度残差、多评分融合（温度系数）、虚拟邻居、分数平滑等
    model = ResFlowGAD(
        hid_dim=cfg.get("hid_dim") if cfg.get("hid_dim") else None,
        ae_dropout=cfg["ae_dropout"],
        ae_lr=cfg["ae_lr"],
        ae_alpha=ae_alpha_cfg,
        proto_alpha=cfg.get("proto_alpha", 0.01),
        weight=cfg.get("weight", 1.0),
        residual_scale=float(cfg.get("residual_scale", 10.0)),
        sample_steps=int(cfg.get("sample_steps", 100)),
        verbose=True,
        use_nll_score=cfg.get("use_nll_score", False),
        use_energy_score=cfg.get("use_energy_score", False),
        use_guided_recon=cfg.get("use_guided_recon", False),
        use_multi_scale_residual=cfg.get("use_multi_scale_residual", True),
        use_adaptive_residual_scale=cfg.get("use_adaptive_residual_scale", True),
        use_multi_score_fusion=cfg.get("use_multi_score_fusion", True),
        score_fusion_temperature=float(cfg.get("score_fusion_temperature", 1.0)),
        use_virtual_neighbors=cfg.get("use_virtual_neighbors", True),
        virtual_degree_threshold=int(cfg.get("virtual_degree_threshold", 5)),
        virtual_k=int(cfg.get("virtual_k", 5)),
        use_hard_negative_mining=cfg.get("use_hard_negative_mining", False),
        use_curriculum_learning=cfg.get("use_curriculum_learning", False),
        curriculum_warmup_epochs=int(cfg.get("curriculum_warmup_epochs", 100)),
        use_score_smoothing=cfg.get("use_score_smoothing", True),
        score_smoothing_alpha=float(cfg.get("score_smoothing_alpha", 0.3)),
        ensemble_score=cfg.get("ensemble_score", True),
        num_trial=args.num_trial if args.num_trial is not None else int(cfg.get("num_trial", 3)),
    )

    print("Running FMGADself on dataset:", dset, "num_trial:", model.num_trial, flush=True)
    t0 = time.perf_counter()
    out = model(dset)
    elapsed = time.perf_counter() - t0
    print("FMGADself_TIME_SEC\t{:.1f}".format(elapsed), flush=True)
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump({"dataset": dset, "time_sec": elapsed, **out}, f, indent=2)
    return out


if __name__ == "__main__":
    main()