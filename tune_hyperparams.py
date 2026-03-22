#!/usr/bin/env python3
"""
FMGAD 超参数调优脚本
- 支持 5 个数据集：weibo, reddit, disney, books, enron
- 多卡并行：每个数据集分配独立 GPU，可同时调参
- 后台运行：建议用 nohup 或 screen/tmux 启动，退出后仍继续运行
- 跑完后自动生成报告到 reports/FMGAD_tuning_report.md
"""
import os
import sys
import json
import yaml
import copy
import itertools
import random
import subprocess
import tempfile
import argparse
import math
from pathlib import Path
from typing import Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 项目根目录
FMGAD_ROOT = Path(__file__).resolve().parent
FINAL_DIR = Path("/mnt/yehang") / "fmgad_outputs"
REPORTS_DIR = FINAL_DIR  # 报告保存到 /mnt/yehang
BEST_CONFIGS_DIR = FINAL_DIR / "best_configs"  # 最佳配置 yaml 保存目录
RESULTS_DIR = FINAL_DIR / "results"  # 单次实验 JSON 保存目录
LOGS_DIR = FMGAD_ROOT / "logs"
CONFIGS_DIR = FMGAD_ROOT / "configs"

# 5 个目标数据集
DATASETS = ["weibo", "reddit", "disney", "books", "enron"]

# 搜索空间：固定参数，减少动态自适应，便于找到最佳配置
# 每个数据集会在此空间内搜索
# 完整空间约 3*3*3*3*3*2*2*2*2 = 3888 组，用随机采样控制数量
SEARCH_SPACE = {
    "ae_dropout": [0.2, 0.3, 0.4],
    "ae_lr": [0.005, 0.01, 0.02],
    "ae_alpha": [0.6, 0.8, 1.0],
    "proto_alpha": [0.001, 0.01, 0.05],
    "weight": [0.5, 1.0, 1.5],
    "residual_scale": [5.0, 10.0, 20.0],
    "sample_steps": [50, 100],
    "use_adaptive_residual_scale": [False],
    "use_multi_score_fusion": [False],  # 固定关闭多评分机制
    "use_virtual_neighbors": [True, False],
    # ---- 新增调参项 ----
    "virtual_k": [2, 5, 8],                  # 虚拟邻居补充数量 k
    "virtual_degree_threshold": [3, 5, 10],  # 判定为低度节点的阈值
    # ------------------
    "use_score_smoothing": [True, False],
}

# 精简搜索空间（用于快速测试，约 16 组）
REDUCED_SPACE = {
    "ae_dropout": [0.2, 0.3],
    "ae_lr": [0.01],
    "ae_alpha": [0.8],
    "proto_alpha": [0.01],
    "weight": [0.5, 1.0, 1.5],
    "residual_scale": [10.0],
    "sample_steps": [100],
    "use_adaptive_residual_scale": [False],
    "use_multi_score_fusion": [False],  # 固定关闭多评分机制
    "use_virtual_neighbors": [True, False],
    # 快速测试：固定虚拟邻居参数
    "virtual_k": [5],
    "virtual_degree_threshold": [5],
    "use_score_smoothing": [True, False],
    "use_joint_training": [True],
    "joint_ae_weight": [0.05, 0.1, 0.2],
    "joint_fm_weight": [0.5, 1.0, 2.0],
    "joint_warmup_epochs": [0, 50, 100],
}

# 默认每组数据集最多尝试的配置数（随机采样，避免全网格爆炸）
DEFAULT_MAX_CONFIGS = 128  # 调参更详细

# 仅调 virtual_k / virtual_degree_threshold 时的网格（其余参数来自「最佳」基线 yaml）
# 关闭分数平滑由 force_disable_score_smoothing 统一注入，不占用搜索维度
K_ONLY_SPACE = {
    "virtual_k": [2, 3, 5, 8, 10, 12],
    "virtual_degree_threshold": [2, 3, 5, 8, 10],
}
DEFAULT_K_ONLY_MAX_CONFIGS = 512  # 全网格 6*6=36，留余量


def _dict_product(d):
    """将搜索空间展开为所有组合的列表"""
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _sample_configs(space: dict, max_configs: int, seed: int) -> list:
    """从搜索空间中采样最多 max_configs 个配置（随机采样，可复现）"""
    all_configs = list(_dict_product(space))
    if len(all_configs) <= max_configs:
        return all_configs
    rng = random.Random(seed)
    return rng.sample(all_configs, max_configs)


def _sanitize_for_json(obj):
    """将 NaN/Inf 等不可 JSON 化的值转换为 None，递归处理 dict/list。"""
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # 兜底：尽量转成可序列化类型
    try:
        if hasattr(obj, "item"):
            return _sanitize_for_json(obj.item())
    except Exception:
        pass
    return str(obj)


def _load_base_config(dataset: str, base_dir: Optional[Path] = None) -> dict:
    if base_dir is not None:
        p = base_dir / f"{dataset}.yaml"
        if p.exists():
            with open(p, "r") as f:
                return yaml.load(f, Loader=yaml.Loader)
    cfg_path = CONFIGS_DIR / f"{dataset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _run_single_experiment(
    dataset: str,
    config_overrides: dict,
    device: int,
    seed: int = 42,
    result_dir: Path = None,
    base_dir: Optional[Path] = None,
    force_disable_score_smoothing: bool = False,
) -> dict:
    """
    运行单次实验，返回 {config, auc_mean, auc_std, ...} 或 {error: str}
    """
    base = _load_base_config(dataset, base_dir)
    cfg = copy.deepcopy(base)
    cfg.update(config_overrides)
    if force_disable_score_smoothing:
        cfg["use_score_smoothing"] = False
        cfg["score_smoothing_alpha"] = 0.0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(FMGAD_ROOT)
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        tmp_config = f.name

    result_file = None
    if result_dir:
        result_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = (
            hash(json.dumps({"cfg": cfg, "seed": seed}, sort_keys=True, ensure_ascii=False))
            % (10 ** 8)
        )
        result_file = result_dir / f"{dataset}_{cfg_hash}.json"

    try:
        cmd = [
            sys.executable,
            str(FMGAD_ROOT / "main_train.py"),
            "--config", tmp_config,
            "--device", str(device),
            "--seed", str(seed),
        ]
        # main_train.py 的 --result-file 仅写指标，这里我们自己在末尾写“可复现 JSON”
        metric_tmp_file = None
        if result_file:
            metric_tmp_file = str(result_file) + ".metrics_tmp.json"
            cmd.extend(["--result-file", metric_tmp_file])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)

        proc = subprocess.run(
            cmd,
            cwd=str(FMGAD_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        os.unlink(tmp_config)

        metrics = {}
        if metric_tmp_file and os.path.exists(metric_tmp_file):
            with open(metric_tmp_file, "r") as f:
                metrics = json.load(f)
            try:
                os.unlink(metric_tmp_file)
            except Exception:
                pass

        err_tail = (proc.stderr or proc.stdout or "")[-4000:]
        ok = proc.returncode == 0 and bool(metrics)

        # 写“可复现 JSON”：包含 base+overrides 合成后的完整 cfg、seed、以及指标（含 auc）
        if result_file:
            payload = {
                "dataset": dataset,
                "seed": seed,
                "config": cfg,  # 完整配置（可直接复现）
                "overrides": config_overrides,  # 本次调参覆盖项（便于阅读）
                "metrics": _sanitize_for_json(metrics),
                "returncode": proc.returncode,
                "ok": bool(ok),
                "error_tail": err_tail if not ok else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
            }
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)

        if ok:
            return {
                "config": config_overrides,
                "seed": seed,
                "auc_mean": metrics.get("auc_mean", 0.0),
                "auc_std": metrics.get("auc_std", 0.0),
                "ap_mean": metrics.get("ap_mean", 0.0),
            }
        else:
            return {
                "config": config_overrides,
                "seed": seed,
                "auc_mean": 0.0,
                "error": err_tail or "Unknown error",
            }
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_config)
        except Exception:
            pass
        # 超时也落盘（若有 result_file）
        if result_file:
            payload = {
                "dataset": dataset,
                "seed": seed,
                "config": cfg,
                "overrides": config_overrides,
                "metrics": {},
                "returncode": None,
                "ok": False,
                "error_tail": "Timeout",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
            }
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        return {"config": config_overrides, "seed": seed, "error": "Timeout", "auc_mean": 0.0}
    except Exception as e:
        try:
            os.unlink(tmp_config)
        except Exception:
            pass
        if result_file:
            payload = {
                "dataset": dataset,
                "seed": seed,
                "config": cfg,
                "overrides": config_overrides,
                "metrics": {},
                "returncode": None,
                "ok": False,
                "error_tail": str(e) + "\n" + traceback.format_exc(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
            }
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        return {
            "config": config_overrides,
            "seed": seed,
            "error": str(e) + "\n" + traceback.format_exc(),
            "auc_mean": 0.0,
        }


def _tune_dataset_shard(args):
    """
    对单个数据集的一部分配置进行调参，在指定 GPU 上顺序运行。
    args: (dataset, device, configs, seed, result_dir, base_dir, force_disable_score_smoothing)
    """
    dataset, device, configs, seed, result_dir, base_dir, force_disable_score_smoothing = args
    results = []
    for i, cfg in enumerate(configs):
        print(f"[{dataset}] GPU{device} shard config {i+1}/{len(configs)}", flush=True)
        r = _run_single_experiment(
            dataset=dataset,
            config_overrides=cfg,
            device=device,
            seed=seed,
            result_dir=result_dir,
            base_dir=base_dir,
            force_disable_score_smoothing=force_disable_score_smoothing,
        )
        r["dataset"] = dataset
        results.append(r)
        if "error" in r:
            print(f"  -> ERROR: {r['error'][:200]}", flush=True)
        else:
            print(f"  -> AUC: {r['auc_mean']:.4f}", flush=True)
    return dataset, results


def _split_list(items: list, num_splits: int) -> list:
    """尽量均匀地把 items 切成 num_splits 份（允许空）。"""
    if num_splits <= 1:
        return [items]
    splits = [[] for _ in range(num_splits)]
    for idx, it in enumerate(items):
        splits[idx % num_splits].append(it)
    return splits


def _load_or_init_report_header(report_path: Path) -> tuple:
    """
    返回 (generated_time_str, existing_change_log_lines)。
    若报告不存在，则生成新的生成时间并返回空修改记录。
    """
    if not report_path.exists():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S"), []
    try:
        txt = report_path.read_text(encoding="utf-8")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S"), []

    gen = None
    change_lines = []
    for line in txt.splitlines():
        if line.startswith("生成时间:"):
            gen = line.split("生成时间:", 1)[1].strip()
        if line.startswith("- 修改时间:"):
            change_lines.append(line)
    return gen or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), change_lines


def main():
    parser = argparse.ArgumentParser(description="FMGAD 超参数调优")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="要调参的数据集列表；默认：全部 5 个；--k-only 时默认为 books enron",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="使用的 GPU 列表，按顺序分配给各数据集",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="使用精简搜索空间（快速测试）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="实验结果目录，默认 logs/tune_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行进程数（用于多 GPU 分片并行）",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=DEFAULT_MAX_CONFIGS,
        help=f"每个数据集最多尝试的配置数（默认 {DEFAULT_MAX_CONFIGS}，超出则随机采样）",
    )
    parser.add_argument(
        "--k-only",
        action="store_true",
        help="仅搜索 virtual_k 与 virtual_degree_threshold；其余超参来自 --base-config-dir 下的最佳 yaml（并强制关闭分数平滑）",
    )
    parser.add_argument(
        "--base-config-dir",
        type=str,
        default="/home/yehang/exp/configs",
        help="--k-only 时读取 {dataset}.yaml 的目录（含 books.yaml / enron.yaml 等最佳基线）",
    )
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = ["books", "enron"] if args.k_only else DATASETS

    space = (
        K_ONLY_SPACE
        if args.k_only
        else (REDUCED_SPACE if args.reduced else SEARCH_SPACE)
    )
    k_only_base = Path(args.base_config_dir).resolve() if args.k_only else None
    force_disable_score_smoothing = bool(args.k_only)
    if args.k_only and not k_only_base.exists():
        print(f"警告: --base-config-dir 不存在 {k_only_base}，将回退到项目内 configs/", flush=True)
        k_only_base = None
    # 输出统一写到 /mnt/yehang
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else FINAL_DIR / f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存搜索空间
    with open(output_dir / "search_space.json", "w") as f:
        json.dump(space, f, indent=2)

    datasets = [d for d in args.datasets if d in DATASETS]
    if not datasets:
        datasets = DATASETS

    # 为每个数据集分配 GPU（按配置分片到多 GPU）
    max_configs = args.max_configs
    tasks = []
    # 每个数据集独立采样一次，再平均分到多张 GPU 上
    num_gpus = max(1, len(args.gpus))
    # 两个数据集时，默认把 GPU 平均分成两组（如 8 卡 -> 每个数据集 4 卡）
    gpus_per_dataset = max(1, num_gpus // max(1, len(datasets)))
    for di, ds in enumerate(datasets):
        cfgs = _sample_configs(space, max_configs, args.seed + di)
        start = di * gpus_per_dataset
        assigned = args.gpus[start:start + gpus_per_dataset] or [args.gpus[di % num_gpus]]
        shards = _split_list(cfgs, len(assigned))
        for gpu, shard_cfgs in zip(assigned, shards):
            if not shard_cfgs:
                continue
            tasks.append(
                (
                    ds,
                    gpu,
                    shard_cfgs,
                    args.seed,
                    result_dir,
                    k_only_base,
                    force_disable_score_smoothing,
                )
            )

    print("=" * 60)
    print("FMGAD 超参数调优" + ("（K-only，分数平滑已关）" if args.k_only else ""))
    print("数据集:", datasets)
    if args.k_only:
        print("基线配置目录:", k_only_base or CONFIGS_DIR)
    print("GPU:", [t[1] for t in tasks])
    print("输出目录:", output_dir)
    print("=" * 60, flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        futures = {ex.submit(_tune_dataset_shard, t): (t[0], t[1]) for t in tasks}
        for fut in as_completed(futures):
            ds, gpu = futures[fut]
            try:
                ds, results = fut.result()
                all_results.setdefault(ds, []).extend(results)
                print(f"[DONE] {ds} GPU{gpu}: {len(results)} configs", flush=True)
            except Exception as e:
                print(f"[FAIL] {ds} GPU{gpu}: {e}", flush=True)
                all_results.setdefault(ds, [])

    # 汇总最佳配置
    best_per_dataset = {}
    for ds, results in all_results.items():
        valid = [r for r in results if "error" not in r and r.get("auc_mean", 0) > 0]
        if valid:
            best = max(valid, key=lambda x: x["auc_mean"])
            best_per_dataset[ds] = {
                "config": best["config"],
                "auc_mean": best["auc_mean"],
                "auc_std": best.get("auc_std", 0),
                "ap_mean": best.get("ap_mean", 0),
            }
        else:
            best_per_dataset[ds] = {"config": {}, "auc_mean": 0, "error": "No valid runs"}

    # 写 JSON 结果
    summary_path = output_dir / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_per_dataset": best_per_dataset, "all_results": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 保存最佳配置 yaml 到 /mnt/yehang/fmgad_outputs/best_configs/
    for ds in datasets:
        b = best_per_dataset.get(ds, {})
        if "error" not in b and b.get("config"):
            base = _load_base_config(ds, k_only_base if args.k_only else None)
            full_cfg = copy.deepcopy(base)
            full_cfg.update(b["config"])
            full_cfg["use_multi_score_fusion"] = False  # 确保关闭多评分
            if args.k_only:
                full_cfg["use_score_smoothing"] = False
                full_cfg["score_smoothing_alpha"] = 0.0
            yaml_path = BEST_CONFIGS_DIR / f"{ds}_best_tuned.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(full_cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"  已保存最佳配置: {yaml_path}", flush=True)

    # 生成 Markdown 报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "FMGAD_tuning_report.md"
    generated_time, old_change_lines = _load_or_init_report_header(report_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_title = (
        "# FMGAD 超参数调优报告（K-only，分数平滑关闭，关闭多评分）"
        if args.k_only
        else "# FMGAD 超参数调优报告（关闭多评分机制）"
    )
    lines = [
        report_title,
        "",
        f"生成时间: {generated_time}",
        f"修改时间: {now_str}",
        f"输出目录: {output_dir}",
        f"最佳配置 yaml 目录: {BEST_CONFIGS_DIR}",
        f"单次实验 JSON 目录: {result_dir}",
        "",
        "## 最佳 AUC 汇总",
        "",
        "| 数据集 | AUC | AUC±std | AP |",
        "|--------|-----|---------|-----|",
    ]
    for ds in datasets:
        b = best_per_dataset.get(ds, {})
        if "error" in b:
            lines.append(f"| {ds} | - | - | - |")
        else:
            lines.append(f"| {ds} | {b['auc_mean']:.4f} | ±{b.get('auc_std', 0):.4f} | {b.get('ap_mean', 0):.4f} |")
    lines.extend([
        "",
        "## 各数据集最佳参数详情",
        "",
    ])
    for ds in datasets:
        b = best_per_dataset.get(ds, {})
        lines.append(f"### {ds}")
        lines.append("")
        lines.append(f"- **本段落修改时间**: {now_str}")
        if "error" in b:
            lines.append(f"- **错误**: {b['error']}")
        else:
            lines.append(f"- **AUC**: {b['auc_mean']:.4f} ± {b.get('auc_std', 0):.4f}")
            lines.append(f"- **AP**: {b.get('ap_mean', 0):.4f}")
            lines.append("- **最佳配置**:")
            for k, v in b["config"].items():
                lines.append(f"  - `{k}`: `{v}`")
        lines.append("")

    lines.extend([
        "## 修改记录",
        "",
    ])
    # 保留历史修改记录，并追加本次修改
    for ln in old_change_lines:
        lines.append(ln)
    lines.append(f"- 修改时间: {now_str}（重新生成报告并写入本次结果）")
    lines.extend([
        "",
        "## 搜索空间",
        "",
        "```json",
        json.dumps(space, indent=2, ensure_ascii=False),
        "```",
        "",
    ])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 60)
    print("调参完成！报告已保存至:", report_path)
    print("=" * 60)
    for ds, b in best_per_dataset.items():
        if "error" not in b:
            print(f"  {ds}: AUC={b['auc_mean']:.4f}")
        else:
            print(f"  {ds}: ERROR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
