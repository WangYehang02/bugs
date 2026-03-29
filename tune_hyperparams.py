#!/usr/bin/env python3
"""
FMGAD 超参数调优脚本
- 支持 5 个数据集：weibo, reddit, disney, books, enron
- 固定项与搜索空间与 run_tune_refined.py 共用 tuning_search_space.py，避免两脚本结论不一致
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
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from tuning_search_space import get_fixed_overrides, get_reduced_search_space, get_refined_search_space

# 项目根目录
FMGAD_ROOT = Path(__file__).resolve().parent
FINAL_DIR = Path("/mnt/yehang/final")
REPORTS_DIR = FINAL_DIR  # 报告保存到 ~/final
BEST_CONFIGS_DIR = FINAL_DIR / "best_configs"  # 最佳配置 yaml 保存目录
LOGS_DIR = FMGAD_ROOT / "logs"
CONFIGS_DIR = FMGAD_ROOT / "configs"

# 5 个目标数据集
DATASETS = ["weibo", "reddit", "disney", "books", "enron"]

# 默认每组数据集最多尝试的配置数（随机采样，避免全网格爆炸）
DEFAULT_MAX_CONFIGS = 128  # 调参更详细


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


def _load_base_config(dataset: str) -> dict:
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
) -> dict:
    """
    运行单次实验，返回 {config, auc_mean, auc_std, ...} 或 {error: str}
    """
    base = _load_base_config(dataset)
    cfg = copy.deepcopy(base)
    cfg.update(get_fixed_overrides(dataset))
    cfg.update(config_overrides)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(FMGAD_ROOT)
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        tmp_config = f.name

    result_file = None
    if result_dir:
        result_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = hash(json.dumps(cfg, sort_keys=True)) % (10 ** 8)
        result_file = result_dir / f"{dataset}_{cfg_hash}.json"

    try:
        cmd = [
            sys.executable,
            str(FMGAD_ROOT / "main_train.py"),
            "--config", tmp_config,
            "--device", str(device),
            "--seed", str(seed),
        ]
        if result_file:
            cmd.extend(["--result-file", str(result_file)])

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

        if proc.returncode != 0:
            return {
                "config": config_overrides,
                "error": proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:],
                "auc_mean": 0.0,
            }

        if result_file and result_file.exists():
            with open(result_file, "r") as f:
                out = json.load(f)
            return {
                "config": config_overrides,
                "auc_mean": out.get("auc_mean", 0.0),
                "auc_std": out.get("auc_std", 0.0),
                "ap_mean": out.get("ap_mean", 0.0),
            }
        else:
            # 解析 stdout 中的 AUC（备用）
            return {
                "config": config_overrides,
                "auc_mean": 0.0,
                "error": "No result file",
            }
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_config)
        except Exception:
            pass
        return {"config": config_overrides, "error": "Timeout", "auc_mean": 0.0}
    except Exception as e:
        try:
            os.unlink(tmp_config)
        except Exception:
            pass
        return {
            "config": config_overrides,
            "error": str(e) + "\n" + traceback.format_exc(),
            "auc_mean": 0.0,
        }


def _tune_dataset(args):
    """
    对单个数据集进行调参，在指定 GPU 上顺序运行所有配置。
    args: (dataset, device, search_space, seed, result_dir, max_configs)
    config_overrides 仅含网格项；固定项由 get_fixed_overrides 在 _run_single_experiment 中合并。
    """
    dataset, device, space, seed, result_dir, max_configs = args
    configs = _sample_configs(space, max_configs, seed)
    results = []
    for i, cfg in enumerate(configs):
        print(f"[{dataset}] GPU{device} config {i+1}/{len(configs)}", flush=True)
        r = _run_single_experiment(
            dataset=dataset,
            config_overrides=cfg,
            device=device,
            seed=seed,
            result_dir=result_dir,
        )
        r["dataset"] = dataset
        results.append(r)
        if "error" in r:
            print(f"  -> ERROR: {r['error'][:200]}", flush=True)
        else:
            print(f"  -> AUC: {r['auc_mean']:.4f}", flush=True)
    return dataset, results


def main():
    parser = argparse.ArgumentParser(description="FMGAD 超参数调优")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="要调参的数据集列表",
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
        default=5,
        help="并行数据集数量（每个数据集一个进程）",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=DEFAULT_MAX_CONFIGS,
        help=f"每个数据集最多尝试的配置数（默认 {DEFAULT_MAX_CONFIGS}，超出则随机采样）",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("/mnt/yehang") / f"fmgad_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in args.datasets if d in DATASETS]
    if not datasets:
        datasets = DATASETS

    space_fn = get_reduced_search_space if args.reduced else get_refined_search_space
    per_dataset_space = {ds: space_fn(ds) for ds in datasets}

    # 保存搜索空间（含每数据集固定项说明）
    with open(output_dir / "search_space.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_dataset": {
                    ds: {
                        "fixed_overrides": get_fixed_overrides(ds),
                        "grid": per_dataset_space[ds],
                    }
                    for ds in datasets
                },
                "reduced": args.reduced,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 为每个数据集分配 GPU
    max_configs = args.max_configs
    tasks = []
    for i, ds in enumerate(datasets):
        gpu = args.gpus[i % len(args.gpus)]
        tasks.append((ds, gpu, per_dataset_space[ds], args.seed, output_dir, max_configs))

    print("=" * 60)
    print("FMGAD 超参数调优")
    print("数据集:", datasets)
    print("GPU:", [t[1] for t in tasks])
    print("输出目录:", output_dir)
    print("=" * 60, flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        futures = {ex.submit(_tune_dataset, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                ds, results = fut.result()
                all_results[ds] = results
                print(f"[DONE] {ds}: {len(results)} configs", flush=True)
            except Exception as e:
                print(f"[FAIL] {ds}: {e}", flush=True)
                all_results[ds] = []

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

    # 保存最佳配置 yaml 到 ~/final/best_configs/
    BEST_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        b = best_per_dataset.get(ds, {})
        if "error" not in b and b.get("config"):
            base = _load_base_config(ds)
            full_cfg = copy.deepcopy(base)
            full_cfg.update(get_fixed_overrides(ds))
            full_cfg.update(b["config"])
            yaml_path = BEST_CONFIGS_DIR / f"{ds}_best_tuned.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(full_cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"  已保存最佳配置: {yaml_path}", flush=True)

    # 生成 Markdown 报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "FMGAD_tuning_report.md"
    lines = [
        "# FMGAD 超参数调优报告（关闭多评分机制）",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"输出目录: {output_dir}",
        f"最佳配置 yaml 目录: {BEST_CONFIGS_DIR}",
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
        "## 搜索空间（每数据集固定项 + 网格，见 tuning_search_space.py）",
        "",
        "```json",
        json.dumps(per_dataset_space, indent=2, ensure_ascii=False),
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
