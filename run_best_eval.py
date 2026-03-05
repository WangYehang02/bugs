#!/usr/bin/env python3
"""
用各数据集最优配置，多 seed 跑多遍，多卡并行，记录最佳 AUC/AP 与 seed，生成报告。
用法: conda activate fmgad && python run_best_eval.py --gpus 0 1 2 3 4 --seeds 42 66 123 256 512
"""
import os
import sys
import json
import subprocess
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

FMGAD_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "/home/yehang/reports"))
CONFIGS_DIR = FMGAD_ROOT / "configs"
DATASETS = ["weibo", "reddit", "disney", "books", "enron"]
DEFAULT_SEEDS = [42, 66, 123, 256, 512]
DEFAULT_GPUS = [0, 1, 2, 3, 4]


def _config_path(dataset: str, config_dir: Path = None) -> Path:
    """优先使用 config_dir 下的 {dataset}_best_tuned.yaml，否则用 configs/{dataset}_best.yaml 或 .yaml"""
    if config_dir is not None:
        p = config_dir / f"{dataset}_best_tuned.yaml"
        if p.exists():
            return p
    best = CONFIGS_DIR / f"{dataset}_best.yaml"
    if best.exists():
        return best
    return CONFIGS_DIR / f"{dataset}.yaml"


def _run_one_seed(dataset: str, config_path: Path, gpu: int, seed: int, result_dir: Path, num_trial: int = 1) -> dict:
    """跑单次（单 dataset + 单 seed），返回 {seed, auc_mean, ap_mean} 或 {seed, error}。"""
    result_file = result_dir / f"{dataset}_seed{seed}.json"
    cmd = [
        sys.executable,
        str(FMGAD_ROOT / "main_train.py"),
        "--config", str(config_path),
        "--device", str(gpu),
        "--seed", str(seed),
        "--result-file", str(result_file),
        "--num_trial", str(num_trial),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(FMGAD_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if proc.returncode != 0:
            return {"seed": seed, "error": (proc.stderr or proc.stdout or "")[-800:]}
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                out = json.load(f)
            return {
                "seed": seed,
                "auc_mean": float(out.get("auc_mean", 0.0)),
                "ap_mean": float(out.get("ap_mean", 0.0)),
            }
        return {"seed": seed, "error": "No result file"}
    except subprocess.TimeoutExpired:
        return {"seed": seed, "error": "Timeout"}
    except Exception as e:
        return {"seed": seed, "error": str(e)}


def _worker_one_dataset(args_tuple):
    """单数据集：在指定 GPU 上跑所有 seeds，返回 (dataset, runs, best_auc, best_ap, best_seed)。"""
    dataset, gpu, seeds, result_dir, num_trial, config_dir = args_tuple
    config_path = _config_path(dataset, config_dir)
    if not config_path.exists():
        return (dataset, [], 0.0, 0.0, None, f"Config not found: {config_path}")
    runs = []
    for seed in seeds:
        r = _run_one_seed(dataset, config_path, gpu, seed, result_dir, num_trial)
        runs.append(r)
    valid = [x for x in runs if "auc_mean" in x and x.get("auc_mean", 0) > 0]
    if not valid:
        err = runs[0].get("error", "No valid run") if runs else "No runs"
        return (dataset, runs, 0.0, 0.0, None, err)
    best = max(valid, key=lambda x: x["auc_mean"])
    return (dataset, runs, best["auc_mean"], best["ap_mean"], best["seed"], None)


def main():
    parser = argparse.ArgumentParser(description="最优配置多 seed 评估，多卡并行，生成报告")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="数据集列表")
    parser.add_argument("--gpus", nargs="+", type=int, default=DEFAULT_GPUS, help="GPU 列表，与数据集一一对应并行")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="随机种子列表，每数据集每个 seed 跑一遍")
    parser.add_argument("--num-trial", type=int, default=1, help="每次运行的 num_trial")
    parser.add_argument("--output-dir", type=str, default=None, help="结果目录，默认 logs/best_eval_YYYYMMDD_HHMMSS")
    parser.add_argument("--report", type=str, default=None, help="报告路径，默认 reports/FMGAD_best_eval_report_YYYYMMDD_HHMMSS.md")
    parser.add_argument("--config-dir", type=str, default=None, help="最佳配置目录，使用 {dataset}_best_tuned.yaml")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir = Path(args.config_dir) if args.config_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else FMGAD_ROOT / "logs" / f"best_eval_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else REPORTS_DIR / f"FMGAD_best_eval_report_{stamp}.md"

    datasets = [d for d in args.datasets if _config_path(d, config_dir).exists()]
    if not datasets:
        print("No dataset configs found.", file=sys.stderr)
        return 1

    # 每个数据集分配一个 GPU，并行跑
    max_workers = min(len(datasets), len(args.gpus))
    tasks = []
    for i, ds in enumerate(datasets):
        gpu = args.gpus[i % len(args.gpus)]
        tasks.append((ds, gpu, args.seeds, output_dir, args.num_trial, config_dir))

    print("Best-eval: {} datasets x {} seeds = {} runs ({} GPUs in parallel)".format(
        len(datasets), len(args.seeds), len(datasets) * len(args.seeds), max_workers))
    print("Output dir:", output_dir)
    print("Report:", report_path)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker_one_dataset, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            ds = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                dataset, runs, best_auc, best_ap, best_seed, err = res
                if err:
                    print("[{}] failed: {}".format(dataset, err), flush=True)
                else:
                    print("[{}] best AUC={:.4f} AP={:.4f} seed={}".format(dataset, best_auc, best_ap, best_seed), flush=True)
            except Exception as e:
                print("[{}] exception: {}".format(ds, e), flush=True)
                results.append((ds, [], 0.0, 0.0, None, str(e)))

    # 按数据集名排序
    results.sort(key=lambda x: (x[0],))

    # 写 MD 报告
    lines = [
        "# FMGAD 最优配置多 seed 评估报告",
        "",
        "生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "结果目录: " + str(output_dir),
        "配置: 使用最佳配置，seeds=42,66,123,256,512，每数据集各跑 5 遍。",
        "",
        "## 各数据集性能（mean±std / 最佳）",
        "",
        "| 数据集 | AUC mean±std | 最佳 AUC | 最佳 seed |",
        "|--------|--------------|----------|-----------|",
    ]
    for r in results:
        dataset, runs, best_auc, best_ap, best_seed, err = r
        if err:
            lines.append("| {} | - | - | - ({}) |".format(dataset, err[:40]))
        else:
            aucs = [x["auc_mean"] for x in runs if "auc_mean" in x and x.get("auc_mean", 0) > 0]
            if len(aucs) >= 2:
                auc_mean, auc_std = statistics.mean(aucs), statistics.stdev(aucs)
                lines.append("| {} | {:.4f}±{:.4f} | {:.4f} | {} |".format(dataset, auc_mean, auc_std, best_auc, best_seed))
            else:
                lines.append("| {} | {:.4f} | {:.4f} | {} |".format(dataset, best_auc, best_auc, best_seed))

    lines.extend([
        "",
        "## 各 seed 明细（AUC / AP）",
        "",
    ])
    for r in results:
        dataset, runs, best_auc, best_ap, best_seed, err = r
        lines.append("### " + dataset)
        lines.append("")
        if not runs:
            lines.append("- 无有效运行。")
        else:
            for x in runs:
                if "auc_mean" in x:
                    lines.append("- seed {}: AUC={:.4f}, AP={:.4f}".format(x["seed"], x["auc_mean"], x["ap_mean"]))
                else:
                    lines.append("- seed {}: 失败 ({})".format(x["seed"], (x.get("error") or "")[:60]))
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Report written to:", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
