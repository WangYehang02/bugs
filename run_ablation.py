#!/usr/bin/env python3
"""
FMGAD 消融实验脚本（多卡并行）

示例:
conda run -n fmgad python run_ablation.py \
  --datasets weibo reddit disney books enron \
  --gpus 0 1 2 3 \
  --seeds 42 66 123 \
  --config-dir /path/to/configs
"""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

FMGAD_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FMGAD_ROOT / "configs"
DEFAULT_DATASETS = ["weibo", "reddit", "disney", "books", "enron"]
DEFAULT_SEEDS = [42, 66, 123]
DEFAULT_GPUS = [0, 1, 2, 3]


def _config_path(dataset: str, config_dir: Path = None) -> Path:
    if config_dir is not None:
        tuned = config_dir / f"{dataset}_best_tuned.yaml"
        if tuned.exists():
            return tuned
    best = CONFIGS_DIR / f"{dataset}_best.yaml"
    if best.exists():
        return best
    return CONFIGS_DIR / f"{dataset}.yaml"


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _build_variants() -> dict:
    # Full 统一打开关键模块，作为消融基线
    return {
        "Full_Model": {
            "use_multi_scale_residual": True,
            "use_adaptive_residual_scale": True,
            "use_virtual_neighbors": True,
            "use_multi_score_fusion": True,
            "use_score_smoothing": True,
            "flow_t_sampling": "logit_normal",
        },
        "w_o_multi_scale_residual": {"use_multi_scale_residual": False},
        "w_o_adaptive_scale": {"use_adaptive_residual_scale": False},
        "w_o_virtual_neighbors": {"use_virtual_neighbors": False},
        "w_o_multi_score_fusion": {"use_multi_score_fusion": False},
        "w_o_score_smoothing": {"use_score_smoothing": False},
        "w_o_logit_normal_sampling": {"flow_t_sampling": "uniform"},
        "w_o_proto_guidance": {"weight": 0.0},
    }


def _run_one(task: tuple) -> dict:
    (
        dataset,
        variant_name,
        overrides,
        config_path,
        seed,
        gpu,
        result_dir,
        timeout_sec,
        num_trial,
    ) = task

    try:
        base_cfg = _load_config(config_path)
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        cfg["dataset"] = dataset
        cfg["exp_tag"] = f"{dataset}_{variant_name}_seed{seed}".replace("/", "_")

        tmp_cfgs = Path(result_dir) / "tmp_cfgs"
        tmp_session = Path(result_dir) / "tmp_session"
        tmp_cfgs.mkdir(parents=True, exist_ok=True)
        tmp_session.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(tmp_cfgs), encoding="utf-8"
        ) as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            tmp_cfg = f.name

        safe_variant = variant_name.replace("/", "_")
        result_file = result_dir / f"{dataset}__{safe_variant}__seed{seed}.json"
        cmd = [
            sys.executable,
            str(FMGAD_ROOT / "main_train.py"),
            "--config",
            tmp_cfg,
            "--device",
            str(gpu),
            "--seed",
            str(seed),
            "--result-file",
            str(result_file),
            "--num_trial",
            str(num_trial),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["FMGAD_MODEL_ROOT"] = str(Path(result_dir) / "model_cache")
        env["TMPDIR"] = str(tmp_session)
        env["XDG_CACHE_HOME"] = str(Path(result_dir) / "xdg_cache")
        proc = subprocess.run(
            cmd,
            cwd=str(FMGAD_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        os.unlink(tmp_cfg)

        if proc.returncode != 0:
            return {
                "dataset": dataset,
                "variant": variant_name,
                "seed": seed,
                "auc": None,
                "gpu": gpu,
                "error": (proc.stderr or proc.stdout or "")[-1200:],
            }
        if not result_file.exists():
            return {
                "dataset": dataset,
                "variant": variant_name,
                "seed": seed,
                "auc": None,
                "gpu": gpu,
                "error": "No result file produced",
            }
        with open(result_file, "r", encoding="utf-8") as f:
            out = json.load(f)
        auc_mean = float(out.get("auc", out.get("auc_mean", 0.0)))
        return {
            "dataset": dataset,
            "variant": variant_name,
            "seed": seed,
            "auc": auc_mean,
            "auc_mean": auc_mean,
            "ap_mean": float(out.get("ap_mean", 0.0)),
            "time_sec": float(out.get("time_sec", 0.0)),
        }
    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset,
            "variant": variant_name,
            "seed": seed,
            "auc": None,
            "gpu": gpu,
            "error": f"Timeout ({timeout_sec}s)",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "dataset": dataset,
            "variant": variant_name,
            "seed": seed,
            "auc": None,
            "gpu": gpu,
            "error": str(e),
        }


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _run_key(r: dict) -> tuple:
    return (r["dataset"], r["variant"], int(r["seed"]))


def _summarize(all_runs: list, variants: dict, datasets: list) -> dict:
    out = {"by_dataset": {}}
    for ds in datasets:
        out["by_dataset"][ds] = {}
        full_auc = None
        full_ap = None
        for vname in variants.keys():
            rs = [r for r in all_runs if r["dataset"] == ds and r["variant"] == vname and "error" not in r]
            aucs = [r["auc_mean"] for r in rs]
            aps = [r["ap_mean"] for r in rs]
            auc_mean, auc_std = _mean_std(aucs)
            ap_mean, ap_std = _mean_std(aps)
            rec = {
                "num_ok": len(rs),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "ap_mean": ap_mean,
                "ap_std": ap_std,
            }
            out["by_dataset"][ds][vname] = rec
            if vname == "Full_Model":
                full_auc, full_ap = auc_mean, ap_mean
        if full_auc is not None:
            for vname, rec in out["by_dataset"][ds].items():
                rec["delta_auc_vs_full"] = rec["auc_mean"] - full_auc
                rec["delta_ap_vs_full"] = rec["ap_mean"] - full_ap
    return out


def _write_report(report_path: Path, summary: dict, all_runs: list, variants: dict, datasets: list, output_dir: Path):
    lines = [
        "# FMGAD Ablation Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output dir: {output_dir}",
        "",
        "## 1) Variant Definitions",
        "",
    ]
    for name, cfg in variants.items():
        lines.append(f"- `{name}`: `{json.dumps(cfg, ensure_ascii=False)}`")
    lines.extend(["", "## 2) Dataset-wise Results (mean±std across seeds)", ""])

    for ds in datasets:
        lines.extend(
            [
                f"### {ds}",
                "",
                "| Variant | AUC | AP | Delta AUC vs Full | Delta AP vs Full |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        ds_map = summary["by_dataset"].get(ds, {})
        for vname in variants.keys():
            rec = ds_map.get(vname, {})
            lines.append(
                "| {v} | {auc:.4f}±{aucstd:.4f} | {ap:.4f}±{apstd:.4f} | {da:+.4f} | {dp:+.4f} |".format(
                    v=vname,
                    auc=rec.get("auc_mean", 0.0),
                    aucstd=rec.get("auc_std", 0.0),
                    ap=rec.get("ap_mean", 0.0),
                    apstd=rec.get("ap_std", 0.0),
                    da=rec.get("delta_auc_vs_full", 0.0),
                    dp=rec.get("delta_ap_vs_full", 0.0),
                )
            )
        lines.append("")

    lines.extend(["## 3) Failed Runs", ""])
    failed = [r for r in all_runs if "error" in r]
    if not failed:
        lines.append("- None")
    else:
        for r in failed:
            lines.append(
                f"- {r['dataset']} / {r['variant']} / seed {r['seed']} / gpu {r['gpu']}: {(r['error'] or '')[:200]}"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="FMGAD ablation runner")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gpus", nargs="+", type=int, default=DEFAULT_GPUS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-trial", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--max-workers", type=int, default=None, help="default=len(gpus)")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(CONFIGS_DIR),
        help="默认识别仓库内 configs/；若有 tuned yaml 可指向含 *_best_tuned.yaml 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="默认写入 /mnt/yehang/fmgad_ablation_<时间戳>",
    )
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument(
        "--retry-failed-from",
        type=str,
        default=None,
        help="Path to previous ablation_runs.json; only failed runs will be retried and merged.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("/mnt/yehang") / f"fmgad_ablation_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp_cfgs").mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp_session").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(output_dir / "tmp_session")
    report_path = Path(args.report) if args.report else output_dir / "FMGAD_ablation_report.md"
    config_dir = Path(args.config_dir) if args.config_dir else None

    variants = _build_variants()
    datasets = [d for d in args.datasets if _config_path(d, config_dir).exists()]
    if not datasets:
        print("No valid dataset configs found.", file=sys.stderr)
        return 1

    previous_runs = None
    tasks = []
    run_id = 0
    if args.retry_failed_from:
        with open(args.retry_failed_from, "r", encoding="utf-8") as f:
            previous_runs = json.load(f)
        failed_runs = [r for r in previous_runs if "error" in r]
        for r in failed_runs:
            ds = r["dataset"]
            vname = r["variant"]
            seed = int(r["seed"])
            if ds not in datasets:
                continue
            if vname not in variants:
                continue
            gpu = args.gpus[run_id % len(args.gpus)]
            run_id += 1
            cfg_path = _config_path(ds, config_dir)
            tasks.append((ds, vname, variants[vname], cfg_path, seed, gpu, output_dir, args.timeout_sec, args.num_trial))
    else:
        for ds in datasets:
            cfg_path = _config_path(ds, config_dir)
            for vname, ov in variants.items():
                for seed in args.seeds:
                    gpu = args.gpus[run_id % len(args.gpus)]
                    run_id += 1
                    tasks.append(
                        (ds, vname, ov, cfg_path, seed, gpu, output_dir, args.timeout_sec, args.num_trial)
                    )

    if not tasks:
        print("No tasks to run.", flush=True)
        return 0

    max_workers = args.max_workers if args.max_workers else len(args.gpus)
    max_workers = min(max_workers, len(tasks))
    if args.retry_failed_from:
        print(f"Retry mode: {len(tasks)} failed runs to rerun", flush=True)
    else:
        print(
            "Ablation runs: {} datasets x {} variants x {} seeds = {} total".format(
                len(datasets), len(variants), len(args.seeds), len(tasks)
            ),
            flush=True,
        )
    print("GPUs:", args.gpus, "workers:", max_workers, flush=True)
    print("Output:", output_dir, flush=True)
    print("Report:", report_path, flush=True)

    all_runs = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            ds, vn, _, _, seed, gpu, *_ = t
            try:
                r = fut.result()
                all_runs.append(r)
                if "error" in r:
                    print(f"[FAIL] {ds}/{vn}/seed{seed}/gpu{gpu}: {r['error'][:120]}", flush=True)
                else:
                    print(
                        f"[OK] {ds}/{vn}/seed{seed}/gpu{gpu}: AUC={r['auc_mean']:.4f}, AP={r['ap_mean']:.4f}",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                all_runs.append(
                    {
                        "dataset": ds,
                        "variant": vn,
                        "seed": seed,
                        "auc": None,
                        "gpu": gpu,
                        "error": f"Future exception: {e}",
                    }
                )
                print(f"[FAIL] {ds}/{vn}/seed{seed}/gpu{gpu}: {e}", flush=True)

    if previous_runs is not None:
        latest = {_run_key(r): r for r in previous_runs}
        for r in all_runs:
            latest[_run_key(r)] = r
        all_runs = list(latest.values())

    with open(output_dir / "ablation_runs.json", "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)

    summary = _summarize(all_runs, variants, datasets)
    with open(output_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_report(report_path, summary, all_runs, variants, datasets, output_dir)
    print("Done. report:", report_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
