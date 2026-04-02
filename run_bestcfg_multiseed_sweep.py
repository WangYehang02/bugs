#!/usr/bin/env python3
"""用 tuning_runs 里各数据集单次 AUC 最高的 full_config，在多个 seed 上复跑并写汇总 JSON + Markdown。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tuning_search_space import get_fixed_overrides

FMGAD_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FMGAD_ROOT / "configs"


def _config_path(dataset: str) -> Path:
    p_best = CONFIGS_DIR / f"{dataset}_best.yaml"
    if p_best.exists():
        return p_best
    return CONFIGS_DIR / f"{dataset}.yaml"


def _stable_cfg_id(dataset: str, cfg: dict) -> str:
    payload = json.dumps({"dataset": dataset, "cfg": cfg}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _merge_training_cfg(dataset: str, full_config: dict) -> dict:
    with open(_config_path(dataset), "r", encoding="utf-8") as f:
        base = yaml.load(f, Loader=yaml.Loader)
    cfg = copy.deepcopy(base)
    cfg.update(get_fixed_overrides(dataset))
    fc = copy.deepcopy(full_config)
    fc.pop("exp_tag", None)
    for k, v in fc.items():
        if k == "dataset":
            continue
        cfg[k] = v
    cfg["dataset"] = dataset
    if cfg.get("hid_dim") in ("", None):
        cfg["hid_dim"] = None
    if "num_trial" not in cfg or cfg["num_trial"] is None:
        cfg["num_trial"] = 1
    return cfg


def _pick_best_run(runs: list, dataset: str) -> dict:
    candidates = [r for r in runs if r.get("dataset") == dataset and r.get("auc") is not None]
    if not candidates:
        raise ValueError(f"No successful runs for dataset={dataset}")
    return max(candidates, key=lambda x: float(x["auc"]))


def _run_one(
    dataset: str,
    cfg: dict,
    seed: int,
    gpu: int,
    output_dir: Path,
    timeout_sec: int,
    num_trial: int,
) -> dict:
    tmp_cfgs = output_dir / "tmp_cfgs"
    tmp_cfgs.mkdir(parents=True, exist_ok=True)
    repro_dir = output_dir / "reproduce_cfgs"
    repro_dir.mkdir(parents=True, exist_ok=True)
    tmp_session = output_dir / "tmp_session" / f"{dataset}__multiseed__seed{seed}"
    tmp_session.mkdir(parents=True, exist_ok=True)
    xdg_one = output_dir / "xdg_cache" / f"{dataset}__multiseed__seed{seed}"
    xdg_one.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(cfg)
    cfg_id = _stable_cfg_id(dataset, cfg)
    cfg["exp_tag"] = f"{dataset}_{cfg_id}_seed{seed}_multiseed"

    repro_yaml = repro_dir / f"{dataset}__bestcfg__seed{seed}.yaml"
    with open(repro_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(tmp_cfgs), encoding="utf-8"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        tmp_cfg = f.name

    result_file = output_dir / "runs" / f"{dataset}__bestcfg__seed{seed}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
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
    env["FMGAD_MODEL_ROOT"] = str(output_dir / "model_cache")
    env["TMPDIR"] = str(tmp_session)
    env["XDG_CACHE_HOME"] = str(xdg_one)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(FMGAD_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    finally:
        try:
            os.unlink(tmp_cfg)
        except OSError:
            pass

    if proc.returncode != 0:
        return {
            "dataset": dataset,
            "seed": seed,
            "auc": None,
            "ap": None,
            "cfg_id": cfg_id,
            "error": (proc.stderr or proc.stdout or "")[-2000:],
            "result_file": str(result_file),
        }
    if not result_file.exists():
        return {
            "dataset": dataset,
            "seed": seed,
            "auc": None,
            "ap": None,
            "cfg_id": cfg_id,
            "error": "No result file",
            "result_file": str(result_file),
        }
    with open(result_file, "r", encoding="utf-8") as f:
        out = json.load(f)
    auc_val = float(out.get("auc", out.get("auc_mean", 0.0)))
    ap_val = float(out.get("ap_mean", out.get("ap", 0.0)))
    record = {
        "dataset": dataset,
        "seed": seed,
        "auc": auc_val,
        "ap": ap_val,
        "time_sec": float(out.get("time_sec", 0.0)),
        "cfg_id": cfg_id,
        "full_config": cfg,
        "source_tuning_auc": None,
        "source_tuning_seed": None,
        "result_file": str(result_file),
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def _write_markdown(
    output_dir: Path,
    datasets: list[str],
    meta: dict,
    by_dataset: dict[str, dict],
) -> None:
    lines: list[str] = []
    lines.append("# 各数据集 AUC 最高配置 × 多 Seed 复现\n\n")
    lines.append(
        f"**生成时间（UTC）**：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
    )
    lines.append(f"**输出目录**：`{output_dir}`\n\n")
    lines.append(f"**调参来源**：`{meta.get('tuning_runs_path','')}`\n\n")
    lines.append(
        "**规则**：对每个数据集，在 `tuning_runs.json` 中取 **单次 run AUC 最大** 的一条，"
        "用其 **`full_config`**（与 `configs/{dataset}_best.yaml` 合并后）在 seeds "
        f"{meta.get('seeds', [])} 上各跑一遍；`num_trial={meta.get('num_trial', 1)}`。\n\n"
    )
    lines.append("---\n\n")

    for ds in datasets:
        info = by_dataset[ds]
        lines.append(f"## {ds}\n\n")
        lines.append(
            f"- **调参时最佳单次 AUC**：{info['source_auc']:.6f}（seed={info['source_seed']}, cfg_id=`{info['source_cfg_id']}`）\n"
        )
        lines.append(f"- **本次复现实验 cfg_id**：`{info['merged_cfg_id']}`\n\n")
        lines.append("| seed | AUC | AP | time_sec | 结果 JSON |\n")
        lines.append("|------|-----|-----|----------|----------|\n")
        for row in info["rows"]:
            err = row.get("error")
            if err:
                auc_s = "FAIL"
                ap_s = "—"
                ts = "—"
            else:
                auc_s = f"{row['auc']:.6f}" if row.get("auc") is not None else "—"
                ap_s = f"{row['ap']:.6f}" if row.get("ap") is not None else "—"
                ts = f"{row.get('time_sec', 0):.1f}" if row.get("time_sec") is not None else "—"
            base = Path(row.get("result_file", "")).name
            lines.append(f"| {row['seed']} | {auc_s} | {ap_s} | {ts} | `{base}` |\n")
        if info.get("error_note"):
            lines.append(f"\n**错误摘录**：\n\n```\n{info['error_note'][:1500]}\n```\n")
        lines.append("\n**完整超参（YAML，与训练一致）**：\n\n```yaml\n")
        lines.append(yaml.dump(info["yaml_cfg"], default_flow_style=False, allow_unicode=True))
        lines.append("```\n\n---\n\n")

    (output_dir / "bestcfg_multiseed_report.md").write_text("".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tuning-runs",
        type=Path,
        default=Path("/mnt/yehang/fmgad_tune_4ds_5seed_gpu234567_20260330_150153/tuning_runs.json"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["weibo", "books", "reddit", "enron"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 66, 123, 256, 512])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-trial", type=int, default=1)
    p.add_argument("--timeout-sec", type=int, default=28800)
    p.add_argument(
        "--append-to-combined-report",
        type=Path,
        default=None,
        help="跑完后调用 merge_multiseed_into_combined_report，把本目录结果并入该总 MD",
    )
    args = p.parse_args()

    with open(args.tuning_runs, "r", encoding="utf-8") as f:
        runs = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "tuning_runs_path": str(args.tuning_runs.resolve()),
        "seeds": args.seeds,
        "num_trial": args.num_trial,
        "gpu": args.gpu,
    }
    (args.output_dir / "RUN_META.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_dataset: dict[str, dict] = {}
    all_rows: list[dict] = []

    for ds in args.datasets:
        best = _pick_best_run(runs, ds)
        full_config = best.get("full_config") or {}
        if not full_config:
            raise RuntimeError(f"{ds}: best run missing full_config")
        merged = _merge_training_cfg(ds, full_config)
        merged_cfg_id = _stable_cfg_id(ds, merged)

        rows = []
        err_note = ""
        for seed in args.seeds:
            rec = _run_one(
                ds,
                merged,
                seed,
                args.gpu,
                args.output_dir,
                args.timeout_sec,
                args.num_trial,
            )
            rec["source_tuning_auc"] = float(best["auc"])
            rec["source_tuning_seed"] = int(best["seed"])
            rec["source_tuning_cfg_id"] = best.get("cfg_id", "")
            rf = rec.get("result_file")
            if rf and rec.get("error") is None:
                Path(rf).write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            all_rows.append(rec)
            rows.append(rec)
            if rec.get("error") and not err_note:
                err_note = rec["error"]

        yaml_cfg = copy.deepcopy(merged)
        yaml_cfg["exp_tag"] = f"{ds}_{merged_cfg_id}_seed<SEED>_multiseed"

        by_dataset[ds] = {
            "source_auc": float(best["auc"]),
            "source_seed": int(best["seed"]),
            "source_cfg_id": best.get("cfg_id", ""),
            "merged_cfg_id": merged_cfg_id,
            "rows": rows,
            "yaml_cfg": yaml_cfg,
            "error_note": err_note,
        }

    (args.output_dir / "multiseed_results.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir, list(args.datasets), meta, by_dataset)
    print("Done:", args.output_dir / "bestcfg_multiseed_report.md")

    if args.append_to_combined_report:
        merge_py = FMGAD_ROOT / "merge_multiseed_into_combined_report.py"
        subprocess.run(
            [
                sys.executable,
                str(merge_py),
                "--combined",
                str(args.append_to_combined_report.resolve()),
                "--sweep-dir",
                str(args.output_dir.resolve()),
                "--tuning-runs",
                str(args.tuning_runs.resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
