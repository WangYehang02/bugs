#!/usr/bin/env python3
"""
1) 补跑旧 manifest #43–#50（enron top10 中未跑完的后 8 条）
2) 复现 20260327 Disney 单次 AUC>0.8 的前 10 条 run（full_config 写临时 YAML）

输出：/home/yehang/补跑复现报告.md + /mnt/yehang/repro_backfill_*.log（由 shell 重定向）
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

FMGAD_ROOT = Path(__file__).resolve().parent
PY = Path("/home/yehang/miniconda3/envs/fmgad/bin/python")
OLD_MANIFEST = Path("/home/yehang/fmgad_repro_top10_singleauc_20260325/manifest.json")
TUNE_20260327 = Path("/mnt/yehang/fmgad_refined_tune_20260327_165203/tuning_runs.json")
OUT_MD = Path("/home/yehang/补跑复现报告.md")
WORKDIR = Path("/mnt/yehang/repro_backfill_work")
TOL_RELAX = 2e-3
DISNEY_FLOOR = 0.8


def _run_one(yaml_path: Path, seed: int, record_auc: float | None) -> tuple[float | None, str | None]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        rf = Path(tf.name)
    cmd = [
        str(PY),
        str(FMGAD_ROOT / "main_train.py"),
        "--config",
        str(yaml_path),
        "--device",
        "0",
        "--seed",
        str(seed),
        "--num_trial",
        "1",
        "--result-file",
        str(rf),
    ]
    err = None
    try:
        proc = subprocess.run(cmd, cwd=str(FMGAD_ROOT), capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-1500:]
    except subprocess.TimeoutExpired:
        err = "timeout 3600s"
    na = None
    if rf.exists():
        try:
            with open(rf, "r", encoding="utf-8") as f:
                o = json.load(f)
            na = float(o.get("auc", o.get("auc_mean", 0.0)))
        except Exception as e:  # noqa: BLE001
            err = str(e)
    rf.unlink(missing_ok=True)
    return na, err


def main() -> int:
    print("run_backfill_repro_batch: start", flush=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    ydir = WORKDIR / "disney_yamls"
    ydir.mkdir(exist_ok=True)

    lines = [
        "# 补跑与 Disney（AUC>0.8）复现校验",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Python：`{PY}`",
        f"- 宽松数值一致：|ΔAUC| ≤ {TOL_RELAX}",
        f"- Disney 目标：重跑后 AUC 仍 ≥ {DISNEY_FLOOR}（不追求与记录逐位相同）",
        "",
    ]

    # ----- Part A: enron manifest 43–50 -----
    with open(OLD_MANIFEST, "r", encoding="utf-8") as f:
        old_meta = json.load(f)
    tail = old_meta["entries"][42:50]
    lines.append("## A. 旧 top10 manifest 补跑（enron rank3–10，原 #43–#50）")
    lines.append("")
    lines.append("| # | rank | seed | cfg_id | 记录 AUC | 重跑 AUC | Δ | 宽松一致 |")
    lines.append("|---:|---:|---:|:---|---:|---:|---:|:---|")

    for j, e in enumerate(tail, 43):
        yp = Path(e["yaml"])
        if not yp.is_file():
            lines.append(f"| {j} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | {e['auc']:.6f} | — | — | 缺 YAML |")
            continue
        na, err = _run_one(yp, int(e["seed"]), float(e["auc"]))
        if na is None:
            lines.append(f"| {j} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | {e['auc']:.6f} | — | — | 失败 |")
            if err:
                lines.append(f"\n```\n{err[:800]}\n```\n")
            print(f"[A {j}] FAIL", flush=True)
            continue
        diff = na - float(e["auc"])
        ok = abs(diff) <= TOL_RELAX
        lines.append(
            f"| {j} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | {e['auc']:.6f} | {na:.6f} | {diff:+.6f} | {'是' if ok else '否'} |"
        )
        print(f"[A {j}] auc {e['auc']:.4f} -> {na:.4f} relaxed={ok}", flush=True)

    lines.append("")

    # ----- Part B: Disney auc > 0.8 top 10 -----
    with open(TUNE_20260327, "r", encoding="utf-8") as f:
        runs = json.load(f)
    dis = [
        r
        for r in runs
        if r.get("dataset") == "disney"
        and r.get("auc") is not None
        and "error" not in r
        and float(r["auc"]) > DISNEY_FLOOR
    ]
    dis.sort(key=lambda x: -float(x["auc"]))
    top10 = dis[:10]

    lines.append("## B. Disney（20260327）单次 AUC>0.8 前十名重跑")
    lines.append("")
    lines.append("| i | 记录 AUC | 重跑 AUC | Δ | 仍≥0.8 | 宽松|Δ|≤2e-3 | cfg_id | seed | YAML |")
    lines.append("|---:|---:|---:|---:|:---|:---|:---|---:|:---|")

    for i, r in enumerate(top10, 1):
        fc = r.get("full_config")
        if not fc:
            lines.append(f"| {i} | {r['auc']:.6f} | — | — | — | — | `{r['cfg_id']}` | {r['seed']} | 无 full_config |")
            print(f"[B {i}] no full_config", flush=True)
            continue
        cfg = dict(fc)
        cfg["dataset"] = "disney"
        cfg["exp_tag"] = f"repro_disney_top{i}_{r['cfg_id']}_s{r['seed']}"
        yp = ydir / f"disney_top{i:02d}_{r['cfg_id']}_seed{r['seed']}.yaml"
        with open(yp, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        rec = float(r["auc"])
        na, err = _run_one(yp, int(r["seed"]), rec)
        if na is None:
            lines.append(f"| {i} | {rec:.6f} | — | — | — | — | `{r['cfg_id']}` | {r['seed']} | `{yp}` |")
            if err:
                lines.append(f"\n```\n{err[:600]}\n```\n")
            print(f"[B {i}] FAIL", flush=True)
            continue
        diff = na - rec
        ge08 = na >= DISNEY_FLOOR
        num_ok = abs(diff) <= TOL_RELAX
        lines.append(
            f"| {i} | {rec:.6f} | {na:.6f} | {diff:+.6f} | {'是' if ge08 else '否'} | "
            f"{'是' if num_ok else '否'} | `{r['cfg_id']}` | {r['seed']} | `{yp}` |"
        )
        print(f"[B {i}] {rec:.4f}->{na:.4f} >=0.8={ge08} relaxed={num_ok}", flush=True)

    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- A 部分使用此前导出的 enron YAML（20260325 口径）。")
    lines.append("- B 部分使用 20260327 `tuning_runs.json` 中的 `full_config`，与当时训练一致。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT_MD, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
