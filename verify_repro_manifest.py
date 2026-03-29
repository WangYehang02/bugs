#!/usr/bin/env python3
"""按 manifest.json 逐项跑 main_train，比对 AUC，写 /home/yehang/复现.md（每条跑完即更新）。"""
import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

FMGAD_ROOT = Path(__file__).resolve().parent
PY = Path("/home/yehang/miniconda3/envs/fmgad/bin/python")
MANIFEST = Path("/home/yehang/fmgad_repro_top10_singleauc_20260325/manifest.json")
OUT_MD = Path("/home/yehang/复现.md")
TOL_STRICT = 1e-4
TOL_RELAXED = 2e-3


def _write_header(f, manifest_path: Path, n: int, limited: bool):
    f.write("# 导出 YAML 复现校验\n\n")
    f.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- manifest：`{manifest_path.resolve()}`\n")
    f.write(f"- 代码目录：`{FMGAD_ROOT}`\n")
    f.write(f"- Python：`{PY}`\n")
    f.write("- 命令：`main_train.py --num_trial 1 --device 0`（与当时 `run_tune_refined` 默认一致），`--seed` 与 manifest 一致。\n")
    f.write(f"- 校验条数：**{n}**" + ("（`--limit` 截断）\n" if limited else "\n"))
    f.write("\n## 判定说明\n\n")
    f.write(f"- **严格一致**：|ΔAUC| ≤ {TOL_STRICT}\n")
    f.write(f"- **宽松一致**（GPU/算子常见漂移）：|ΔAUC| ≤ {TOL_RELAXED}\n")
    f.write("- 若仅宽松通过，说明配置对齐但数值有千分级浮动，写论文时可报「均值±std」而非单次点值。\n\n")
    f.write("## 结果表\n\n")
    f.write("| # | dataset | rank | seed | cfg_id | 记录 AUC | 重跑 AUC | Δ | 严格 | 宽松 |\n")
    f.write("|---:|:---|---:|---:|:---|---:|---:|---:|:---|:---|\n")
    f.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta["entries"]
    if args.limit:
        entries = entries[: args.limit]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    n_strict = 0
    n_relaxed_only = 0
    n_fail_relaxed = 0
    n_crash = 0

    with open(args.out, "w", encoding="utf-8") as out:
        _write_header(out, args.manifest, len(entries), bool(args.limit))

        err_notes = []
        for i, e in enumerate(entries, 1):
            yaml_path = Path(e["yaml"])
            if not yaml_path.is_file():
                out.write(
                    f"| {i} | {e['dataset']} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | {e['auc']:.6f} | — | — | 失败 | 失败 |\n"
                )
                out.flush()
                n_crash += 1
                continue

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                result_json = Path(tf.name)

            cmd = [
                str(PY),
                str(FMGAD_ROOT / "main_train.py"),
                "--config",
                str(yaml_path),
                "--device",
                "0",
                "--seed",
                str(e["seed"]),
                "--num_trial",
                "1",
                "--result-file",
                str(result_json),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(FMGAD_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
            except subprocess.TimeoutExpired:
                proc = None

            new_auc = None
            err = None
            if proc is None:
                err = "timeout 1800s"
            elif proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "")[-1200:]
            elif result_json.exists():
                with open(result_json, "r", encoding="utf-8") as f:
                    outj = json.load(f)
                new_auc = float(outj.get("auc", outj.get("auc_mean", 0.0)))
            else:
                err = "no result json"

            result_json.unlink(missing_ok=True)

            if new_auc is None:
                out.write(
                    f"| {i} | {e['dataset']} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | {e['auc']:.6f} | — | — | 失败 | 失败 |\n"
                )
                out.flush()
                if err:
                    err_notes.append(f"### #{i} {e['dataset']} r{e['rank']}\n```\n{err}\n```\n")
                n_crash += 1
                print(f"[{i}/{len(entries)}] FAIL", flush=True)
                continue

            diff = new_auc - float(e["auc"])
            s_ok = abs(diff) <= TOL_STRICT
            r_ok = abs(diff) <= TOL_RELAXED
            if s_ok:
                n_strict += 1
            elif r_ok:
                n_relaxed_only += 1
            else:
                n_fail_relaxed += 1

            out.write(
                f"| {i} | {e['dataset']} | {e['rank']} | {e['seed']} | `{e['cfg_id']}` | "
                f"{e['auc']:.6f} | {new_auc:.6f} | {diff:+.6f} | "
                f"{'是' if s_ok else '否'} | {'是' if r_ok else '否'} |\n"
            )
            out.flush()
            print(
                f"[{i}/{len(entries)}] {e['dataset']} r{e['rank']} strict={s_ok} relaxed={r_ok} d={diff:+.6f}",
                flush=True,
            )

        out.write("\n## 汇总\n\n")
        out.write(f"- 条目数：{len(entries)}\n")
        out.write(f"- 严格通过（|Δ|≤{TOL_STRICT}）：**{n_strict}**\n")
        out.write(f"- 仅宽松通过（{TOL_STRICT}<|Δ|≤{TOL_RELAXED}）：**{n_relaxed_only}**\n")
        out.write(f"- 宽松仍不通过（|Δ|>{TOL_RELAXED}）：**{n_fail_relaxed}**\n")
        out.write(f"- 运行失败/超时/无结果：**{n_crash}**\n")
        out.write(
            f"- 宽松意义下可复现合计：**{n_strict + n_relaxed_only}** / {len(entries)}\n"
        )
        if err_notes:
            out.write("\n## 失败日志\n\n")
            out.write("\n".join(err_notes))

    print("Wrote", args.out)
    return 0 if n_crash == 0 and n_fail_relaxed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
