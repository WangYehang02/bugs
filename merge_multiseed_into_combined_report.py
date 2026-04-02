#!/usr/bin/env python3
"""将某次 run_bestcfg_multiseed_sweep 的输出（multiseed_results.json）合并进总报告 MD。"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import yaml


def _rows_to_by_dataset(rows: list) -> tuple[list[str], dict]:
    by_ds: dict[str, list] = {}
    for r in rows:
        d = r["dataset"]
        by_ds.setdefault(d, []).append(r)
    order = sorted(by_ds.keys())
    out: dict[str, dict] = {}
    for ds in order:
        rs = sorted(by_ds[ds], key=lambda x: x["seed"])
        first = rs[0]
        yaml_cfg = copy.deepcopy(first.get("full_config") or {})
        yaml_cfg["exp_tag"] = f"{ds}_{first.get('cfg_id', '')}_seed<SEED>_multiseed"
        err_note = ""
        for row in rs:
            if row.get("error") and not err_note:
                err_note = row["error"]
        out[ds] = {
            "source_auc": float(first.get("source_tuning_auc", 0)),
            "source_seed": int(first.get("source_tuning_seed", 0)),
            "source_cfg_id": first.get("source_tuning_cfg_id", ""),
            "merged_cfg_id": first.get("cfg_id", ""),
            "rows": rs,
            "yaml_cfg": yaml_cfg,
            "error_note": err_note,
        }
    return order, out


def _dataset_sections_md(datasets: list[str], by_dataset: dict) -> str:
    lines: list[str] = []
    for ds in datasets:
        info = by_dataset[ds]
        lines.append(f"## {ds}\n\n")
        lines.append(
            f"- **调参时最佳单次 AUC**：{info['source_auc']:.6f}（seed={info['source_seed']}, "
            f"cfg_id=`{info['source_cfg_id']}`）\n"
        )
        lines.append(f"- **本次复现实验 cfg_id**：`{info['merged_cfg_id']}`\n\n")
        lines.append("| seed | AUC | AP | time_sec | 结果 JSON |\n")
        lines.append("|------|-----|-----|----------|----------|\n")
        for row in info["rows"]:
            err = row.get("error")
            if err:
                auc_s, ap_s, ts = "FAIL", "—", "—"
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
    return "".join(lines)


def _patch_header_for_yelpchi(text: str, yelpchi_tune_json: str, yelpchi_out_dir: str) -> str:
    if "**调参来源（YelpChi）**" in text:
        text = re.sub(
            r"\*\*输出目录（YelpChi）\*\*：`[^`]*`",
            f"**输出目录（YelpChi）**：`{yelpchi_out_dir}`",
            text,
            count=1,
        )
        text = re.sub(
            r"\*\*调参来源（YelpChi）\*\*：`[^`]*`",
            f"**调参来源（YelpChi）**：`{yelpchi_tune_json}`",
            text,
            count=1,
        )
        return text

    text = text.replace(
        "# 各数据集 AUC 最高配置 × 多 Seed 复现\n",
        "# 各数据集 AUC 最高配置 × 多 Seed 复现（四数据集 + YelpChi）\n",
        1,
    )
    text = text.replace("**输出目录**：", "**输出目录（四数据集）**：", 1)
    text = re.sub(
        r"(\*\*输出目录（四数据集）\*\*：`[^`]+`)\n",
        rf"\1\n\n**输出目录（YelpChi）**：`{yelpchi_out_dir}`\n",
        text,
        count=1,
    )
    text = text.replace("**调参来源**：", "**调参来源（四数据集）**：", 1)
    text = re.sub(
        r"(\*\*调参来源（四数据集）\*\*：`[^`]+`)\n",
        rf"\1\n\n**调参来源（YelpChi）**：`{yelpchi_tune_json}`\n",
        text,
        count=1,
    )
    return text


def _load_sweep_rows(sweep_dir: Path) -> list:
    results_path = sweep_dir / "multiseed_results.json"
    if results_path.is_file():
        return json.loads(results_path.read_text(encoding="utf-8"))
    run_dir = sweep_dir / "runs"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"缺少 {results_path} 且无 {run_dir}")
    rows = []
    for p in sorted(run_dir.glob("*__bestcfg__seed*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    rows.sort(key=lambda x: (str(x.get("dataset", "")), int(x.get("seed", 0))))
    if not rows:
        raise FileNotFoundError(f"缺少 {results_path} 且 {run_dir} 下无结果 JSON")
    return rows


def merge(
    combined_md: Path,
    sweep_dir: Path,
    tuning_runs_path: str | None = None,
) -> None:
    rows = _load_sweep_rows(sweep_dir)
    meta_path = sweep_dir / "RUN_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    tune_path = tuning_runs_path or meta.get("tuning_runs_path", "")
    yelpchi_out = str(sweep_dir.resolve())

    datasets, by_dataset = _rows_to_by_dataset(rows)
    new_sections = _dataset_sections_md(datasets, by_dataset)

    text = combined_md.read_text(encoding="utf-8")
    for ds in datasets:
        text = re.sub(rf"\n## {re.escape(ds)}\n[\s\S]*?(?=\n## |\Z)", "\n", text, count=1)

    text = text.rstrip() + "\n"
    if "yelpchi" in datasets and tune_path:
        text = _patch_header_for_yelpchi(text, tune_path, yelpchi_out)

    if not text.endswith("\n---\n\n"):
        if text.endswith("---\n"):
            text = text.rstrip() + "\n\n"
        elif not text.endswith("\n\n"):
            text += "\n"

    text = text.rstrip() + "\n\n---\n\n" + new_sections.rstrip() + "\n"
    combined_md.write_text(text, encoding="utf-8")
    print("Updated:", combined_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, required=True, help="总报告，如 fmgad_4ds_bestcfg_multiseed_report.md")
    ap.add_argument("--sweep-dir", type=Path, required=True, help="某次 multiseed sweep 输出目录")
    ap.add_argument("--tuning-runs", type=str, default=None, help="覆盖 RUN_META 中的调参 JSON 路径（写进报告头）")
    args = ap.parse_args()
    merge(args.combined, args.sweep_dir, args.tuning_runs)


if __name__ == "__main__":
    main()
