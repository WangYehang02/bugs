#!/usr/bin/env python3
"""
从某次 run_tune_refined 输出目录导出「可复现」完整 YAML。

旧版 runs/*.json 往往只有指标，没有超参；本脚本用：
  base configs/*.yaml + search_space_refined.json 的 fixed_overrides + tuning_runs.json 里每条记录的 config

用法示例（五数据集、单次 AUC 前十）：
  python export_repro_yamls_from_tune_dir.py \\
    --tune-dir /mnt/yehang/fmgad_refined_tune_20260325_125628 \\
    --out-dir /home/yehang/fmgad_repro_top10_singleauc_20260325 \\
    --rank-by single_auc
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import yaml

FMGAD_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FMGAD_ROOT / "configs"


def _config_path(dataset: str) -> Path:
    p_best = CONFIGS_DIR / f"{dataset}_best.yaml"
    if p_best.exists():
        return p_best
    return CONFIGS_DIR / f"{dataset}.yaml"


def _load_fixed(tune_dir: Path) -> dict:
    p = tune_dir / "search_space_refined.json"
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 旧目录只有全局 fixed_overrides；新目录可能是 per_dataset
    if "fixed_overrides" in meta:
        return meta["fixed_overrides"]
    raise ValueError(f"{p} 中未找到 fixed_overrides，请检查目录是否为 refined tune 输出。")


def _merge_full_cfg(dataset: str, grid: dict, fixed: dict) -> dict:
    with open(_config_path(dataset), "r", encoding="utf-8") as f:
        base = yaml.load(f, Loader=yaml.Loader)
    cfg = copy.deepcopy(base)
    cfg.update(fixed)
    cfg.update(grid)
    cfg["dataset"] = dataset
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--rank-by",
        choices=("single_auc", "cfg_mean_auc"),
        default="single_auc",
        help="single_auc: 每数据集按单次 run 的 AUC 取前十；cfg_mean_auc: 按 cfg_id 多 seed 均值取前十（每个 cfg 导出 1 个 YAML，需自行对 seeds 各跑一次）",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["weibo", "reddit", "disney", "books", "enron"],
    )
    args = parser.parse_args()

    tune_dir = args.tune_dir.resolve()
    runs_path = tune_dir / "tuning_runs.json"
    with open(runs_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    fixed = _load_fixed(tune_dir)
    valid = [r for r in runs if "error" not in r and r.get("auc") is not None]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    if args.rank_by == "single_auc":
        for ds in args.datasets:
            sub = [r for r in valid if r["dataset"] == ds]
            sub.sort(key=lambda x: x["auc"], reverse=True)
            top = sub[:10]
            for i, r in enumerate(top, 1):
                cfg = _merge_full_cfg(ds, r["config"], fixed)
                cfg["exp_tag"] = f"{ds}_repro_rank{i:02d}_{r['cfg_id']}_seed{r['seed']}"
                fname = f"{ds}_rank{i:02d}_seed{r['seed']}_cfg{r['cfg_id']}_auc{r['auc']:.6f}.yaml"
                out_path = out_dir / fname
                with open(out_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
                manifest.append(
                    {
                        "dataset": ds,
                        "rank": i,
                        "auc": r["auc"],
                        "ap": r.get("ap"),
                        "seed": r["seed"],
                        "cfg_id": r["cfg_id"],
                        "yaml": str(out_path),
                    }
                )
    else:
        grouped = defaultdict(list)
        for r in valid:
            if r["dataset"] not in args.datasets:
                continue
            grouped[(r["dataset"], r["cfg_id"])].append(r)

        rows = []
        for (ds, cfg_id), items in grouped.items():
            aucs = [x["auc"] for x in items]
            aps = [x["ap"] for x in items]
            rows.append(
                {
                    "dataset": ds,
                    "cfg_id": cfg_id,
                    "auc_mean": sum(aucs) / len(aucs),
                    "ap_mean": sum(aps) / len(aps),
                    "seeds": sorted({x["seed"] for x in items}),
                    "config": items[0]["config"],
                }
            )

        for ds in args.datasets:
            sub = [x for x in rows if x["dataset"] == ds]
            sub.sort(key=lambda x: x["auc_mean"], reverse=True)
            top = sub[:10]
            for i, row in enumerate(top, 1):
                cfg = _merge_full_cfg(ds, row["config"], fixed)
                cfg["exp_tag"] = f"{ds}_repro_mean_top{i:02d}_{row['cfg_id']}"
                fname = f"{ds}_meanrank{i:02d}_cfg{row['cfg_id']}_aucmean{row['auc_mean']:.6f}.yaml"
                out_path = out_dir / fname
                with open(out_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
                manifest.append(
                    {
                        "dataset": ds,
                        "rank": i,
                        "auc_mean": row["auc_mean"],
                        "ap_mean": row["ap_mean"],
                        "seeds_to_run": row["seeds"],
                        "cfg_id": row["cfg_id"],
                        "yaml": str(out_path),
                    }
                )

    man_path = out_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tune_dir": str(tune_dir),
                "rank_by": args.rank_by,
                "fixed_overrides_used": fixed,
                "entries": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    readme = out_dir / "怎么复现.md"
    py = "python main_train.py"
    if args.rank_by == "single_auc":
        body = f"""# 复现说明（单次 AUC 前十）

- 这些 YAML 来自：`{tune_dir}`
- 已合并：**base 配置 + 该次实验的 fixed_overrides + 网格 config**
- 每条对应 **一个** seed；文件名里已带 `seed` 与 `auc`。

在 `{FMGAD_ROOT}` 下执行（请用你训练时的 Python 环境，例如 conda `fmgad`）：

```bash
cd {FMGAD_ROOT}
# 示例：把下面换成 manifest.json 里某条的 yaml 路径与 seed
{py} --config /path/to/某个.yaml --device 0 --seed 42 --result-file /tmp/out.json
```

`--seed` **必须与文件名中的 seed 一致**，才能对齐当时那次 run 的 AUC。
"""
    else:
        body = f"""# 复现说明（cfg 多 seed 均值前十）

- 每个 YAML 对应一个 `cfg_id`；应用 **manifest.json** 里该条的 `seeds_to_run`，**每个 seed 各跑一次**，再对 AUC 取平均，可与当时的「均值排名」对齐。
"""

    with open(readme, "w", encoding="utf-8") as f:
        f.write(body)

    print("导出完成:", out_dir)
    print("清单:", man_path)
    print("说明:", readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
