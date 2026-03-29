#!/usr/bin/env python3
"""
生成 ~/复现配置：保证五个数据集各自至少 10 个 YAML（按目录合计）

- 50 个：20260325 全局 fixed 下，各数据集 cfg 多 seed 均值 **前十**（5×10）
- 15 个：20260327 Disney 单次 AUC>0.8（full_config，现有数据共 15 条）
- 10+10：20260327 books / enron 的 cfg 均值 **前十**

合计 85。各集个数：weibo 10、reddit 10、disney 10+15=25、books 10+10=20、enron 10+10=20。
"""
from __future__ import annotations

import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from tuning_search_space import get_fixed_overrides

FMGAD_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FMGAD_ROOT / "configs"
OUT_ROOT = Path.home() / "复现配置"

RUN_25 = Path("/mnt/yehang/fmgad_refined_tune_20260325_125628")
SEARCH_25 = RUN_25 / "search_space_refined.json"
RUNS_25 = RUN_25 / "tuning_runs.json"

RUN_27 = Path("/mnt/yehang/fmgad_refined_tune_20260327_165203")
RUNS_27 = RUN_27 / "tuning_runs.json"

DATASETS_25 = ["weibo", "reddit", "disney", "books", "enron"]
TOP_PER_DS_25 = 10
TOP_BOOKS_ENRON_27 = 10

OLD_DIRS = [
    "20260325_全局fixed_各数据集_cfg均值前五",
    "20260327_books_cfg均值前五",
    "20260327_enron_cfg均值前五",
]


def _config_path(dataset: str) -> Path:
    p = CONFIGS_DIR / f"{dataset}_best.yaml"
    return p if p.exists() else CONFIGS_DIR / f"{dataset}.yaml"


def _merge_25(dataset: str, grid: dict, fixed: dict) -> dict:
    with open(_config_path(dataset), "r", encoding="utf-8") as f:
        base = yaml.load(f, Loader=yaml.Loader)
    cfg = copy.deepcopy(base)
    cfg.update(fixed)
    cfg.update(grid)
    cfg["dataset"] = dataset
    return cfg


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in OLD_DIRS:
        p = OUT_ROOT / name
        if p.is_dir():
            shutil.rmtree(p)

    d25 = OUT_ROOT / "20260325_全局fixed_各数据集_cfg均值前十"
    d27d = OUT_ROOT / "20260327_Disney_单次AUC大于0.8"
    d27b = OUT_ROOT / "20260327_books_cfg均值前十"
    d27e = OUT_ROOT / "20260327_enron_cfg均值前十"
    for d in (d25, d27d, d27b, d27e):
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    with open(SEARCH_25, "r", encoding="utf-8") as f:
        fixed_25 = json.load(f)["fixed_overrides"]
    with open(RUNS_25, "r", encoding="utf-8") as f:
        runs25 = json.load(f)
    valid25 = [r for r in runs25 if "error" not in r and r.get("auc") is not None]
    g25 = defaultdict(list)
    for r in valid25:
        g25[(r["dataset"], r["cfg_id"])].append(r)
    rows25 = []
    for (ds, cid), items in g25.items():
        aucs = [x["auc"] for x in items]
        rows25.append(
            {
                "dataset": ds,
                "cfg_id": cid,
                "auc_mean": sum(aucs) / len(aucs),
                "grid": items[0]["config"],
                "n_seeds": len(items),
            }
        )

    idx = 0
    for ds in DATASETS_25:
        sub = [x for x in rows25 if x["dataset"] == ds]
        sub.sort(key=lambda x: -x["auc_mean"])
        for rank, row in enumerate(sub[:TOP_PER_DS_25], 1):
            idx += 1
            cfg = _merge_25(ds, row["grid"], fixed_25)
            cfg["exp_tag"] = f"repro25_{ds}_meanrank{rank}_{row['cfg_id']}"
            fn = f"{idx:02d}_{ds}_meanrank{rank}_{row['cfg_id']}_aucmean{row['auc_mean']:.6f}.yaml"
            yp = d25 / fn
            with open(yp, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            manifest.append(
                {
                    "id": idx,
                    "protocol": "20260325_global_fixed",
                    "dataset": ds,
                    "note": "各数据集 cfg 多 seed 均值排名前十；Disney 在此协议下约 0.6 档",
                    "auc_mean_recorded": row["auc_mean"],
                    "cfg_id": row["cfg_id"],
                    "seeds_in_tune": row["n_seeds"],
                    "yaml": str(yp),
                }
            )

    with open(RUNS_27, "r", encoding="utf-8") as f:
        runs27 = json.load(f)

    dis = [
        r
        for r in runs27
        if r.get("dataset") == "disney"
        and r.get("auc") is not None
        and "error" not in r
        and float(r["auc"]) > 0.8
    ]
    dis.sort(key=lambda x: -float(x["auc"]))
    for rank, r in enumerate(dis, 1):
        fc = r.get("full_config")
        if not fc:
            continue
        idx += 1
        cfg = copy.deepcopy(dict(fc))
        cfg["dataset"] = "disney"
        cfg["exp_tag"] = f"repro27_disney_aucgt08_rank{rank}_{r['cfg_id']}_s{r['seed']}"
        fn = f"{idx:02d}_disney_rank{rank}_seed{r['seed']}_{r['cfg_id']}_auc{r['auc']:.6f}.yaml"
        yp = d27d / fn
        with open(yp, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        manifest.append(
            {
                "id": idx,
                "protocol": "20260327_dataset_specific",
                "dataset": "disney",
                "note": "单次 AUC>0.8（按 AUC 降序）；按数据集 fixed + 搜 weight 等",
                "auc_single_recorded": float(r["auc"]),
                "seed": r["seed"],
                "cfg_id": r["cfg_id"],
                "yaml": str(yp),
            }
        )

    def topk_mean(dataset: str, out_sub: Path, tag: str, k: int):
        nonlocal idx
        g = defaultdict(list)
        for r in runs27:
            if r.get("dataset") != dataset or "error" in r or r.get("auc") is None:
                continue
            g[r["cfg_id"]].append(r)
        rows = []
        for cid, items in g.items():
            aucs = [x["auc"] for x in items]
            rows.append(
                {
                    "cfg_id": cid,
                    "auc_mean": sum(aucs) / len(aucs),
                    "grid": items[0].get("config") or {},
                    "full": items[0].get("full_config"),
                    "n_seeds": len(items),
                }
            )
        rows.sort(key=lambda x: -x["auc_mean"])
        for rank, row in enumerate(rows[:k], 1):
            idx += 1
            if row["full"]:
                cfg = copy.deepcopy(dict(row["full"]))
            else:
                fo = get_fixed_overrides(dataset)
                with open(_config_path(dataset), "r", encoding="utf-8") as f:
                    base = yaml.load(f, Loader=yaml.Loader)
                cfg = copy.deepcopy(base)
                cfg.update(fo)
                cfg.update(row["grid"])
            cfg["dataset"] = dataset
            cfg["exp_tag"] = f"repro27_{dataset}_meanrank{rank}_{row['cfg_id']}"
            fn = f"{idx:02d}_{dataset}_meanrank{rank}_{row['cfg_id']}_aucmean{row['auc_mean']:.6f}.yaml"
            yp = out_sub / fn
            with open(yp, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            manifest.append(
                {
                    "id": idx,
                    "protocol": "20260327_dataset_specific",
                    "dataset": dataset,
                    "note": tag,
                    "auc_mean_recorded": row["auc_mean"],
                    "cfg_id": row["cfg_id"],
                    "seeds_in_tune": row["n_seeds"],
                    "yaml": str(yp),
                }
            )

    topk_mean("books", d27b, "books cfg 多 seed 均值前十", TOP_BOOKS_ENRON_27)
    topk_mean("enron", d27e, "enron cfg 多 seed 均值前十", TOP_BOOKS_ENRON_27)

    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n = len(manifest)
    readme = f"""# 复现配置汇总（{n} 个 YAML）

根目录：`{OUT_ROOT}`

详细说明见 **[说明.md](./说明.md)**。

## 两套协议（摘要）

| 子目录 | 数量 | 含义 |
|--------|-----:|------|
| `20260325_全局fixed_各数据集_cfg均值前十/` | 50 | 旧实验全局 fixed；**每数据集 10 个**（cfg 均值排名）。Disney 约 0.6 档。 |
| `20260327_Disney_单次AUC大于0.8/` | 15 | 新实验；Disney 单次 AUC>0.8 全部记录。 |
| `20260327_books_cfg均值前十/` | 10 | 新实验 books **均值前十**。 |
| `20260327_enron_cfg均值前十/` | 10 | 新实验 enron **均值前十**。 |

**各数据集 YAML 个数（合计）**：weibo 10、reddit 10、disney 25、books 20、enron 20（均 ≥10）。

## 训练示例

```bash
cd {FMGAD_ROOT}
python main_train.py --config <yaml> --device 0 --seed <seed> --num_trial 1
```

## 清单

`manifest.json`（{n} 条）。
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("Wrote", OUT_ROOT, "entries", n)


if __name__ == "__main__":
    main()
