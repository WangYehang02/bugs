#!/usr/bin/env python3
"""
FMGAD 精细调参（多卡并行，可复现）

特性:
- 5 数据集并行调参（任务级并行，不是按数据集串行）
- 固定项与搜索空间见 tuning_search_space.py（按数据集名精确映射，与 tune_hyperparams 一致）
- 每条运行结果 JSON 明确包含: dataset / seed / auc / ap / config
- 输出统一写入 /mnt/yehang
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from tuning_search_space import get_fixed_overrides, get_refined_search_space


FMGAD_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = FMGAD_ROOT / "configs"
DEFAULT_DATASETS = ["weibo", "reddit", "disney", "books", "enron"]
DEFAULT_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
DEFAULT_SEEDS = [42, 66, 123]


def _config_path(dataset: str) -> Path:
    p_best = CONFIGS_DIR / f"{dataset}_best.yaml"
    if p_best.exists():
        return p_best
    return CONFIGS_DIR / f"{dataset}.yaml"


def _dict_product(d):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _stable_cfg_id(dataset: str, cfg: dict) -> str:
    payload = json.dumps({"dataset": dataset, "cfg": cfg}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _json_safe(obj):
    """保证可写入 JSON（与 YAML 合并后的配置一致，便于复现）。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _run_one(task: tuple) -> dict:
    dataset, config_path, tune_cfg, seed, gpu, output_dir, timeout_sec, num_trial = task
    tmp_cfgs = Path(output_dir) / "tmp_cfgs"
    tmp_session = Path(output_dir) / "tmp_session"
    tmp_cfgs.mkdir(parents=True, exist_ok=True)
    tmp_session.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            base = yaml.load(f, Loader=yaml.Loader)
        cfg = copy.deepcopy(base)
        cfg.update(get_fixed_overrides(dataset))
        cfg.update(tune_cfg)
        cfg["dataset"] = dataset
        cfg_id = _stable_cfg_id(dataset, cfg)
        cfg["exp_tag"] = f"{dataset}_{cfg_id}_seed{seed}"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(tmp_cfgs), encoding="utf-8"
        ) as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            tmp_cfg = f.name

        result_file = output_dir / "runs" / f"{dataset}__{cfg_id}__seed{seed}.json"
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
        # 模型与进程临时文件全部落在 output_dir（默认在 /mnt/yehang）
        env["FMGAD_MODEL_ROOT"] = str(Path(output_dir) / "model_cache")
        env["TMPDIR"] = str(tmp_session)
        env["XDG_CACHE_HOME"] = str(Path(output_dir) / "xdg_cache")

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
                "seed": seed,
                "auc": None,
                "gpu": gpu,
                "cfg_id": cfg_id,
                "config": tune_cfg,
                "error": (proc.stderr or proc.stdout or "")[-1200:],
            }
        if not result_file.exists():
            return {
                "dataset": dataset,
                "seed": seed,
                "auc": None,
                "gpu": gpu,
                "cfg_id": cfg_id,
                "config": tune_cfg,
                "error": "No result file produced",
            }
        with open(result_file, "r", encoding="utf-8") as f:
            out = json.load(f)
        auc_val = float(out.get("auc", out.get("auc_mean", 0.0)))
        fixed_ov = get_fixed_overrides(dataset)
        full_record = {
            **out,
            "cfg_id": cfg_id,
            "full_config": _json_safe(cfg),
            "fixed_overrides": _json_safe(fixed_ov),
            "grid_overrides": _json_safe(tune_cfg),
            "base_config_path": str(config_path),
            "seed": int(seed),
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(_json_safe(full_record), f, indent=2, ensure_ascii=False)

        return {
            "dataset": dataset,
            "seed": seed,
            "auc": auc_val,
            "gpu": gpu,
            "cfg_id": cfg_id,
            "config": tune_cfg,
            "full_config": _json_safe(cfg),
            "fixed_overrides": _json_safe(fixed_ov),
            "grid_overrides": _json_safe(tune_cfg),
            "base_config_path": str(config_path),
            "ap": float(out.get("ap_mean", 0.0)),
            "time_sec": float(out.get("time_sec", 0.0)),
            "result_file": str(result_file),
        }
    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset,
            "seed": seed,
            "auc": None,
            "gpu": gpu,
            "config": tune_cfg,
            "error": f"Timeout ({timeout_sec}s)",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "dataset": dataset,
            "seed": seed,
            "auc": None,
            "gpu": gpu,
            "config": tune_cfg,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="FMGAD refined tuning")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gpus", nargs="+", type=int, default=DEFAULT_GPUS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-configs", type=int, default=120, help="Per-dataset sampled configs")
    parser.add_argument("--sampler-seed", type=int, default=20260324)
    parser.add_argument("--num-trial", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--output-root", type=str, default="/mnt/yehang")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"fmgad_refined_tune_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp_cfgs").mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp_session").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(out_dir / "tmp_session")

    tasks = []
    rng = random.Random(args.sampler_seed)
    run_id = 0
    for ds in args.datasets:
        cfg_path = _config_path(ds)
        if not cfg_path.exists():
            continue
        space = get_refined_search_space(ds)
        all_cfgs = list(_dict_product(space))
        cfgs = all_cfgs if len(all_cfgs) <= args.max_configs else rng.sample(all_cfgs, args.max_configs)
        for cfg in cfgs:
            for seed in args.seeds:
                gpu = args.gpus[run_id % len(args.gpus)]
                run_id += 1
                tasks.append((ds, cfg_path, cfg, seed, gpu, out_dir, args.timeout_sec, args.num_trial))

    if not tasks:
        print("No tasks to run.", flush=True)
        return 1

    with open(out_dir / "search_space_refined.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "module": "tuning_search_space.py",
                "per_dataset": {
                    ds: {
                        "fixed_overrides": get_fixed_overrides(ds),
                        "refined_search_space": get_refined_search_space(ds),
                    }
                    for ds in args.datasets
                },
                "sampler_seed": args.sampler_seed,
                "max_configs_per_dataset": args.max_configs,
                "seeds": args.seeds,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Tasks: {len(tasks)} | datasets={args.datasets} | seeds={args.seeds}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    all_runs = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        futures = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            all_runs.append(r)
            if "error" in r:
                print(f"[FAIL] {r['dataset']} seed{r['seed']}: {r['error'][:120]}", flush=True)
            else:
                print(
                    f"[OK] {r['dataset']} seed{r['seed']} cfg={r['cfg_id']} AUC={r['auc']:.4f} AP={r['ap']:.4f}",
                    flush=True,
                )

    with open(out_dir / "tuning_runs.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(all_runs), f, indent=2, ensure_ascii=False)

    # 聚合：同一数据集+cfg_id 按多 seed 均值选最优
    grouped = {}
    for r in all_runs:
        if "error" in r:
            continue
        key = (r["dataset"], r["cfg_id"])
        grouped.setdefault(key, {"dataset": r["dataset"], "cfg_id": r["cfg_id"], "config": r["config"], "runs": []})
        grouped[key]["runs"].append({"seed": r["seed"], "auc": r["auc"], "ap": r["ap"]})

    best_by_dataset = {}
    for ds in args.datasets:
        cands = []
        for (d, _), rec in grouped.items():
            if d != ds:
                continue
            if not rec["runs"]:
                continue
            aucs = [x["auc"] for x in rec["runs"]]
            aps = [x["ap"] for x in rec["runs"]]
            cands.append(
                {
                    "cfg_id": rec["cfg_id"],
                    "config": rec["config"],
                    "num_seeds": len(rec["runs"]),
                    "auc_mean": sum(aucs) / len(aucs),
                    "ap_mean": sum(aps) / len(aps),
                    "seed_runs": rec["runs"],
                }
            )
        if cands:
            best_by_dataset[ds] = max(cands, key=lambda x: x["auc_mean"])
        else:
            best_by_dataset[ds] = {"error": "No valid runs"}

    with open(out_dir / "best_by_dataset.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(best_by_dataset), f, indent=2, ensure_ascii=False)

    # 保存最佳 YAML 到 /mnt/yehang
    best_cfg_dir = out_dir / "best_configs"
    best_cfg_dir.mkdir(parents=True, exist_ok=True)
    for ds, b in best_by_dataset.items():
        if "error" in b:
            continue
        with open(_config_path(ds), "r", encoding="utf-8") as f:
            base = yaml.load(f, Loader=yaml.Loader)
        final_cfg = copy.deepcopy(base)
        final_cfg.update(get_fixed_overrides(ds))
        final_cfg.update(b["config"])
        final_cfg["dataset"] = ds
        final_cfg["exp_tag"] = f"{ds}_best_refined"
        with open(best_cfg_dir / f"{ds}_best_refined.yaml", "w", encoding="utf-8") as f:
            yaml.dump(final_cfg, f, default_flow_style=False, allow_unicode=True)

    print("Done. runs:", out_dir / "tuning_runs.json", flush=True)
    print("Done. best:", out_dir / "best_by_dataset.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

