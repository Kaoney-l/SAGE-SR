"""Run explicitly supplied NPZ benchmark tasks and aggregate task-level metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from sage_sr import SAGESRRegressor, SearchConfig, LLMController
from sage_sr.data import load_npz
from sage_sr.estimator import _json_safe


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path, help="JSON list of tasks: id, domain, data, optional context")
parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
args = parser.parse_args()
torch.set_num_threads(1)
tasks = json.loads(args.manifest.read_text(encoding="utf-8"))
if not isinstance(tasks, list) or not tasks:
    parser.error("Provide a nonempty task list.")
if len({task["id"] for task in tasks}) != len(tasks):
    parser.error("Task identifiers must be unique.")
base = json.loads(args.config.read_text(encoding="utf-8"))
args.output.mkdir(parents=True, exist_ok=True)
results = []
for task in tasks:
    dataset = load_npz(args.manifest.parent / task["data"])
    # Stable across task sharding and independent of the search seed.
    split_seed = int.from_bytes(hashlib.sha256(task["id"].encode()).digest()[:4], "big")
    safe_id = hashlib.sha256(task["id"].encode()).hexdigest()[:16]
    for seed in args.seeds:
        config = SearchConfig.from_dict({**base, "seed": seed, "validation_seed": split_seed})
        model = SAGESRRegressor(config, LLMController()).fit(dataset.X, dataset.y,
                    feature_names=dataset.feature_names, context=task.get("context", {}))
        metrics = {}
        for split in ["id", "ood"]:
            X, y = getattr(dataset, "X_test_" + split), getattr(dataset, "y_test_" + split)
            if X is not None:
                metrics[split] = model.evaluate(X, y)
        output = args.output / f"{safe_id}-seed{seed}.json"
        model.save(output)
        result = json.loads(output.read_text(encoding="utf-8"))
        result.update(task_id=task["id"], domain=task.get("domain", "all"), test_metrics=metrics)
        output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"domain": task.get("domain", "all"), "seed": seed, "metrics": metrics})
        print(task["id"], seed, _json_safe(metrics), flush=True)
summary = []
for domain in sorted({row["domain"] for row in results}):
    for seed in args.seeds:
        group = [row for row in results if row["domain"] == domain and row["seed"] == seed]
        for split in ["id", "ood"]:
            scores = [row["metrics"][split] for row in group if split in row["metrics"]]
            if scores:
                summary.append({"domain": domain, "seed": seed, "split": split,
                                "tasks": len(scores), "mean_nmse": float(np.mean([s["nmse"] for s in scores])),
                                "accuracy_0.01_percent": 100 * float(np.mean([s["success"] for s in scores]))})
(args.output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
