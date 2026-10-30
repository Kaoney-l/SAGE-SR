"""Portable CSV/NPZ entry point for search and component ablations."""
import argparse
import json
from pathlib import Path

from .config import SearchConfig
from .data import load_csv, load_npz
from .estimator import SAGESRRegressor, _json_safe
from .priors import LLMController, StaticController, StructuredPrior


ABLATIONS = {
    "full": {}, "no-token-bias": {"use_token_bias": False},
    "no-film": {"use_film": False}, "uniform-mutations": {"use_semantic_mutation": False},
    "static-prior": {"update_semantic_prior": False}, "fixed-trust": {"calibrate_trust": False},
    "no-elite-inheritance": {"inherit_elites": False}, "no-evolutionary-edits": {"gp_rate": 0.0},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAGE-SR symbolic expression discovery")
    parser.add_argument("data", type=Path, help="Headered CSV or numeric NPZ file")
    parser.add_argument("--target", help="Target column for CSV input")
    parser.add_argument("--features", nargs="+", help="Optional ordered CSV input columns")
    parser.add_argument("--config", type=Path, help="SearchConfig JSON file")
    parser.add_argument("--context", type=Path, help="Task semantics as JSON")
    parser.add_argument("--controller", choices=["llm", "static"], default="llm")
    parser.add_argument("--prior", type=Path, help="StructuredPrior JSON for the explicit static controller")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/run.json"))
    args = parser.parse_args(argv)
    import torch
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.data.suffix.lower() == ".npz":
        dataset = load_npz(args.data)
    else:
        if not args.target:
            parser.error("CSV input requires --target")
        dataset = load_csv(args.data, args.target, args.features)
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    values.update(ABLATIONS[args.ablation])
    if args.seed is not None:
        values["seed"] = args.seed
    config = SearchConfig.from_dict(values)
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else {}
    if args.controller == "static":
        if not args.prior:
            parser.error("--controller static requires an explicit --prior JSON")
        controller = StaticController(StructuredPrior.from_dict(json.loads(args.prior.read_text(encoding="utf-8"))))
    else:
        controller = LLMController()
    model = SAGESRRegressor(config, controller).fit(dataset.X, dataset.y,
                    feature_names=dataset.feature_names, context=context)
    model.save(args.output)
    output = json.loads(args.output.read_text(encoding="utf-8"))
    output["controller"] = type(controller).__name__
    output["ablation"] = args.ablation
    output["test_metrics"] = {}
    for name in ["id", "ood"]:
        X, y = getattr(dataset, "X_test_" + name), getattr(dataset, "y_test_" + name)
        if X is not None:
            output["test_metrics"][name] = model.evaluate(X, y)
    args.output.write_text(json.dumps(_json_safe(output), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(model.expression_)
    print(f"Evaluations: {model.n_evaluations_}; results: {args.output}")


if __name__ == "__main__":
    main()
