"""JSON-configured command line entry point."""
import argparse
import json
import logging
from pathlib import Path
from ..data.loaders import load_dataset
from ..evaluation.pipeline import ExperimentConfig
from ..evaluation.cross_validation import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/binary.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--estimators", nargs="+")
    parser.add_argument("--learners", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--counterfactual-folds", type=int)
    parser.add_argument("--nuisance-folds", type=int)
    parser.add_argument("--n-samples", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spec = json.loads(Path(args.config).read_text())
    settings = spec.get("experiment", {})
    for key in [
        "output_dir",
        "estimators",
        "learners",
        "variants",
        "counterfactual_folds",
        "nuisance_folds",
    ]:
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.n_samples is not None:
        if spec["dataset"].get("name") != "synthetic":
            parser.error("--n-samples is only for synthetic data")
        spec["dataset"]["n_samples"] = args.n_samples
    cfg = ExperimentConfig(**settings)
    bundle = load_dataset(spec["dataset"], cfg.seed)
    mode = spec.get("mode", "binary")
    if mode == "binary":
        run_experiment(bundle, cfg)
    elif mode in ("multi_general", "multi_reproduction"):
        from .multi_treatment import run_multi

        run_multi(bundle, cfg, mode, spec.get("advanced", {}))
    elif mode in ("continuous_reproduction", "continuous_general"):
        from .continuous_treatment import run_continuous

        run_continuous(bundle, cfg, mode, spec.get("advanced", {}))
    else:
        parser.error(f"Unknown mode {mode}")
if __name__ == "__main__":
    main()
