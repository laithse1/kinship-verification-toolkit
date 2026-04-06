from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinship.algorithms.classical import run_classical_verification
from kinship.algorithms.family_deep_native import run_family_deep
from kinship.algorithms.gae_native import run_gae
from kinship.algorithms.kinver import run_kinver
from kinship.configs import (
    benchmark_config_paths,
    experiment_config_paths,
    load_benchmark_config,
    load_experiment_config,
)
from kinship.registry import algorithm_names
from kinship.runner import run_benchmark, run_experiment


def _dump(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def _classical_command(args: argparse.Namespace) -> int:
    config = _inline_experiment_config(
        name=f"classical-{args.relation}-{args.method}",
        algorithm="classical",
        parameters={
            "dataset": args.dataset,
            "relation": args.relation,
            "method": args.method,
            "limit": args.limit,
        },
    )
    if args.output_root:
        run_info = run_experiment(config=config, output_root=Path(args.output_root))
        _dump(
            {
                "run_dir": run_info["run_dir"],
                "result": run_info["payload"]["result"],
            }
        )
        return 0

    result = run_classical_verification(
        relation=args.relation,
        dataset=args.dataset,
        method=args.method,
        limit=args.limit,
    )
    payload = {
        "algorithm": "classical",
        "dataset": result.dataset,
        "relation": result.relation,
        "method": result.method,
        "fold_scores": result.fold_scores,
        "mean_accuracy": result.mean_accuracy,
    }
    _dump(payload)
    return 0


def _inline_experiment_config(name: str, algorithm: str, parameters: dict) -> object:
    from kinship.configs import ExperimentConfig

    return ExperimentConfig(name=name, algorithm=algorithm, parameters=parameters)


def _kinver_payload(result) -> dict:
    return {
        "algorithm": "kinver",
        "dataset": result.dataset,
        "relation": result.relation,
        "fold_scores": result.fold_scores,
        "mean_accuracy": result.mean_accuracy,
        "beta_means": result.beta_means,
        "n_components": result.n_components,
    }


def _family_deep_command(args: argparse.Namespace) -> int:
    parameters = {
        "mode": args.mode,
        "dataset_name": args.dataset_name,
        "data_path": args.data_path,
        "model_name": args.model_name,
        "gpu": args.gpu,
        "lr": args.lr,
        "bs": args.bs,
        "num_epochs": args.num_epochs,
        "img1": args.img1,
        "img2": args.img2,
        "pair_type": args.pair_type,
        "checkpoints_dir": args.checkpoints_dir,
        "vgg_weights": args.vgg_weights,
    }
    config = _inline_experiment_config(
        name=f"family-deep-{args.dataset_name}-{args.model_name}-{args.mode}",
        algorithm="family-deep",
        parameters=parameters | {"output_dir": args.output_dir},
    )
    if args.output_root:
        run_info = run_experiment(config=config, output_root=Path(args.output_root))
        _dump(
            {
                "run_dir": run_info["run_dir"],
                "result": run_info["payload"]["result"],
            }
        )
        return 0
    _dump(run_family_deep(**parameters, output_dir=args.output_dir))
    return 0


def _gae_command(args: argparse.Namespace) -> int:
    parameters = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "variant": args.variant,
        "numfac": args.numfac,
        "nummap": args.nummap,
        "learnrate": args.learnrate,
        "numepochs": args.numepochs,
        "donorm": not args.no_norm,
        "verbose": not args.quiet,
        "random_state": args.random_state,
        "subspace_dims": args.subspace_dims,
    }
    config = _inline_experiment_config(
        name=f"gae-{args.variant}-{Path(args.input_path).stem}",
        algorithm="gae",
        parameters=parameters,
    )
    if args.output_root:
        run_info = run_experiment(config=config, output_root=Path(args.output_root))
        _dump(
            {
                "run_dir": run_info["run_dir"],
                "result": run_info["payload"]["result"],
            }
        )
        return 0
    _dump(
        {
            "algorithm": "gae",
            "variant": result.variant,
            "input_path": result.input_path,
            "output_path": result.output_path,
            "numfac": result.numfac,
            "nummap": result.nummap,
            "numepochs": result.numepochs,
            "reconstruction_error": result.reconstruction_error,
            "history": result.history,
        }
        if (result := run_gae(**parameters))
        else {}
    )
    return 0


def _list_command(args: argparse.Namespace) -> int:
    payload = {
        "algorithms": algorithm_names(),
        "experiment_configs": [path.stem for path in experiment_config_paths()],
        "benchmark_configs": [path.stem for path in benchmark_config_paths()],
    }
    _dump(payload)
    return 0


def _run_config_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    run_info = run_experiment(config, output_root=Path(args.output_root) if args.output_root else None)
    _dump(
        {
            "run_dir": run_info["run_dir"],
            "result": run_info["payload"]["result"],
        }
    )
    return 0


def _kinver_command(args: argparse.Namespace) -> int:
    config = _inline_experiment_config(
        name=f"kinver-{args.relation}",
        algorithm="kinver",
        parameters={
            "dataset": args.dataset,
            "relation": args.relation,
            "use_vggface": not args.no_vggface,
            "use_vggf": not args.no_vggf,
            "use_lbp": args.use_lbp,
            "use_hog": args.use_hog,
            "use_feature_selection": not args.no_feature_selection,
            "use_pca": not args.no_pca,
            "use_mnrml": not args.no_mnrml,
            "iterations": args.iterations,
            "knn": args.knn,
        },
    )
    if args.output_root:
        run_info = run_experiment(config=config, output_root=Path(args.output_root))
        _dump(
            {
                "run_dir": run_info["run_dir"],
                "result": run_info["payload"]["result"],
            }
        )
        return 0

    result = run_kinver(
        relation=args.relation,
        dataset=args.dataset,
        use_vggface=not args.no_vggface,
        use_vggf=not args.no_vggf,
        use_lbp=args.use_lbp,
        use_hog=args.use_hog,
        use_feature_selection=not args.no_feature_selection,
        use_pca=not args.no_pca,
        use_mnrml=not args.no_mnrml,
        iterations=args.iterations,
        knn=args.knn,
    )
    payload = _kinver_payload(result)
    _dump(payload)
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    run_info = run_benchmark(config, output_root=Path(args.output_root) if args.output_root else None)
    _dump(
        {
            "run_dir": run_info["run_dir"],
            "summary_rows": run_info["summary_rows"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kinship verification toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classical = subparsers.add_parser("classical", help="Run the classical HOG/LBP pipelines")
    classical.add_argument("--dataset", default="KinFaceW-I")
    classical.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    classical.add_argument("--method", choices=["random", "kfold", "chisq"], default="random")
    classical.add_argument("--limit", type=int, default=None)
    classical.add_argument("--output-root", default=None)
    classical.set_defaults(func=_classical_command)

    kinver = subparsers.add_parser("kinver", help="Run the KinVer metric-learning pipeline")
    kinver.add_argument("--dataset", default="KinFaceW-II")
    kinver.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    kinver.add_argument("--no-vggface", action="store_true")
    kinver.add_argument("--no-vggf", action="store_true")
    kinver.add_argument("--use-lbp", action="store_true")
    kinver.add_argument("--use-hog", action="store_true")
    kinver.add_argument("--no-feature-selection", action="store_true")
    kinver.add_argument("--no-pca", action="store_true")
    kinver.add_argument("--no-mnrml", action="store_true")
    kinver.add_argument("--iterations", type=int, default=4)
    kinver.add_argument("--knn", type=int, default=6)
    kinver.add_argument("--output-root", default=None)
    kinver.set_defaults(func=_kinver_command)

    family_deep = subparsers.add_parser(
        "family-deep",
        help="Run the native deep-learning kinship family models",
    )
    family_deep.add_argument("--mode", choices=["train", "test", "demo"], default="test")
    family_deep.add_argument("--dataset-name", choices=["fiw", "kinfacew"], default="fiw")
    family_deep.add_argument("--data-path", default=None)
    family_deep.add_argument(
        "--model-name",
        choices=[
            "kin_facenet",
            "small_face_model",
            "small_siamese_face_model",
            "vgg_multichannel",
            "vgg_siamese",
        ],
        default="kin_facenet",
    )
    family_deep.add_argument("--gpu", type=int, default=0)
    family_deep.add_argument("--lr", type=float, default=1e-3)
    family_deep.add_argument("--bs", type=int, default=40)
    family_deep.add_argument("--num-epochs", type=int, default=16)
    family_deep.add_argument("--img1", default=None)
    family_deep.add_argument("--img2", default=None)
    family_deep.add_argument("--pair-type", choices=["fd", "fs", "md", "ms"], default="ms")
    family_deep.add_argument("--output-dir", default=None)
    family_deep.add_argument("--output-root", default=None)
    family_deep.add_argument("--checkpoints-dir", default=None)
    family_deep.add_argument("--vgg-weights", default=None)
    family_deep.set_defaults(func=_family_deep_command)

    gae = subparsers.add_parser(
        "gae",
        help="Run the native GAE family feature mapper over an input .mat file",
    )
    gae.add_argument("input_path")
    gae.add_argument("--output-path", default=None)
    gae.add_argument("--variant", choices=["standard", "multiview"], default="standard")
    gae.add_argument("--numfac", type=int, default=600)
    gae.add_argument("--nummap", type=int, default=400)
    gae.add_argument("--learnrate", type=float, default=0.01)
    gae.add_argument("--numepochs", type=int, default=100)
    gae.add_argument("--no-norm", action="store_true")
    gae.add_argument("--quiet", action="store_true")
    gae.add_argument("--random-state", type=int, default=1)
    gae.add_argument("--subspace-dims", type=int, default=2)
    gae.add_argument("--output-root", default=None)
    gae.set_defaults(func=_gae_command)

    list_cmd = subparsers.add_parser("list", help="List algorithms and available configs")
    list_cmd.set_defaults(func=_list_command)

    run_config = subparsers.add_parser(
        "run-config",
        help="Run an experiment from a TOML config under configs/experiments",
    )
    run_config.add_argument("config")
    run_config.add_argument("--output-root", default=None)
    run_config.set_defaults(func=_run_config_command)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run a benchmark preset from a TOML config under configs/benchmarks",
    )
    benchmark.add_argument("config")
    benchmark.add_argument("--output-root", default=None)
    benchmark.set_defaults(func=_benchmark_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
