from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinship.algorithms.classical import run_classical_verification
from kinship.algorithms.facornet_native import run_facornet
from kinship.algorithms.face_embeddings import ALL_BACKBONES, run_face_embedding_verification
from kinship.algorithms.family_deep_native import run_family_deep
from kinship.algorithms.forestnn_native import (
    ALIGNMENT_MODES,
    FUSION_MODELS,
    FUSION_TRAIN_SCOPES,
    GLOBAL_BACKBONES,
    LOCAL_ENCODERS,
    MODEL_SELECTION_METRICS,
    run_forestnn,
)
from kinship.benchmark_report import generate_benchmark_report
from kinship.algorithms.gae_native import run_gae
from kinship.algorithms.kinver import run_kinver
from kinship.algorithms.relation_aware_native import SUPPORT_RELATION_MODES, run_relation_aware
from kinship.configs import (
    benchmark_config_paths,
    experiment_config_paths,
    load_benchmark_config,
    load_experiment_config,
)
from kinship.datasets.mydataset import (
    export_mydataset_inventory,
    export_mydataset_pairs,
    export_mydataset_summary,
    summarize_mydataset,
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


def _face_embed_command(args: argparse.Namespace) -> int:
    parameters = {
        "dataset": args.dataset,
        "relation": args.relation,
        "backbone": args.backbone,
        "calibration_objective": args.calibration_objective,
        "far_targets": args.far_targets,
        "image_batch_size": args.image_batch_size,
    }
    config = _inline_experiment_config(
        name=f"face-embed-{args.backbone}-{args.relation}",
        algorithm="face-embed",
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

    result = run_face_embedding_verification(**parameters)
    _dump(result.__dict__)
    return 0


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
        "pair_types": args.pair_types,
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


def _forestnn_command(args: argparse.Namespace) -> int:
    parameters = {
        "dataset": args.dataset,
        "relation": args.relation,
        "global_backbone": args.global_backbone,
        "fusion_model": args.fusion_model,
        "fusion_train_scope": args.fusion_train_scope,
        "local_encoder": args.local_encoder,
        "alignment_mode": args.alignment_mode,
        "image_size": args.image_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "calibration_objective": args.calibration_objective,
        "model_selection_metric": args.model_selection_metric,
        "far_targets": args.far_targets,
        "random_state": args.random_state,
        "limit_pairs": args.limit_pairs,
    }
    config = _inline_experiment_config(
        name=f"forestnn-{args.dataset}-{args.relation}",
        algorithm="forestnn",
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

    result = run_forestnn(**parameters)
    _dump(result.__dict__)
    return 0


def _facornet_command(args: argparse.Namespace) -> int:
    parameters = {
        "dataset": args.dataset,
        "relation": args.relation,
        "global_backbone": args.global_backbone,
        "fusion_model": args.fusion_model,
        "fusion_train_scope": args.fusion_train_scope,
        "alignment_mode": args.alignment_mode,
        "image_size": args.image_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "calibration_objective": args.calibration_objective,
        "model_selection_metric": args.model_selection_metric,
        "far_targets": args.far_targets,
        "random_state": args.random_state,
        "limit_pairs": args.limit_pairs,
    }
    config = _inline_experiment_config(
        name=f"facornet-{args.dataset}-{args.relation}",
        algorithm="facornet",
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

    result = run_facornet(**parameters)
    _dump(result.__dict__)
    return 0


def _relation_aware_command(args: argparse.Namespace) -> int:
    parameters = {
        "dataset": args.dataset,
        "relation": args.relation,
        "support_relations": args.support_relations,
        "global_backbone": args.global_backbone,
        "fusion_model": args.fusion_model,
        "fusion_train_scope": args.fusion_train_scope,
        "alignment_mode": args.alignment_mode,
        "image_size": args.image_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "relation_dim": args.relation_dim,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "calibration_objective": args.calibration_objective,
        "model_selection_metric": args.model_selection_metric,
        "ranking_loss_weight": args.ranking_loss_weight,
        "far_targets": args.far_targets,
        "random_state": args.random_state,
        "limit_pairs": args.limit_pairs,
    }
    config = _inline_experiment_config(
        name=f"relation-aware-{args.dataset}-{args.relation}",
        algorithm="relation-aware",
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

    result = run_relation_aware(**parameters)
    _dump(result.__dict__)
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
            "relation_winners": run_info["relation_winners"],
        }
    )
    return 0


def _report_benchmark_command(args: argparse.Namespace) -> int:
    payload = generate_benchmark_report(Path(args.run_dir))
    _dump(payload)
    return 0


def _mydataset_summary_command(args: argparse.Namespace) -> int:
    summary = summarize_mydataset()
    payload = {
        "root": summary.root,
        "subset_count": summary.subset_count,
        "family_count": summary.family_count,
        "person_count": summary.person_count,
        "image_count": summary.image_count,
        "subsets": summary.subsets,
    }
    if args.output_path:
        export_mydataset_summary(Path(args.output_path))
    _dump(payload)
    return 0


def _mydataset_inventory_command(args: argparse.Namespace) -> int:
    output_path = export_mydataset_inventory(Path(args.output_path))
    _dump({"output_path": str(output_path)})
    return 0


def _mydataset_pairs_command(args: argparse.Namespace) -> int:
    output_path = export_mydataset_pairs(
        output_path=Path(args.output_path),
        subset=args.subset,
        max_positive_pairs_per_person_pair=args.max_positive_pairs_per_person_pair,
        negative_ratio=args.negative_ratio,
        random_state=args.random_state,
    )
    _dump({"output_path": str(output_path)})
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

    face_embed = subparsers.add_parser(
        "face-embed",
        help="Run a face-embedding verification baseline with official KinFaceW folds",
    )
    face_embed.add_argument("--dataset", default="KinFaceW-II")
    face_embed.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    face_embed.add_argument("--backbone", choices=list(ALL_BACKBONES), default="vggface-precomputed")
    face_embed.add_argument(
        "--calibration-objective",
        choices=["balanced_accuracy", "f1", "eer", "accuracy"],
        default="balanced_accuracy",
    )
    face_embed.add_argument("--far-targets", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    face_embed.add_argument("--image-batch-size", type=int, default=32)
    face_embed.add_argument("--output-root", default=None)
    face_embed.set_defaults(func=_face_embed_command)

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
    family_deep.add_argument("--pair-types", nargs="+", choices=["fd", "fs", "md", "ms"], default=None)
    family_deep.add_argument("--output-dir", default=None)
    family_deep.add_argument("--output-root", default=None)
    family_deep.add_argument("--checkpoints-dir", default=None)
    family_deep.add_argument("--vgg-weights", default=None)
    family_deep.set_defaults(func=_family_deep_command)

    forestnn = subparsers.add_parser(
        "forestnn",
        help="Run the native FNN-style kinship model over official KinFaceW folds",
    )
    forestnn.add_argument("--dataset", default="KinFaceW-II")
    forestnn.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    forestnn.add_argument(
        "--global-backbone",
        choices=list(GLOBAL_BACKBONES),
        default="vggface-precomputed",
    )
    forestnn.add_argument(
        "--fusion-model",
        choices=list(FUSION_MODELS),
        default="engineered-logreg",
    )
    forestnn.add_argument("--fusion-train-scope", choices=list(FUSION_TRAIN_SCOPES), default="trainpool")
    forestnn.add_argument("--local-encoder", choices=list(LOCAL_ENCODERS), default="residual")
    forestnn.add_argument("--alignment-mode", choices=list(ALIGNMENT_MODES), default="adaptive")
    forestnn.add_argument("--image-size", type=int, default=64)
    forestnn.add_argument("--embedding-dim", type=int, default=64)
    forestnn.add_argument("--hidden-dim", type=int, default=96)
    forestnn.add_argument("--batch-size", type=int, default=16)
    forestnn.add_argument("--num-epochs", type=int, default=6)
    forestnn.add_argument("--learning-rate", type=float, default=1e-3)
    forestnn.add_argument("--weight-decay", type=float, default=1e-4)
    forestnn.add_argument("--val-fraction", type=float, default=0.2)
    forestnn.add_argument(
        "--calibration-objective",
        choices=["balanced_accuracy", "f1", "eer", "accuracy"],
        default="balanced_accuracy",
    )
    forestnn.add_argument("--model-selection-metric", choices=list(MODEL_SELECTION_METRICS), default="roc_auc")
    forestnn.add_argument("--far-targets", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    forestnn.add_argument("--random-state", type=int, default=42)
    forestnn.add_argument("--limit-pairs", type=int, default=None)
    forestnn.add_argument("--output-root", default=None)
    forestnn.set_defaults(func=_forestnn_command)

    facornet = subparsers.add_parser(
        "facornet",
        help="Run a native FaCoRNet-inspired component-relation kinship model",
    )
    facornet.add_argument("--dataset", default="KinFaceW-II")
    facornet.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    facornet.add_argument("--global-backbone", choices=list(GLOBAL_BACKBONES), default="vggface+vggf-precomputed")
    facornet.add_argument("--fusion-model", choices=list(FUSION_MODELS), default="engineered-logreg")
    facornet.add_argument("--fusion-train-scope", choices=list(FUSION_TRAIN_SCOPES), default="trainpool")
    facornet.add_argument("--alignment-mode", choices=list(ALIGNMENT_MODES), default="adaptive")
    facornet.add_argument("--image-size", type=int, default=32)
    facornet.add_argument("--embedding-dim", type=int, default=16)
    facornet.add_argument("--hidden-dim", type=int, default=24)
    facornet.add_argument("--num-heads", type=int, default=4)
    facornet.add_argument("--batch-size", type=int, default=16)
    facornet.add_argument("--num-epochs", type=int, default=1)
    facornet.add_argument("--learning-rate", type=float, default=1e-3)
    facornet.add_argument("--weight-decay", type=float, default=1e-4)
    facornet.add_argument("--val-fraction", type=float, default=0.2)
    facornet.add_argument(
        "--calibration-objective",
        choices=["balanced_accuracy", "f1", "eer", "accuracy"],
        default="balanced_accuracy",
    )
    facornet.add_argument("--model-selection-metric", choices=list(MODEL_SELECTION_METRICS), default="roc_auc")
    facornet.add_argument("--far-targets", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    facornet.add_argument("--random-state", type=int, default=42)
    facornet.add_argument("--limit-pairs", type=int, default=None)
    facornet.add_argument("--output-root", default=None)
    facornet.set_defaults(func=_facornet_command)

    relation_aware = subparsers.add_parser(
        "relation-aware",
        help="Run the native relation-aware cross-relation kinship model",
    )
    relation_aware.add_argument("--dataset", default="KinFaceW-II")
    relation_aware.add_argument("--relation", choices=["fd", "fs", "md", "ms"], required=True)
    relation_aware.add_argument("--support-relations", choices=list(SUPPORT_RELATION_MODES), default="all")
    relation_aware.add_argument(
        "--global-backbone",
        choices=list(GLOBAL_BACKBONES),
        default="vggface+vggf-precomputed",
    )
    relation_aware.add_argument("--fusion-model", choices=list(FUSION_MODELS), default="engineered-logreg")
    relation_aware.add_argument("--fusion-train-scope", choices=list(FUSION_TRAIN_SCOPES), default="trainpool")
    relation_aware.add_argument("--alignment-mode", choices=list(ALIGNMENT_MODES), default="adaptive")
    relation_aware.add_argument("--image-size", type=int, default=64)
    relation_aware.add_argument("--embedding-dim", type=int, default=64)
    relation_aware.add_argument("--hidden-dim", type=int, default=96)
    relation_aware.add_argument("--relation-dim", type=int, default=16)
    relation_aware.add_argument("--num-heads", type=int, default=4)
    relation_aware.add_argument("--batch-size", type=int, default=16)
    relation_aware.add_argument("--num-epochs", type=int, default=6)
    relation_aware.add_argument("--learning-rate", type=float, default=1e-3)
    relation_aware.add_argument("--weight-decay", type=float, default=1e-4)
    relation_aware.add_argument("--val-fraction", type=float, default=0.2)
    relation_aware.add_argument(
        "--calibration-objective",
        choices=["accuracy", "balanced_accuracy", "f1"],
        default="balanced_accuracy",
    )
    relation_aware.add_argument(
        "--model-selection-metric",
        choices=list(MODEL_SELECTION_METRICS),
        default="roc_auc",
    )
    relation_aware.add_argument("--ranking-loss-weight", type=float, default=0.10)
    relation_aware.add_argument("--far-targets", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    relation_aware.add_argument("--random-state", type=int, default=42)
    relation_aware.add_argument("--limit-pairs", type=int, default=None)
    relation_aware.add_argument("--output-root", default=None)
    relation_aware.set_defaults(func=_relation_aware_command)

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

    report_benchmark = subparsers.add_parser(
        "report-benchmark",
        help="Generate publication-friendly tables and plots from an existing benchmark run directory",
    )
    report_benchmark.add_argument("run_dir")
    report_benchmark.set_defaults(func=_report_benchmark_command)

    mydataset = subparsers.add_parser(
        "mydataset",
        help="Inspect and export manifests for the local private mydataset collection",
    )
    mydataset_subparsers = mydataset.add_subparsers(dest="mydataset_command", required=True)

    mydataset_summary = mydataset_subparsers.add_parser(
        "summary",
        help="Summarize the local mydataset collection",
    )
    mydataset_summary.add_argument("--output-path", default=None)
    mydataset_summary.set_defaults(func=_mydataset_summary_command)

    mydataset_inventory = mydataset_subparsers.add_parser(
        "export-inventory",
        help="Export an image-level manifest for mydataset",
    )
    mydataset_inventory.add_argument(
        "--output-path",
        default="outputs/mydataset/mydataset_inventory.csv",
    )
    mydataset_inventory.set_defaults(func=_mydataset_inventory_command)

    mydataset_pairs = mydataset_subparsers.add_parser(
        "export-pairs",
        help="Export a pair manifest from mydataset for downstream experiments",
    )
    mydataset_pairs.add_argument(
        "--output-path",
        default="outputs/mydataset/mydataset_pairs.csv",
    )
    mydataset_pairs.add_argument("--subset", default=None)
    mydataset_pairs.add_argument("--max-positive-pairs-per-person-pair", type=int, default=20)
    mydataset_pairs.add_argument("--negative-ratio", type=float, default=1.0)
    mydataset_pairs.add_argument("--random-state", type=int, default=42)
    mydataset_pairs.set_defaults(func=_mydataset_pairs_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
