"""Inference-only evaluation from saved fold checkpoints."""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from pancdr_pipeline.config import PANCDRPipelineConfig, load_config, resolve_path, save_config
from pancdr_pipeline.data_io import load_raw_inputs
from pancdr_pipeline.datasets import build_fold_data_bundle
from pancdr_pipeline.drug_graph_adapter import build_drug_graphs
from pancdr_pipeline.drug_index import build_drug_index
from pancdr_pipeline.evaluation import EVAL_NAMES, evaluate_fold
from pancdr_pipeline.features import align_source_target_features
from pancdr_pipeline.hyperparams import load_pancdr_hyperparams
from pancdr_pipeline.model_adapter import PANCDRModelAdapter
from pancdr_pipeline.reports import (
    build_cross_fold_summary,
    build_ensemble_predictions,
    ensure_dir,
    fold_dir,
    write_csv,
    write_json,
)
from pancdr_pipeline.schema import normalize_all_inputs
from pancdr_pipeline.splits import FoldSplit, SourceSplits, build_source_splits


def load_manifest(checkpoint_dir):
    path = Path(checkpoint_dir) / "manifest.json"
    with open(str(path), "r") as f:
        return json.load(f)


def load_source_splits_from_reports(checkpoint_dir, source_response, n_splits, source_test_size, seed):
    split_path = Path(checkpoint_dir) / "reports" / "source_split.csv"
    assign_path = Path(checkpoint_dir) / "reports" / "fold_assignments.csv"
    if assign_path.is_file():
        assign = pd.read_csv(str(assign_path))
        folds = []
        for fold_id in sorted(assign["fold"].unique()):
            sub = assign[assign["fold"] == fold_id]
            folds.append(
                FoldSplit(
                    fold=int(fold_id),
                    train_sample_ids=sub.loc[sub["split"] == "train", "sample_id"].astype(str).tolist(),
                    valid_sample_ids=sub.loc[sub["split"] == "valid", "sample_id"].astype(str).tolist(),
                    source_test_sample_ids=sub.loc[
                        sub["split"] == "source_test", "sample_id"
                    ].astype(str).tolist(),
                )
            )
        source_split = pd.read_csv(str(split_path)) if split_path.is_file() else assign
        return SourceSplits(folds=folds, source_split=source_split, fold_assignments=assign)
    return build_source_splits(source_response, n_splits, source_test_size, seed)


def run_target_inference(config, checkpoint_dir, output_dir=None, eval_prefixes=None, skip_reports=False):
    # type: (...) -> Path
    checkpoint_dir = Path(checkpoint_dir)
    out = ensure_dir(output_dir or resolve_path(config, config.output_dir))
    reports_dir = ensure_dir(out / "reports")
    save_config(config, out / "config.json")

    raw = load_raw_inputs(config)
    normalized = normalize_all_inputs(raw, config, reports_dir)
    aligned = align_source_target_features(raw.source_omics, raw.target_omics, config)
    drug_index_result = build_drug_index(
        normalized["source_response"],
        normalized["target_eval_sets"],
        normalized["drug_smiles"],
    )
    drug_graph_bundle = build_drug_graphs(drug_index_result, out, force_rebuild=True)

    manifest = load_manifest(checkpoint_dir)
    splits = load_source_splits_from_reports(
        checkpoint_dir,
        normalized["source_response"],
        config.n_splits,
        config.source_test_size,
        config.seed,
    )

    params = load_pancdr_hyperparams(config)
    if eval_prefixes is None:
        eval_prefixes = [n for n in EVAL_NAMES if n != "source_test"]

    for fold_split in splits.folds:
        ckpt = checkpoint_dir / "fold_{}".format(fold_split.fold) / "best_model.pt"
        if not ckpt.is_file():
            continue
        bundle = build_fold_data_bundle(
            fold_split,
            normalized["source_response"],
            aligned.source_omics,
            aligned.target_omics,
            normalized["target_eval_sets"],
            drug_graph_bundle,
            aligned.common_features,
            debug_rows=config.debug_rows,
        )
        adapter = PANCDRModelAdapter(
            bundle.source_train.gexpr.shape[1],
            drug_graph_bundle.atom_feature_dim,
            params,
            config.device,
        ).build_models()
        adapter.load_checkpoint(str(ckpt))

        eval_results = evaluate_fold(adapter, bundle, drug_index_result, config, fold_split.fold)
        fdir = fold_dir(out, fold_split.fold)
        for eval_name, payload in eval_results.items():
            if eval_name not in eval_prefixes and eval_name != "source_test":
                continue
            write_csv(payload["predictions"], fdir / "{}_prediction_results.csv".format(eval_name))
            if not skip_reports:
                write_csv(payload["summary"], fdir / "{}_metrics_summary.csv".format(eval_name))
                write_csv(payload["per_drug"], fdir / "{}_metrics_per_drug.csv".format(eval_name))

    if not skip_reports:
        build_ensemble_predictions(out, eval_prefixes, config.threshold)
        build_cross_fold_summary(out, eval_prefixes)

    write_json(
        {
            "mode": "inference_only",
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(out),
            "eval_prefixes": eval_prefixes,
            "tcga_labels_in_training": False,
        },
        out / "manifest.json",
    )
    return out
