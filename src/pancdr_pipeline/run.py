"""PANCDR pipeline orchestrator."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pancdr_pipeline.config import PANCDRPipelineConfig, resolve_path, save_config, validate_config
from pancdr_pipeline.data_io import load_raw_inputs
from pancdr_pipeline.datasets import build_fold_data_bundle
from pancdr_pipeline.drug_graph_adapter import build_drug_graphs
from pancdr_pipeline.drug_index import TARGET_EVAL_NAMES, build_drug_index
from pancdr_pipeline.evaluation import EVAL_NAMES, evaluate_fold
from pancdr_pipeline.features import align_source_target_features
from pancdr_pipeline.fid import build_cross_fold_fid_summary
from pancdr_pipeline.hyperparams import load_pancdr_hyperparams, save_hyperparams
from pancdr_pipeline.kmeans import build_cross_fold_kmeans_summary
from pancdr_pipeline.latent import run_fold_latent_analysis
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
from pancdr_pipeline.splits import build_source_splits
from pancdr_pipeline.trainer_wrapper import train_one_fold


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(config):
    # type: (PANCDRPipelineConfig) -> Path
    validate_config(config)
    _set_global_seed(config.seed)

    out = ensure_dir(resolve_path(config, config.output_dir))
    reports_dir = ensure_dir(out / "reports")
    save_config(config, out / "config.json")

    raw = load_raw_inputs(config)
    normalized = normalize_all_inputs(raw, config, reports_dir)
    write_csv(normalized["validation_report"], reports_dir / "schema_validation_report.csv")
    write_csv(normalized["skipped_report"], reports_dir / "skipped_rows_schema.csv")

    if config.debug_rows is not None:
        normalized["source_response"] = normalized["source_response"].head(config.debug_rows)
        for name in list(normalized["target_eval_sets"].keys()):
            normalized["target_eval_sets"][name] = normalized["target_eval_sets"][name].head(
                config.debug_rows
            )

    aligned = align_source_target_features(
        raw.source_omics, raw.target_omics, config
    )
    write_csv(aligned.report, reports_dir / "feature_alignment_report.csv")

    drug_index_result = build_drug_index(
        normalized["source_response"],
        normalized["target_eval_sets"],
        normalized["drug_smiles"],
    )
    write_csv(drug_index_result.drug_index, reports_dir / "drug_index.csv")
    write_csv(drug_index_result.availability_report, reports_dir / "drug_availability_report.csv")
    write_csv(drug_index_result.zero_shot_report, reports_dir / "target_eval_zero_shot_drug_report.csv")

    drug_graph_bundle = build_drug_graphs(drug_index_result, out)
    write_csv(drug_graph_bundle.availability_report, reports_dir / "drug_graph_availability_report.csv")
    write_csv(drug_graph_bundle.edge_report, reports_dir / "drug_graph_edge_report.csv")

    splits = build_source_splits(
        normalized["source_response"],
        config.n_splits,
        config.source_test_size,
        config.seed,
    )
    write_csv(splits.source_split, reports_dir / "source_split.csv")
    write_csv(splits.fold_assignments, reports_dir / "fold_assignments.csv")

    params = load_pancdr_hyperparams(config)
    save_hyperparams(params, out)

    fold_manifest = []
    eval_names = [n for n in EVAL_NAMES if n == "source_test" or n in normalized["target_eval_sets"]]

    folds_to_run = splits.folds
    if config.debug_rows is not None:
        folds_to_run = splits.folds[:1]

    for fold_split in folds_to_run:
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
        train_result = train_one_fold(
            fold_split.fold, bundle, params, config, out
        )
        fold_manifest.append(
            {
                "fold": fold_split.fold,
                "checkpoint": "fold_{}/best_model.pt".format(fold_split.fold),
                "best_source_valid_auc": train_result.best_source_valid_auc,
            }
        )

        adapter = PANCDRModelAdapter(
            bundle.source_train.gexpr.shape[1],
            drug_graph_bundle.atom_feature_dim,
            params,
            config.device,
        ).build_models()
        adapter.load_checkpoint(train_result.checkpoint_path)

        eval_results = evaluate_fold(
            adapter, bundle, drug_index_result, drug_graph_bundle, config, fold_split.fold
        )
        fdir = fold_dir(out, fold_split.fold)
        for eval_name, payload in eval_results.items():
            write_csv(payload["predictions"], fdir / "{}_prediction_results.csv".format(eval_name))
            write_csv(payload["summary"], fdir / "{}_metrics_summary.csv".format(eval_name))
            write_csv(payload["per_drug"], fdir / "{}_metrics_per_drug.csv".format(eval_name))

        run_fold_latent_analysis(adapter, bundle, out, config)

    build_ensemble_predictions(out, eval_names, config.threshold)
    build_cross_fold_summary(out, eval_names)
    if config.run_fid:
        build_cross_fold_fid_summary(out)
    if config.run_kmeans:
        build_cross_fold_kmeans_summary(out)

    target_eval_specs = [
        {"name": s.name, "path": s.path}
        for s in __import__("pancdr_pipeline.data_io", fromlist=["load_target_eval_config"]).load_target_eval_config(config)
    ]
    manifest = {
        "model": "PANCDR",
        "n_splits": config.n_splits,
        "threshold": config.threshold,
        "tcga_labels_in_training": False,
        "model_selection_metric": "source_valid_auroc",
        "target_eval_sets": target_eval_specs,
        "drug_smiles_path": config.drug_smiles_path,
        "hyperparams_path": config.hyperparams_path,
        "output_dir": str(out),
        "folds": fold_manifest,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(manifest, out / "manifest.json")
    return out
