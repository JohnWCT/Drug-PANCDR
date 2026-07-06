"""Export encoder latent representations and optional fold-level analysis."""

from pathlib import Path

import pandas as pd
import torch

from pancdr_pipeline.config import PANCDRPipelineConfig
from pancdr_pipeline.fid import compute_fid_for_fold
from pancdr_pipeline.kmeans import run_kmeans_for_fold
from pancdr_pipeline.model_adapter import PANCDRModelAdapter
from pancdr_pipeline.reports import ensure_dir, write_csv
from pancdr_pipeline.tsne import run_tsne_for_fold


def _latent_df(sample_ids, latent, domain, eval_name, cancer_types=None):
    n, dim = latent.shape
    cols = {"sample_id": sample_ids, "domain": domain, "eval_name": eval_name}
    if cancer_types is not None:
        cols["cancer_type"] = cancer_types
    for j in range(dim):
        cols["latent_{}".format(j)] = latent[:, j]
    return pd.DataFrame(cols)


def _export_labeled_latent(adapter, split_data, domain, eval_name, latent_type, out_dir):
    if len(split_data.sample_ids) == 0:
        return
    gexpr = torch.FloatTensor(split_data.gexpr).to(adapter.device)
    lat = adapter.extract_latent(gexpr, use_mu=(latent_type == "mu"))
    key = "encoder_mu" if latent_type == "mu" else "encoder_z"
    df = _latent_df(
        split_data.sample_ids,
        lat[key],
        domain,
        eval_name,
        split_data.cancer_types,
    )
    write_csv(df, out_dir / "{}_{}_encoder_{}.csv".format(eval_name, domain, latent_type))


def _export_unlabeled_latent(adapter, bundle, latent_dir, latent_type):
    if len(bundle.target_unlabeled.sample_ids) == 0:
        return
    gexpr = torch.FloatTensor(bundle.target_unlabeled.gexpr).to(adapter.device)
    lat = adapter.extract_latent(gexpr, use_mu=(latent_type == "mu"))
    key = "encoder_mu" if latent_type == "mu" else "encoder_z"
    df = _latent_df(
        bundle.target_unlabeled.sample_ids,
        lat[key],
        "target",
        "target_unlabeled",
    )
    write_csv(df, latent_dir / "target_unlabeled_encoder_{}.csv".format(latent_type))


def export_fold_latents(adapter, bundle, output_dir):
    # type: (PANCDRModelAdapter, object, Path) -> None
    latent_dir = ensure_dir(Path(output_dir) / "fold_{}".format(bundle.fold_id) / "latent")

    splits = [
        ("source_train", bundle.source_train, "source"),
        ("source_valid", bundle.source_valid, "source"),
        ("source_test", bundle.source_test, "source"),
    ]
    for eval_name, split_data, domain in splits:
        _export_labeled_latent(adapter, split_data, domain, eval_name, "mu", latent_dir)
        _export_labeled_latent(adapter, split_data, domain, eval_name, "z", latent_dir)

    _export_unlabeled_latent(adapter, bundle, latent_dir, "mu")
    _export_unlabeled_latent(adapter, bundle, latent_dir, "z")

    for eval_name, split_data in bundle.target_eval_sets.items():
        _export_labeled_latent(adapter, split_data, "target", eval_name, "mu", latent_dir)
        _export_labeled_latent(adapter, split_data, "target", eval_name, "z", latent_dir)


def run_fold_latent_analysis(adapter, bundle, output_dir, config):
    # type: (PANCDRModelAdapter, object, Path, PANCDRPipelineConfig) -> None
    if not config.run_latent:
        return
    export_fold_latents(adapter, bundle, output_dir)
    latent_dir = Path(output_dir) / "fold_{}".format(bundle.fold_id) / "latent"
    if config.run_fid:
        fid_df = compute_fid_for_fold(latent_dir, bundle.fold_id)
        write_csv(fid_df, latent_dir / "fid_summary.csv")
    if config.run_kmeans:
        run_kmeans_for_fold(latent_dir, bundle.fold_id, seed=config.seed)
    if config.run_tsne:
        run_tsne_for_fold(latent_dir, bundle.fold_id, seed=config.seed)
