"""Export encoder latent representations."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from pancdr_pipeline.datasets import FoldDataBundle, LabeledSplitData, TargetUnlabeledData
from pancdr_pipeline.model_adapter import PANCDRModelAdapter
from pancdr_pipeline.reports import ensure_dir, write_csv


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


def export_fold_latents(adapter, bundle, output_dir):
    # type: (PANCDRModelAdapter, FoldDataBundle, Path) -> None
    fold_id = bundle.fold_id
    latent_dir = ensure_dir(Path(output_dir) / "fold_{}".format(fold_id) / "latent")

    splits = [
        ("source_train", bundle.source_train, "source"),
        ("source_valid", bundle.source_valid, "source"),
        ("source_test", bundle.source_test, "source"),
    ]
    for eval_name, split_data, domain in splits:
        _export_labeled_latent(adapter, split_data, domain, eval_name, "mu", latent_dir)
        if eval_name == "source_test":
            _export_labeled_latent(adapter, split_data, domain, eval_name, "z", latent_dir)

    if len(bundle.target_unlabeled.sample_ids):
        gexpr = torch.FloatTensor(bundle.target_unlabeled.gexpr).to(adapter.device)
        lat = adapter.extract_latent(gexpr, use_mu=True)
        df = _latent_df(
            bundle.target_unlabeled.sample_ids,
            lat["encoder_mu"],
            "target",
            "target_unlabeled",
        )
        write_csv(df, latent_dir / "target_unlabeled_encoder_mu.csv")

    for eval_name, split_data in bundle.target_eval_sets.items():
        _export_labeled_latent(adapter, split_data, "target", eval_name, "mu", latent_dir)
        _export_labeled_latent(adapter, split_data, "target", eval_name, "z", latent_dir)
