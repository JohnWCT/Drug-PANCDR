"""Fréchet distance between source and target latent distributions."""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from pancdr_pipeline.reports import write_csv


def _frechet_distance(x, y):
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    sigma1 = np.cov(x, rowvar=False)
    sigma2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean = np.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fid_for_fold(latent_dir, fold_id):
    rows = []
    latent_dir = Path(latent_dir)
    pairs = [
        ("source_train", "target_unlabeled"),
        ("source_train", "target_primary"),
        ("source_train", "target_only"),
        ("source_train", "dapl"),
        ("source_train", "target_aacdr"),
        ("source_train", "target_aacdr_only"),
    ]
    for source_name, target_name in pairs:
        src_path = latent_dir / "{}_source_encoder_mu.csv".format(source_name)
        tgt_path = latent_dir / "{}_target_encoder_mu.csv".format(target_name)
        if target_name == "target_unlabeled":
            tgt_path = latent_dir / "target_unlabeled_encoder_mu.csv"
        if not src_path.is_file() or not tgt_path.is_file():
            continue
        src = pd.read_csv(str(src_path))
        tgt = pd.read_csv(str(tgt_path))
        latent_cols = [c for c in src.columns if c.startswith("latent_")]
        if not latent_cols:
            continue
        x = src[latent_cols].values
        y = tgt[latent_cols].values
        warning = ""
        if len(x) < 10 or len(y) < 10:
            warning = "small_sample_warning"
        try:
            fid = _frechet_distance(x, y)
        except Exception as exc:
            fid = float("nan")
            warning = str(exc)
        rows.append(
            {
                "fold": fold_id,
                "source_name": source_name,
                "target_name": target_name,
                "fid": fid,
                "n_source": len(x),
                "n_target": len(y),
                "latent_type": "encoder_mu",
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def build_cross_fold_fid_summary(output_dir):
    frames = []
    output_dir = Path(output_dir)
    for fold_path in sorted(output_dir.glob("fold_*")):
        fold_id = int(fold_path.name.split("_")[1])
        fid_path = fold_path / "latent" / "fid_summary.csv"
        if fid_path.is_file():
            frames.append(pd.read_csv(str(fid_path)))
        else:
            df = compute_fid_for_fold(fold_path / "latent", fold_id)
            if not df.empty:
                write_csv(df, fid_path)
                frames.append(df)
    if frames:
        cross = pd.concat(frames, ignore_index=True)
        write_csv(cross, output_dir / "summary" / "fid_cross_fold_summary.csv")
        return cross
    return pd.DataFrame()
