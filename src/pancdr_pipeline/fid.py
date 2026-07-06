"""Fréchet distance between source and target latent distributions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from pancdr_pipeline.reports import write_csv


def _frechet_distance(x, y, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("FID expects 2D arrays: x={}, y={}".format(x.shape, y.shape))

    if x.shape[1] != y.shape[1]:
        raise ValueError("Latent dims mismatch: x={}, y={}".format(x.shape[1], y.shape[1]))

    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)

    sigma1 = np.atleast_2d(np.cov(x, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(y, rowvar=False))

    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(
            (sigma1 + np.eye(sigma1.shape[0]) * eps).dot(
                sigma2 + np.eye(sigma2.shape[0]) * eps
            )
        )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


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
    if not frames:
        return pd.DataFrame()

    cross = pd.concat(frames, ignore_index=True)
    write_csv(cross, output_dir / "summary" / "fid_cross_fold_summary.csv")

    agg = (
        cross.groupby(["source_name", "target_name", "latent_type"])
        .agg(
            fid_mean=("fid", "mean"),
            fid_std=("fid", "std"),
            fid_min=("fid", "min"),
            fid_max=("fid", "max"),
            n_folds=("fold", "nunique"),
            n_valid_folds=("fid", lambda s: int(s.notna().sum())),
            n_nan_folds=("fid", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    write_csv(agg, output_dir / "summary" / "fid_cross_fold_aggregated_summary.csv")
    return cross
