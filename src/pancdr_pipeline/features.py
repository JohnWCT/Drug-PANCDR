"""Align source/target omics to common gene features."""

import warnings
from dataclasses import dataclass
from typing import List

import pandas as pd

from pancdr_pipeline.config import PANCDRPipelineConfig
from pancdr_pipeline.schema import normalize_source_sample_id, normalize_tcga_patient_id


@dataclass
class FeatureAlignmentResult:
    source_omics: pd.DataFrame
    target_omics: pd.DataFrame
    common_features: List[str]
    report: pd.DataFrame


def _sample_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def align_source_target_features(source_omics, target_omics, config):
    # type: (pd.DataFrame, pd.DataFrame, PANCDRPipelineConfig) -> FeatureAlignmentResult
    source_sample_col = _sample_col(source_omics, ["Sample_ID", "sample_id"])
    target_sample_col = _sample_col(target_omics, [config.target_omics_sample_col, "tissue_id"])

    source_features = [c for c in source_omics.columns if c != source_sample_col]
    target_features = [c for c in target_omics.columns if c != target_sample_col]
    common = sorted(set(source_features) & set(target_features))

    if len(common) == 0:
        raise ValueError("No common gene features between source and target omics")
    if len(common) < 100:
        warnings.warn(
            "Only {} common features between source and target omics".format(len(common))
        )

    source_aligned = source_omics[[source_sample_col] + common].copy()
    source_aligned["sample_id"] = source_aligned[source_sample_col].map(normalize_source_sample_id)
    source_aligned = source_aligned.drop(columns=[source_sample_col])

    target_aligned = target_omics[[target_sample_col] + common].copy()
    target_aligned["sample_id"] = target_aligned[target_sample_col].map(normalize_tcga_patient_id)
    target_aligned = target_aligned.drop(columns=[target_sample_col])
    # Collapse multiple tissue barcodes per patient to one row (mean expression).
    target_aligned = target_aligned.groupby("sample_id", as_index=False)[common].mean()

    report = pd.DataFrame(
        [
            {
                "source_feature_count": len(source_features),
                "target_feature_count": len(target_features),
                "common_feature_count": len(common),
                "source_only_count": len(set(source_features) - set(target_features)),
                "target_only_count": len(set(target_features) - set(source_features)),
            }
        ]
    )
    return FeatureAlignmentResult(
        source_omics=source_aligned,
        target_omics=target_aligned,
        common_features=common,
        report=report,
    )
