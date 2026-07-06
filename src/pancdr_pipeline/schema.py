"""Schema validation and canonical column normalization."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pancdr_pipeline.config import PANCDRPipelineConfig

_TCGA_PREFIX = re.compile(r"^TCGA-", re.IGNORECASE)

BINARY_POSITIVE = {
    "1", "1.0", "true", "True", "TRUE", "sensitive", "Sensitive", "SENSITIVE",
    "response", "Response", "RESPONSE", "CR", "PR",
}
BINARY_NEGATIVE = {
    "0", "0.0", "false", "False", "FALSE", "resistant", "Resistant", "RESISTANT",
    "non-response", "Non-response", "NON-RESPONSE", "PD", "SD",
}


def normalize_drug_key(value):
    # type: (Any) -> str
    return str(value).strip().lower()


def normalize_tcga_patient_id(value):
    # type: (Any) -> str
    sid = str(value).strip()
    parts = sid.split("-")
    if _TCGA_PREFIX.match(sid) and len(parts) >= 3:
        return "-".join(parts[:3])
    return sid


def normalize_source_sample_id(value):
    # type: (Any) -> str
    return str(value).strip()


def _find_column(df, candidates):
    # type: (pd.DataFrame, List[str]) -> Optional[str]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _label_to_binary(value):
    # type: (Any) -> Tuple[Optional[int], str]
    if pd.isna(value):
        return None, "nan_label"
    if isinstance(value, (bool, np.bool_)):
        return int(value), ""
    s = str(value).strip()
    if s in BINARY_POSITIVE:
        return 1, ""
    if s in BINARY_NEGATIVE:
        return 0, ""
    try:
        num = float(s)
        if num in (0.0, 1.0):
            return int(num), ""
        return None, "non_binary_numeric:{}".format(s)
    except ValueError:
        return None, "unrecognized_label:{}".format(s)


def normalize_source_response(df, config, reports_dir):
    # type: (pd.DataFrame, PANCDRPipelineConfig, Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    sample_col = _find_column(df, [config.source_sample_col, "Sample_ID", "sample_id"]) or config.source_sample_col
    drug_col = _find_column(df, [config.source_drug_col, "drug_name", "DRUG_NAME"]) or config.source_drug_col
    label_col = _find_column(df, [config.source_label_col, "Label", "label"]) or config.source_label_col

    validation_rows = []
    skipped_rows = []
    out_rows = []

    for idx, row in df.iterrows():
        sample_id = normalize_source_sample_id(row.get(sample_col, ""))
        drug_name_raw = row.get(drug_col, "")
        drug_key = normalize_drug_key(drug_name_raw)
        label, reason = _label_to_binary(row.get(label_col))
        validation_rows.append(
            {
                "row_index": idx,
                "dataset": "source_response",
                "sample_col": sample_col,
                "drug_col": drug_col,
                "label_col": label_col,
                "sample_id": sample_id,
                "drug_name": drug_name_raw,
                "label_raw": row.get(label_col),
                "label_binary": label,
                "status": "ok" if label is not None else "skipped",
                "reason": reason,
            }
        )
        if label is None:
            skipped_rows.append(validation_rows[-1])
            continue
        out_rows.append(
            {
                "sample_id": sample_id,
                "drug_name": str(drug_name_raw).strip(),
                "drug_key": drug_key,
                "label": label,
                "source_file": "source_response",
            }
        )

    out = pd.DataFrame(out_rows)
    validation_df = pd.DataFrame(validation_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    return out, validation_df, skipped_df


def normalize_target_eval(df, dataset_name, config):
    # type: (pd.DataFrame, str, PANCDRPipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    sample_col = _find_column(
        df, [config.target_sample_col, "Patient_id", "patient", "sample_id"]
    ) or config.target_sample_col
    drug_col = _find_column(df, [config.target_drug_col, "drug_name", "DRUG_NAME"]) or config.target_drug_col
    label_col = _find_column(df, [config.target_label_col, "Label", "label"]) or config.target_label_col

    validation_rows = []
    skipped_rows = []
    out_rows = []

    for idx, row in df.iterrows():
        sample_id = normalize_tcga_patient_id(row.get(sample_col, ""))
        drug_name_raw = row.get(drug_col, "")
        drug_key = normalize_drug_key(drug_name_raw)
        label, reason = _label_to_binary(row.get(label_col))
        cancer_type = row.get("cancers", row.get("cancer_type", np.nan))
        smiles = row.get("smiles", row.get("SMILES", np.nan))
        pubchem_id = row.get("pubchem", row.get("pubchem_id", np.nan))
        validation_rows.append(
            {
                "row_index": idx,
                "dataset": dataset_name,
                "sample_col": sample_col,
                "drug_col": drug_col,
                "label_col": label_col,
                "sample_id": sample_id,
                "drug_name": drug_name_raw,
                "label_raw": row.get(label_col),
                "label_binary": label,
                "status": "ok" if label is not None else "skipped",
                "reason": reason,
            }
        )
        if label is None:
            skipped_rows.append(validation_rows[-1])
            continue
        out_rows.append(
            {
                "sample_id": sample_id,
                "drug_name": str(drug_name_raw).strip(),
                "drug_key": drug_key,
                "label": label,
                "cancer_type": cancer_type,
                "smiles": smiles,
                "pubchem_id": pubchem_id,
                "source_file": dataset_name,
            }
        )

    return pd.DataFrame(out_rows), pd.DataFrame(validation_rows), pd.DataFrame(skipped_rows)


def normalize_drug_smiles_table(df, config):
    # type: (pd.DataFrame, PANCDRPipelineConfig) -> pd.DataFrame
    drug_col = _find_column(
        df, [config.smiles_drug_col, "drug_name", "DRUG_NAME", "name"]
    ) or config.smiles_drug_col
    smiles_col = _find_column(df, [config.smiles_col, "SMILES", "smiles"]) or config.smiles_col
    out = df.copy()
    out["drug_name"] = out[drug_col].astype(str).str.strip()
    out["drug_key"] = out["drug_name"].map(normalize_drug_key)
    out["smiles"] = out[smiles_col].astype(str).str.strip()
    if "pubchem_id" not in out.columns and "pubchem" in out.columns:
        out["pubchem_id"] = out["pubchem"]
    return out[["drug_name", "drug_key", "smiles"] + [c for c in ["pubchem_id"] if c in out.columns]]


def normalize_all_inputs(raw, config, reports_dir):
    # type: (...) -> Dict[str, Any]
    source_resp, src_val, src_skip = normalize_source_response(
        raw.source_response, config, reports_dir
    )
    target_eval = {}
    all_validation = [src_val]
    all_skipped = [src_skip]
    for name, df in raw.target_eval_sets.items():
        norm, val, skip = normalize_target_eval(df, name, config)
        target_eval[name] = norm
        all_validation.append(val)
        all_skipped.append(skip)

    drug_smiles = normalize_drug_smiles_table(raw.drug_smiles, config)
    validation_report = pd.concat(all_validation, ignore_index=True, sort=False)
    skipped_report = pd.concat(all_skipped, ignore_index=True, sort=False)

    return {
        "source_response": source_resp,
        "target_eval_sets": target_eval,
        "drug_smiles": drug_smiles,
        "validation_report": validation_report,
        "skipped_report": skipped_report,
    }
