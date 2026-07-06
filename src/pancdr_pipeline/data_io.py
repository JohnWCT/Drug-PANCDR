"""Unified data loading for PANCDR pipeline."""

import json
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from pancdr_pipeline.config import PANCDRPipelineConfig, resolve_path


@dataclass
class TargetEvalSpec:
    name: str
    path: str


@dataclass
class RawPANCDRInputs:
    source_omics: pd.DataFrame
    source_response: pd.DataFrame
    target_omics: pd.DataFrame
    target_eval_sets: Dict[str, pd.DataFrame]
    drug_smiles: pd.DataFrame
    ccle_cancer_info: pd.DataFrame
    tcga_cancer_info: pd.DataFrame


def load_source_omics(path):
    # type: (str) -> pd.DataFrame
    return pd.read_csv(path, low_memory=False)


def load_target_omics(path):
    # type: (str) -> pd.DataFrame
    return pd.read_csv(path, low_memory=False)


def load_source_response(path):
    # type: (str) -> pd.DataFrame
    return pd.read_csv(path, low_memory=False)


def load_drug_smiles(path):
    # type: (str) -> pd.DataFrame
    return pd.read_csv(path, low_memory=False)


def load_cancer_info(path):
    # type: (str) -> pd.DataFrame
    return pd.read_csv(path, low_memory=False)


def load_target_eval_config(config):
    # type: (PANCDRPipelineConfig) -> List[TargetEvalSpec]
    cfg_path = resolve_path(config, config.target_eval_config)
    with open(str(cfg_path), "r") as f:
        payload = json.load(f)
    specs = []
    for item in payload["target_eval_sets"]:
        specs.append(TargetEvalSpec(name=item["name"], path=item["path"]))
    return specs


def load_target_eval_sets(config):
    # type: (PANCDRPipelineConfig) -> Dict[str, pd.DataFrame]
    out = {}
    for spec in load_target_eval_config(config):
        path = resolve_path(config, spec.path)
        out[spec.name] = pd.read_csv(str(path), low_memory=False)
    return out


def load_raw_inputs(config):
    # type: (PANCDRPipelineConfig) -> RawPANCDRInputs
    return RawPANCDRInputs(
        source_omics=load_source_omics(str(resolve_path(config, config.source_omics_path))),
        source_response=load_source_response(
            str(resolve_path(config, config.source_response_path))
        ),
        target_omics=load_target_omics(str(resolve_path(config, config.target_omics_path))),
        target_eval_sets=load_target_eval_sets(config),
        drug_smiles=load_drug_smiles(str(resolve_path(config, config.drug_smiles_path))),
        ccle_cancer_info=load_cancer_info(
            str(resolve_path(config, config.ccle_cancer_info_path))
        ),
        tcga_cancer_info=load_cancer_info(
            str(resolve_path(config, config.tcga_cancer_info_path))
        ),
    )
