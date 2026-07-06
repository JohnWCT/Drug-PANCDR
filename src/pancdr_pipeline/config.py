"""CLI and configuration for the PANCDR evaluation pipeline."""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PANCDRPipelineConfig:
    workspace_root: str = "/workspace"

    source_omics_path: str = "DAPL-master/data/pretrain_ccle.csv"
    source_response_path: str = "DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv"
    target_omics_path: str = "DAPL-master/data/TCGA/pretrain_tcga.csv"

    target_eval_config: str = "PANCDR/configs/target_eval_sets.json"

    drug_smiles_path: str = "DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"

    ccle_cancer_info_path: str = "DAPL-master/data/ccle_sample_info_df.csv"
    tcga_cancer_info_path: str = "DAPL-master/data/TCGA/xena_sample_info_df.csv"

    output_dir: str = "PANCDR/outputs_pancdr_5fold_multitcga"

    n_splits: int = 5
    source_test_size: float = 0.10
    seed: int = 0
    device: str = "cuda"

    hyperparams_path: str = "PANCDR/src/tuned_hyperparameters/TCGA_CV_params.csv"

    max_epochs: int = 1000
    early_stop_patience: int = 10
    threshold: float = 0.5

    run_latent: bool = True
    run_fid: bool = True
    run_kmeans: bool = True
    run_tsne: bool = True

    debug_rows: Optional[int] = None

    # Column hints for DAPL-style CSVs
    source_sample_col: str = "Sample_ID"
    source_drug_col: str = "drug_name"
    source_label_col: str = "Label"
    target_sample_col: str = "Patient_id"
    target_drug_col: str = "drug_name"
    target_label_col: str = "Label"
    target_omics_sample_col: str = "tissue_id"
    smiles_drug_col: str = "drug_name"
    smiles_col: str = "SMILES"

    extra: Dict[str, Any] = field(default_factory=dict)


def resolve_path(config, path):
    # type: (PANCDRPipelineConfig, Optional[str]) -> Optional[Path]
    if path is None or path == "":
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(config.workspace_root) / p


def build_arg_parser():
    # type: () -> argparse.ArgumentParser
    p = argparse.ArgumentParser(
        description="PANCDR 5-fold source training + 5 TCGA target eval pipeline"
    )
    p.add_argument("--workspace_root", default="/workspace")
    p.add_argument("--source_omics_path", default="DAPL-master/data/pretrain_ccle.csv")
    p.add_argument(
        "--source_response_path",
        default="DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv",
    )
    p.add_argument("--target_omics_path", default="DAPL-master/data/TCGA/pretrain_tcga.csv")
    p.add_argument(
        "--target_eval_config",
        default="PANCDR/configs/target_eval_sets.json",
    )
    p.add_argument(
        "--drug_smiles_path",
        default="DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv",
    )
    p.add_argument(
        "--ccle_cancer_info_path",
        default="DAPL-master/data/ccle_sample_info_df.csv",
    )
    p.add_argument(
        "--tcga_cancer_info_path",
        default="DAPL-master/data/TCGA/xena_sample_info_df.csv",
    )
    p.add_argument("--output_dir", default="PANCDR/outputs_pancdr_5fold_multitcga")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--source_test_size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--hyperparams_path",
        default="PANCDR/src/tuned_hyperparameters/TCGA_CV_params.csv",
    )
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no_latent", action="store_true")
    p.add_argument("--no_fid", action="store_true")
    p.add_argument("--no_kmeans", action="store_true")
    p.add_argument("--no_tsne", action="store_true")
    p.add_argument("--debug_rows", type=int, default=None)
    return p


def config_from_args(args):
    # type: (argparse.Namespace) -> PANCDRPipelineConfig
    return PANCDRPipelineConfig(
        workspace_root=args.workspace_root,
        source_omics_path=args.source_omics_path,
        source_response_path=args.source_response_path,
        target_omics_path=args.target_omics_path,
        target_eval_config=args.target_eval_config,
        drug_smiles_path=args.drug_smiles_path,
        ccle_cancer_info_path=args.ccle_cancer_info_path,
        tcga_cancer_info_path=args.tcga_cancer_info_path,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        source_test_size=args.source_test_size,
        seed=args.seed,
        device=args.device,
        hyperparams_path=args.hyperparams_path,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        threshold=args.threshold,
        run_latent=not args.no_latent,
        run_fid=not args.no_fid,
        run_kmeans=not args.no_kmeans,
        run_tsne=not args.no_tsne,
        debug_rows=args.debug_rows,
    )


def validate_config(config):
    # type: (PANCDRPipelineConfig) -> None
    if not (0.0 < config.source_test_size < 1.0):
        raise ValueError("source_test_size must be in (0, 1)")
    if config.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    required = [
        ("source_omics_path", resolve_path(config, config.source_omics_path)),
        ("source_response_path", resolve_path(config, config.source_response_path)),
        ("target_omics_path", resolve_path(config, config.target_omics_path)),
        ("target_eval_config", resolve_path(config, config.target_eval_config)),
        ("drug_smiles_path", resolve_path(config, config.drug_smiles_path)),
        ("hyperparams_path", resolve_path(config, config.hyperparams_path)),
    ]
    for name, path in required:
        if path is None or not path.exists():
            raise FileNotFoundError("Missing required file for {}: {}".format(name, path))


def config_to_dict(config):
    # type: (PANCDRPipelineConfig) -> Dict[str, Any]
    return asdict(config)


def save_config(config, path):
    # type: (PANCDRPipelineConfig, Path) -> None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(config_to_dict(config), f, indent=2)


def load_config(path):
    # type: (Path) -> PANCDRPipelineConfig
    with open(str(path), "r") as f:
        data = json.load(f)
    try:
        fields = PANCDRPipelineConfig.__dataclass_fields__
    except AttributeError:
        fields = getattr(PANCDRPipelineConfig, "__annotations__", {})
    filtered = {k: v for k, v in data.items() if k in fields}
    return PANCDRPipelineConfig(**filtered)
