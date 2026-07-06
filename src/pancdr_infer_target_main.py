#!/usr/bin/env python
"""Inference-only TCGA target evaluation from pretrained PANCDR fold checkpoints."""

import argparse
import json
import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from pancdr_pipeline.config import PANCDRPipelineConfig, load_config
from pancdr_pipeline.infer import run_target_inference


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_eval_config", default=None)
    p.add_argument("--drug_smiles_path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--eval_prefixes",
        default="target_primary,target_only,dapl,target_aacdr,target_aacdr_only",
    )
    p.add_argument("--skip_reports", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base = load_config(os.path.join(args.checkpoint_dir, "config.json"))

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.target_eval_config:
        overrides["target_eval_config"] = args.target_eval_config
    if args.drug_smiles_path:
        overrides["drug_smiles_path"] = args.drug_smiles_path
    if args.device:
        overrides["device"] = args.device
    overrides["run_latent"] = False
    overrides["run_fid"] = False
    overrides["run_kmeans"] = False
    overrides["run_tsne"] = False

    merged = dict(base.__dict__)
    merged.update(overrides)
    config = PANCDRPipelineConfig(**merged)

    eval_prefixes = [x.strip() for x in args.eval_prefixes.split(",") if x.strip()]
    out = run_target_inference(
        config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        eval_prefixes=eval_prefixes,
        skip_reports=args.skip_reports,
    )
    print(json.dumps({"output_dir": str(out), "eval_prefixes": eval_prefixes}, indent=2))


if __name__ == "__main__":
    main()
