#!/usr/bin/env python
"""Thin CLI entry point for PANCDR 5-fold + multi-TCGA pipeline."""

import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from pancdr_pipeline.config import build_arg_parser, config_from_args, validate_config
from pancdr_pipeline.run import run_pipeline


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    validate_config(config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
