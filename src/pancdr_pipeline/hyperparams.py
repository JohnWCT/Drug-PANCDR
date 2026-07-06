"""Load shared PANCDR hyperparameters from tuned CSV."""

import ast
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pancdr_pipeline.config import PANCDRPipelineConfig, resolve_path


def load_pancdr_hyperparams(config):
    # type: (PANCDRPipelineConfig) -> Dict[str, Any]
    path = resolve_path(config, config.hyperparams_path)
    df = pd.read_csv(str(path))
    row = df.loc[(df["Model"] == "PANCDR") & (df["Classification"] == "T"), "Best_params"].values[0]
    if isinstance(row, str):
        try:
            params = ast.literal_eval(row)
        except (ValueError, SyntaxError):
            params = eval(row)  # noqa: S307 - legacy upstream format
    else:
        params = row
    return dict(params)


def save_hyperparams(params, output_dir):
    # type: (Dict[str, Any], Path) -> None
    path = Path(output_dir) / "hyperparams.json"
    with open(str(path), "w") as f:
        json.dump(params, f, indent=2)
