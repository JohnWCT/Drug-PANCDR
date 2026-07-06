"""Global drug index across source and target eval sets."""

from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd

from pancdr_pipeline.schema import normalize_drug_key


TARGET_EVAL_NAMES = [
    "target_primary",
    "target_only",
    "dapl",
    "target_aacdr",
    "target_aacdr_only",
]


@dataclass
class DrugIndexResult:
    drug_index: pd.DataFrame
    availability_report: pd.DataFrame
    zero_shot_report: pd.DataFrame
    drug_keys: List[str]
    smiles_map: Dict[str, str]


def build_drug_index(source_response, target_eval_sets, drug_smiles):
    # type: (...) -> DrugIndexResult
    source_drugs = set(source_response["drug_key"].astype(str))
    target_sets = {}
    for name in TARGET_EVAL_NAMES:
        if name in target_eval_sets:
            target_sets[name] = set(target_eval_sets[name]["drug_key"].astype(str))
        else:
            target_sets[name] = set()

    all_drugs = set(source_drugs)
    for s in target_sets.values():
        all_drugs |= s
    all_drugs = sorted(all_drugs)

    smiles_map = {}
    has_smiles = set()
    for _, row in drug_smiles.iterrows():
        key = str(row["drug_key"])
        smi = str(row.get("smiles", "")).strip()
        if smi and smi.lower() not in ("nan", "none", ""):
            smiles_map[key] = smi
            has_smiles.add(key)

    rows = []
    for drug_key in all_drugs:
        in_source = drug_key in source_drugs
        flags = {name: drug_key in target_sets[name] for name in TARGET_EVAL_NAMES}
        in_any_target = any(flags.values())
        rows.append(
            {
                "drug_key": drug_key,
                "drug_name": drug_key,
                "in_source": int(in_source),
                **{"in_{}".format(k): int(flags[k]) for k in TARGET_EVAL_NAMES},
                "in_any_target": int(in_any_target),
                "is_target_only_drug": int((not in_source) and in_any_target),
                "has_smiles": int(drug_key in has_smiles),
                "has_graph": 0,
            }
        )

    drug_index = pd.DataFrame(rows)
    availability_report = drug_index.copy()
    zero_shot = drug_index[drug_index["is_target_only_drug"] == 1].copy()
    return DrugIndexResult(
        drug_index=drug_index,
        availability_report=availability_report,
        zero_shot_report=zero_shot,
        drug_keys=all_drugs,
        smiles_map=smiles_map,
    )
