"""Instance counting: return the number of unique instances after 3D clustering."""

from __future__ import annotations

import numpy as np


def count_unique_instances(results_3d: dict) -> dict:
    """Count unique instances from 3D localization results (after clustering).

    Returns:
        dict with total_unique and obj_id_list.
    """
    obj_id_list = results_3d.get("obj_id_list", [])
    return {
        "total_unique": len(obj_id_list),
        "obj_id_list": obj_id_list,
    }


def print_counting_summary(counting_result: dict, text_prompt: str) -> None:
    print(f'Scene has {counting_result["total_unique"]} unique "{text_prompt}" instances')
    print(f'Instance IDs: {counting_result["obj_id_list"]}')
