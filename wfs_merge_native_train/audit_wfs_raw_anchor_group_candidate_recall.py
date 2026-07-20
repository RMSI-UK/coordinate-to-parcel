#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import pandas as pd

from train_wfs_raw_anchor_group_model import (
    DEFAULT_TARGET_GPKG,
    DEFAULT_TARGET_LAYER,
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_CLEAN_GPKG,
    DEFAULT_WFS_CLEAN_LAYER,
    _add_uprn_counts,
    _build_edges,
    _build_source_indexes,
    _collect_anchor_pool,
    _enumerate_anchor_groups,
    _enumerate_anchor_groups_with_shape_supplement,
    _ids_text,
    _parse_bbox,
    _parse_id_set,
    _read_clean_wfs,
    _read_targets,
    _safe_ratio,
    _target_clean_set,
)


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_full_v1/"
    "wfs_raw_anchor_group_model_v1.joblib"
)
DEFAULT_OUTPUT_JSON = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_full_v1/"
    "wfs_raw_anchor_group_candidate_recall_audit.json"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _param(args: argparse.Namespace, params: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return params.get(name, default)


def _is_connected(clean_ids: frozenset[int], adjacency: dict[int, list[tuple[int, float]]]) -> bool:
    if len(clean_ids) <= 1:
        return True
    remaining = set(int(v) for v in clean_ids)
    stack = [remaining.pop()]
    seen = {stack[0]}
    while stack:
        current = stack.pop()
        for neighbor, _shared in adjacency.get(int(current), []):
            neighbor = int(neighbor)
            if neighbor not in clean_ids or neighbor in seen:
                continue
            seen.add(neighbor)
            if neighbor in remaining:
                remaining.remove(neighbor)
            stack.append(neighbor)
    return not remaining


def _failure_reason(
    *,
    missing_source_ids: set[int],
    anchor_clean_ids: frozenset[int],
    target_clean_ids: frozenset[int],
    eligible_clean_ids: set[int],
    pool: set[int],
    target_area: float,
    max_group_size: int,
    max_candidate_area: float,
    adjacency: dict[int, list[tuple[int, float]]],
) -> str:
    if missing_source_ids:
        return "missing_source_in_clean_wfs"
    if not anchor_clean_ids:
        return "anchor_missing_in_clean_wfs"
    if not target_clean_ids:
        return "target_clean_empty"
    if len(target_clean_ids) > int(max_group_size):
        return "target_exceeds_max_group_size"
    if float(target_area) > float(max_candidate_area):
        return "target_exceeds_max_candidate_area"
    missing_eligible = target_clean_ids - set(int(v) for v in eligible_clean_ids)
    if missing_eligible:
        return "target_contains_non_plot_eligible_clean"
    missing_pool = target_clean_ids - set(int(v) for v in pool)
    if missing_pool:
        return "target_not_reached_by_anchor_pool"
    if not _is_connected(target_clean_ids, adjacency):
        return "target_not_connected_in_shared_edge_graph"
    return "enumeration_limit_or_ordering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit production candidate generation recall against council labels.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--neighbor-depth", type=int, default=None)
    parser.add_argument("--max-pool-size", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument("--max-candidate-area", type=float, default=None)
    parser.add_argument("--per-anchor-candidate-limit", type=int, default=None)
    parser.add_argument("--shape-supplement-pool-limit", type=int, default=None)
    parser.add_argument("--shape-supplement-keep", type=int, default=None)
    parser.add_argument("--top-neighbors", type=int, default=None)
    parser.add_argument("--min-shared-edge", type=float, default=None)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.model)
    params = dict(payload.get("training_params", {})) if isinstance(payload, dict) else {}
    neighbor_depth = int(_param(args, params, "neighbor_depth", 3))
    max_pool_size = int(_param(args, params, "max_pool_size", 22))
    max_group_size = int(_param(args, params, "max_group_size", 7))
    max_candidate_area = float(_param(args, params, "max_candidate_area", 2500.0))
    per_anchor_candidate_limit = int(_param(args, params, "per_anchor_candidate_limit", 80))
    shape_supplement_pool_limit = int(_param(args, params, "shape_supplement_pool_limit", 0))
    shape_supplement_keep = int(_param(args, params, "shape_supplement_keep", 0))
    top_neighbors = int(_param(args, params, "top_neighbors", 14))
    min_shared_edge = float(_param(args, params, "min_shared_edge", 0.05))

    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    target = _read_targets(Path(args.target_gpkg), str(args.target_layer), bbox, int(args.max_target_rows))
    source_to_clean, _source_by_clean = _build_source_indexes(wfs)
    nodes = wfs[wfs["plot_eligible"].astype(bool)].copy()
    eligible_clean_ids = set(nodes["clean_fid"].astype(int))
    _edges, adjacency, _shared_by_pair = _build_edges(
        nodes,
        min_shared_edge=min_shared_edge,
        top_neighbors=top_neighbors,
        query_chunk_size=int(args.edge_query_chunk_size),
        edge_calc_chunk_size=int(args.edge_calc_chunk_size),
    )
    area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
    perimeter_by_clean = wfs.set_index("clean_fid")["perimeter"].astype(float).to_dict()

    records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    generated_exact_count = 0
    target_count = 0
    for row_index, row in enumerate(target.itertuples(index=False), start=1):
        target_count += 1
        target_source_ids = set(int(v) for v in getattr(row, "target_source_set", _parse_id_set(row.source_wfs_fids)))
        anchor_source_fid = int(row.anchor_source_fid)
        target_train_component_id = int(row.train_component_id)
        anchor_clean_ids = frozenset(source_to_clean.get(anchor_source_fid, []))
        target_clean_ids, missing_source_ids = _target_clean_set(target_source_ids, source_to_clean)
        target_area = sum(float(area_by_clean.get(clean_fid, 0.0)) for clean_fid in target_clean_ids)
        pool = _collect_anchor_pool(
            anchor_clean_ids=anchor_clean_ids,
            positive_clean_ids=frozenset(),
            adjacency=adjacency,
            eligible_clean_ids=eligible_clean_ids,
            max_depth=neighbor_depth,
            max_pool_size=max_pool_size,
        )
        generated_exact = False
        generated_group_count = 0
        if not missing_source_ids and anchor_clean_ids and target_clean_ids:
            if int(shape_supplement_pool_limit) > int(per_anchor_candidate_limit):
                groups = set(
                    _enumerate_anchor_groups_with_shape_supplement(
                        anchor_clean_ids=anchor_clean_ids,
                        pool=pool,
                        adjacency=adjacency,
                        shared_by_pair=_shared_by_pair,
                        area_by_clean=area_by_clean,
                        perimeter_by_clean=perimeter_by_clean,
                        max_group_size=max_group_size,
                        max_candidate_area=max_candidate_area,
                        per_anchor_limit=per_anchor_candidate_limit,
                        shape_supplement_pool_limit=shape_supplement_pool_limit,
                        shape_supplement_keep=shape_supplement_keep,
                    )
                )
            else:
                groups = _enumerate_anchor_groups(
                    anchor_clean_ids=anchor_clean_ids,
                    pool=pool,
                    adjacency=adjacency,
                    area_by_clean=area_by_clean,
                    max_group_size=max_group_size,
                    max_candidate_area=max_candidate_area,
                    per_anchor_limit=per_anchor_candidate_limit,
                )
            generated_group_count = len(groups)
            generated_exact = target_clean_ids in groups
        if generated_exact:
            generated_exact_count += 1
            reason = "generated_exact"
        else:
            reason = _failure_reason(
                missing_source_ids=missing_source_ids,
                anchor_clean_ids=anchor_clean_ids,
                target_clean_ids=target_clean_ids,
                eligible_clean_ids=eligible_clean_ids,
                pool=pool,
                target_area=target_area,
                max_group_size=max_group_size,
                max_candidate_area=max_candidate_area,
                adjacency=adjacency,
            )
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        missing_pool_clean = target_clean_ids - set(int(v) for v in pool)
        records.append(
            {
                "target_train_component_id": target_train_component_id,
                "anchor_source_fid": anchor_source_fid,
                "target_source_fids": _ids_text(target_source_ids),
                "target_clean_fids": _ids_text(target_clean_ids),
                "anchor_clean_fids": _ids_text(anchor_clean_ids),
                "generated_exact": int(generated_exact),
                "failure_reason": reason,
                "target_source_count": int(len(target_source_ids)),
                "target_clean_count": int(len(target_clean_ids)),
                "anchor_clean_count": int(len(anchor_clean_ids)),
                "target_area": float(target_area),
                "pool_size": int(len(pool)),
                "missing_pool_clean_count": int(len(missing_pool_clean)),
                "missing_pool_clean_fids": _ids_text(missing_pool_clean),
                "missing_source_count": int(len(missing_source_ids)),
                "missing_source_fids": _ids_text(missing_source_ids),
                "generated_group_count": int(generated_group_count),
            }
        )
        if row_index % 5000 == 0:
            _log(
                "[INFO] Audited targets "
                f"{row_index:,}/{len(target):,}; generated_exact={generated_exact_count:,}; "
                f"coverage={_safe_ratio(generated_exact_count, target_count):.4f}"
            )

    audit = pd.DataFrame.from_records(records)
    summary = {
        "model": str(args.model),
        "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
        "wfs_clean_layer": str(args.wfs_clean_layer),
        "target_gpkg": str(args.target_gpkg),
        "target_layer": str(args.target_layer),
        "bbox": str(args.bbox),
        "params": {
            "neighbor_depth": neighbor_depth,
            "max_pool_size": max_pool_size,
            "max_group_size": max_group_size,
            "max_candidate_area": max_candidate_area,
            "per_anchor_candidate_limit": per_anchor_candidate_limit,
            "shape_supplement_pool_limit": shape_supplement_pool_limit,
            "shape_supplement_keep": shape_supplement_keep,
            "top_neighbors": top_neighbors,
            "min_shared_edge": min_shared_edge,
        },
        "target_rows": int(target_count),
        "generated_exact_rows": int(generated_exact_count),
        "generation_exact_recall": _safe_ratio(float(generated_exact_count), float(target_count)),
        "failure_reason_counts": {
            str(key): int(value)
            for key, value in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if str(args.output_csv).strip():
        output_csv = Path(args.output_csv)
    else:
        output_csv = output_json.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False)

    _log("[DONE] Candidate generation recall audit complete")
    _log(json.dumps(summary, indent=2))
    _log(f"[DONE] output_json={output_json}")
    _log(f"[DONE] output_csv={output_csv}")


if __name__ == "__main__":
    main()
