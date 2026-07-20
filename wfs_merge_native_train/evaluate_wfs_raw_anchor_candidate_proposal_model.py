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
import numpy as np
import pandas as pd

from train_wfs_raw_anchor_candidate_proposal_model import cheap_candidate_features
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
    _enumerate_anchor_groups_ordered,
    _ids_text,
    _parse_bbox,
    _read_clean_wfs,
    _read_targets,
    _safe_ratio,
    _target_clean_set,
)


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_full_sampled_v1/"
    "wfs_raw_anchor_candidate_proposal_model_v1.joblib"
)
DEFAULT_OUTPUT_JSON = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_full_sampled_v1/"
    "wfs_raw_anchor_candidate_proposal_expanded_rank_eval.json"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_int_list(value: str) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _rank_position(scores: np.ndarray, fast_scores: np.ndarray, exact_idx: int) -> int:
    order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), -float(fast_scores[idx]), idx))
    for rank, idx in enumerate(order, start=1):
        if int(idx) == int(exact_idx):
            return int(rank)
    return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proposal model on the true expanded candidate pool.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--target-id-mod", type=int, default=0)
    parser.add_argument("--target-id-remainders", default="")
    parser.add_argument("--neighbor-depth", type=int, default=3)
    parser.add_argument("--max-pool-size", type=int, default=22)
    parser.add_argument("--max-group-size", type=int, default=10)
    parser.add_argument("--max-candidate-area", type=float, default=6000.0)
    parser.add_argument("--expanded-candidate-limit", type=int, default=1500)
    parser.add_argument("--top-neighbors", type=int, default=14)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--rank-recall-at", default="20,40,80,120,160,200")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--log-every-targets", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.model)
    if not isinstance(payload, dict) or payload.get("model_kind") != "wfs_raw_anchor_candidate_proposal_ranker":
        raise RuntimeError("--model must be a wfs_raw_anchor_candidate_proposal_ranker payload.")
    pipeline = payload["pipeline"]
    feature_cols = list(payload["feature_cols"])
    ks = [int(part.strip()) for part in str(args.rank_recall_at).split(",") if part.strip()]

    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    target = _read_targets(Path(args.target_gpkg), str(args.target_layer), bbox, int(args.max_target_rows))
    if int(args.target_id_mod) > 0:
        remainders = _parse_int_list(str(args.target_id_remainders))
        if not remainders:
            raise ValueError("--target-id-remainders is required when --target-id-mod is set")
        target = target[
            target["train_component_id"].astype(int).mod(int(args.target_id_mod)).isin(remainders)
        ].copy()
        _log(f"[INFO] Target id modulo filter applied: rows={len(target):,}")

    source_to_clean, source_by_clean = _build_source_indexes(wfs)
    nodes = wfs[wfs["plot_eligible"].astype(bool)].copy()
    eligible_clean_ids = set(nodes["clean_fid"].astype(int))
    _edges, adjacency, shared_by_pair = _build_edges(
        nodes,
        min_shared_edge=float(args.min_shared_edge),
        top_neighbors=int(args.top_neighbors),
        query_chunk_size=int(args.edge_query_chunk_size),
        edge_calc_chunk_size=int(args.edge_calc_chunk_size),
    )
    attrs_by_clean = wfs.set_index("clean_fid").drop(columns="geometry").to_dict("index")
    area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
    perimeter_by_clean = wfs.set_index("clean_fid")["perimeter"].astype(float).to_dict()

    rows: list[dict[str, Any]] = []
    counts = {
        "target_rows": 0,
        "skipped_missing_source_targets": 0,
        "skipped_no_anchor_targets": 0,
        "generated_exact_targets": 0,
        "expanded_group_count": 0,
    }
    topk_ok = {int(k): 0 for k in ks}
    for row_index, row in enumerate(target.itertuples(index=False), start=1):
        counts["target_rows"] += 1
        target_source_ids = set(int(v) for v in getattr(row, "target_source_set"))
        anchor_source_fid = int(row.anchor_source_fid)
        target_train_component_id = int(row.train_component_id)
        anchor_clean_ids = frozenset(source_to_clean.get(anchor_source_fid, []))
        target_clean_ids, missing_source_ids = _target_clean_set(target_source_ids, source_to_clean)
        if missing_source_ids:
            counts["skipped_missing_source_targets"] += 1
            rows.append(
                {
                    "target_train_component_id": target_train_component_id,
                    "anchor_source_fid": anchor_source_fid,
                    "status": "missing_source",
                    "exact_rank": -1,
                    "expanded_group_count": 0,
                }
            )
            continue
        if not anchor_clean_ids or not target_clean_ids:
            counts["skipped_no_anchor_targets"] += 1
            rows.append(
                {
                    "target_train_component_id": target_train_component_id,
                    "anchor_source_fid": anchor_source_fid,
                    "status": "no_anchor_or_target",
                    "exact_rank": -1,
                    "expanded_group_count": 0,
                }
            )
            continue

        pool = _collect_anchor_pool(
            anchor_clean_ids=anchor_clean_ids,
            positive_clean_ids=frozenset(),
            adjacency=adjacency,
            eligible_clean_ids=eligible_clean_ids,
            max_depth=int(args.neighbor_depth),
            max_pool_size=int(args.max_pool_size),
        )
        groups = _enumerate_anchor_groups_ordered(
            anchor_clean_ids=anchor_clean_ids,
            pool=pool,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=int(args.max_group_size),
            max_candidate_area=float(args.max_candidate_area),
            per_anchor_limit=int(args.expanded_candidate_limit),
        )
        counts["expanded_group_count"] += int(len(groups))
        if target_clean_ids not in set(groups):
            rows.append(
                {
                    "target_train_component_id": target_train_component_id,
                    "anchor_source_fid": anchor_source_fid,
                    "status": "exact_not_generated",
                    "exact_rank": -1,
                    "expanded_group_count": int(len(groups)),
                    "target_clean_fids": _ids_text(target_clean_ids),
                    "target_source_fids": _ids_text(target_source_ids),
                }
            )
            continue

        counts["generated_exact_targets"] += 1
        exact_idx = groups.index(target_clean_ids)
        records = [
            cheap_candidate_features(
                anchor_source_fid=anchor_source_fid,
                anchor_clean_ids=anchor_clean_ids,
                candidate_clean_ids=group,
                enum_rank=rank,
                target_train_component_id=target_train_component_id,
                source_by_clean=source_by_clean,
                attrs_by_clean=attrs_by_clean,
                area_by_clean=area_by_clean,
                perimeter_by_clean=perimeter_by_clean,
                adjacency=adjacency,
                shared_by_pair=shared_by_pair,
            )
            for rank, group in enumerate(groups, start=1)
        ]
        frame = pd.DataFrame.from_records(records)
        for column in feature_cols:
            if column not in frame.columns:
                frame[column] = np.nan
        scores = pipeline.predict_proba(frame[feature_cols])[:, 1]
        fast_scores = pd.to_numeric(frame["fast_shape_score"], errors="coerce").fillna(0.0).to_numpy()
        rank = _rank_position(scores, fast_scores, exact_idx)
        for k in ks:
            topk_ok[int(k)] += int(0 < rank <= int(k))
        rows.append(
            {
                "target_train_component_id": target_train_component_id,
                "anchor_source_fid": anchor_source_fid,
                "status": "generated_exact",
                "exact_rank": int(rank),
                "expanded_group_count": int(len(groups)),
                "exact_proposal_proba": float(scores[exact_idx]),
                "target_clean_fids": _ids_text(target_clean_ids),
                "target_source_fids": _ids_text(target_source_ids),
            }
        )
        if row_index % int(args.log_every_targets) == 0:
            _log(
                "[INFO] Evaluated proposal targets "
                f"{row_index:,}/{len(target):,}; generated_exact={counts['generated_exact_targets']:,}"
            )

    generated = int(counts["generated_exact_targets"])
    summary = {
        "model": str(args.model),
        "target_gpkg": str(args.target_gpkg),
        "target_layer": str(args.target_layer),
        "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
        "wfs_clean_layer": str(args.wfs_clean_layer),
        "bbox": str(args.bbox),
        "max_target_rows": int(args.max_target_rows),
        "target_id_mod": int(args.target_id_mod),
        "target_id_remainders": sorted(_parse_int_list(str(args.target_id_remainders))),
        "params": {
            "neighbor_depth": int(args.neighbor_depth),
            "max_pool_size": int(args.max_pool_size),
            "max_group_size": int(args.max_group_size),
            "max_candidate_area": float(args.max_candidate_area),
            "expanded_candidate_limit": int(args.expanded_candidate_limit),
            "top_neighbors": int(args.top_neighbors),
            "min_shared_edge": float(args.min_shared_edge),
        },
        **{key: int(value) for key, value in counts.items()},
        "generation_exact_recall_all_targets": _safe_ratio(
            float(counts["generated_exact_targets"]),
            float(counts["target_rows"]),
        ),
        "generation_exact_recall_available_targets": _safe_ratio(
            float(counts["generated_exact_targets"]),
            float(counts["target_rows"] - counts["skipped_missing_source_targets"] - counts["skipped_no_anchor_targets"]),
        ),
        "rank_recall_generated_targets": {
            f"recall_at_{int(k)}": {
                "ok": int(topk_ok[int(k)]),
                "denom": generated,
                "recall": _safe_ratio(float(topk_ok[int(k)]), float(generated)),
            }
            for k in ks
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if str(args.output_csv).strip():
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(rows).to_csv(output_csv, index=False)
        _log(f"[DONE] output_csv={output_csv}")
    _log("[DONE] Expanded proposal rank evaluation complete")
    _log(json.dumps(summary, indent=2))
    _log(f"[DONE] output_json={output_json}")


if __name__ == "__main__":
    main()
