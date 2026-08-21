"""Check whether a BodyPressureSD installation can support mesh/v2vP metrics."""

import argparse
import json
from pathlib import Path

import numpy as np

from BodyPressure.pose_estimation.dataset import BodyPressureSDPoseDataset, _load_pickle
from BodyPressure.pose_estimation.train import read_file_list


def audit_assets(data_root, file_list, smpl_root=None):
    root = BodyPressureSDPoseDataset.resolve_data_root(data_root)
    candidates = (root / "GT_BP_DATA" / "bp2", root / "GT_BP_data" / "bp2")
    gt_root = next((path for path in candidates if path.is_dir()), candidates[0])
    entries, ready = [], True
    for filename in file_list:
        stem = Path(filename).stem
        source = root / "synth" / filename
        vertices = gt_root / f"{stem}_gt_vertices.npy"
        pmaps = gt_root / f"{stem}_gt_pmaps.npy"
        source_count = len(_load_pickle(source)["images"]) if source.is_file() else None
        vertex_count = int(np.load(vertices, mmap_mode="r").shape[0]) if vertices.is_file() else None
        pmap_count = int(np.load(pmaps, mmap_mode="r").shape[0]) if pmaps.is_file() else None
        counts = (source_count, vertex_count, pmap_count)
        aligned = None not in counts and len(set(counts)) == 1
        ready &= aligned
        entries.append({
            "source": str(source), "source_samples": source_count,
            "vertices": str(vertices), "vertices_samples": vertex_count,
            "pmaps": str(pmaps), "pmaps_samples": pmap_count, "aligned": aligned,
        })
    smpl_files = {}
    if smpl_root:
        smpl_root = Path(smpl_root).expanduser().resolve()
        for gender in ("MALE", "FEMALE"):
            path = smpl_root / f"SMPL_{gender}.pkl"
            smpl_files[gender.lower()] = str(path) if path.is_file() else None
        ready &= all(smpl_files.values())
    parsed_files = {}
    for name in ("EA1.npy", "EA2.npy"):
        path = root / "parsed" / name
        parsed_files[name] = str(path) if path.is_file() else None
    ready &= all(parsed_files.values())
    return {
        "ready_for_pve_v2vp": bool(ready), "data_root": str(root),
        "gt_root": str(gt_root), "smpl_models": smpl_files,
        "parsed_indexes": parsed_files, "files": entries,
        "requirements": {
            "mpjpe": "markers_xyz_m",
            "pve": "SMPL model/parameters and *_gt_vertices.npy",
            "body_dimensions": "predicted vertices and SHAPY measurement dependencies",
            "v2vP": "predicted per-vertex pressure, *_gt_pmaps.npy, and parsed EA indexes",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--smpl_root")
    parser.add_argument("--out", default="mesh_asset_audit.json")
    args = parser.parse_args()
    result = audit_assets(args.data_root, read_file_list(args.files), args.smpl_root)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["ready_for_pve_v2vp"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
