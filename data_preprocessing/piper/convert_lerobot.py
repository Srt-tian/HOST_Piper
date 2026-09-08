"""Manifest-driven LeRobot v2.1 -> HOST conversion; never fabricates alignment.

No network access. Raw videos remain immutable and are symlinked into the output.
Manifest conventions and action semantics must be explicitly confirmed before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

CAMERAS = ("cam_front", "cam_left", "cam_right")
SIDES = ("left", "right")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite a prior conversion/alignment run.
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)


def keys(prefix):
    return [f"{prefix}_{side}_{part}" for side in SIDES
            for part in ("position", "rotation", "gripper")]


def validate_manifest(manifest):
    c = manifest["conventions"]
    if c.get("confirmed") is not True:
        raise ValueError("Confirm units, RPY, TCP and target_source in manifest first")
    if c.get("rpy") != "Rz(yaw)Ry(pitch)Rx(roll)":
        raise ValueError("Only explicitly confirmed Rz(yaw)Ry(pitch)Rx(roll) is supported")
    if c.get("position_unit") not in ("m", "mm") or c.get("angle_unit") not in ("rad", "deg"):
        raise ValueError("Explicit position_unit=m/mm and angle_unit=rad/deg required")
    if not c.get("end_frame"):
        raise ValueError("Explicit flange/TCP end_frame required")
    if c.get("target_source") not in ("executed_state", "command"):
        raise ValueError("Explicit executed_state or command target_source required")
    if c["target_source"] == "command" and not c.get("command_semantics_confirmed"):
        raise ValueError("Confirm action_end/real_action correspondence and coordinate frames")
    seen = set()
    for s in manifest["sources"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", s["id"]) or s["id"] in seen:
            raise ValueError("Each source needs a unique filesystem-safe id")
        seen.add(s["id"])
        if not s.get("task") or not s.get("instruction"):
            raise ValueError("Explicit semantic task and instruction required")
        if not Path(s["root"]).is_absolute():
            raise ValueError("Source root must be absolute")


def array(table, name, width):
    result = np.asarray(table[name].to_pylist(), dtype=np.float64)
    if result.shape != (len(table), width) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: shape={result.shape}; expected {(len(table), width)}")
    return result


def make_rows(table, conventions):
    state = array(table, "state_end", 12)
    qpos = array(table, "observation.qpos", 14)
    command = array(table, "action_end", 12)
    real = array(table, "real_action", 14)
    times = np.asarray(table["timestamp"].to_pylist(), dtype=float).reshape(-1)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode timestamps must be finite and strictly increasing")
    pos_scale = {"m": 1.0, "mm": 0.001}[conventions["position_unit"]]
    rot_scale = {"rad": 1.0, "deg": np.pi / 180}[conventions["angle_unit"]]
    rows = []
    for i, timestamp in enumerate(times):
        row = {"timestamp": float(timestamp), "source_frame_index": i}
        for prefix, poses, joints in (("follow", state, qpos), ("master", command, real)):
            for side, p, g in (("left", 0, 6), ("right", 6, 13)):
                row[f"{prefix}_{side}_position"] = (poses[i, p:p+3] * pos_scale).tolist()
                row[f"{prefix}_{side}_rotation"] = (poses[i, p+3:p+6] * rot_scale).tolist()
                row[f"{prefix}_{side}_gripper"] = float(joints[i, g])
        rows.append(row)
    return rows, times


def assign_splits(episodes, val_fraction, seed):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = {}
    for ep in episodes:
        groups.setdefault(ep["task"], []).append(ep)
    for task, group in groups.items():
        # Each split needs >=2 independent episodes for same-task reference pairing.
        if len(group) < 4:
            raise ValueError(f"{task}: at least four episodes required for split-local peers")
        ordered = sorted(group, key=lambda e: hashlib.sha256(
            f"{seed}:{e['source_id']}:{e['episode_index']}".encode()).hexdigest())
        nval = max(2, min(len(group)-2, round(len(group)*val_fraction)))
        for i, ep in enumerate(ordered):
            ep["split"] = "val" if i < nval else "train"


def normalization(episodes, target_source):
    bounds = {}
    for ep in episodes:
        if ep["split"] != "train":
            continue
        for key in keys("follow") + keys("master"):
            data = np.array([np.atleast_1d(row[key]) for row in ep["rows"]])
            low, high = data.min(0), data.max(0)
            if key in bounds:
                low = np.minimum(bounds[key][0], low)
                high = np.maximum(bounds[key][1], high)
            bounds[key] = low, high
    nmd = {k: {"min": low.tolist(), "delta": np.maximum(high-low, 1e-6).tolist()}
           for k, (low, high) in bounds.items()}
    prefix = "follow" if target_source == "executed_state" else "master"
    for side in SIDES:
        key, ref = f"{prefix}_{side}_position", f"follow_{side}_position"
        low = bounds[key][0] - bounds[ref][1]
        high = bounds[key][1] - bounds[ref][0]
        nmd[key+"_relative"] = {"min": low.tolist(), "delta": np.maximum(high-low, 1e-6).tolist()}
    return {"action_keys": keys(prefix), "joint_keys": keys("follow"), "norm_min_delta": nmd}


def convert(manifest, output):
    validate_manifest(manifest)
    output = Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError("Output must be a new absolute directory")
    episodes = []
    for source in manifest["sources"]:
        root = Path(source["root"]).resolve()
        info = read_json(root / "meta/info.json")
        if info["codebase_version"] != "v2.1" or info["fps"] != 30:
            raise ValueError("This profile requires LeRobot v2.1 / actual 30Hz metadata")
        eps = [json.loads(line) for line in (root/"meta/episodes.jsonl").read_text().splitlines() if line]
        bad_file = root/"meta/is_bad_data.jsonl"
        if bad_file.exists() and bad_file.read_text().strip():
            raise ValueError(f"Review nonempty quality marks before conversion: {bad_file}")
        for meta in eps:
            idx = meta["episode_index"]
            fmt = {"episode_index": idx, "episode_chunk": idx//info["chunks_size"]}
            path = root / info["data_path"].format(**fmt)
            table = pq.read_table(path)
            if len(table) != meta["length"]:
                raise ValueError(f"Parquet/metadata length mismatch: {path}")
            rows, times = make_rows(table, manifest["conventions"])
            videos = {}
            import av
            for cam in CAMERAS:
                video = root / info["video_path"].format(**fmt, video_key=f"observation.images.{cam}")
                # Decode count, not only container metadata; failure cannot silently create black frames.
                with av.open(str(video)) as container:
                    count = sum(1 for _ in container.decode(video=0))
                if count != len(rows):
                    raise ValueError(f"Video/trajectory length mismatch: {video}: {count}/{len(rows)}")
                videos[cam] = str(video)
            episodes.append(dict(source_id=source["id"], task=source["task"],
                instruction=source["instruction"], episode_index=idx, rows=rows, videos=videos,
                timestamp_max_gap=float(np.diff(times).max()), parquet=str(path),
                parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assign_splits(episodes, manifest.get("val_fraction", 0.1), manifest.get("seed", 42))
    norm = normalization(episodes, manifest["conventions"]["target_source"])
    for ep in episodes:
        ep["output"] = str(output/"episodes"/ep["split"]/ep["source_id"]/f"episode_{ep['episode_index']:06d}")
    for ep in episodes:
        dest = Path(ep["output"])
        write_json(dest/(dest.name+".json"), {"data": ep["rows"]})
        with (dest/"instruction.txt").open("x") as f:
            f.write(ep["instruction"]+"\n")
        for cam, source_path in ep["videos"].items():
            (dest/f"{cam}.mp4").symlink_to(source_path)
        peers = [p["output"] for p in episodes if p is not ep
                 and p["task"] == ep["task"] and p["split"] == ep["split"]]
        write_json(dest/"task_paths.json", {"same": peers})
        write_json(dest/"provenance.json", {k:v for k,v in ep.items() if k != "rows"})
    cam_mapping = {str(Path(ep["output"]).parent): list(CAMERAS) for ep in episodes}
    for split in ("train", "val"):
        write_json(output/split/"piper_video_paths.json", [ep["output"] for ep in episodes if ep["split"] == split])
    write_json(output/"cam_mapping/piper_cam_mapping.json", cam_mapping)
    write_json(output/"joint_action_mapping/piper_joint_action_mapping.json", {"piper": norm})
    write_json(output/"manifest.json", manifest)
    write_json(output/"conversion_summary.json", {
        "episodes": len(episodes), "frames": sum(len(ep["rows"]) for ep in episodes),
        "progress_generated": False, "text_embeddings_generated": False,
        "time_policy": "preserve source row/image index; no 30-to-32Hz relabeling",
        "norm_policy": "train-only extrema; relative position uses conservative difference bounds",
        "split_policy": "deterministic episode holdout within semantic task; not session holdout"})
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = convert(read_json(args.manifest), args.output)
    print(json.dumps({"converted_episodes": len(result), "output": args.output}))
