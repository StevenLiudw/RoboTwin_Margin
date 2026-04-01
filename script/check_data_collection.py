#!/usr/bin/env python3
"""
Audit RoboTwin collected datasets for structural/data completeness.

Checks:
1) Episode coverage across data/video/traj/instruction files
2) HDF5 schema consistency (observation cameras, rgb/depth, joint_action keys)
3) Optional video frame-count consistency vs HDF5 frame count
4) Depth sanity and unit inference from sampled values

Example:
    python script/check_data_collection.py \
      /home/steven/Projects/Margin/RoboTwin/data/pick_diverse_bottles \
      --task-config demo_randomized
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


DRIVE_QPOS_KEYS = [
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
    "vector",
]

REAL_QPOS_KEYS = [
    "left_arm_real",
    "left_gripper_real",
    "right_arm_real",
    "right_gripper_real",
    "vector_real",
]


@dataclass
class DepthStats:
    sample_count: int = 0
    valid_pixel_count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    medians: List[float] = field(default_factory=list)

    def update(self, arr: np.ndarray) -> None:
        self.sample_count += 1
        valid = arr[np.isfinite(arr) & (arr > 0)]
        if valid.size == 0:
            return
        self.valid_pixel_count += int(valid.size)
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        self.min_value = min(self.min_value, vmin)
        self.max_value = max(self.max_value, vmax)
        self.medians.append(float(np.median(valid)))

    def summary(self) -> Dict[str, Optional[float]]:
        median_of_medians = float(np.median(self.medians)) if self.medians else None
        unit = infer_depth_unit(median_of_medians)
        return {
            "sample_count": self.sample_count,
            "valid_pixel_count": self.valid_pixel_count,
            "min_value": (None if self.min_value == float("inf") else self.min_value),
            "max_value": (None if self.max_value == float("-inf") else self.max_value),
            "median_of_medians": median_of_medians,
            "inferred_unit": unit,
        }


@dataclass
class AuditResult:
    split_name: str
    hdf5_count: int = 0
    episode_indices: List[int] = field(default_factory=list)
    video_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    depth_stats: DepthStats = field(default_factory=DepthStats)
    camera_names: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def infer_depth_unit(median_value: Optional[float]) -> str:
    if median_value is None:
        return "unknown"
    # Typical tabletop depth ~0.3m to 1.5m. If values are ~300-1500, this is millimeters.
    if median_value > 20:
        return "millimeters (mm)"
    if median_value > 0:
        return "meters (m)"
    return "unknown"


def parse_episode_idx(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"episode(\d+)(?:_[^.]*)?\.(hdf5|mp4|pkl|json)$", name)
    if not m:
        raise ValueError(f"Cannot parse episode index from: {path}")
    return int(m.group(1))


def sorted_episode_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    return sorted(files, key=lambda p: parse_episode_idx(p))


def camera_video_path(video_dir: str, episode_name: str, camera_name: str) -> str:
    if camera_name == "head_camera":
        return os.path.join(video_dir, f"{episode_name}.mp4")
    return os.path.join(video_dir, f"{episode_name}_{camera_name}.mp4")


def check_video_frame_count(video_path: str) -> Tuple[Optional[int], Optional[str]]:
    if cv2 is None:
        return None, "opencv-python not available; skip frame-count check"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "failed to open video"
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames, None


def sample_depth_frames(depth_ds: h5py.Dataset, max_samples_per_episode: int = 3) -> List[np.ndarray]:
    n = int(depth_ds.shape[0])
    if n <= 0:
        return []
    candidate = [0, n // 2, n - 1]
    uniq = []
    seen = set()
    for idx in candidate:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    uniq = uniq[:max_samples_per_episode]
    return [np.asarray(depth_ds[idx]) for idx in uniq]


def audit_split(
    split_name: str,
    root: str,
    expect_depth: bool,
    check_video_frames: bool,
    strict_video_frames: bool,
) -> AuditResult:
    result = AuditResult(split_name=split_name)
    data_dir = os.path.join(root, "data")
    video_dir = os.path.join(root, "video")

    if not os.path.isdir(data_dir):
        result.add_warning(f"{split_name}: data dir missing: {data_dir}")
        return result

    hdf5_files = sorted_episode_files(os.path.join(data_dir, "episode*.hdf5"))
    result.hdf5_count = len(hdf5_files)
    result.episode_indices = [parse_episode_idx(p) for p in hdf5_files]

    video_files = sorted_episode_files(os.path.join(video_dir, "episode*.mp4")) if os.path.isdir(video_dir) else []
    result.video_count = len(video_files)

    for hdf5_path in hdf5_files:
        episode_idx = parse_episode_idx(hdf5_path)
        episode_name = f"episode{episode_idx}"
        prefix = f"{split_name}:{episode_name}"

        try:
            with h5py.File(hdf5_path, "r") as f:
                if "observation" not in f:
                    result.add_error(f"{prefix}: missing group `observation`")
                    continue

                observation = f["observation"]
                camera_names = list(observation.keys())
                if not camera_names:
                    result.add_error(f"{prefix}: no cameras in observation")
                    continue
                if not result.camera_names:
                    result.camera_names = camera_names

                frame_count = None
                for cam in camera_names:
                    cgrp = observation[cam]
                    if "rgb" not in cgrp:
                        result.add_error(f"{prefix}:{cam}: missing `rgb` dataset")
                        continue

                    rgb_n = int(cgrp["rgb"].shape[0])
                    if frame_count is None:
                        frame_count = rgb_n
                    elif rgb_n != frame_count:
                        result.add_error(f"{prefix}:{cam}: rgb frame mismatch {rgb_n} != {frame_count}")

                    if expect_depth:
                        if "depth" not in cgrp:
                            result.add_error(f"{prefix}:{cam}: missing `depth` dataset")
                        else:
                            depth_n = int(cgrp["depth"].shape[0])
                            if frame_count is not None and depth_n != frame_count:
                                result.add_error(
                                    f"{prefix}:{cam}: depth frame mismatch {depth_n} != {frame_count}"
                                )
                            for depth_frame in sample_depth_frames(cgrp["depth"]):
                                result.depth_stats.update(depth_frame)
                    else:
                        if "depth" in cgrp:
                            # Informational only
                            result.add_warning(
                                f"{prefix}:{cam}: depth exists while expect_depth=False (config likely changed)"
                            )

                if "joint_action" not in f:
                    result.add_error(f"{prefix}: missing group `joint_action`")
                else:
                    jkeys = set(f["joint_action"].keys())
                    missing_drive = [k for k in DRIVE_QPOS_KEYS if k not in jkeys]
                    missing_real = [k for k in REAL_QPOS_KEYS if k not in jkeys]
                    if missing_drive:
                        result.add_error(f"{prefix}: missing drive qpos keys: {missing_drive}")
                    if missing_real:
                        result.add_warning(f"{prefix}: missing real qpos keys: {missing_real}")

                    if frame_count is not None:
                        for k in (DRIVE_QPOS_KEYS + REAL_QPOS_KEYS):
                            if k not in f["joint_action"]:
                                continue
                            ds = f["joint_action"][k]
                            if ds.shape and int(ds.shape[0]) != frame_count:
                                result.add_error(
                                    f"{prefix}: joint_action/{k} frame mismatch {ds.shape[0]} != {frame_count}"
                                )

                # Video checks per camera
                if not os.path.isdir(video_dir):
                    result.add_error(f"{prefix}: video dir missing: {video_dir}")
                else:
                    for cam in camera_names:
                        vpath = camera_video_path(video_dir, episode_name, cam)
                        if not os.path.exists(vpath):
                            result.add_error(f"{prefix}:{cam}: missing video {vpath}")
                            continue
                        if check_video_frames and frame_count is not None:
                            vframes, verr = check_video_frame_count(vpath)
                            if verr is not None:
                                result.add_warning(f"{prefix}:{cam}: {verr} ({vpath})")
                                continue
                            if vframes is None:
                                continue
                            if strict_video_frames:
                                ok = (vframes == frame_count)
                            else:
                                ok = (abs(vframes - frame_count) <= 1)
                            if not ok:
                                result.add_error(
                                    f"{prefix}:{cam}: video frame mismatch {vframes} vs hdf5 {frame_count}"
                                )
        except Exception as exc:
            result.add_error(f"{prefix}: failed to read hdf5: {type(exc).__name__}: {exc}")

    return result


def resolve_collection_root(dataset_root: str, task_config: Optional[str]) -> str:
    dataset_root = os.path.abspath(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, "data")):
        return dataset_root
    if task_config:
        candidate = os.path.join(dataset_root, task_config)
        if os.path.isdir(candidate):
            return candidate
    # Best effort fallback: first child that looks like a collection root.
    for child in sorted(os.listdir(dataset_root)):
        candidate = os.path.join(dataset_root, child)
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "data")):
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve collection root from {dataset_root}. "
        f"Pass --task-config if dataset_root is task-level."
    )


def print_result(result: AuditResult) -> None:
    print(f"\n=== {result.split_name.upper()} ===")
    print(f"HDF5 episodes: {result.hdf5_count}")
    print(f"Video files:   {result.video_count}")
    if result.episode_indices:
        print(
            f"Episode index range: {min(result.episode_indices)} .. {max(result.episode_indices)}"
        )
    if result.camera_names:
        print(f"Cameras: {result.camera_names}")

    dsum = result.depth_stats.summary()
    if dsum["sample_count"] > 0:
        print(
            "Depth samples: "
            f"{dsum['sample_count']} frames, "
            f"valid_pixels={dsum['valid_pixel_count']}, "
            f"min={dsum['min_value']:.3f}, "
            f"median~={dsum['median_of_medians']:.3f}, "
            f"max={dsum['max_value']:.3f}"
        )
        print(f"Depth inferred unit: {dsum['inferred_unit']}")
        if dsum["inferred_unit"] == "millimeters (mm)":
            print("Depth note: values are metric but stored in millimeters; divide by 1000 for meters.")
    else:
        print("Depth samples: none")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors[:200]:
            print(f"  - {e}")
        if len(result.errors) > 200:
            print(f"  ... truncated {len(result.errors) - 200} more errors")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:200]:
            print(f"  - {w}")
        if len(result.warnings) > 200:
            print(f"  ... truncated {len(result.warnings) - 200} more warnings")

    if not result.errors and not result.warnings:
        print("No issues detected.")


def build_overview(collection_root: str) -> Dict[str, object]:
    seed_file = os.path.join(collection_root, "seed.txt")
    seed_count = 0
    if os.path.exists(seed_file):
        with open(seed_file, "r", encoding="utf-8") as f:
            seed_count = len([x for x in f.read().split() if x.strip()])

    traj_success_count = len(glob.glob(os.path.join(collection_root, "_traj_data", "episode*.pkl")))
    traj_failed_count = len(glob.glob(os.path.join(collection_root, "_traj_data_failed", "episode*.pkl")))
    instruction_count = len(glob.glob(os.path.join(collection_root, "instructions", "episode*.json")))

    failed_records_path = os.path.join(collection_root, "failed_episode_records.json")
    failed_records_count = 0
    failed_collected_true = 0
    failed_collected_no_error = 0
    failed_collected_with_error = 0
    if os.path.exists(failed_records_path):
        with open(failed_records_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        if isinstance(rec, list):
            failed_records_count = len(rec)
            failed_collected_true = sum(1 for r in rec if bool(r.get("collected", False)))
            failed_collected_no_error = sum(
                1 for r in rec if bool(r.get("collected", False)) and not r.get("collection_error")
            )
            failed_collected_with_error = sum(
                1 for r in rec if bool(r.get("collected", False)) and bool(r.get("collection_error"))
            )

    return {
        "seed_count": seed_count,
        "traj_success_count": traj_success_count,
        "traj_failed_count": traj_failed_count,
        "instruction_count": instruction_count,
        "failed_records_count": failed_records_count,
        "failed_records_collected_true": failed_collected_true,
        "failed_records_collected_no_error": failed_collected_no_error,
        "failed_records_collected_with_error": failed_collected_with_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RoboTwin dataset collection completeness and depth quality.")
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Task-level path or collection root (e.g., data/pick_diverse_bottles or .../demo_randomized).",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="demo_randomized",
        help="Task config folder under dataset_root when dataset_root is task-level.",
    )
    parser.add_argument(
        "--expect-depth",
        action="store_true",
        help="Require depth dataset per camera in every episode.",
    )
    parser.add_argument(
        "--check-video-frames",
        action="store_true",
        help="Check video frame count against HDF5 frame count (requires opencv-python).",
    )
    parser.add_argument(
        "--strict-video-frames",
        action="store_true",
        help="Require exact video frame match (default allows +/-1).",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional path to save audit summary JSON.",
    )
    args = parser.parse_args()

    collection_root = resolve_collection_root(args.dataset_root, args.task_config)
    print(f"Collection root: {collection_root}")

    overview = build_overview(collection_root)
    print("\n=== OVERVIEW ===")
    for k, v in overview.items():
        print(f"{k}: {v}")

    success_result = audit_split(
        split_name="success",
        root=collection_root,
        expect_depth=args.expect_depth,
        check_video_frames=args.check_video_frames,
        strict_video_frames=args.strict_video_frames,
    )
    failed_result = audit_split(
        split_name="failed",
        root=os.path.join(collection_root, "failed"),
        expect_depth=args.expect_depth,
        check_video_frames=args.check_video_frames,
        strict_video_frames=args.strict_video_frames,
    )

    print_result(success_result)
    print_result(failed_result)

    global_errors: List[str] = []
    global_warnings: List[str] = []

    # Dataset-level consistency checks.
    seed_count = int(overview.get("seed_count", 0))
    traj_success_count = int(overview.get("traj_success_count", 0))
    instruction_count = int(overview.get("instruction_count", 0))
    failed_collected_no_error = int(overview.get("failed_records_collected_no_error", 0))
    failed_collected_with_error = int(overview.get("failed_records_collected_with_error", 0))

    if seed_count > 0 and success_result.hdf5_count != seed_count:
        global_errors.append(
            f"Success HDF5 count ({success_result.hdf5_count}) != seed count ({seed_count})."
        )
    if traj_success_count > 0 and success_result.hdf5_count != traj_success_count:
        global_errors.append(
            f"Success HDF5 count ({success_result.hdf5_count}) != _traj_data count ({traj_success_count})."
        )
    if instruction_count > 0 and success_result.hdf5_count != instruction_count:
        global_warnings.append(
            f"Success HDF5 count ({success_result.hdf5_count}) != instruction count ({instruction_count})."
        )

    if failed_result.hdf5_count != failed_collected_no_error:
        global_warnings.append(
            "Failed HDF5 count "
            f"({failed_result.hdf5_count}) != failed records collected with no error ({failed_collected_no_error})."
        )
    if failed_collected_with_error > 0:
        global_warnings.append(
            f"{failed_collected_with_error} failed records are marked collected but have collection_error "
            "(usually unstable/timeout/skipped episodes)."
        )

    for split_result in (success_result, failed_result):
        if split_result.camera_names:
            expected_videos = split_result.hdf5_count * len(split_result.camera_names)
            if split_result.video_count != expected_videos:
                global_warnings.append(
                    f"{split_result.split_name}: video count ({split_result.video_count}) != "
                    f"hdf5_count*cameras ({expected_videos})."
                )

    total_errors = len(success_result.errors) + len(failed_result.errors) + len(global_errors)
    total_warnings = len(success_result.warnings) + len(failed_result.warnings) + len(global_warnings)

    print("\n=== SUMMARY ===")
    print(f"Total errors:   {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if global_errors:
        print("\nGlobal errors:")
        for e in global_errors:
            print(f"  - {e}")
    if global_warnings:
        print("\nGlobal warnings:")
        for w in global_warnings:
            print(f"  - {w}")

    # Explain depth-scale source-of-truth in code path.
    print(
        "\nDepth scale implementation note: "
        "`envs/camera/camera.py:get_depth` computes `-Position.z * 1000`, "
        "so stored depth is metric in millimeters."
    )

    if args.report_json:
        report = {
            "collection_root": collection_root,
            "overview": overview,
            "global_errors": global_errors,
            "global_warnings": global_warnings,
            "success": {
                "hdf5_count": success_result.hdf5_count,
                "video_count": success_result.video_count,
                "errors": success_result.errors,
                "warnings": success_result.warnings,
                "depth": success_result.depth_stats.summary(),
            },
            "failed": {
                "hdf5_count": failed_result.hdf5_count,
                "video_count": failed_result.video_count,
                "errors": failed_result.errors,
                "warnings": failed_result.warnings,
                "depth": failed_result.depth_stats.summary(),
            },
        }
        out_path = os.path.abspath(args.report_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report: {out_path}")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
