import argparse
import glob
import os
import re
import subprocess

import cv2
import h5py
import numpy as np


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, timeout_s: float = 300.0) -> None:
    if not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, 3).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, height, width, _ = imgs.shape

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            out_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _, stderr = ffmpeg.communicate(input=imgs.tobytes(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        ffmpeg.kill()
        raise TimeoutError(f"ffmpeg timeout while writing {out_path}")

    if ffmpeg.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore") if stderr else ""
        raise RuntimeError(f"ffmpeg failed to write {out_path}. {err_msg}")

    print(f"Saved {out_path} ({n_frames} frames, {width}x{height}).")


def _decode_jpeg_buffer(raw) -> np.ndarray:
    raw_bytes = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
    # RGB frames are stored as fixed-width JPEG byte strings in HDF5 and padded with trailing nulls.
    # Trim only right padding; do not split on first null because JPEG streams may contain internal null bytes.
    raw_bytes = raw_bytes.rstrip(b"\0")
    decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode JPEG buffer from HDF5 rgb dataset.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _decode_rgb_dataset(rgb_dataset) -> np.ndarray:
    # Some datasets may already store raw RGB frames as uint8 arrays.
    if rgb_dataset.dtype == np.uint8 and rgb_dataset.ndim == 4 and rgb_dataset.shape[-1] in (3, 4):
        frames = rgb_dataset[()]
        if frames.shape[-1] == 4:
            frames = frames[:, :, :, :3]
        return frames

    frames = [_decode_jpeg_buffer(item) for item in rgb_dataset]
    if not frames:
        raise ValueError("RGB dataset is empty.")
    return np.stack(frames, axis=0)


def _episode_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else stem


def _camera_video_path(video_dir: str, episode_name: str, camera_name: str) -> str:
    if camera_name == "head_camera":
        return os.path.join(video_dir, f"{episode_name}.mp4")
    return os.path.join(video_dir, f"{episode_name}_{camera_name}.mp4")


def extract_episode_videos(hdf5_path: str, video_dir: str, fps: float = 30.0, overwrite: bool = False) -> int:
    episode_name = os.path.splitext(os.path.basename(hdf5_path))[0]

    with h5py.File(hdf5_path, "r") as handle:
        observation = handle.get("observation")
        if observation is None:
            raise KeyError(f"`observation` group not found in {hdf5_path}.")

        camera_names = list(observation.keys())
        if "head_camera" in camera_names:
            camera_names = ["head_camera"] + [name for name in camera_names if name != "head_camera"]

        exported = 0
        for camera_name in camera_names:
            camera_group = observation[camera_name]
            if "rgb" not in camera_group:
                continue

            out_path = _camera_video_path(video_dir, episode_name, camera_name)
            if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"Skip existing {out_path}")
                continue

            print(f"Extract {episode_name}::{camera_name} -> {out_path}")
            frames = _decode_rgb_dataset(camera_group["rgb"])
            images_to_video(frames, out_path, fps=fps)
            exported += 1

    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera videos (head + wrist) from collected HDF5 episodes."
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Dataset root path, e.g. data/handover_block/demo_randomized",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--pattern", type=str, default="episode*.hdf5", help="HDF5 file glob pattern.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing video files.")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    data_dir = os.path.join(dataset_root, "data")
    video_dir = os.path.join(dataset_root, "video")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, args.pattern)), key=_episode_sort_key)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir} with pattern {args.pattern}")

    total_exported = 0
    total_hdf5 = len(hdf5_files)
    for idx, hdf5_path in enumerate(hdf5_files):
        print(f"[{idx + 1}/{total_hdf5}] Processing {hdf5_path}")
        exported = extract_episode_videos(hdf5_path, video_dir, fps=args.fps, overwrite=args.overwrite)
        total_exported += exported

    print(f"Done. Exported {total_exported} camera videos from {len(hdf5_files)} episodes.")


if __name__ == "__main__":
    main()
