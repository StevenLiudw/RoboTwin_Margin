from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

sys.path.append("./")
sys.path.append("./policy")

try:
    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
except Exception:
    CONFIGS_PATH = None

    class UnStableError(Exception):
        pass


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as exc:
        raise SystemExit("No such task") from exc
    return env_instance


def eval_function_decorator(policy_name: str, func_name: str):
    module = importlib.import_module(policy_name)
    return getattr(module, func_name)


def get_embodiment_config(robot_file: str):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_seed_list(path: Path) -> list[int]:
    if not path.exists():
        return []
    raw = path.read_text().strip()
    if raw == "":
        return []
    # Backward compatibility if a JSON list was written previously.
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [int(x) for x in payload]
    except Exception:
        pass
    seeds: list[int] = []
    for tok in raw.split():
        try:
            seeds.append(int(tok))
        except Exception:
            continue
    return seeds


def _save_seed_list(path: Path, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = " ".join(str(int(s)) for s in seeds)
    if content:
        content += " "
    path.write_text(content)


def _next_hdf5_index(root: Path) -> int:
    idx = 0
    while (root / "data" / f"episode{idx}.hdf5").exists():
        idx += 1
    return idx


def _load_task_args(task_name: str, task_config: str) -> dict[str, Any]:
    if CONFIGS_PATH is None:
        raise ModuleNotFoundError(
            "RoboTwin environment dependencies are unavailable in this Python env. "
            "Activate the RoboTwin runtime env first."
        )
    config_path = Path(f"./task_config/{task_config}.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    args = yaml.safe_load(config_path.read_text())
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f.read())

    def get_embodiment_file(embodiment_item: str) -> str:
        robot_file = embodiment_types[embodiment_item]["file_path"]
        if robot_file is None:
            raise ValueError("Missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("Embodiment config length must be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    return args


def _load_policy_args(args: argparse.Namespace, task_args: dict[str, Any]) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.policy_config).read_text())
    cfg["policy_name"] = args.policy_name
    cfg["task_name"] = args.task_name
    cfg["task_config"] = args.task_config
    cfg["seed"] = args.seed
    cfg["ckpt_setting"] = args.ckpt_setting

    cfg["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    cfg["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])

    if args.partial_credit_root is not None:
        cfg["partial_credit_root"] = args.partial_credit_root
    if args.pc_checkpoint is not None:
        cfg["pc_checkpoint"] = args.pc_checkpoint
    if args.device is not None:
        cfg["device"] = args.device
    if args.execute_horizon is not None:
        cfg["execute_horizon"] = args.execute_horizon
    if args.num_inference_steps is not None:
        cfg["num_inference_steps"] = args.num_inference_steps
    if args.advantage_bin is not None:
        cfg["advantage_bin"] = args.advantage_bin
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.depth_unit is not None:
        cfg["depth_unit"] = args.depth_unit
    if args.depth_norm_max_m is not None:
        cfg["depth_norm_max_m"] = args.depth_norm_max_m
    if args.state_joint_key is not None:
        cfg["state_joint_key"] = args.state_joint_key

    return cfg


def _move_episode_artifacts(src_root: Path, src_ep_idx: int, dst_root: Path, dst_ep_idx: int) -> bool:
    src_data = src_root / "data" / f"episode{src_ep_idx}.hdf5"
    src_video = src_root / "video" / f"episode{src_ep_idx}.mp4"
    if not src_data.exists():
        return False

    (dst_root / "data").mkdir(parents=True, exist_ok=True)
    (dst_root / "video").mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_data), str(dst_root / "data" / f"episode{dst_ep_idx}.hdf5"))
    if src_video.exists():
        shutil.move(str(src_video), str(dst_root / "video" / f"episode{dst_ep_idx}.mp4"))
    return True


def _run_episode(
    task_env,
    eval_func,
    reset_func,
    model,
    *,
    setup_args: dict[str, Any],
    episode_idx: int,
    seed: int,
    max_steps: int,
) -> tuple[bool, dict[str, Any]]:
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **setup_args)
    if max_steps > 0:
        if task_env.step_lim is None:
            task_env.step_lim = int(max_steps)
        else:
            task_env.step_lim = min(int(task_env.step_lim), int(max_steps))

    reset_func(model)
    task_env._take_picture()  # initial frame

    while task_env.take_action_cnt < task_env.step_lim:
        obs = task_env.get_obs()
        eval_func(task_env, model, obs)
        task_env._take_picture()
        if task_env.eval_success:
            break

    success = bool(task_env.check_success())
    ep_info = {
        "seed": int(seed),
        "success": success,
        "take_action_cnt": int(task_env.take_action_cnt),
        "step_lim": int(task_env.step_lim) if task_env.step_lim is not None else None,
        "eval_success_flag": bool(task_env.eval_success),
    }
    if hasattr(task_env, "info"):
        ep_info["info"] = task_env.info
    return success, ep_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RoboTwin rollouts using a learned policy.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional collection root path. If unset, uses task_config save_path/task_name/task_config.",
    )
    parser.add_argument("--policy-name", type=str, default="Your_Policy")
    parser.add_argument("--policy-config", type=str, default="policy/Your_Policy/deploy_policy.yml")
    parser.add_argument("--ckpt-setting", type=str, default="pc_art")
    parser.add_argument("--seed", type=int, default=0, help="Used only to derive default seed_start.")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Max number of rollout attempts. Collection may stop earlier when target conditions are met.",
    )
    parser.add_argument(
        "--target-success",
        type=int,
        default=None,
        help="Stop when newly saved successful episodes reaches this value.",
    )
    parser.add_argument(
        "--target-total-saved",
        type=int,
        default=None,
        help="Stop when newly saved total episodes (success + failed saved) reaches this value.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--clear-cache-freq", type=int, default=None)
    parser.add_argument("--save-failed", action=argparse.BooleanOptionalAction, default=True)

    # Policy adapter overrides
    parser.add_argument("--partial-credit-root", type=str, default=None)
    parser.add_argument("--pc-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--advantage-bin", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--depth-unit", type=str, default=None, choices=["mm", "m"])
    parser.add_argument("--depth-norm-max-m", type=float, default=None)
    parser.add_argument("--state-joint-key", type=str, default=None, choices=["vector_real", "vector"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_args = _load_task_args(args.task_name, args.task_config)
    default_save_root = Path(task_args["save_path"]).resolve()
    save_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else default_save_root
    )
    failed_root = save_root / "failed"
    tmp_root = save_root / "_policy_tmp"

    max_attempts = int(args.num_episodes) if args.num_episodes is not None else int(task_args["episode_num"])
    if max_attempts <= 0:
        raise ValueError("num_episodes must be > 0")
    target_success = int(args.target_success) if args.target_success is not None else None
    target_total_saved = int(args.target_total_saved) if args.target_total_saved is not None else None
    if target_success is not None and target_success <= 0:
        raise ValueError("target_success must be > 0 when provided.")
    if target_total_saved is not None and target_total_saved <= 0:
        raise ValueError("target_total_saved must be > 0 when provided.")

    seed_start = int(args.seed_start) if args.seed_start is not None else int(100000 * (1 + args.seed))
    clear_cache_freq = (
        int(args.clear_cache_freq)
        if args.clear_cache_freq is not None
        else int(task_args.get("clear_cache_freq", 5))
    )

    policy_args = _load_policy_args(args, task_args)
    get_model = eval_function_decorator(args.policy_name, "get_model")
    eval_func = eval_function_decorator(args.policy_name, "eval")
    reset_func = eval_function_decorator(args.policy_name, "reset_model")
    model = get_model(policy_args)

    task_env = class_decorator(args.task_name)

    setup_args = dict(task_args)
    setup_args["need_plan"] = False
    setup_args["save_data"] = True
    setup_args["eval_mode"] = True
    setup_args["render_freq"] = int(args.render_freq)
    setup_args["save_path"] = str(tmp_root)
    setup_args["eval_video_save_dir"] = None

    seed_file = save_root / "seed.txt"
    failed_records_path = save_root / "failed_episode_records.json"
    success_scene_info_path = save_root / "scene_info.json"
    failed_scene_info_path = failed_root / "scene_info.json"

    success_seeds = _load_seed_list(seed_file)
    failed_records = _load_json(failed_records_path, [])
    if not isinstance(failed_records, list):
        failed_records = []
    success_scene_db = _load_json(success_scene_info_path, {})
    if not isinstance(success_scene_db, dict):
        success_scene_db = {}
    failed_scene_db = _load_json(failed_scene_info_path, {})
    if not isinstance(failed_scene_db, dict):
        failed_scene_db = {}

    success_idx = _next_hdf5_index(save_root)
    failed_idx = _next_hdf5_index(failed_root)
    existing_failed_ids = []
    for rec in failed_records:
        try:
            existing_failed_ids.append(int(rec.get("failed_episode_idx", -1)))
        except Exception:
            continue
    failed_record_next_idx = max(existing_failed_ids + [failed_idx - 1]) + 1
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"Task Name: {args.task_name}")
    print(f"Task Config: {args.task_config}")
    print(f"Policy: {args.policy_name}")
    print(f"Output root: {save_root}")
    print(f"Max attempts: {max_attempts}")
    print(f"Stop target success: {target_success}")
    print(f"Stop target total_saved: {target_total_saved}")
    print(f"Seed start: {seed_start}")
    print(f"Success start index: {success_idx}")
    print(f"Failed start index: {failed_idx}")

    success_count = 0
    fail_count = 0
    saved_success = 0
    saved_total = 0
    attempts_done = 0

    def _stop_reached() -> bool:
        cond_success = target_success is not None and saved_success >= target_success
        cond_total = target_total_saved is not None and saved_total >= target_total_saved
        return cond_success or cond_total

    attempt_idx = 0
    while attempt_idx < max_attempts:
        if _stop_reached():
            break
        seed = seed_start + attempt_idx
        tmp_ep_idx = attempt_idx
        reason = "task_check_failed"
        err_msg = None
        err_tb = None

        print(f"[{attempt_idx + 1}/{max_attempts}] seed={seed}")
        attempts_done += 1
        try:
            success, ep_info = _run_episode(
                task_env,
                eval_func,
                reset_func,
                model,
                setup_args=setup_args,
                episode_idx=tmp_ep_idx,
                seed=seed,
                max_steps=int(args.max_steps),
            )

            do_clear_cache = bool(clear_cache_freq) and (((attempt_idx + 1) % int(clear_cache_freq)) == 0)
            task_env.close_env(clear_cache=do_clear_cache)
            task_env.merge_pkl_to_hdf5_video()
            task_env.remove_data_cache()

            if success:
                moved = _move_episode_artifacts(tmp_root, tmp_ep_idx, save_root, success_idx)
                if not moved:
                    raise RuntimeError("Missing temporary episode artifacts for success episode.")
                success_scene_db[f"episode_{success_idx}"] = ep_info
                success_seeds.append(seed)
                success_idx += 1
                success_count += 1
                saved_success += 1
                saved_total += 1
                print("  -> success")
            else:
                if args.save_failed:
                    moved = _move_episode_artifacts(tmp_root, tmp_ep_idx, failed_root, failed_idx)
                    if moved:
                        failed_scene_db[f"episode_{failed_idx}"] = ep_info
                        failed_records.append(
                            {
                                "failed_episode_idx": failed_idx,
                                "seed": seed,
                                "failure_reason": reason,
                                "error": None,
                                "traceback": None,
                                "collected": True,
                                "collection_replay_mode": "policy_rollout",
                                "collection_success": False,
                                "collection_error": None,
                            }
                        )
                        failed_record_next_idx = max(failed_record_next_idx, failed_idx + 1)
                        failed_idx += 1
                        saved_total += 1
                fail_count += 1
                print("  -> fail")

        except UnStableError as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            err_tb = traceback.format_exc()
            fail_count += 1
            print(f"  -> unstable: {err_msg}")
            try:
                task_env.close_env()
            except Exception:
                pass
        except Exception as exc:
            reason = "runtime_exception"
            err_msg = f"{type(exc).__name__}: {exc}"
            err_tb = traceback.format_exc()
            fail_count += 1
            print(f"  -> error: {err_msg}")
            try:
                task_env.close_env()
            except Exception:
                pass
        finally:
            # Remove any remaining temporary artifacts for this episode index.
            for p in (
                tmp_root / "data" / f"episode{tmp_ep_idx}.hdf5",
                tmp_root / "video" / f"episode{tmp_ep_idx}.mp4",
                tmp_root / ".cache" / f"episode{tmp_ep_idx}",
            ):
                if p.is_file():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)

            if err_msg is not None and args.save_failed:
                record_idx = failed_record_next_idx
                failed_record_next_idx += 1
                failed_records.append(
                    {
                        "failed_episode_idx": record_idx,
                        "seed": seed,
                        "failure_reason": reason,
                        "error": err_msg,
                        "traceback": err_tb,
                        "collected": False,
                        "collection_replay_mode": "policy_rollout",
                        "collection_success": False,
                        "collection_error": err_msg,
                    }
                )

            _save_seed_list(seed_file, success_seeds)
            _save_json(failed_records_path, failed_records)
            _save_json(success_scene_info_path, success_scene_db)
            if args.save_failed:
                _save_json(failed_scene_info_path, failed_scene_db)

        if _stop_reached():
            reasons = []
            if target_success is not None and saved_success >= target_success:
                reasons.append(f"saved_success={saved_success} reached target_success={target_success}")
            if target_total_saved is not None and saved_total >= target_total_saved:
                reasons.append(f"saved_total={saved_total} reached target_total_saved={target_total_saved}")
            print("Stop condition met: " + " OR ".join(reasons))
            break
        attempt_idx += 1

    # Cleanup temporary root if empty.
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(
        f"Done. attempts={attempts_done} "
        f"success={success_count} fail={fail_count} "
        f"(newly saved success={saved_success}, newly saved total={saved_total}, "
        f"dataset success total={len(success_seeds)}, failed records total={len(failed_records)})"
    )


if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv:
        try:
            from test_render import Sapien_TEST

            Sapien_TEST()
        except Exception:
            pass
    main()
