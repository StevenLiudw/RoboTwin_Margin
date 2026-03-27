import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import pickle
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _save_failed_traj_data(save_path, idx, task_env=None, left_joint_path=None, right_joint_path=None):
    file_path = os.path.join(save_path, "_traj_data_failed", f"episode{idx}.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if task_env is not None:
        left_joint_path = getattr(task_env, "left_joint_path", [])
        right_joint_path = getattr(task_env, "right_joint_path", [])
    traj_data = {
        "left_joint_path": [] if left_joint_path is None else left_joint_path,
        "right_joint_path": [] if right_joint_path is None else right_joint_path,
    }
    with open(file_path, "wb") as f:
        pickle.dump(traj_data, f)


def _load_failed_traj_data(save_path, idx):
    file_path = os.path.join(save_path, "_traj_data_failed", f"episode{idx}.pkl")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
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
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mRenderer Config:\033[0m " + str(args.get("renderer_mode", "rt")) + ", denoiser=" +
          str(args.get("ray_tracing_denoiser", "none")))
    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    failed_records_path = os.path.join(args["save_path"], "failed_episode_records.json")
    failed_records = _load_json(failed_records_path, [])
    if not isinstance(failed_records, list):
        failed_records = []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True
        fail_num = len(failed_records)

        if os.path.exists(seed_file_path):
            with open(seed_file_path, "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            setup_done = False
            failure_reason = None
            failure_error = None
            failure_traceback = None

            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                setup_done = True
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    failure_reason = ("planning_failed" if not TASK_ENV.plan_success else "task_check_failed")
                    print(f"simulate data episode {suc_num} fail! (seed = {epid}, reason = {failure_reason})")
                    fail_idx = len(failed_records)
                    fail_num += 1
                    _save_failed_traj_data(args["save_path"], fail_idx, task_env=TASK_ENV)
                    failed_records.append({
                        "failed_episode_idx": fail_idx,
                        "seed": epid,
                        "failure_reason": failure_reason,
                        "error": None,
                        "traceback": None,
                        "collected": False,
                    })

                TASK_ENV.close_env()

                if args["render_freq"] and hasattr(TASK_ENV, "viewer"):
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                failure_reason = "unstable_error"
                failure_error = f"{type(e).__name__}: {str(e)}"
                failure_traceback = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")

                fail_idx = len(failed_records)
                fail_num += 1
                if setup_done:
                    _save_failed_traj_data(args["save_path"], fail_idx, task_env=TASK_ENV)
                else:
                    _save_failed_traj_data(args["save_path"], fail_idx)
                failed_records.append({
                    "failed_episode_idx": fail_idx,
                    "seed": epid,
                    "failure_reason": failure_reason,
                    "error": failure_error,
                    "traceback": failure_traceback,
                    "collected": False,
                })

                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass

                if args["render_freq"] and hasattr(TASK_ENV, "viewer"):
                    try:
                        TASK_ENV.viewer.close()
                    except Exception:
                        pass
                time.sleep(0.3)
            except Exception as e:
                failure_reason = "runtime_exception"
                failure_error = f"{type(e).__name__}: {str(e)}"
                failure_traceback = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")

                fail_idx = len(failed_records)
                fail_num += 1
                if setup_done:
                    _save_failed_traj_data(args["save_path"], fail_idx, task_env=TASK_ENV)
                else:
                    _save_failed_traj_data(args["save_path"], fail_idx)
                failed_records.append({
                    "failed_episode_idx": fail_idx,
                    "seed": epid,
                    "failure_reason": failure_reason,
                    "error": failure_error,
                    "traceback": failure_traceback,
                    "collected": False,
                })

                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass

                if args["render_freq"] and hasattr(TASK_ENV, "viewer"):
                    try:
                        TASK_ENV.viewer.close()
                    except Exception:
                        pass
                time.sleep(1)

            epid += 1

            with open(seed_file_path, "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)
            _save_json(failed_records_path, failed_records)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(seed_file_path, "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        fail_num = len(failed_records)

    # =========== Collect Data ===========
    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        def exist_hdf5(path, idx):
            file_path = os.path.join(path, "data", f"episode{idx}.hdf5")
            return os.path.exists(file_path)

        # Success episodes (original behavior)
        st_idx = 0
        while exist_hdf5(args["save_path"], st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")
            if not os.path.exists(info_file_path):
                _save_json(info_file_path, {})

            info_db = _load_json(info_file_path, {})
            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info
            _save_json(info_file_path, info_db)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

            if not TASK_ENV.check_success():
                print(f"\033[91mWarning:\033[0m collect replay failed for success episode {episode_idx}.")

        # Failed episodes collection (new behavior)
        if len(failed_records) > 0:
            failed_save_path = os.path.join(args["save_path"], "failed")
            os.makedirs(failed_save_path, exist_ok=True)

            failed_idx = 0
            while exist_hdf5(failed_save_path, failed_idx):
                failed_idx += 1

            print("\033[93m" + "[Start Failed Episode Collection]" + "\033[0m")
            print(f"Failed episodes to collect: {len(failed_records)} (skip first {failed_idx} already collected)")

            for idx in range(failed_idx, len(failed_records)):
                record = failed_records[idx]
                seed = record["seed"]
                replay_mode = "replan"
                print(
                    f"\033[34mTask name: {args['task_name']}\033[0m "
                    + f"| failed episode {idx} (seed={seed}, reason={record.get('failure_reason')})"
                )

                failed_args = dict(args)
                failed_args["save_path"] = failed_save_path

                traj_data = _load_failed_traj_data(args["save_path"], idx)
                if traj_data is not None:
                    left_path = traj_data.get("left_joint_path", [])
                    right_path = traj_data.get("right_joint_path", [])
                    if len(left_path) > 0 or len(right_path) > 0:
                        replay_mode = "saved_traj"
                        failed_args["left_joint_path"] = left_path
                        failed_args["right_joint_path"] = right_path
                        failed_args["need_plan"] = False
                    else:
                        failed_args["left_joint_path"] = []
                        failed_args["right_joint_path"] = []
                        failed_args["need_plan"] = True
                else:
                    failed_args["left_joint_path"] = []
                    failed_args["right_joint_path"] = []
                    failed_args["need_plan"] = True

                try:
                    TASK_ENV.setup_demo(now_ep_num=idx, seed=seed, **failed_args)
                    TASK_ENV.set_path_lst(failed_args)

                    failed_info_file_path = os.path.join(failed_save_path, "scene_info.json")
                    if not os.path.exists(failed_info_file_path):
                        _save_json(failed_info_file_path, {})

                    failed_info_db = _load_json(failed_info_file_path, {})
                    info = TASK_ENV.play_once()
                    collect_success = TASK_ENV.check_success()
                    failed_info_db[f"episode_{idx}"] = {
                        "seed": seed,
                        "simulation_failure_reason": record.get("failure_reason"),
                        "replay_mode": replay_mode,
                        "collect_success": bool(collect_success),
                        "info": info,
                    }
                    _save_json(failed_info_file_path, failed_info_db)

                    TASK_ENV.close_env(clear_cache=((idx + 1) % clear_cache_freq == 0))
                    TASK_ENV.merge_pkl_to_hdf5_video()
                    TASK_ENV.remove_data_cache()

                    record["collected"] = True
                    record["collection_replay_mode"] = replay_mode
                    record["collection_success"] = bool(collect_success)
                    record["collection_error"] = None
                except Exception as e:
                    print(
                        f"\033[91mFailed episode collection error:\033[0m "
                        + f"episode={idx}, seed={seed}, error={type(e).__name__}: {e}"
                    )
                    record["collected"] = False
                    record["collection_replay_mode"] = replay_mode
                    record["collection_success"] = False
                    record["collection_error"] = f"{type(e).__name__}: {str(e)}"
                    record["collection_traceback"] = traceback.format_exc()
                    try:
                        TASK_ENV.close_env()
                    except Exception:
                        pass

                _save_json(failed_records_path, failed_records)

        command = (
            f"cd description && bash gen_episode_instructions.sh "
            + f"{args['task_name']} {args['task_config']} {args['language_num']}"
        )
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
