#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "$task_name" "$task_config"

dataset_root="data/${task_name}/${task_config}"
failed_dataset_root="${dataset_root}/failed"

python script/extract_camera_videos.py "${dataset_root}" || \
  echo "[WARN] Camera extraction failed for ${dataset_root}"

if [ -d "${failed_dataset_root}/data" ]; then
  python script/extract_camera_videos.py "${failed_dataset_root}" || \
    echo "[WARN] Camera extraction failed for ${failed_dataset_root}"
fi

rm -rf "${dataset_root}/.cache" "${failed_dataset_root}/.cache"
