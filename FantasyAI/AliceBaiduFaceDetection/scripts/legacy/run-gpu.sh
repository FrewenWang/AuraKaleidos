#!/bin/bash
###############################################################
## 注意-- 注意--注意 ##
## K8S GPU 类型作业示例 ##
## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk ##
## ##
###############################################################
#!/bin/bash
###############################################################
## 注意-- 注意--注意 ##
## K8S 单机作业示例 ##
###############################################################
cur_time=`date +"%Y%m%d%H%M"`
job_name=vehicle_test_job${cur_time}
# 作业参数
group_name="iov-cv-32g-0-yq01-k8s-gpu-v100-8" # 将作业提交到group_name指定的组，必填
job_version="paddle-v2.0.2"
start_cmd="PYTHONPATH=src python -m paddle.distributed.launch scripts/legacy/train.py"
k8s_gpu_cards=8
wall_time="10:00:00"
k8s_priority="normal"
file_dir="."
ak="${PADDLECLOUD_AK:?Set PADDLECLOUD_AK before submitting a job}"
sk="${PADDLECLOUD_SK:?Set PADDLECLOUD_SK before submitting a job}"
paddlecloud job --ak ${ak} --sk ${sk} \
train --job-name ${job_name} \
 --job-conf configs/legacy.ini \
 --group-name ${group_name} \
 --start-cmd "${start_cmd}" \
 --file-dir ${file_dir} \
 --job-version ${job_version} \
 --k8s-gpu-cards ${k8s_gpu_cards} \
 --k8s-priority ${k8s_priority} \
 --wall-time ${wall_time} \
 --is-standalone 1
