# #!/bin/bash
# ###############################################################
# ## 注意-- 注意--注意 ##
# ## K8S GPU 类型作业示例 ##
# ## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk ##
# ## ##
# ###############################################################
# #!/bin/bash
# ###############################################################
# ## 注意-- 注意--注意 ##
# ## K8S 单机作业示例 ##
# ###############################################################
# cur_time=`date +"%Y%m%d%H%M"`
# # job_name=facedetect_2d4f5904V2_yb08v100_finetunev2_job${cur_time}
# job_name=facedetect_2d4f5904V2_addJidu_2_job${cur_time}
# # 作业参数
# group_name="iov-cv-32g-0-yq01-k8s-gpu-v100-8" # 将作业提交到group_name指定的组，必填
# # group_name="iov-cv-40g-0-yq01-k8s-gpu-a100-16" # 将作业提交到group_name指定的组，必填
# job_version="paddle-v2.2.2"
# start_cmd="sh scripts/legacy/train_facedetect_cloud.sh"
# #algo_id="algo-8dffbb743aaf4046"
# # algo_id="algo-a64e38bc77c5481d"
# algo_id="algo-9e7bf0820e6c40a0"
# k8s_gpu_cards=2
# wall_time="10:00:00"
# k8s_priority="normal"
# file_dir="."
# Historical credentials removed. Supply PADDLECLOUD_AK/PADDLECLOUD_SK at runtime.
# #image_addr="registry.baidu.com/paddlecloud-runenv-ubuntu18.04:ubuntu18.04-gcc8.2.0-cuda11.0-cudnn8-python3.7.10-paddle2.1.2"

# paddlecloud job --ak ${ak} --sk ${sk} \
# train --job-name ${job_name} \
#  --job-conf configs/legacy.ini \
#  --group-name ${group_name} \
#  --start-cmd "${start_cmd}" \
#  --file-dir ${file_dir} \
#  --job-version ${job_version} \
#  --k8s-gpu-cards ${k8s_gpu_cards} \
#  --k8s-priority ${k8s_priority} \
#  --algo-id ${algo_id} \
#  --wall-time ${wall_time} \
#  --is-standalone 1
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
# job_name=facedetect_2d4f5904V2_yb08v100_finetunev2_job${cur_time}
job_name=facedetect_2d4f5904V2_addJidu_2_job${cur_time}
# 作业参数
group_name="iov-cv-32g-0-yq01-k8s-gpu-v100-8" # 将作业提交到group_name指定的组，必填
# group_name="iov-cv-40g-0-yq01-k8s-gpu-a100-16" # 将作业提交到group_name指定的组，必填
job_version="paddle-v2.2.2"
start_cmd="sh scripts/legacy/train_facedetect_cloud.sh"
#algo_id="algo-8dffbb743aaf4046"
# algo_id="algo-a64e38bc77c5481d"
algo_id="algo-9d5de86dc5764a48"
k8s_gpu_cards=2
wall_time="10:00:00"
k8s_priority="normal"
file_dir="."
ak="${PADDLECLOUD_AK:?Set PADDLECLOUD_AK before submitting a job}"
sk="${PADDLECLOUD_SK:?Set PADDLECLOUD_SK before submitting a job}"
#image_addr="registry.baidu.com/paddlecloud-runenv-ubuntu18.04:ubuntu18.04-gcc8.2.0-cuda11.0-cudnn8-python3.7.10-paddle2.1.2"

paddlecloud job --ak ${ak} --sk ${sk} \
train --job-name ${job_name} \
 --job-conf configs/legacy.ini \
 --group-name ${group_name} \
 --start-cmd "${start_cmd}" \
 --file-dir ${file_dir} \
 --job-version ${job_version} \
 --k8s-gpu-cards ${k8s_gpu_cards} \
 --k8s-priority ${k8s_priority} \
 --algo-id ${algo_id} \
 --wall-time ${wall_time} \
 --is-standalone 1
