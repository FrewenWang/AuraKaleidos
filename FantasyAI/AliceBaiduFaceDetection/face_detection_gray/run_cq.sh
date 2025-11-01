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
# start_cmd="sh train_facedetect_cloud.sh"
# #algo_id="algo-8dffbb743aaf4046"  
# # algo_id="algo-a64e38bc77c5481d"
# algo_id="algo-9e7bf0820e6c40a0"
# k8s_gpu_cards=2
# wall_time="10:00:00"
# k8s_priority="normal"
# file_dir="."
# #ak="9ca5b1fcc92657fdbeb46ead0473df0c"
# #sk="075c6507244450c3998d7adce441603d"
# # ak="2ec41daaf1775536850019007f5834a7"
# # sk="19f0acc7cd015069a070ff5bec8b2694"
# ak="7647a7d8381e5656a87f8273127b68f2"
# sk="4ffc536395ac55619bd0ccad208280bd"
# #image_addr="registry.baidu.com/paddlecloud-runenv-ubuntu18.04:ubuntu18.04-gcc8.2.0-cuda11.0-cudnn8-python3.7.10-paddle2.1.2"

# paddlecloud job --ak ${ak} --sk ${sk} \
# train --job-name ${job_name} \
#  --job-conf config.ini \
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
start_cmd="sh train_facedetect_cloud.sh"
#algo_id="algo-8dffbb743aaf4046"  
# algo_id="algo-a64e38bc77c5481d"
algo_id="algo-9d5de86dc5764a48"
k8s_gpu_cards=2
wall_time="10:00:00"
k8s_priority="normal"
file_dir="."
ak="9c977979a51f51a8b54ca9292cc78019"
sk="977a95647c6e5824b8d9ac1f05500fdd"
# ak="2ec41daaf1775536850019007f5834a7"
# sk="19f0acc7cd015069a070ff5bec8b2694"
# ak="a6602956f46a516b91d495f37c13e808"
# sk="532fcdf49d01506ab391bff99311b447"
#image_addr="registry.baidu.com/paddlecloud-runenv-ubuntu18.04:ubuntu18.04-gcc8.2.0-cuda11.0-cudnn8-python3.7.10-paddle2.1.2"

paddlecloud job --ak ${ak} --sk ${sk} \
train --job-name ${job_name} \
 --job-conf config.ini \
 --group-name ${group_name} \
 --start-cmd "${start_cmd}" \
 --file-dir ${file_dir} \
 --job-version ${job_version} \
 --k8s-gpu-cards ${k8s_gpu_cards} \
 --k8s-priority ${k8s_priority} \
 --algo-id ${algo_id} \
 --wall-time ${wall_time} \
 --is-standalone 1
