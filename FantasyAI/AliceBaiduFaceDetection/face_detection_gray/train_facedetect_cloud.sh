# 执行链接AFS的文件系统
sh each_node.sh

# shellcheck disable=SC2006
date=`date +%Y%m%d-%H%M%S`
echo "start_copy_files" $date
echo 'checkdir: /root/paddlejob/data_train'

# 执行
if [ ! -d '/root/paddlejob/data_train' ]; then
  mkdir  /root/paddlejob/data_train
  mkdir  /root/paddlejob/data_train/train_mid_img
  mkdir /root/paddlejob/workspace/log/modelsave
  mkdir /root/paddlejob/data_train/all_back_from_ImgNet
  echo "checkdir: mkdir -p /root/paddlejob/data_train"
else
  echo "checkdir: exist /root/paddlejob/data_train"
fi

date=`date +%Y%m%d-%H%M%S`
echo "cp  start! " $date
# 执行copy操作，将挂载的目录的中的facedata.tar拷贝到/root/paddlejob/data_train训练的文件目录中。
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/facedata.tar /root/paddlejob/data_train &
PID1=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/oms_facedata.tar /root/paddlejob/data_train &
PID2=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/val_resize.tar /root/paddlejob/data_train &
PID3=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/1000persionBiao2.tar /root/paddlejob/data_train &
PID3_1=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/kouzhao_mult.tar /root/paddlejob/data_train &
PID3_2=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/widerMoreFace.tar /root/paddlejob/data_train &
PID_wider=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/openImageMoreface.tar /root/paddlejob/data_train &
PID_open=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data.tar  /root/paddlejob/data_train &
PID4=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_1107.tar  /root/paddlejob/data_train &
PID5=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_1108.tar  /root/paddlejob/data_train &
PID6=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0112.tar  /root/paddlejob/data_train &
PID11=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0227.tar  /root/paddlejob/data_train &
PID12=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0227_fl.tar  /root/paddlejob/data_train &
PID13=$!

cd /root/paddlejob/data_train/all_back_from_ImgNet
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/back_from_ImgNet.tar  . &
PID7=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/wanou.tar  . &
PID8=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/zicai_wanou.tar  . &
PID9=$!
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/zicai_wanou_di2pi.tar  . &
PID10=$!

# 等待上述所有的后台进程子任务子任务执行完毕
wait $PID1
wait $PID2
wait $PID3
wait $PID3_1
wait $PID3_2
wait $PID_wider
wait $PID_open
wait $PID4
wait $PID5
wait $PID6
wait $PID7
wait $PID8
wait $PID9
wait $PID10
wait $PID11
wait $PID12
wait $PID13

date=`date +%Y%m%d-%H%M%S`
echo "cp  end! " $date

date=`date +%Y%m%d-%H%M%S`
echo "start_tar_xf files!" $date
# 解压所有的文件
tar -xf back_from_ImgNet.tar
tar -xf wanou.tar
cd /root/paddlejob/data_train
tar -xf facedata.tar
tar -xf oms_facedata.tar
tar -xf val_resize.tar
tar -xf 1000persionBiao2.tar
tar -xf kouzhao_mult.tar
tar -xf widerMoreFace.tar
tar -xf openImageMoreface.tar
tar -xf jidu_data.tar
tar -xf jidu_data_1107.tar
tar -xf jidu_data_1108.tar
tar -xf jidu_data_0112.tar
tar -xf jidu_data_0227.tar
tar -xf jidu_data_0227_fl.tar


# TODO 拷贝所有的训练标注文件？？？？
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/trainV2.txt  /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/widerMoreFace_476.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/openImageMoreface3700.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_1107.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_1108.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0112.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0227.txt   /root/paddlejob/data_train/filelist/train_filelist
cp /root/paddlejob/workspace/env_run/afs_aicv/imgdata_tar/jidu_data_0227_fl.txt   /root/paddlejob/data_train/filelist/train_filelist

echo "end_tar_xf files!" $date
tree -L 1

cd /root/paddlejob/workspace/env_run
date=`date +%Y%m%d-%H%M%S`
echo "start train face_detect model!" $date
pip install opencv-python==4.0.1.24
pip install  paddleslim==2.2.2
# python modify_quant_layers.py

export CUDA_VISIBLE_DEVICES=0,1
# 调用python执行训练脚本
python -m paddle.distributed.launch  train_face1_cloud.py
#python   train_face1_cloud.py
date=`date +%Y%m%d-%H%M%S`
echo "end train face_detect model!" $date

# date=`date +%Y%m%d-%H%M%S`
# echo "start eval face_detect!" $date
# python predict_images_cloud.py
# python evalScript/count_face_PR.py > /root/paddlejob/workspace/log/pr_rslt.txt
# python evalScript/count_loss.py
# date=`date +%Y%m%d-%H%M%S`
# echo "end eval face_detect!" $date
sleep  15000h
