# http://wiki.baidu.com/pages/viewpage.action?pageId=246288341
function raw_afs_mount() {
    fs_name=$1
    username=$2
    password=$3
    afs_local_mount_point=$4
    afs_remote_mount_point=$5
    # 创建本地挂载点目录
    mkdir -p ${afs_local_mount_point}
    # 使用原生 afs 提供的挂载工具进行手动挂载
    pushd /opt/afs_mount
    nohup ./bin/afs_mount \
                --username=${username} \
                --password=${password} ${afs_local_mount_point} ${fs_name}${afs_remote_mount_point} 1>my_mount.log 2>&1 &
    popd
}
fs_name="afs://shaolin.afs.baidu.com:9902"
username="iov_cv_data"
password="iovcv@baidu123"
afs_local_mount_point_01="/root/paddlejob/workspace/env_run/afs_aicv/"
afs_remote_mount_point_01="/user/iov_cv_data/vision_tasks/face_detect"
#iafs_local_mount_point_02="/root/paddlejob/workspace/env_run/afs_02/"
#afs_remote_mount_point_02="/user/SYS_KM_Data/yzw/test/demo/fit-a-line/train"
# mount afs 01
raw_afs_mount ${fs_name} ${username} ${password} ${afs_local_mount_point_01} ${afs_remote_mount_point_01}
sleep 2s
tree -L 2 ${afs_local_mount_point_01}
# mount afs 02
#raw_afs_mount ${fs_name} ${username} ${password} ${afs_local_mount_point_02} ${afs_remote_mount_point_02}
#sleep 2s
#tree -L 2 ${afs_local_mount_point_02}
