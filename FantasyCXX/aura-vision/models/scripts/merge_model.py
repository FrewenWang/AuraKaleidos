# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from sys import platform
import argparse

# 李文栋服务器
# SERVER_IP = "172.20.68.20"  # 李文栋的服务器的IP地址
# SERVER_USER = "baiduiov"
# MODEL_HUB_ROOT = "/home/baiduiov/work/vision-space/ModelHub"

# 构建服务器
SERVER_IP = "172.20.68.21"
SERVER_USER = "iov"
MODEL_HUB_ROOT = "/home/iov/work/vision-space/ModelHub"

# 注意IP地址更新完成之后，需要记得向此服务器添加SSH公钥
# ssh-copy-id -i ~/.ssh/id_rsa.pub iov@172.20.68.10


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_CACHE = os.path.abspath(SCRIPT_PATH + "/../cache")
GENERATED_PATH = os.path.abspath(SCRIPT_PATH + "/../generated")

# 需要根据自己机器的路径进行配置。开发主机目前只支持: OSX、Linux.
host_os = "osx"
if platform == "linux" or platform == "linux2":
    host_os = "linux"
elif platform == "darwin":
    host_os = "osx"

# 设置模型合并工具工程编译的目录
MERGE_TOOL_BUILT_PATH_DEBUG = os.path.abspath(SCRIPT_PATH + "/../../build/" + host_os +
"-x86_64-debug/vision/tools/model_packer/merge_model_tool")
MERGE_TOOL_BUILT_PATH_RELEASE = os.path.abspath(SCRIPT_PATH + "/../../build/" + host_os +
"-x86_64-release/vision/tools/model_packer/merge_model_tool")

# 设置模型合并工具预编译的目录
MERGE_TOOL_PREBUILT_PATH_DEBUG = os.path.abspath(SCRIPT_PATH + "/../../tools/model_packer/prebuilt/" + host_os +
                                        "-x86_64-debug/merge_model_tool")
MERGE_TOOL_PREBUILT_PATH_RELEASE = os.path.abspath(SCRIPT_PATH + "/../../tools/model_packer/prebuilt/" + host_os +
                                          "-x86_64-release/merge_model_tool")


def merge_models(model_config, target_os, target_arch):
    """
    进行模型合并和加密打包
    :param model_config:    模型配置
    :param target_os:       目标系统版本
    :param target_arch:     目标平台架构
    :return:
    """
    arch = "x86"
    if target_os == "qnx" or target_os == "android" or target_os == "ios":
        arch = "arm"
    elif target_os == "osx" and target_arch == "arm":
        arch = "arm"

    if not os.path.isfile(MERGE_TOOL_BUILT_PATH_RELEASE):
        if not os.path.isfile(MERGE_TOOL_BUILT_PATH_DEBUG):
            if not os.path.isfile(MERGE_TOOL_PREBUILT_PATH_RELEASE):
                if not os.path.isfile(MERGE_TOOL_PREBUILT_PATH_DEBUG):
                    print("[MergeModel] merge model tool NOT FOUND! tool path: " + MERGE_TOOL_PREBUILT_PATH_DEBUG)
                    return False
                else:
                    merge_tool = MERGE_TOOL_PREBUILT_PATH_RELEASE
            else:
                merge_tool = MERGE_TOOL_PREBUILT_PATH_RELEASE
        else:
            merge_tool = MERGE_TOOL_BUILT_PATH_DEBUG
    else:
        merge_tool = MERGE_TOOL_BUILT_PATH_RELEASE

    print("[MergeModel] merge_tool: " + merge_tool)
    # 如果合并模型的目录不存在,则进行目录创建
    if not os.path.exists(GENERATED_PATH):
        os.makedirs(GENERATED_PATH)

    # os.system(merge_tool + " " + model_config + " " + LOCAL_CACHE + " " + GENERATED_PATH)
    cmd = merge_tool + " " + model_config + " " + LOCAL_CACHE + " " + GENERATED_PATH + " " \
          + SERVER_IP + " " + SERVER_USER + " " + MODEL_HUB_ROOT + " " + "1" + " " + arch
    print(cmd)
    os.system(cmd)
    return True


def main(model_config, target_os, target_arch):
    """
    进行模型合并和加密打包
    :param model_config:    模型配置
    :param target_os:       目标系统版本
    :param target_arch:     目标平台架构
    :return:
    """
    if not os.path.isfile(model_config):
        print("[MergeModel] model_config file is NOT ACCESSIBLE! model path:" + model_config)
        return

    merge_models(model_config, target_os, target_arch)
    if os.path.isfile(GENERATED_PATH + "/vision_model.bin"):
        model_len = os.path.getsize(GENERATED_PATH + "/vision_model.bin")
        print("[MergeModel] model length=" + str(model_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="the model config path")
    parser.add_argument("--target_os", help="the target os")
    parser.add_argument("--target_arch", help="the target arch")
    args = parser.parse_args()
    if not args.config:
        parser.print_help()
        sys.exit()
    main(args.config, args.target_os, args.target_arch)
