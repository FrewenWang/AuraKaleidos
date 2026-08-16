# -*- coding: utf-8 -*-
"""
虚拟视频播放器控制模块 - 跨平台实现
使用subprocess和pathlib统一处理，最小化平台差异
"""
import os
import sys
import time
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from src.config.platform_compat import get_platform, is_windows


def convert2second(time_str: str) -> int:
    """将时间字符串转换为秒数"""
    parts = time_str.split(":")
    if len(parts) == 3:
        return 3600 * int(parts[0]) + 60 * int(parts[1]) + int(parts[2])
    return 0


def get_video_config(file_path: str) -> dict:
    """从XML文件获取视频配置"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到配置文件 {file_path}")
        return {}

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        config = {}
        for elem in root:
            video_type = elem.attrib.get('type', '')
            video_time = elem.attrib.get('time', '')
            video_dir = elem.attrib.get('dir', '')
            if video_type:
                config[video_type] = [video_time, video_dir]
        return config
    except ET.ParseError as e:
        print(f"解析XML失败: {e}")
        return {}


def play_video(file_path: str, video_list: list, video_time: int) -> bool:
    """播放视频文件 - 使用跨平台播放器"""
    platform = get_platform()

    # 构建视频文件路径
    video_files = [
        str(Path(file_path) / video)
        for video in video_list
        if (Path(file_path) / video).exists()
    ]

    if not video_files:
        print("没有找到视频文件")
        return False

    print(f"播放视频: {video_files}")

    try:
        if is_windows():
            # Windows: 使用VCamDemo或默认播放器
            exe_path = str(Path(__file__).parent / 'VCamDemo.exe')
            if os.path.exists(exe_path):
                cmd = [exe_path] + video_files
            else:
                # 降级到默认播放器
                cmd = video_files[0]
                subprocess.Popen(cmd, shell=True)
                time.sleep(video_time)
                return True
        else:
            # macOS/Linux: 尝试ffplay
            try:
                cmd = ['ffplay', '-nodisp', '-autoexit'] + video_files
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                # ffplay不可用，尝试系统默认播放器
                for video in video_files:
                    platform.open_file(video)
                    time.sleep(video_time)
                return True

        # 等待播放完成
        time.sleep(video_time)
        return True

    except Exception as e:
        print(f"播放视频失败: {e}")
        return False


def cleanup():
    """清理播放器进程"""
    platform = get_platform()
    platform.kill_process('VCamDemo.exe')
    platform.kill_process('ffplay')


def run_vcamtest(name: str = ''):
    """运行虚拟摄像头测试"""
    platform = get_platform()

    # 检查是否为Windows系统
    if not is_windows():
        print("虚拟摄像头功能目前仅在Windows系统上支持")
        print("在macOS/Linux上，此功能将被跳过")
        time.sleep(5)  # 模拟等待
        return

    xml_path = str(Path(__file__).parent / 'videoconfig.xml')
    video_config = get_video_config(xml_path)

    if not video_config:
        print("未找到视频配置")
        return

    for video_type, (video_time_str, video_dir) in video_config.items():
        video_path = str(Path(__file__).parent / video_dir)
        video_list = []
        for _, _, files in os.walk(video_path):
            video_list = files
            break

        if not video_list:
            continue

        video_time = convert2second(video_time_str)
        play_video(video_path, video_list, video_time)


# 获取当前文件所在目录
current_path = Path(__file__).parent


if __name__ == "__main__":
    run_vcamtest("")
