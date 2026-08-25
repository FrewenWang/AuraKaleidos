# YOLO11 与卡尔曼目标跟踪

这是一个使用 YOLO11 检测结果和卡尔曼滤波器跟踪单个或多个目标的教学工程。跟踪器支持
欧式距离和马氏距离关联，可在短时漏检时继续预测目标位置，也可选用包含加速度的运动模型。

实现参考 [kalman-object-tracker](https://github.com/CherifiImene/kalman-object-tracker)，并已整理为
标准 Python `src/` 包布局。模型权重、输入视频和运行输出不纳入 Git。

## 环境与安装

支持 Python 3.10 及以上版本。OpenCV/YOLO 的 GUI、摄像头和硬件加速能力取决于宿主平台；
纯卡尔曼滤波和命令行解析测试可在 Windows、macOS 与 Linux 离线运行。

```bash
cd FantasyAutoDrive/kalman_filter_with_yolo11_objects_tracker
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

## 运行

安装后可使用 `alice-object-tracker`，或直接执行模块入口：

```bash
# 查看参数
alice-object-tracker --help

# 摄像头单目标跟踪，默认跟踪 person
alice-object-tracker --mode single --video-source 0 --target-class 0

# 视频文件多目标跟踪
alice-object-tracker --mode multi --video-source ./input.mp4 \
  --target-class 0 --target-class 2 --association-metric mahalanobis
```

首次运行 Ultralytics 可能下载默认模型；离线环境应提前准备模型缓存。摄像头编号 `0` 仅适用于
已授权访问摄像头的桌面环境，无显示服务的 Linux CI 不应运行视频入口。

## 测试

```bash
python -m pytest -q
python -m compileall -q src tests
```

测试不下载模型、不连接摄像头，覆盖类别配置解析、错误输入、卡尔曼预测确定性，以及观测校正
是否降低位置误差。pytest 已配置 `src/` 导入路径，因此无需先执行 editable 安装也能收集测试。

## 目录

```text
src/app/          命令行入口与视频应用
src/ai/tracker/   卡尔曼滤波和目标关联实现
tests/            离线单元测试
models/           本地模型目录（不提交大模型）
outputs/          本地运行结果（不提交）
```
