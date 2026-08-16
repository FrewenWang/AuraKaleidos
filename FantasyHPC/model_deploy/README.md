# model_deploy

PyTorch 图像分类和 YOLOv3 检测模型的 Flask 服务化示例集合。各目录是独立 demo，不共享
统一依赖或启动入口。

## 目录

| 目录 | 用途 |
|---|---|
| `flask-test/` | 最小 Flask 服务和 ResNet50 请求示例 |
| `deploy-pytorch-model-master/` | 历史分类服务副本 |
| `deploy_pytorch_model/` | YOLOv3 Flask 服务封装 |
| `PyTorch-YOLO-V3/` | YOLOv3 训练、测试和检测源码 |
| `docs/` | 部署、精度测试和 Docker 笔记 |

## 快速验证

```bash
cd FantasyHPC/model_deploy/flask-test/sample
python -m pip install -r requirements.txt
python run_pytorch_server.py
# 另开终端，在相同目录执行：
python simple_request.py
```

服务会监听本地端口并可能加载或下载模型。运行前检查端口、GPU 开关、权重和图片路径；
不要把服务直接暴露到公网。YOLOv3 示例依赖历史版本 PyTorch，模型权重和数据集需自行
准备。扩展文档位于 [`docs/`](docs/)。
