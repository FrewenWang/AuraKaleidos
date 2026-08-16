# FantasyAI 子工程盘点

| 子工程 | 类型 | 主要入口/分区 | 结构判断 |
|---|---|---|---|
| AliceAILearn | Notebook 教程 | 卷积/池化 Notebook、`demo_machine_learning/` | Notebook 为主，新内容放 `notebooks/` |
| AliceAutoDriving | 算法 demo | `lane_line/01.demo_ransac_regressor.py` | 仅一个示例，后续归入 `examples/` |
| AliceBaiduFaceDetection | Paddle 训练 | `face_detection_gray/train*.py`, `predict*.py` | 历史平铺代码；迁移前需先补测试 |
| AliceInference | 多后端推理 | `alice_vision_detection/`, `baidu_face_detection/` | 按 ONNX/Paddle/C++ 后端分组，保留现有布局 |
| AliceKaleidoYolo | 上游/演进版本 | `yolo-v3/`, `yolo-v8/`, `yolo-v11/` | 上游代码不重排；本地文档放 `docs/` |
| AliceModelConvert | 转换脚本 | `onnx/`, `qnn/`, `run_qnn.sh` | 按后端分组，输出必须放 `outputs/` |
| AliceOpenCV | C++/Python demo | `OpenCVCXX/`, `OpenCVPython/` | 语言边界清晰；CMake 不得硬编码路径 |
| AlicePaddlePaddle | Paddle 教程 | 基础 Notebook、手写数字、垃圾分类 | 按任务分组，`common_operation/test.py` 是 demo |
| AlicePyTorch | PyTorch 教程/任务 | Notebook、`face_rect_train/`, `object_detection/` | 混合教程与独立项目，子项目保留 README |
| AliceTensorFlow | TensorFlow 1.x | `tensorflow/*.py` | 历史 API 示例，与 TF2 环境隔离 |
| AliceTensorflow2.0 | TensorFlow 2.x | `demos/`, `examples/` | 已接近教程布局，训练产物应迁出 Git |

## 保留例外

- `yolo-v8/src/` 等上游包结构不强制改成新模板。
- 子组件可以在自身根目录保留 README，例如 `face_rect_train/README.md`。
- `test.py` 等历史评估入口本次不直接重命名，以免破坏现有调用；后续新增真正单测时使用 `tests/`。

## 当前主要技术债务

1. 人脸训练/推理脚本数量大，并带有个人数据集绝对路径。
2. TensorFlow 2.0 目录跟踪了 TensorBoard event、profile 和 H5 模型等运行产物。
3. YOLOv8 跟踪了权重和运行结果，YOLOv11 跟踪了上游 zip。
4. 多数子工程没有可重现的依赖锁定和小型测试数据。

本次先用 README、`docs/`、统一 ignore 规则和结构检查阻止债务继续扩大。删除已跟踪的模型/运行产物会改变现有使用方式，应在确认外部存储位置后单独处理。
