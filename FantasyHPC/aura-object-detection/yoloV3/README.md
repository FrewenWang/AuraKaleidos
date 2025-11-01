
# 基于yoloV3的活体检测项目



## 目录结构

```c++
(base) ~/03.ProgramSpace/01.WorkSpace/AuraKaleidoScope/AuraKaleidoHPC/aura-object-detection/yoloV3 git:[main]
tree -L 2
.
├── README.md
├── assets
│   ├── dog.png
│   ├── giraffe.png
│   ├── messi.png
│   └── traffic.png
├── checkpoints
├── config
│   ├── coco.data										// 定义训练集、验证集
│   ├── create_custom_model.sh
│   ├── custom.data
│   ├── yolov3-tiny.cfg
│   └── yolov3.cfg									// 模型的网络结构的配置
├── data
│   ├── coco
│   ├── coco.names									// coco的类别的名称分布
│   ├── custom
│   ├── get_coco_dataset.sh
│   └── samples
├── detect.py
├── logs
├── models.py
├── output
│   └── samples
├── test.py
├── train.py
├── utils
│   ├── __init__.py
│   ├── augmentations.py
│   ├── datasets.py
│   ├── logger.py
│   ├── parse_config.py
│   └── utils.py
└── weights             // 整个网络模型的权重文件，下面的不同的模型文件
    ├── darknet53.conv.74					// 
    ├── download_weights.sh				// 
    ├── yolov3-tiny.weights				//  简单的网络结构 mAP值可能比较小
    └── yolov3.weights						// 
```





