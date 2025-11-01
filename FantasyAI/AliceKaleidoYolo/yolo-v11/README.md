



# 安装pytorch

```shell
	
```

# 安装Ultralytics

```shell
pip install ultralytics
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple #网络不好的话

```

# 进行预测

首先，进行下载预训练权重文件，
```shell
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

得到结果

![result](./docs/images/result.png)

