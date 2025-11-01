## QuantCalibGenerator
### 概述
`QuantCalibGenerator`工具用于生成 Int8 post-training 方式量化 calibration set的工具；  
post-training量化依赖样本集，通常由模型的测试集中选取具有代表性的几十至几百张左右的样本图片。  
如SNPE框架，其量化工具输入为样本图片归一化之后的raw data二进制数据。  
QuantCalibGenerator工具可以读取指定目录下的图片集合，对其执行指定人脸或手势模型的预处理流程，
并保存预处理后的raw data数据，作为量化样本集。  
**注意量化样本集的输入尺寸必须与量化模型的输入尺寸保持一致，
在使用QuantCalibGenerator工具生成量化样本集时，务必关注模型的input shape配置！**

### 量化所需数据集
离线量化所需的样本集原始图片托管地址（由算法组测试集中选取）：  
**企业云盘**：[存储路径](https://ecloud.baidu.com/index.html#/team/116299868)  
**本地服务器**：iov@172.20.72.11:~/vision-space/VisionImages/quantize_sample_images
- 人脸数据集：face_sample_images_for_quantize.zip
- 手势数据集：gesture_type_sample_images_for_quantize.zip

### 量化工具编译
- 在../CMakeLists.txt中，取消注释行：add_subdirectory(quant_calib_generator)  
- 编译vision，quant_calib_generator将作为包含模块被编译

`osx-x86_64已预编译该工具：prebuilt/osx-x86_64/QuantCalibGenerator`

### 量化工具使用
```shell script
./QuantCalibGenerator src_img_dir [-f] [-g]
Usage:
  src_img_dir:    image set used to generate the calibration set, REQUIRED
  -f --face:      prepare calibration set for face related abilities, OPTIONAL
  -g --gesture:   prepare calibration set for gesture related abilities, OPTIONAL
```
首先，需要设置用于产生标定数据集的源图片路径

* 如果生成人脸相关的标定数据，
```shell script
./QuantCalibGenerator src_img_dir -f
```
* 如果生成手势相关的标定数据，
```shell script
./QuantCalibGenerator src_img_dir -g
```
* 如果源图片支持同时用于人脸和手势检测（源图片可以检出人脸及手势），
```shell script
./QuantCalibGenerator src_img_dir -f -g
```

### 量化样本raw数据保存
输出结果将保存于QuantCalibGenerator可执行程序的同级目录下。  
如faceRect，将保存于faceRect文件夹下，其中：  
raw数据将写入faceRect/val路径下。  
图片路径集合将写入faceRect/val_list.txt文件中