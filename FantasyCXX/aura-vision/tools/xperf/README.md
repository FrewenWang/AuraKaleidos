# XPERF
## 1. 概述
    Xperf 用于 libvision 工程部署时的各项性能测试，包括模型推理效率测试、模型精度指标测试；

## 2. speed_test 模型推理效率测试
    对指定图片进行多次检测，统计每个模型的平均耗时

## 3. accuracy_test 模型精度指标测试工具
    遍历模型测试集(算法组提供)推理并输出，根据算法组提供的每个模型的指标统计方式，
    计算测试集整体指标。
### 3.1 模型测试集
    测试集托管服务器：iov@172.20.72.11
    存储路径：/home/iov/vision_space/data/VisionImages/test_images
相应目录下为算法组提供的每个模型测试集
```shell script
ll /vision_space/data/VisionImages/test_images
drwxrwxr-x  8 iov iov        4096 Mar 15 17:10 dms/
drwxrwxr-x  6 iov iov        4096 Dec 17  2020 face_attribute/
drwxrwxr-x  6 iov iov        4096 Dec  9  2020 face_call/
drwxrwxr-x  9 iov iov        4096 Jan  7  2021 face_cover/
drwxrwxr-x  7 iov iov        4096 Jan  7  2021 face_emotion/
drwxr-xr-x  6 iov iov        4096 Jan 11  2021 face_eye_center/
drwxrwxr-x  3 iov iov        4096 Oct 19  2020 face_landmark_eye_close/
drwxrwxr-x  6 iov iov        4096 Nov 30  2020 face_liveness_ir/
drwxrwxr-x  6 iov iov        4096 Nov 30  2020 face_liveness_rgb/
drwxrwxr-x  5 iov iov        4096 Nov 25  2020 face_recognize/
drwxrwxr-x  5 iov iov        4096 Feb 22 11:07 gesture_type/
```
每个测试集目录下，已将算法提供的原始数据集，按照便于处理的方式做过数据归纳，并生成cvs格式的标签文件。
同时提供random_select.py脚本, 支持随机选取部分样本作为测试集。如打电话检测模型：
```
python random_select.py \
            --sample_rate=<sample_rate> \
            --random_seed=[random_seed]
sample_rate：选取样本比率
random_seed：随机种子
将随机选取num_of_test_data * sample_rate的样本，生成
face_call_sample_images/            ：样本数据集   
face_call_sample_images.csv         ：样本数据集标签文件
face_call_sample_images.tar.gz      ：样本数据集压缩包
```
```shell script
ll /vision_space/data/VisionImages/test_images/face_call
# 打电话模型的全量测试集标签文件
-rw-rw-r-- 1 iov iov     104805 Dec  9  2020 call.csv
-rw-r--r-- 1 iov iov       1711 Dec  9  2020 random_select.py
drwxrwxr-x 2 iov iov     192512 Dec  9  2020 face_call_sample_images/
-rw-rw-r-- 1 iov iov     104805 Dec  9  2020 face_call_sample_images.csv
-rw-rw-r-- 1 iov iov 1137315840 Dec  9  2020 face_call_sample_images.tar.gz
```

### 3.2 测试实现 (./py_eval/eval.py)
#### 3.2.1 获取测试数据到本地：
    功能测试类(如face_dms_eval.py)继承自BaseEval类, 实现prepare_eval_data函数,
    从iov@172.20.72.11获取对应的测试样本数据，存放于./test_data/dms
#### 3.2.2 eval.py 参数：
    --bin_path:                 xperf程序路径
    --batch_size:               每个批次检测数据量
    --eval_feature:             需要统计指标的功能
    --use_local_detect_result:  是否使用本地检测结果文件
#### 3.2.3 eval.py流程：
    - 根据测试数据集大小和batch_size，将测试数据集分为多个批次
    for i in batch:
        - 建立./test_data/eval_feature/buffer_image目录，
          将当前批次的图片拷贝进来，并push到车机eval_img_path
        - 执行车机/data/local/tmp/xperf-run/xperf程序，  
          传入参数[**eval_img_path, eval_result_path**]   
          检测eval_img_path目录下的图片，
          并将检测结果(json)追加写入到eval_result_path文件中
    - 全部batch检测结束，获取车机eval_result_path的所有批次检测结果文件
    - 功能测试类实现eval函数，读取检测结果文件，计算测试指标
    - 功能测试类实现print_result函数，展示指标结果  
```
默认使用./test_data作为测试数据存放目录，  
使用CLion时，右键./test_data->mark directory as->Excluded,  
以避免CLion产生大量文件index占用磁盘空间。
```  

### 3.3 测试配置
    我们的检测程序libvision中，在单帧检测结果之上封装了滑窗策略，且使用了人脸追踪策略。
    但在对数据集进行模型指标计算时，需要统计单张图片的输出结果。
    开启BENCHMARK_TEST，配置如下：
#### 3.3.1 发版模式设置
- PRODUCT ：生产环境发版，人脸追踪
- BENCHMARK_TEST ： 测试集指标测试发版，无人脸追踪，输出单帧结果
```c++
    VisionService *g_service = new VisionService;
    g_service->setConfig(ParamKey::RELEASE_MODE, BENCHMARK_TEST);
```
#### 3.3.2 设置最多检测人脸数为1
```c++
    g_service->setConfig(ParamKey::FACE_MAX_COUNT, 1);
    service->setConfig(ParamKey::FACE_NEED_CHECK_COUNT, 1);
```
    
## 4. 编译与运行
### 4.1 编译libvision
### 4.2 编译xperf工具
如：`./build.sh -r -t 2`  
如编译SNPE版本：`./build.sh -s -r -t 2`
如编译QNN版本：`./build.sh -s -q -r -t 2`
### 4.3 运行
#### 4.3.1 运行speed_test:  
`./run.sh -t 2 -s`  
#### 4.3.2 运行accuracy_test,检测dms五分类数据集并统计模型指标:  
`./run.sh -t 2 -a -f dms`
#### 4.3.3 运行accuracy_test,统计本地的dms模型json结果文件，并计算指标:  
`./run.sh -t 2 -a -f dms -l`
