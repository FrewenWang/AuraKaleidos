# va_cv模块单元测试

## 1. 编译
```
编译vision-ability工程
开启BUILD_VACV_TEST选项后会编译vacv_test工具
```

## 2. 测试
### 2.1 测试main函数
    src/test_main.cpp，选择测试vacv函数  
### 2.2 测试类实现
    `src/impl`
### 2.3 测试参数设置
    src/profile/cv_profile.cpp  
```
_k_test_times    :  测试次数    
k_log_batch_size :  输出日志间隔次数
``` 
### 2.4 测试实现选取
    如crop算子测试，在Crop::crop函数中，通过宏定义设置naive或arm-neon实现
### 2.5 测试脚本
    run_vacv_test.sh

## 3. 测试实现
测试基于opencv原生实现，对相同图像的相同操作，  
与vacv对应算子的输出结果与运算效率进行对比。  
输出对比使用余弦相似度。

## 4. vacv支持算子及平台
1. yuv2bgr
    - arm neon
2. crop
    - arm neon
    - naive
3. layout_change
    - arm neon
    - naive
4. dtype_change(int8/fp32)
    - arm neon
    - naive
5. resize
    - inter linear
        - arm neon
    - inter cubic
        - arm neon
6. mean_stddev
    - arm neon
    - naive
7. normalize
    - arm neon
    - naive
8. warp_affine
    - naive
    
    