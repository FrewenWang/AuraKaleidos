# Post-training Quantize
## 概述
`quant_post_paddle.py`脚本用于 paddle-lite模型的INT8量化。

1）首先需要准备用于post-training方式量化的标定集，可使用 vision/tools/quant_calib_generator 工具生成；

2）准备 paddle 格式的预训练模型， 一般位于fluid/inference_model下

3）调用quant_post_paddle.py脚本执行量化

4）使用 opt 工具将量化版本的模型转换为 paddle-lite 的 .nb 模型格式
```shell script
opt_mac --model_file=__model__ --param_file=__params__ --valid_targets=arm --optimize_out_type=naive_buffer --optimize_out=model_opt
```

## 使用
```shell script
python quant_post_paddle.py \
--img_dir faceFeature \
--model_path FaceRecognize0603V4Main/paddle_lite/fluid/inference_model \
--save_path ./FaceRecognize0603V4Main_int8 \
```
* img_dir：标定集图片路径
* model_path：paddle格式的与训练模型路径
* save_path：量化模型的存放名称
