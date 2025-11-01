QUANT_DIR=/home/baiduiov/tool/QNN/qnn-v1.15.0.220706112757_38277/target/x86_64-linux-clang/bin
echo ${QUANT_DIR}


#  初始化QNN编译环境（需要自行在命令行初始化）
# sh qnn1110
# conda activate qnn17


# 进行缓存数据清理
rm -rf ./OnnxResult/*
rm -rf ./QnnResult/Result_0

# 初始化模型转换变量
modelName="convnets_modified"
rawName="new_dms.raw"
shapeHeight=112
shapeWidth=112
channel=1

# 进行ONNX模型的逐层网络输出
echo "================start run_onnx_all.py================="
 python run_onnx_all.py  ./OnnxModels/${modelName}.onnx  ./input/${rawName}  ${shapeHeight} ${shapeWidth} ${channel}


#  模型转换 
#  下面的参数为禁止优化batchnorm网络结构
#   --disable_batchnorm_folding  \
#echo "================start qnn-onnx-converter================="
qnn-onnx-converter  \
    --input_network  ./OnnxModels/${modelName}.onnx   \
    --input_list ./input_onnx/input_list.txt               \
    -o ${QUANT_DIR}/QnnModels/quant/cpp/${modelName}.cpp \
    --input_layout NCHW \
    --disable_batchnorm_folding

# 模型lib生成
#echo "================start qnn-model-lib-generator================="
 qnn-model-lib-generator \
     -c ./QnnModels/quant/cpp/${modelName}.cpp \
     -b ./QnnModels/quant/cpp/${modelName}.bin  \
     -o ./QnnModels/quant/so/ \
     -t x86_64-linux-clang

# generate context cache bin file
qnn-context-binary-generator \
     --backend ../lib/libQnnHtp.so \
     --model ./QnnModels/quant/so/x86_64-linux-clang/lib${modelName}.so \
     --binary_file ${modelName}.bin \
     --output_dir ./QnnModels/quant/so/x86_64-linux-clang

#  注意:qnn-net-run的
echo "================start qnn-net-run================="
 qnn-net-run \
     --backend ../lib/libQnnHtp.so \
     --model ./QnnModels/quant/so/x86_64-linux-clang/lib${modelName}.so \
     --input_list ./input/input_list.txt \
     --output_dir ./QnnResult \
     --debug

# 进行ONNX模型的逐层网络和QNN模型逐层网络对齐结果
echo "================start compare_layer_by_layer================="
 jsonPath="QnnModels/quant/cpp/${modelName}_net.json"
 python compare_layer_by_layer.py ${jsonPath}
