#!/usr/bin/env bash

# 开关控制
EXEC_ONNX="y"
EXEC_QNN="y"
JUST_COMPARE="n"
USE_OLD_FILE="y"

CURRENT_DIR=$(pwd)

echo "-- ONNX SDK Directory : " ${ONNX_ROOT}
echo "-- QNN SDK Directory : " ${QNN_SDK_BIN}
echo "-- CURRENT Directory : " ${CURRENT_DIR}

# 初始化模型转换变量
MODEL_TYPE=""
MODEL_DIR=""
MODEL_NAME=""
TEST_RAW_IMAGE=""
TEST_RAW_DIR=""
INPUT_HEIGHT=""
INPUT_WIDTH=""
INPUT_CHANNEL=""

MODEL_ONNX=""
MODEL_QNN_DIR=""
RESULT_ONNX=""
RESULT_QNX=""
INPUT_LIST_QUAN=""
INPUT_LIST_RUN=""

function configEnv {
    MODEL_ONNX="${MODEL_NAME}.onnx"
    MODEL_QNN_DIR="${MODEL_DIR}/Result/QnnModel"
    RESULT_ONNX="${MODEL_DIR}/Result/OnnxResult/"
    RESULT_QNX="${MODEL_DIR}/Result/QnnResult/"
    INPUT_LIST_QUAN=${MODEL_QNN_DIR}/input_list_quant.txt
    INPUT_LIST_RUN=${MODEL_QNN_DIR}/input_list_run.txt

    if [[ $MODEL_TYPE == "8" ]]; then
        MODEL_TYPE="int8"
    elif [[ $MODEL_TYPE == "16" ]]; then
        MODEL_TYPE="float16"
    fi

    echo "-- MODEL_TYPE      = " ${MODEL_TYPE}
    echo "-- MODEL_DIR       = " ${MODEL_DIR}
    echo "-- MODEL_NAME      = " ${MODEL_NAME}
    echo "-- TEST_RAW_IMAGE  = " ${TEST_RAW_IMAGE}
    echo "-- TEST_RAW_DIR    = " ${TEST_RAW_DIR}
    echo "-- INPUT_HEIGHT    = " ${INPUT_HEIGHT}
    echo "-- INPUT_WIDTH     = " ${INPUT_WIDTH}
    echo "-- INPUT_CHANNEL   = " ${INPUT_CHANNEL}

    echo "-- MODEL_ONNX      = " ${MODEL_ONNX}
    echo "-- MODEL_QNN_DIR   = " ${MODEL_QNN_DIR}
    echo "-- RESULT_ONNX     = " ${RESULT_ONNX}
    echo "-- RESULT_QNX      = " ${RESULT_QNX}
    echo "-- INPUT_LIST_QUAN = " ${INPUT_LIST_QUAN}
    echo "-- INPUT_LIST_RUN  = " ${INPUT_LIST_RUN}

    if [[ ! -d ${MODEL_QNN_DIR} ]]; then
        mkdir -p ${MODEL_QNN_DIR}
    fi
}

# 进行 ONNX 模型的逐层网络输出
function execOnnx {
    if [[ $EXEC_ONNX == "y" ]]; then
        echo -e "\n================ ONNX 模型的逐层网络输出 ================="
        python exec_onnx.py ${MODEL_DIR} ${MODEL_ONNX} ${TEST_RAW_IMAGE} ${INPUT_HEIGHT} ${INPUT_WIDTH} ${INPUT_CHANNEL}
    fi
}


function prepareInputListQuant {
    mkdir -p ${MODEL_QNN_DIR}
    files=$(ls $TEST_RAW_DIR)
    rm -f ${INPUT_LIST_QUAN}
    touch ${INPUT_LIST_QUAN}
    for filename in $files
    do
       echo "$TEST_RAW_DIR/$filename" >> ${INPUT_LIST_QUAN}
    done
}


function prepareInputListRun {
    mkdir -p ${MODEL_QNN_DIR}
    rm -f ${INPUT_LIST_RUN}
    touch ${INPUT_LIST_RUN}
    echo "${MODEL_DIR}/${TEST_RAW_IMAGE}" >> ${INPUT_LIST_RUN}
}


function execQnnInt8 {
    echo -e "\n>>>> Step 1 : 生成 input_list_quant&run.txt"
    prepareInputListQuant
    prepareInputListRun

    # --input_network  model.onnx
    # 此命令生成的产物是：model.bin、model.cpp、model_net.json
    # --disable_batchnorm_folding # 该参数为禁止优化 batchnorm 网络结构
    echo -e "\n>>>> Step 2 : 执行 qnn-onnx-converter"
    ${QNN_SDK_BIN}/qnn-onnx-converter \
        --input_network ${MODEL_DIR}/${MODEL_ONNX} \
        --input_list ${INPUT_LIST_QUAN} \
        -o ${MODEL_QNN_DIR}/int8/cpp/${MODEL_NAME}.cpp \
        --disable_batchnorm_folding


    # -c model.cpp
    # -b model.bin
    # -o output
    echo -e "\n>>>> Step 3 : 执行 qnn-model-lib-generator"
    ${QNN_SDK_BIN}/qnn-model-lib-generator \
         -c ${MODEL_QNN_DIR}/int8/cpp/${MODEL_NAME}.cpp \
         -b ${MODEL_QNN_DIR}/int8/cpp/${MODEL_NAME}.bin \
         -o ${MODEL_QNN_DIR}/int8/so/ \
         -t x86_64-linux-clang


    echo -e "\n>>>> Step 4 : 执行 qnn-context-binary-generator"
    ${QNN_SDK_BIN}/qnn-context-binary-generator \
         --backend ${QNN_SDK_BIN}/../lib/libQnnHtp.so \
         --model ${MODEL_QNN_DIR}/int8/so/x86_64-linux-clang/lib${MODEL_NAME}.so \
         --binary_file ${MODEL_NAME}_int8 \
         --output_dir ${MODEL_QNN_DIR}/int8/so/x86_64-linux-clang


    echo -e "\n>>>> Step 5 : 执行 qnn-net-run"
    ${QNN_SDK_BIN}/qnn-net-run \
        --backend ${QNN_SDK_BIN}/../lib/libQnnHtp.so \
        --model ${MODEL_QNN_DIR}/int8/so/x86_64-linux-clang/lib${MODEL_NAME}.so \
        --input_list ${INPUT_LIST_RUN} \
        --output_dir ${RESULT_QNX} \
        --debug
}


function execQnnFloat16 {
    echo -e "\n>>>> Step 1 : 执行 qnn-onnx-converter"
    ${QNN_SDK_BIN}/qnn-onnx-converter \
        --input_network ${MODEL_DIR}/${MODEL_ONNX} \
        -o ${MODEL_QNN_DIR}/float16/cpp/${MODEL_NAME}.cpp \
        --disable_batchnorm_folding # 该参数为禁止优化 batchnorm 网络结构


    echo -e "\n>>>> Step 2 : 执行 qnn-model-lib-generator"
    ${QNN_SDK_BIN}/qnn-model-lib-generator \
         -c ${MODEL_QNN_DIR}/float16/cpp/${MODEL_NAME}.cpp \
         -b ${MODEL_QNN_DIR}/float16/cpp/${MODEL_NAME}.bin \
         -o ${MODEL_QNN_DIR}/float16/so/ \
         -t x86_64-linux-clang


    echo -e "\n>>>> Step 3 : 生成配置文件"
    configBackend=`cat ${CURRENT_DIR}/qnn_config_fp16_backend_extensions_template.json`
    configGraph=`cat ${CURRENT_DIR}/qnn_config_fp16_graph_template.json`
    configBackend=${configBackend/<CONFIG_FILE>/${MODEL_QNN_DIR}/float16/qnn_config_fp16_graph.json}
    configGraph=${configGraph/<MODEL_NAME>/${MODEL_NAME}}
    touch ${MODEL_QNN_DIR}/float16/qnn_config_fp16_backend_extensions.json
    touch ${MODEL_QNN_DIR}/float16/qnn_config_fp16_graph.json
    echo -e "${configBackend}" > ${MODEL_QNN_DIR}/float16/qnn_config_fp16_backend_extensions.json
    echo -e "${configGraph}" > ${MODEL_QNN_DIR}/float16/qnn_config_fp16_graph.json


    echo -e "\n>>>> Step 4 : 执行 qnn-context-binary-generator"
    ${QNN_SDK_BIN}/qnn-context-binary-generator \
         --backend ${QNN_SDK_BIN}/../lib/libQnnHtp.so \
         --model ${MODEL_QNN_DIR}/float16/so/x86_64-linux-clang/lib${MODEL_NAME}.so \
         --binary_file ${MODEL_NAME} \
         --output_dir ${MODEL_QNN_DIR}/float16/so/x86_64-linux-clang \
         --config_file ${MODEL_QNN_DIR}/float16/qnn_config_fp16_backend_extensions.json


    echo -e "\n>>>> Step 5 : 生成 input_list_run.txt"
    prepareInputListRun


    echo -e "\n>>>> Step 6 : 执行 qnn-net-run"
    ${QNN_SDK_BIN}/qnn-net-run \
        --backend ${QNN_SDK_BIN}/../lib/libQnnHtp.so \
        --model ${MODEL_QNN_DIR}/float16/so/x86_64-linux-clang/lib${MODEL_NAME}.so \
        --input_list ${INPUT_LIST_RUN} \
        --output_dir ${RESULT_QNX} \
        --config_file ${MODEL_QNN_DIR}/float16/qnn_config_fp16_backend_extensions.json \
        --debug
}

# 进行 QNN 模型的逐层网络输出
function execQnn {
  if [[ $EXEC_QNN == "y" ]]; then
    #         conda activate qnn17
    source ${QNN_SDK_BIN}/envsetup.sh -o ${ONNX_ROOT}

    rm -rf ${RESULT_QNX}
    rm -rf ${MODEL_QNN_DIR}

    if [[ $MODEL_TYPE == "int8" ]]; then
      echo -e "\n================ QNN 模型转换 ->int8 ================="
      execQnnInt8
    elif [[ $MODEL_TYPE == "float16" ]]; then
      echo -e "\n================ QNN 模型转换 ->float16 ================="
      execQnnFloat16
    fi
  fi
}

function compare {
  # 进行ONNX模型的逐层网络和QNN模型逐层网络对齐结果
  echo -e "\n>>>> Step Final : Compare Onnx and Qnn model layer by layer"
  jsonPath="${MODEL_QNN_DIR}/${MODEL_TYPE}/cpp/${MODEL_NAME}_net.json"
  python cos_compare/compare_layer_by_layer.py ${jsonPath} "${MODEL_DIR}/Result"
}


function print_usage {
    set +x
    echo "Usage: $0 [option...]" >&2
    echo "   -h                   Show help message"
    echo "   -mt [8|16]           Set MODEL_TYPE to int8 or float16"
    echo "   -md <dir>            Set MODEL_DIR"
    echo "   -mn <dir>            Set MODEL_NAME"
    echo "   -tr <raw file>       Set TEST_RAW_IMAGE"
    echo "   -trd <raw file dir>  Set TEST_RAW_DIR"
    echo "   -ih <int>            Set INPUT_HEIGHT"
    echo "   -iw <int>            Set INPUT_WIDTH"
    echo "   -ic <int>            Set INPUT_CHANNEL"
    echo "   -onnx [n|y(def)]     Set exec OnnxModel"
    echo "   -qnn  [n|y(def)]     Set exec QnnModel"
    echo "   -jcp  [n(def)|y]     Set just compare onnx and qnn result"
    echo "   -uof  [n|y(def)]     Set use old files if exists instead generate new file"
}


function main {
    # Parse command line.
    if [ $# -eq 0 ] ; then
       print_usage
       exit 1
    fi
    while [ $# != 0 ]
    do
        case "$1" in
            -h)
                print_usage
                exit 1
                ;;
            -mt)
                MODEL_TYPE=$2
                shift
                ;;
            -md)
                MODEL_DIR=$2
                shift
                ;;
            -mn)
                MODEL_NAME=$2
                shift
                 ;;
            -tr)
                TEST_RAW_IMAGE=$2
                shift
                ;;
            -trd)
                TEST_RAW_DIR=$2
                shift
                ;;
            -ih)
                INPUT_HEIGHT=$2
                shift
                ;;
            -iw)
                INPUT_WIDTH=$2
                shift
                ;;
            -ic)
                INPUT_CHANNEL=$2
                shift
                ;;
            -onnx)
                EXEC_ONNX=$2
                shift
                ;;
            -qnn)
                EXEC_QNN=$2
                shift
                ;;
            -jcp)
                JUST_COMPARE=$2
                shift
                ;;
            -uof)
                USE_OLD_FILE=$2
                shift
                ;;
            *)
                echo "Parse arg error"
        esac
        shift
    done
    configEnv
    if [[ $JUST_COMPARE == "n" ]]; then
        execOnnx
        execQnn
    fi
    compare
}

main $@
