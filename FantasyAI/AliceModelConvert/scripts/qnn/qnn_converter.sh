#!/usr/bin/env bash


# 开关控制
EXEC_ONNX="y"
EXEC_QNN="y"
JUST_COMPARE="n"
USE_OLD_FILE="y"

# 初始化模型转换变量
MODEL_TYPE=""
MODEL_DIR=""
MODEL_NAME=""
TEST_RAW_IMAGE=""
TEST_RAW_DIR=""
INPUT_HEIGHT=""
INPUT_WIDTH=""
INPUT_CHANNEL=""

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

function print_usage {
    set +x
    echo "Usage: $0 [option...]" >&2
    echo "   -h                             Show help message"
    echo "   -model_type [8|16]             Set MODEL_TYPE to int8 or float16"
    echo "   -model_dir <dir>               Set MODEL_DIR"
    echo "   -model_name <dir>              Set MODEL_NAME"
    echo "   -raw_img <raw file>            Set TEST_RAW_IMAGE"
    echo "   -raw_dir <raw file dir>        Set TEST_RAW_DIR"
    echo "   -input_h <int>                 Set INPUT_HEIGHT"
    echo "   -input_w <int>                 Set INPUT_WIDTH"
    echo "   -input_c <int>                      Set INPUT_CHANNEL"
    echo "   -onnx [n|y(default)]           Set exec OnnxModel"
    echo "   -qnn  [n|y(default)]           Set exec QnnModel"
    echo "   -compare  [n(default)|y]       Set just compare onnx and qnn result"
    echo "   -uof  [n|y(default)]           Set use old files if exists instead generate new file"
}


function main {
    # $# 表示参数个数，如果参数个数为0 ，则输出提示信息
    if [ $# -eq 0 ];then
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
            -model_type)
                MODEL_TYPE=$2
                shift
                ;;
            -model_dir)
                MODEL_DIR=$2
                shift
                ;;   
            -model_name)
                MODEL_NAME=$2
                shift
                 ;;     
            -raw_img)
                TEST_RAW_IMAGE=$2
                shift
                ;;
            -raw_dir)
                TEST_RAW_DIR=$2
                shift
                ;;    
            -input_h)
                INPUT_HEIGHT=$2
                shift
                ;;
            -input_w)
                INPUT_WIDTH=$2
                shift
                ;;        
            -input_c)
                INPUT_CHANNEL=$2
                shift
                ;;
            -compare)
                JUST_COMPARE=$2
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



main "$@"