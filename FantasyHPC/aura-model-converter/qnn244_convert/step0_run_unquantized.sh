#!/bin/bash

export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/zhangpengfei10/miniforge3/envs/qairt/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$QNN_SDK_ROOT/lib/python


export root_dir=${ENV_PROJECT_ROOT_DIR}
export MODEL_PATH=${root_dir}/onnx/${ENV_ONNX_MODEL_NAME}.onnx
export convert_output=${root_dir}/convert_output/${ENV_ONNX_MODEL_NAME}_fp32_qnn

export test_list=${ENV_TEST_LIST}
export infer_output=${root_dir}/infer_output/${ENV_ONNX_MODEL_NAME}_fp32_qnn

# 1: detailed dump, 0: only output
detailed=${1:-1}

# Main function to orchestrate the entire process
main() {
    echo "${BLUE}Starting the model conversion and inference process...${RESET}"
    # 进行模型转换（转换成QNN浮点模型）
    measure_time convert_model
    measure_time generate_model_lib

    # measure_time generate_context_bin
    # measure_time generate_context_bin_detailed
    measure_time run_inference

    cp $0 ${convert_output}

    echo "${BLUE}Process completed successfully!${RESET}"
}

# Color definitions for colorful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'  # Reset color

# Error handling function
handle_error() {
    echo "${RED}Error: $1${RESET}"
    exit 1
}

# Function to measure and log the execution time of a function
measure_time() {
    local start_time=$(date +%s)
    echo "${YELLOW}Starting $1...${RESET}"
    "$@"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "${GREEN}$1 completed in ${duration} seconds${RESET}"
}

# Function to convert ONNX model using the converter
convert_model() {
    echo "${CYAN}Converting the model using ONNX converter...${RESET}"

    if [ ! -f "${MODEL_PATH}" ]; then
        handle_error "ONNX model file not found at ${MODEL_PATH}"
    fi

    qnn-onnx-converter \
    --input_network ${MODEL_PATH} \                     # 模型路径
    --output_path ${convert_output}/model.cpp \         # 输出的model.cpp的路径
    --preserve_io layout \
     || handle_error "Failed to convert ONNX model"

    # --preserve_io layout
    # --debug
    # --custom_io gets higher precedence than --preserve_io
    # --float_bitwidth 32 \

    echo "${GREEN}Model successfully converted using ONNX converter${RESET}"
}


# Function to generate model libraries
generate_model_lib() {
    echo "${CYAN}Generating model libraries...${RESET}"

    if [ ! -f "${convert_output}/model.cpp" ]; then
        handle_error "Model cpp file not found at ${convert_output}/model.cpp"
    fi

    if [ ! -f "${convert_output}/model.bin" ]; then
        handle_error "Model bin file not found at ${convert_output}/model.bin"
    fi

    qnn-model-lib-generator \
    -c ${convert_output}/model.cpp \
    -b ${convert_output}/model.bin \
    -t x86_64-linux-clang \
    -o ${convert_output} \
    -l libmodel.so || handle_error "Failed to generate model library"

    echo "${GREEN}Model libraries successfully generated${RESET}"
}

# Function to generate the context binary
generate_context_bin() {
    echo "${CYAN}Generating context binary...${RESET}"

    if [ ! -f "${convert_output}/x86_64-linux-clang/libmodel.so" ]; then
        handle_error "Model library not found at ${convert_output}/x86_64-linux-clang/libmodel.so"
    fi

    if [ ! -f "${config_path}" ]; then
        handle_error "Config file not found at ${config_path}"
    fi

    qnn-context-binary-generator \
    --model ${convert_output}/x86_64-linux-clang/libmodel.so \
    --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
    --binary_file model_htp \
    --config_file ${config_path} \
    --output_dir ${convert_output} \
    --log_level warn || handle_error "Failed to generate context binary"

     # "error", "warn", "info" and "verbose"

    echo "${GREEN}Context binary successfully generated${RESET}"
}

# Function to generate the context binary
generate_context_bin_detailed() {
    echo "${CYAN}Generating context binary...${RESET}"

    if [ ! -f "${convert_output}/x86_64-linux-clang/libmodel.so" ]; then
        handle_error "Model library not found at ${convert_output}/x86_64-linux-clang/libmodel.so"
    fi

    if [ ! -f "${config_path}" ]; then
        handle_error "Config file not found at ${config_path}"
    fi

    qnn-context-binary-generator \
    --model ${convert_output}/x86_64-linux-clang/libmodel.so \
    --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
    --binary_file model_htp_detailed \
    --config_file ${config_path} \
    --output_dir ${convert_output} \
    --enable_intermediate_outputs \
    --log_level warn || handle_error "Failed to generate context binary"

    # --set_output_tensors
    # --enable_intermediate_outputs
    # "error", "warn", "info" and "verbose"

    echo "${GREEN}Context binary successfully generated${RESET}"
}

# Function to run the unquantized model inference
run_inference() {
    echo "${CYAN}Executing unquantized model inference...${RESET}"

    if [ ! -f "${test_list}" ]; then
        handle_error "Test list file not found at ${test_list}"
    fi

    # enable debug by default
    ENABLE_DEBUG="--debug"
    if [ "$detailed" = "0" ]; then
        ENABLE_DEBUG=""
    fi
    echo "ENABLE_DEBUG = ${ENABLE_DEBUG}"

    qnn-net-run \
    --model ${convert_output}/x86_64-linux-clang/libmodel.so \
    --input_list ${test_list} \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so \
    --output_dir ${infer_output} \
    --log_level error ${ENABLE_DEBUG}
    #--debug
    # --debug "error", "warn", "info" and "verbose"

    echo "${GREEN}Inference successfully completed${RESET}"
}

# Run the main function
main