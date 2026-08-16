#!/bin/bash

# export QNN_SDK_ROOT=/home/zlf/software/ai-sdk/qairt/2.28.0
# echo $QNN_SDK_ROOT

export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/zhangpengfei10/miniforge3/envs/qairt/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$QNN_SDK_ROOT/lib/python

export root_dir=${ENV_PROJECT_ROOT_DIR}
export MODEL_PATH=${root_dir}/onnx/${ENV_ONNX_MODEL_NAME}.onnx

# override json path
override_file=${1:-$ENV_OVERRIDE_JSON_PATH}
# qat calib list with only 1 raw if full override json is provided
input_list=${2:-$ENV_CALIB_LIST}

export convert_output=${root_dir}/convert_output/${ENV_MODEL_DIR}

export config_path=/data2/wzj/01.WorkSpace/asd_scene_cls_4heads_model_quant/utils/qnn244_convert/config/config.json

# export custom_io=/home/zlf/Documents/projects/o1sdsr/qnn_quant/encoder/onnx/encoder_custom_io.yaml

# qnn-onnx-converter -help && exit 1

# Color definitions for colorful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'  # Reset color

# Main function to orchestrate the entire process
main() {
    echo "${BLUE}Starting the model conversion, quantization, and inference process...${RESET}"
    echo "${YELLOW}override json=${override_file} input list=${input_list}${RESET}"
    measure_time convert_model
    measure_time generate_model_lib
    measure_time generate_context_bin
    measure_time generate_context_bin_detailed
    # mv schematic.bin for optrace
    mv model_schematic.bin ${convert_output}

    # cp $0 ${convert_output}
    # if [ -n "${override_file}" ]; then
    #     cp ${override_file} ${convert_output}/override.json
    # fi
    # measure_time run_inference_so
    # measure_time run_inference
    echo "${BLUE}Process completed successfully!${RESET}"
}

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

    if [ ! -f "${input_list}" ]; then
        handle_error "Input list file not found at ${input_list}"
    fi

    # if [ ! -f "${override_file}" ]; then
    #     handle_error "override_file not found at ${override_file}"
    # fi

    qnn-onnx-converter --input_network ${MODEL_PATH} --input_list ${input_list} \
    --output_path ${convert_output}/model.cpp \
    --act_bitwidth ${ENV_ACT_BITWIDTH} --weights_bitwidth ${ENV_WEIGHT_BITWIDTH} --bias_bitwidth 32 \
    --param_quantizer_calibration min-max \
    --use_per_channel_quantization \
    --act_quantizer_calibration ${ENV_ACT_QUANT} \
    --dump_encoding_json \
    --preserve_io layout \
    --debug || handle_error "ONNX model conversion failed"

    # --act_quantizer_calibration percentile --percentile_calibration_value 99.99 \

    # --restrict_quantization_steps "-0x8000 0x7F7F"  for A16W16
    # --quantization_overrides ${override_file} \
    # --preserve_io layout \
    # --use_per_channel_quantization \
    # --dump_encoding_json --quantization_overrides ${override_file}
    # --dumpIR  --dump_inferred_model --dump_value_info \
    # --preserve_io layout --debug \
    # --debug
    # --expand_lstm_op_structure \
    # --algorithm cle \
    # --use_per_channel_quantization \  # Convolution, Deconvolution, and FullyConnected
    # --use_per_row_quantization \      # row wise quantization of Matmul and FullyConnected ops
    # --param_quantizer_calibration min-max \ min-max (default), sqnr, entropy, mse, percentile
    # --act_quantizer_calibration sqnr \    min-max (default), sqnr, entropy, mse, percentile
    # --act_quantizer_schema asymmetric \   asymmetric (default), symmetric, unsignedsymmetric
    # --param_quantizer_schema symmetric \  asymmetric (default), symmetric, unsignedsymmetric
    # --custom_io ${custom_io} \

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
    --enable_intermediate_outputs \
    --config_file ${config_path} \
    --output_dir ${convert_output} \
    --profiling_option=optrace --profiling_level=detailed \
    --log_level warn || handle_error "Failed to generate context binary"

    # --set_output_tensors 193,215,267,289,341,363,412,434,456,669,690,706,481,499,611,503,617,610,623,660 \
    # --set_output_tensors
    # --enable_intermediate_outputs \
    # "error", "warn", "info" and "verbose"

    echo "${GREEN}Context binary successfully generated${RESET}"
}

# Function to run the quantized model inference
run_inference_bin() {
    echo "${CYAN}Executing quantized model inference...${RESET}"

    if [ ! -f "${convert_output}/model_htp.bin" ]; then
        handle_error "Context binary file not found at ${convert_output}/model_htp_detailed.bin"
    fi

    if [ ! -f "${test_list}" ]; then
        handle_error "Test list file not found at ${test_list}"
    fi

    qnn-net-run \
    --retrieve_context ${convert_output}/model_htp_detailed.bin \
    --input_list ${test_list} \
    --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
    --output_dir ${infer_output} \
    --log_level warn || handle_error "Inference failed"
    # "error", "warn", "info" and "verbose"


    echo "${GREEN}Inference successfully completed${RESET}"
}

# Function to run the quantized model inference
run_inference_so() {

    echo "${CYAN}Executing quantized model inference...${RESET}"
    if [ ! -f "${convert_output}/x86_64-linux-clang/libmodel.so" ]; then
        handle_error "libmodel.so not found at ${convert_output}/x86_64-linux-clang/libmodel.so"
    fi

    if [ ! -f "${test_list}" ]; then
        handle_error "Test list file not found at ${test_list}"
    fi

    qnn-net-run \
    --model ${convert_output}/x86_64-linux-clang/libmodel.so \
    --input_list ${test_list} \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --output_dir ${infer_output} \
    --log_level error --debug || handle_error "Inference failed"
    # --debug "error", "warn", "info" and "verbose"

    echo "${GREEN}Inference successfully completed${RESET}"
}

# Run the main function
main