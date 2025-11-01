message(STATUS "============Build CMakeLists:进行模型的配置 ===========")

# 设置模型文件vision_model.bin文件
set(model_files ${CMAKE_SOURCE_DIR}/models/generated/vision_model.bin)
# 设置获取模型文件的长度的python脚本
set(script_file ${CMAKE_SOURCE_DIR}/models/scripts/get_model_length.py)

# 如果使用外部模型。则设置模型的数据长度为0
if (${USE_EXTERNAL_MODEL} MATCHES "true")
    set(MODEL_LEN 0)
else ()
    # 如果模型文件不存在。则设置模型的数据长度为0
    if (NOT EXISTS ${model_file})
        message(WARNING "模型配置: model file does not exists! The built library will fail when init!")
        set(MODEL_LEN 0)
    else ()
        # 调用python命令：
        execute_process(COMMAND python ${script_file} --model ${model_file} OUTPUT_VARIABLE MODEL_LEN)
        message(STATUS "模型配置: get_model_len=${MODEL_LEN}")
    endif ()
endif ()
