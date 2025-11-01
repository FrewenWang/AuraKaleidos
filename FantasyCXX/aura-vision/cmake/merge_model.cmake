message(STATUS "============Build CMakeLists:进行模型的合并 ===========")

include(cmake/utils.cmake)

# 判断模型文件是否存在。如果合并的模型不存在，则设置NEED_MERGE_MODEL为true
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/models/generated/vision_model.bin")
    set(NEED_MERGE_MODEL true)
else ()
    # 模型配置文件 main.prototxt
    file(MD5 ${MODEL_CONFIG_FILE} CONFIG_CHECKSUM)

    # 判断ConfigCache.txt的文件是否存在。读取文件
    if (EXISTS "${BINARY_DIR}/ConfigCache.txt")
        file(READ "${BINARY_DIR}/ConfigCache.txt" CONFIG_CHECKSUM_CACHE)
    endif ()

    message(STATUS "模型合并: CONFIG_CHECKSUM=${CONFIG_CHECKSUM}")

    # 如果CONFIG_CHECKSUM 和 CONFIG_CHECKSUM_CACHE 的MD5值是否一致
    if (NOT "${CONFIG_CHECKSUM}" STREQUAL "${CONFIG_CHECKSUM_CACHE}")
        file(WRITE "${BINARY_DIR}/ConfigCache.txt" ${CONFIG_CHECKSUM})
        set(NEED_MERGE_MODEL true)
        message(STATUS "No changes found in model config file")
    else ()
        set(NEED_MERGE_MODEL false)
    endif ()
endif ()


# For hack usage, when the library must be compiled with a given model file
set(FORCE_USE_GIVEN_MODEL false)

message(STATUS "模型合并: NEED_MERGE_MODEL = ${NEED_MERGE_MODEL}")
message(STATUS "模型合并: FORCE_USE_GIVEN_MODEL = ${FORCE_USE_GIVEN_MODEL}")

# 如果需要进行模型合并。并且不强制使用外部给的模型
if (NEED_MERGE_MODEL AND NOT FORCE_USE_GIVEN_MODEL)
    message(STATUS "模型合并: 开始合并 config file : ${MODEL_CONFIG_FILE}")
    # 调用合并模型的python脚本
    execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/models/scripts/merge_model.py --config ${MODEL_CONFIG_FILE})
endif ()