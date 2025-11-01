
set(TASK_NAME generate_model)
message(STATUS "============Build CMakeLists:进行模型的打包和生成 ===========")

if (${USE_EXT_MODEL} MATCHES true)
    set(USE_EXTERNAL_MODEL true)
else ()
    set(USE_EXTERNAL_MODEL false)
endif ()

message(STATUS "[{}]: USE_EXTERNAL_MODEL=${USE_EXTERNAL_MODEL}")
message(STATUS "模型打包: CMAKE_COMMAND=${CMAKE_COMMAND}")
message(STATUS "模型打包: CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message(STATUS "模型打包: CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")


# 名称: vision_model
# command:command就是生成目标文件的命令
# 参数一：-DMODEL_CONFIG_FILE=${CMAKE_SOURCE_DIR}/models/config/${PRODUCT}.prototxt
# 参数二：-DBINARY_DIR=${CMAKE_BINARY_DIR}
# WORKING_DIRECTORY：工作目录
# COMMENT: 评论信息
# [VERBATIM] [USES_TERMINAL]:

add_custom_target(
        vision_model

        ${CMAKE_COMMAND} -DMODEL_CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/models/config/${PRODUCT}.prototxt



        -DBINARY_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/merge_model.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "模型打包:开始进行模型合并..."
        VERBATIM
)


# 参数二：-DUSE_EXTERNAL_MODEL=${USE_EXTERNAL_MODEL}  设置是否使用外部模型的参数配置
add_custom_target(
        configure_vision_model
        ${CMAKE_COMMAND} -DTARGET_CONFIG=${CMAKE_BINARY_DIR}/build_config.h
        -DUSE_EXTERNAL_MODEL=${USE_EXTERNAL_MODEL}
        -P ${CMAKE_SOURCE_DIR}/cmake/configure_model.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "模型打包:开始进行模型配置..."
        VERBATIM
)

add_dependencies(configure_vision_model vision_model)