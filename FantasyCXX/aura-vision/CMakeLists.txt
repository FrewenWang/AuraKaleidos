set(TARGET_NAME AuraVision)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")


# ----------------------------------------------------------------------------------------------------------------------
#                                               设置编译工程的配置参数
# ----------------------------------------------------------------------------------------------------------------------
if (NOT DEFINED USE_EXT_MODEL)
    set(USE_EXT_MODEL false)
    message(STATUS "[${TARGET_NAME}] App useExternalModel = ${USE_EXT_MODEL}")
endif ()

# 执行模型打点的相关工作
include(cmake/generate_model.cmake)

add_subdirectory(src/proto)
add_subdirectory(src)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TOOLS)
    add_subdirectory(tools)
endif ()

message(STATUS "[${TARGET_NAME}] ====================Build End============================")