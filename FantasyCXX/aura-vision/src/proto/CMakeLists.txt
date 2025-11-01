if (${HOST_OS} MATCHES osx)
    set(PLAINC_BIN ${CMAKE_SOURCE_DIR}/aura-lightbuffer/prebuilt/osx-x86_64/plainc)
elseif (${HOST_OS} MATCHES linux)
    set(PLAINC_BIN ${CMAKE_SOURCE_DIR}/aura-lightbuffer/prebuilt/linux-x86_64/plainc)
else ()
    # default use osx version
    set(PLAINC_BIN ${CMAKE_SOURCE_DIR}/aura-lightbuffer/prebuilt/osx-x86_64/plainc)
endif ()

set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(compile_proto)

# 通过 cache 策略加速编译，如果 proto 文件没有变更，则不需要重新编译
include(../../../cmake/utils.cmake)
file_changed(${PROTO_SRC_DIR}/model_config.proto model_config_proto ModelConfigChanged)
file_changed(${PROTO_SRC_DIR}/model_port.proto model_port_proto ModelPortChanged)
file_changed(${PROTO_SRC_DIR}/model_def.proto model_def_proto ModelDefChanged)

message(STATUS "[proto]HOST_OS = ${HOST_OS}")
message(STATUS "[proto]PLAINC_BIN = ${PLAINC_BIN}")
message(STATUS "[proto]ModelConfigChanged = ${ModelConfigChanged}")
message(STATUS "[proto]ModelPortChanged = ${ModelPortChanged}")
message(STATUS "[proto]ModelDefChanged = ${ModelDefChanged}")

if (ModelConfigChanged)
    add_custom_command(
            TARGET compile_proto
            COMMAND ${PLAINC_BIN} -I=${PROTO_SRC_DIR} --cpp_out=${PROTO_SRC_DIR} model_config.proto
            COMMENT "[plainc] compiling proto messages: ${PROTO_SRC_DIR}/model_config.proto"
            VERBATIM)
endif ()

if (ModelPortChanged)
    add_custom_command(
            TARGET compile_proto
            COMMAND ${PLAINC_BIN} -I=${PROTO_SRC_DIR} --cpp_out=${PROTO_SRC_DIR} model_port.proto
            COMMENT "[plainc] compiling proto messages: ${PROTO_SRC_DIR}/model_port.proto"
            VERBATIM)
endif ()

if (ModelDefChanged)
    add_custom_command(
            TARGET compile_proto
            COMMAND ${PLAINC_BIN} -I=${PROTO_SRC_DIR} --cpp_out=${PROTO_SRC_DIR} model_def.proto
            COMMENT "[plainc] compiling proto messages: ${PROTO_SRC_DIR}/model_def.proto"
            VERBATIM)
endif ()