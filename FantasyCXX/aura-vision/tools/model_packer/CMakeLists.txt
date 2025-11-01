if (${ICP_TARGET_OS} MATCHES "android" OR ${ICP_TARGET_OS} MATCHES "qnx")
    message(WARNING "ignored compiling tools on ${ICP_TARGET_OS}")
    return()
endif ()

set(CXX g++)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(TOOL_NAME merge_model_tool)

include_directories(
        ${CMAKE_SOURCE_DIR}/vision/src/
        ${CMAKE_SOURCE_DIR}/vision/include/
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/plainbuffer/include
)

list(APPEND packer_src
        main.cpp
     MergeTool.cpp
        model_packer.cpp
        downloader.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/util/file_util.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/util/log.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/util/SystemClock.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/util/VaAllocator.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/util/InferenceConverter.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/model_ir/model_info.cpp
        ${CMAKE_SOURCE_DIR}/vision/src/model_ir/encryption.cpp)
collect_source_files(packer_src GLOB ${CMAKE_SOURCE_DIR}/vision/src/proto)

add_executable(${TOOL_NAME} ${packer_src})
add_dependencies(${TOOL_NAME} compile_proto plainbuffer)
target_link_libraries(${TOOL_NAME} pthread plainbuffer)
