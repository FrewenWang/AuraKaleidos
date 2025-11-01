message(STATUS "[AuraVisionTools] ====================AuraVisionTools Build============================")

if (${TARGET_OS} MATCHES "android" OR ${TARGET_OS} MATCHES "qnx")
    message(WARNING "ignored compiling tools on ${TARGET_OS}")
    return()
endif ()

set(CXX g++)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(TOOL_NAME MergeModelTool)

include_directories(
        ${CMAKE_SOURCE_DIR}/vision/src/
        ${CMAKE_SOURCE_DIR}/vision/include/
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/plainbuffer/include
)

#list(APPEND lib_srcs)
#collect_source_files(lib_srcs GLOB ${CMAKE_SOURCE_DIR}/vision/src/proto)
#
#add_executable(${TOOL_NAME} ${lib_srcs})
#add_dependencies(${TOOL_NAME} compile_proto plainbuffer)
#target_link_libraries(${TOOL_NAME} pthread plainbuffer)

message(STATUS "[AuraVisionTools] ====================AuraVisionTools End============================")