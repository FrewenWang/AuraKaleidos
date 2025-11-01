message(STATUS "[AuraLightBuffer] =================== AuraLightBuffer Build ===================")
cmake_minimum_required(VERSION 3.10)
project(LightBuffer)

# 编译配置
option(BUILD_TEST "build unit test" OFF)                    # 编译单测程序
option(BUILD_EXAMPLE "build example" ON)                   # 编译示例程序
option(BUILD_LIGHT_C "build LightBuffer compiler" ON)        # 编译生成 LightBuffer 的编译程序

# 编译变量设置
set(LIB_NAME light_buffer)
set(COMPILER_NAME LightBufferC)

# 设置工具类库的源码根路径以及include头文件根路径
set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 设置工具类库的文件列表
collect_source_files(LIB_SRCS GLOB_RECURSE .)

message(STATUS "[AuraLightBuffer] include_root:${INCLUDE_ROOT}")
message(STATUS "[AuraLightBuffer] src_root:${SRC_ROOT}")
message(STATUS "[AuraLightBuffer] lib_srcs:${LIB_SRCS}")

list(APPEND src_compiler)

# 设置生成静态库
add_library(${LIB_NAME} STATIC ${LIB_SRCS})
# 设置当前目标物的include文件夹
target_include_directories(${LIB_NAME} PUBLIC include)
target_compile_options(${LIB_NAME} PRIVATE -O3 -std=c++11 -fPIC)

# 设施目标生成物需要连接的三方依赖库
target_link_libraries(${LIB_NAME} PUBLIC
                      ${OPENCV_LIB}
                      AuraLib
                      AuraCV)

if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif()

message(STATUS "[AuraLightBuffer] =================== AuraLightBuffer Build End===================")