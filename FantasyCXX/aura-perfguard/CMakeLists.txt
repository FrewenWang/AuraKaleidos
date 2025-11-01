message(STATUS "[AuraPerfGuard] =================== AuraPerfGuard Build Begin ====================================")
cmake_minimum_required(VERSION 3.10)
project(AuraPerfGuard)

# 编译配置
option(BUILD_TEST           "build unit test"       OFF)                    # 编译单测程序
option(BUILD_EXAMPLE        "build example"         ON)                     # 编译示例程序

# 编译变量设置
set(LIB_NAME PerfGuard)
# 设置工具类库的源码根路径以及include头文件根路径
set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 设置工具类库的文件列表
collect_source_files(LIB_SRCS GLOB_RECURSE .)

message(STATUS "[AuraPerfGuard] include_root:${INCLUDE_ROOT}")
message(STATUS "[AuraPerfGuard] src_root:${SRC_ROOT}")
message(STATUS "[AuraPerfGuard] lib_srcs:${LIB_SRCS}")
message(STATUS "[AuraPerfGuard] build_example:${BUILD_EXAMPLE}")


# 设置生成静态库
add_library(${LIB_NAME} STATIC ${LIB_SRCS} example/main.cpp)
# 设置当前目标物的include文件夹
target_include_directories(${LIB_NAME} PUBLIC include)
target_compile_options(${LIB_NAME} PRIVATE -O3 -std=c++11 -fPIC)

# 设施目标生成物需要连接的三方依赖库
target_link_libraries(${LIB_NAME} PUBLIC
        AuraLib)

if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif()

message(STATUS "[AuraPerfGuard] =================== AuraPerfGuard Build End ======================================")