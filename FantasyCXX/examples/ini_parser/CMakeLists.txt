# This file handles building each task as sub-project.
#
# Each task becomes an independent project in which you can experiment.
#
# Tasks are added as separate sub-projects so as compilation errors in
# one task don't affect another task.

set(TARGET_NAME iniparser)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")


# 设置工具类库的编译选项
option(BUILD_STATIC_LIB         "build the library as static"   OFF)        # 看判断编译静态库还是动态库
option(BUILD_TEST               "build test"                    ON)         # 是否编译测试程序

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
# 设置工具类库的文件列表
collect_source_files(TARGET_SRCS GLOB_RECURSE .)

message(STATUS "[${TARGET_NAME}] INCLUDE_ROOT:${INCLUDE_ROOT}")
message(STATUS "[${TARGET_NAME}] SRC_ROOT:${SRC_ROOT}")
# message(STATUS "[${TARGET_NAME}] TARGET_SRCS:${TARGET_SRCS}")

# 设置编译配置，编译静态库(.a)、或者共享库(.so)
if (BUILD_STATIC_LIB)
    # 编译静态库(.a文件)
    add_library(${TARGET_NAME} STATIC ${TARGET_SRCS})
else ()
    # 编译动态库(.so文件)
    add_library(${TARGET_NAME} SHARED ${TARGET_SRCS})
endif ()

# 设置目标产物的include头文件的路径
target_include_directories(
        ${TARGET_NAME} PUBLIC
        ${INCLUDE_ROOT}
        ${SRC_ROOT})

# 设置附属生成物的编译控制
if (BUILD_TEST)
    add_subdirectory(test)
endif ()

message(STATUS "[${TARGET_NAME}] ====================Build End============================")