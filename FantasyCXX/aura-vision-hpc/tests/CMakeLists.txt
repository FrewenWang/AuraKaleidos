
set(TARGET_NAME AuraVisionHPC-UnitTester)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# 查找 OpenCL 框架（macOS 专用）
find_library(OPENCL_LIBRARY OpenCL)     # macOS 上会自动定位到 /System/Library/Frameworks/OpenCL.framework

# 如果未找到，尝试通用查找（适用于其他平台）
if (NOT OPENCL_LIBRARY)
    find_package(OpenCL REQUIRED)
    set(OPENCL_LIBRARY ${OpenCL_LIBRARIES})
endif()

# 重要：设置工具类库的文件列表
# 设置工具类库的源码根路径以及include头文件根路径
set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
collect_source_files(TEST_TARGET_SRCS GLOB_RECURSE .)
collect_source_files(TEST_TARGET_SRCS GLOB_RECURSE opencl)


message(STATUS "[${TARGET_NAME}] INCLUDE_ROOT: ${INCLUDE_ROOT}")
message(STATUS "[${TARGET_NAME}] SRC_ROOT: ${SRC_ROOT}")
#message(STATUS "[${TARGET_NAME}] TEST_TARGET_SRCS: ${TEST_TARGET_SRCS}")
message(STATUS "[${TARGET_NAME}] GTEST_LIB:${GTEST_LIB}")
# 生成单测程序
add_executable(${TARGET_NAME} ${TEST_TARGET_SRCS}
        src/opencl/utils/KernelUtils.cpp
        src/opencl/utils/KernelUtils.h)


# 设置生成物的include的目录
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_ROOT})
# 设置生成物连接
target_link_libraries(${TARGET_NAME} PRIVATE  ${GTEST_LIB} AuraUtils)
# 将 OpenCL 链接到目标
target_link_libraries(${TARGET_NAME} PRIVATE ${OPENCL_LIBRARY})


message(STATUS "[${TARGET_NAME}] ====================Build End============================")