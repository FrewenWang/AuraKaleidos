message(STATUS "[AuraKaleidoCXX-UnitTest] ==================== AuraKaleidoCXX UnitTest Build ====================")

set(UT_APP_NAME AuraKaleidoCXXTester)


# 查找并添加glog库
# 在开发宿主机上安装glog库
# 注释掉下面语句 add by frewen@20250409 ubuntu20.04暂时没有安装成功glog
# find_package(glog REQUIRED)


# 使用gtest的时候，提示对‘pthread_key_create’未定义的引用这样的错误
# 需要添加如下对于pthread的依赖
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# Enable C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Accelerate")

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
# 如果使用collect_source_files来进行列举出所有源码文件。
# 首先需要在最外层的cmake中include(aura-cmake/common_utils.cmake)
# 其次，需要明确使用"SRC_ROOT"来指定源码文件的最外层根目录
# 最后，需要使用"srcs" 来指定源码的列表
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
collect_source_files(TARGET_SRCS GLOB_RECURSE .)
collect_source_files(TARGET_SRCS GLOB_RECURSE basic)
collect_source_files(TARGET_SRCS GLOB_RECURSE ceres-solver)
collect_source_files(TARGET_SRCS GLOB_RECURSE eigen3)
collect_source_files(TARGET_SRCS GLOB_RECURSE performance)
message(STATUS "[${TARGET_NAME}] INCLUDE_ROOT: ${INCLUDE_ROOT}")
message(STATUS "[${TARGET_NAME}] SRC_ROOT: ${SRC_ROOT}")
#message(STATUS "[${TARGET_NAME}] TARGET_SRCS: ${TEST_TARGET_SRCS}")

# 生成单测程序
add_executable(${UT_APP_NAME} ${TARGET_SRCS})

# 设置目标产物的include包含头文件的路径
target_include_directories(${UT_APP_NAME} PRIVATE ${INCLUDE_ROOT})
# 设置目标产物link链接依赖库
# message(STATUS "[AuraKaleidoCXX-UnitTest] CERES_LIB : ${CERES_LIB}")
target_link_libraries(${UT_APP_NAME} PRIVATE ${GTEST_LIB} ${CERES_LIB})
target_link_libraries(${UT_APP_NAME} PRIVATE AuraUtils)
# 链接glog库
# 注释掉下面语句 add by frewen@20250409 ubuntu20.04暂时没有安装成功glog
#target_link_libraries(${UT_APP_NAME} PRIVATE glog::glog)


message(STATUS "[AuraKaleidoCXX-UnitTest] ==================== AuraKaleidoCXX UnitTest Build End ====================")