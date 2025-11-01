
set(TARGET_NAME AuraAutoDriving-UnitTester)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 重要：设置工具类库的文件列表
# 设置工具类库的源码根路径以及include头文件根路径
set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
collect_source_files(TARGET_SRCS GLOB_RECURSE .)

# 设置工具类库的文件列表
collect_source_files(TARGET_SRCS GLOB_RECURSE kalman_filter)
collect_source_files(TARGET_SRCS GLOB_RECURSE case)
collect_source_files(TARGET_SRCS GLOB_RECURSE bspline)
collect_source_files(TARGET_SRCS GLOB_RECURSE ransac)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils/math)

message(STATUS "[${TARGET_NAME}] INCLUDE_ROOT: ${INCLUDE_ROOT}")
message(STATUS "[${TARGET_NAME}] SRC_ROOT: ${SRC_ROOT}")
#message(STATUS "[${TARGET_NAME}] TARGET_SRCS: ${TEST_TARGET_SRCS}")
# 生成单测程序
add_executable(${TARGET_NAME} ${TARGET_SRCS})


# 设置生成物的include的目录
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_ROOT} ${CMAKE_SOURCE_DIR}/AuraAutoDriving/include)
# 设置生成物连接
target_link_libraries(${TARGET_NAME} PRIVATE  ${GTEST_LIB} AuraUtils AuraAutoDriving)


message(STATUS "[${TARGET_NAME}] ====================Build End============================")