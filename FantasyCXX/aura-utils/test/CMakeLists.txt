# 设置工具类库的名字
set(TARGET_NAME AuraUtils-UnitTester)
logger_d("[${TARGET_NAME}] ====================Start Build============================")

# 设置单测的代码文件列表
collect_source_files_recurse(${CMAKE_CURRENT_SOURCE_DIR}/src TARGET_SRCS)

# 生成单测程序
add_executable(${TARGET_NAME} ${TARGET_SRCS})


# 设置生成物的include的目录
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_ROOT} ${AURA_LIB_INCLUDE_DIRS})
# 设置生成物连接
target_link_libraries(${TARGET_NAME} PRIVATE
                      ${GTEST_LIB}
                      aura_utils)


#logger_d("[${TARGET_NAME}] CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
#logger_d("[${TARGET_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
#logger_d("[${TARGET_NAME}] TARGET_SRCS: ${TARGET_SRCS}")
#logger_d("[${TARGET_NAME}] GTEST_LIB: ${GTEST_LIB}")
#logger_d("[${TARGET_NAME}] GTEST_LINK_PATH: ${GTEST_LINK_PATH}")
# logger_d("[${TARGET_NAME}] AURA_LIB_SRCS: ${AURA_LIB_SRCS}")

logger_d("[${TARGET_NAME}] ====================Build End============================")