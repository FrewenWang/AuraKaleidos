# This file handles building each task as sub-project.
#
# Each task becomes an independent project in which you can experiment.
#
# Tasks are added as separate sub-projects so as compilation errors in
# one task don't affect another task.
cmake_minimum_required(VERSION 3.10)

set(TARGET_NAME DemoIniParser)
message(STATUS "[${TARGET_NAME}] ==================== Start Build============================")

add_executable(${TARGET_NAME} ini_example.cpp)

# 设置生成物的include的目录
target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_ROOT})

target_link_libraries(${TARGET_NAME} PRIVATE iniparser)

message(STATUS "[${TARGET_NAME}] ==================== Build End ====================")
