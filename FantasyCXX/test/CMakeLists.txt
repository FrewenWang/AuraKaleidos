# AuraKaleidoCXX 顶层单元测试
# 依赖 GTest（已内置）与 aura-utils。
set(UT_APP_NAME AuraKaleidoCXXTester)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Accelerate")

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
collect_source_files(TARGET_SRCS GLOB_RECURSE .)

# Ceres 预编译库未随本平台发布，跳过 Ceres 曲线拟合单测
list(REMOVE_ITEM TARGET_SRCS ${SRC_ROOT}/src/ceres-solver/TestCubicCurveFitting.cpp)

add_executable(${UT_APP_NAME} ${TARGET_SRCS})

target_include_directories(${UT_APP_NAME} PRIVATE ${INCLUDE_ROOT})
target_link_libraries(${UT_APP_NAME} PRIVATE ${GTEST_LIB} aura_utils)

message(STATUS "[${UT_APP_NAME}] ====================Build End============================")
