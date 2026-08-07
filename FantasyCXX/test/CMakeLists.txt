# AuraKaleidoCXX 顶层单元测试
# 依赖 GTest（已内置）与 aura-utils。
set(UT_APP_NAME AuraKaleidoCXXTester)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
collect_source_files(TARGET_SRCS GLOB_RECURSE .)

# Ceres 预编译库未随本平台发布，跳过 Ceres 曲线拟合单测
list(FILTER TARGET_SRCS EXCLUDE REGEX ".*/ceres-solver/TestCubicCurveFitting\\.cpp$")
if(NOT BUILD_PERFORMANCE_TESTS)
    list(FILTER TARGET_SRCS EXCLUDE REGEX ".*/performance/.*")
endif()

add_executable(${UT_APP_NAME} ${TARGET_SRCS})

target_include_directories(${UT_APP_NAME} PRIVATE ${INCLUDE_ROOT})
target_link_libraries(${UT_APP_NAME} PRIVATE ${GTEST_LIB} aura_utils Threads::Threads)
add_test(NAME ${UT_APP_NAME} COMMAND ${UT_APP_NAME})

message(STATUS "[${UT_APP_NAME}] ====================Build End============================")
