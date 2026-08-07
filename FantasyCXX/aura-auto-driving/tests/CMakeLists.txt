set(TARGET_NAME AuraAutoDriving-UnitTester)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
collect_source_files(TARGET_SRCS GLOB_RECURSE .)
collect_source_files(TARGET_SRCS GLOB_RECURSE kalman_filter)
collect_source_files(TARGET_SRCS GLOB_RECURSE case)
collect_source_files(TARGET_SRCS GLOB_RECURSE bspline)
collect_source_files(TARGET_SRCS GLOB_RECURSE ransac)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils/math)

# QSQPSplineSmoother 依赖 OsqpEigen（未随工程发布），跳过其单测
list(FILTER TARGET_SRCS EXCLUDE REGEX ".*/bspline/TestFitBSpline\\.cpp$")

add_executable(${TARGET_NAME} ${TARGET_SRCS})

target_include_directories(${TARGET_NAME} PRIVATE
        ${INCLUDE_ROOT}
        ${CMAKE_SOURCE_DIR}/aura-auto-driving/include
        ${CMAKE_SOURCE_DIR}/aura-auto-driving/src
        ${CMAKE_SOURCE_DIR}/aura-config-parser/include)
target_link_libraries(${TARGET_NAME} PRIVATE ${GTEST_LIB} aura_utils AuraAutoDriving)
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})

message(STATUS "[${TARGET_NAME}] ====================Build End============================")
