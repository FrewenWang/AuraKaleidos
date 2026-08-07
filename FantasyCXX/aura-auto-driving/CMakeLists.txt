# AuraAutoDriving
# 自动驾驶基础算法库：卡尔曼滤波、目标关联、几何工具等。
# 依赖 Eigen（头文件库，已内置）与 aura-utils。
#
# 说明：smoother/QSQPSplineSmoother.h 依赖 OsqpEigen（未随工程发布），但它是纯头文件且
# 未被任何 .cpp 直接引用，collect_source_files 不会收集到它，因此不影响编译。
set(TARGET_NAME AuraAutoDriving)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")

option(BUILD_UNIT_TEST_LIB "build unit test" ON)

set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

collect_source_files(TARGET_SRCS GLOB_RECURSE kalman_filter)
collect_source_files(TARGET_SRCS GLOB_RECURSE smoother)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils)
collect_source_files(TARGET_SRCS GLOB_RECURSE utils/math)

# 静态库（避免把 .h 当作源文件加入动态库目标）
add_library(${TARGET_NAME} STATIC ${TARGET_SRCS})
target_include_directories(${TARGET_NAME} PUBLIC
        ${INCLUDE_ROOT}
        ${SRC_ROOT}
        ${CMAKE_SOURCE_DIR}/aura-config-parser/include)
target_link_libraries(${TARGET_NAME} PUBLIC aura_utils)

if (BUILD_UNIT_TEST_LIB)
    add_subdirectory(tests)
endif ()

message(STATUS "[${TARGET_NAME}] ====================Build End============================")
