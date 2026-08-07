# 设置工具类库的名字
set(TARGET_NAME PerfGuard-UnitTester)
message(STATUS "[${TARGET_NAME}] ====================Start Build============================")

file(GLOB_RECURSE test_srcs LIST_DIRECTORIES FALSE
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE LIST_DIRECTORIES FALSE
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

link_directories(
        ${OPENCV_3RDPARTY_LINK_DIR}
        ${OPENCV_LINK_DIR}
)

message(STATUS "[AuraCV:UnitTest] test_srcs ${test_srcs}")
add_executable(${TARGET_NAME} ${test_srcs})

target_compile_options(${TARGET_NAME} PRIVATE -std=c++14)
target_compile_options(${TARGET_NAME} PRIVATE -O3)

target_include_directories(${TARGET_NAME}
                           PRIVATE
                           ${test_include}
                           ${OPENCV_INCLUDE_DIR}
                           ${CMAKE_SOURCE_DIR}/AuraCV/include
                           ${CMAKE_SOURCE_DIR}/AuraUtils/include)

target_link_libraries(${TARGET_NAME}
                      PRIVATE
                      ${OPENCV_LIB}
                      ${GTEST_LIB}
                      AuraCV
                      AuraUtils)

message(STATUS "[${TARGET_NAME}] ====================Build End============================")

