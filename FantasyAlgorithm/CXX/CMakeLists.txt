# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.10)
# 第二步： 设置工程名称
# 设置工具类库的名字
set(TARGET_NAME AliceAlgorithm)

message(STATUS "[${TARGET_NAME}] ====================Start Build============================")
# 第三步：设置CMake设置的C++的版本
project(${TARGET_NAME})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/AuraPlatform.cmake")

add_library(${TARGET_NAME} STATIC src/L0122_best-time-to-buy-and-sell-stock-ii.cc)
target_include_directories(${TARGET_NAME} PUBLIC include)
aura_apply_platform(${TARGET_NAME})

include(CTest)
if(BUILD_TESTING)
    add_executable(${TARGET_NAME}Tests tests/test_stock_profit.cc)
    target_link_libraries(${TARGET_NAME}Tests PRIVATE ${TARGET_NAME})
    aura_apply_platform(${TARGET_NAME}Tests)
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME ${TARGET_NAME}Tests COMMAND ${TARGET_NAME}Tests)
    endif()
endif()


message(STATUS "[${TARGET_NAME}] ====================Build End============================")
