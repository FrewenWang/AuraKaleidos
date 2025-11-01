# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.10)
# 第二步： 设置工程名称
set(PROJECT_NAME AuraUtils)
logger_d("[${PROJECT_NAME}] ====================Start Build============================")
project(${PROJECT_NAME})

# 设置C++语言的版本号
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

# 设置install的目录前缀
if (NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)
endif ()
set(SUB_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/aura-utils")

# =================================== 编译的开关相关的配置选项 ===================================
option(BUILD_STATIC_LIB         "build the library as static"   OFF)  # 看判断编译静态库还是动态库
option(BUILD_UNIT_TEST          "build unit test"               OFF)  # 编译单元测试程序，默认关闭.暂时注释掉，外部工程设置则此地不生效
option(BUILD_EXAMPLES           "build examples"                ON)   # 编译示例代码


set(BUILD_UNIT_TEST ON)

# install
set(AURA_LIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AURA_LIB_PRIVATE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/private)


add_subdirectory(src)

# 设置附属生成物的编译控制
if (BUILD_UNIT_TEST)
    add_subdirectory(test)
endif ()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION ${SUB_INSTALL_PREFIX}/include)

# 导出这个框架工程的包含目录
set_property(GLOBAL PROPERTY AURA_UTILS_INCLUDE_DIRS ${AURA_LIB_INCLUDE_DIRS})


#logger_d("[${PROJECT_NAME}] BUILD_UNIT_TEST: ${BUILD_UNIT_TEST}")
#logger_d("[${PROJECT_NAME}] SUB_INSTALL_PREFIX: ${SUB_INSTALL_PREFIX}")
#logger_d("[${PROJECT_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
#logger_d("[${PROJECT_NAME}] AURA_LIB_PRIVATE_DIRS: ${AURA_LIB_PRIVATE_DIRS}")
logger_d("[${PROJECT_NAME}] ====================Build End============================")