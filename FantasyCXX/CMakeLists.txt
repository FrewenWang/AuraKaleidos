# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.10)

# 第二步： 设置工程名称
set(PROJECT_NAME AuraKaleidoCXX)
message(STATUS "[${PROJECT_NAME}] ================= Build Start==============================")
project(${PROJECT_NAME})

# 第三步：设置CMake设置的C++的版本
set(CMAKE_CXX_STANDARD 17)


# install prefix
# CMake 第一次运行时，如果你没指定 CMAKE_INSTALL_PREFIX，它会写入默认值 /usr/local 到 CMakeCache.txt。
# 之后你在 CMakeLists.txt 里写的这句：由于 CMAKE_INSTALL_PREFIX 已经在缓存里有值了，CMake 就不会改了。
# 加 FORCE（推荐）
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)
endif()


# =================================== 编译的系统相关的变量配置选项 =================================

# =================================== 编译系统相关的变量配置选项 =================================
option(ENABLE_LOG "Enable CMake log output" ON)
set(LOG_LEVEL "DEBUG" CACHE STRING "Log level (DEBUG, INFO, WARN, ERROR, NONE)")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(config_init)
include(logger)

# =================================== 编译的开关相关的配置选项 ===================================
option(BUILD_STATIC_LIB "build the library as static" OFF)      # 看判断编译静态库还是动态库
option(BUILD_EXAMPLES "build examples" ON)       # 编译示例代码
option(BUILD_TOOLS "build tools" ON)       # 编译Tools
option(BUILD_UNIT_TEST "build unit test " ON)       # 编译单测模块


# =================================== 设置编译三方依赖的开关 ====================================
option(BUILD_OPENCV "Use OpenCV" ON)
option(BUILD_GTEST_LIB "open build gtest lib" ON)
option(BUILD_JSON_LIB "open build json lib" ON)
option(BUILD_CERES "open build BUILD_CERES lib" ON)
option(BUILD_EIGEN "open build eigen lib" ON)
option(BUILD_BOOST_LIB "open build boost lib" ON)
option(BUILD_OSQP "open build osqp lib" ON)
option(BUILD_OSQP_EIGEN "open build osqp eigen lib" ON)


# ----------------------------------------------------------------------------------------------------------------------
#                                               设置编译工程的配置参数
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# 执行各模块 cmake
# ----------------------------------------------------------------------------------------------------------------------
# 添加CMAKE的通用工具
include(cmake/macro.cmake)
# 添加CMAKE的第三方依赖库
include(3rdparty/dependence.cmake)

# 添加项目子工程目录的cmake文件
# add_subdirectory(aura-utils)
# add_subdirectory(aura-cv)
#add_subdirectory(aura-auto-driving)
#add_subdirectory(examples)
#add_subdirectory(test)
#add_subdirectory(aura-lightbuffer)
#add_subdirectory(aura-perfguard)
#add_subdirectory(aura-vision)
#add_subdirectory(aura-vision-hpc)
add_subdirectory(aura-cv)


message(STATUS "[${PROJECT_NAME}] ================= Build Start==============================")