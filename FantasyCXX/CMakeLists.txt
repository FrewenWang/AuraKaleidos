# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.14)

# 第二步： 设置工程名称
set(PROJECT_NAME AuraKaleidoCXX)
message(STATUS "[${PROJECT_NAME}] ================= Build Start==============================")
project(${PROJECT_NAME})
include(CTest)

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
option(BUILD_PERFORMANCE_TESTS "build slow/destructive performance experiments" OFF)


# =================================== 设置编译三方依赖的开关 ====================================
option(BUILD_OPENCV "Use OpenCV" ON)
option(BUILD_GTEST_LIB "open build gtest lib" ON)
option(BUILD_JSON_LIB "open build json lib" ON)
option(BUILD_CERES "open build BUILD_CERES lib" ON)
option(BUILD_EIGEN "open build eigen lib" ON)
option(BUILD_BOOST_LIB "open build boost lib" ON)
option(BUILD_OSQP "open build osqp lib" ON)
option(BUILD_OSQP_EIGEN "open build osqp eigen lib" ON)
option(FETCH_MISSING_DEPS "download a pinned dependency when the bundled copy is incomplete" ON)

# aura-cv 依赖 OpenCV（已内置 mac-x86_64 预编译库），源码量大、构建较慢，默认关闭；
# 如需构建请打开此开关（目标平台需与 bundled OpenCV 的 mac-x86_64 一致）。
option(BUILD_AURA_CV "build aura-cv (Mac-compatible, requires bundled OpenCV; large build)" OFF)

# aura-vision / aura-vision-hpc 为设备/嵌入式视觉与 HPC SDK：
# 依赖 Android / NCNN / SNPE / QNN / 设备 OpenCL 等 SDK，无法在 macOS x86_64 上编译。
# 仅在对应设备工具链下通过下面两个开关开启。
option(BUILD_AURA_VISION "build aura-vision (device/embedded vision SDK; needs Android/NCNN/SNPE/QNN toolchains)" OFF)
option(BUILD_AURA_VISION_HPC "build aura-vision-hpc (OpenCL/NEON HPC examples; needs device OpenCL SDK)" OFF)


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
add_subdirectory(aura-utils)
if(BUILD_AURA_CV)
    add_subdirectory(aura-cv)   # Mac 可编译（依赖内置 OpenCV）；源码量大，默认关闭
endif()
add_subdirectory(aura-auto-driving)      # 依赖 Eigen（内置）+ aura-utils + aura-config-parser
add_subdirectory(aura-config-parser)    # 头文件库 INI 解析
add_subdirectory(aura-lightbuffer)      # LightBuffer 序列化库
add_subdirectory(samples)
if(BUILD_UNIT_TEST)
    add_subdirectory(test)              # 依赖 GTest（内置）+ aura-utils
endif()
add_subdirectory(aura-perfguard)
# aura-vision / aura-vision-hpc 为设备/嵌入式 SDK，macOS x86_64 无法编译，默认关闭（见上方 option）
if(BUILD_AURA_VISION)
    add_subdirectory(aura-vision)
endif()
if(BUILD_AURA_VISION_HPC)
    add_subdirectory(aura-vision-hpc)
endif()


message(STATUS "[${PROJECT_NAME}] ================= Build Start==============================")
