# min cmake version
cmake_minimum_required(VERSION 3.13)

set(CMAKE_SYSTEM_NAME "Mac")

# install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

option(AURA_ENABLE_ARM82    "Enable ARMv8.2 support"                         ON)
option(AURA_ENABLE_ASAN     "Enable ASAN, if hwasan, ignore asan"            OFF)
option(AURA_ENABLE_HWASAN   "Enable HWASAN, if hwasan, ignore asan"          OFF) # hwasan requires ndk version 21 or newer
option(AURA_ENABLE_HEXAGON  "Enable Hexagon DSP support"                     OFF)
option(AURA_ENABLE_OPENCL   "Enable OpenCL support"                          OFF)
option(AURA_ENABLE_NN       "Enable NN, if nnlite, nn enabled"               OFF)
option(AURA_ENABLE_NN_LITE  "Enable NN lite, if nnlite, nn enabled"          OFF)
option(AURA_SHARED_LIBRARY  "Build shared library instead of static library" OFF)
option(AURA_BUILD_UNIT_TEST "Build test program"                             OFF)
option(AURA_RELEASE         "Enable release mode"                            OFF)

if(NOT DEFINED AURA_BUILD_TYPE OR AURA_BUILD_TYPE MATCHES "(RELEASE|Release|release)")
    set(AURA_RELEASE ON)
    set(AURA_BUILD_TYPE "release")
else()
    set(AURA_BUILD_TYPE "debug")
endif()

message(STATUS "AURA_BUILD_TYPE = ${AURA_BUILD_TYPE}")

if(AURA_ENABLE_NN_LITE)
    set(AURA_ENABLE_HEXAGON     OFF)
    set(AURA_ENABLE_NN          ON)
    set(AURA_ENABLE_OPENCL      OFF)
    set(AURA_BUILD_UNIT_TEST    OFF)
endif()

# set aura version
if(NOT AURA_VERSION)
    set(AURA_VERSION 1.0.0)
endif()

# project
project(aura)

# set c++ standard
# CMake如何指定C++11标准的基础教程
set(CMAKE_CXX_STANDARD 11)
set(AURA_CXX_STANDARD_11 ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set lib type
if(AURA_SHARED_LIBRARY)
    set(AURA_LIB_TYPE "share")
    set(AURA_API_EXPORTS ON)
else()
    set(AURA_LIB_TYPE "static")
endif()

# set build type
set(CMAKE_BUILD_TYPE "release")

# set lib name
set(AURA_LIB_NAME "aura_cv")

# target
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(AURA_BUILD_WIN ON)
    set(AURA_BUILD_HOST ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if((NOT AURA_BUILD_HEXAGON) AND (NOT AURA_BUILD_XTENSA))
        # 嵌入式Linux
        if(${CMAKE_SYSTEM_NAME} STREQUAL "Embedded-Linux")
            set(AURA_BUILD_EMBEDDED_LINUX ON)
        elseif(AURA_BUILD_XPLORER)
            # 文章参考：https://blog.csdn.net/wangyx1234/article/details/113728110
            # 文章参考：https://blog.csdn.net/whatday/article/details/87268727
            set(AURA_ENABLE_XTENSA ON)
        else()
            set(AURA_BUILD_LINUX ON)
        endif()
        set(AURA_BUILD_HOST ON)
    endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(AURA_BUILD_ANDROID ON)
    set(AURA_BUILD_HOST ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Mac")
    set(AURA_BUILD_MAC ON)
    set(AURA_BUILD_HOST ON)
else()
    message(FATAL_ERROR "unsupported platform")
endif()

# compile & link options
set(AURA_COMPILE_OPTIONS "")
set(AURA_COMPILE_ARCH    "")
set(AURA_LINK_OPTIONS    "")

include(cmake/macros.cmake)

# find python
if("${AURA_ENABLE_OPENCL}" OR "${AURA_ENABLE_HEXAGON}" OR "${AURA_BUILD_HEXAGON}" OR "${AURA_ENABLE_NN}" OR "${AURA_BUILD_XTENSA}")
    # Find Python 3 interpreter
    find_package(Python3 REQUIRED Interpreter)
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "No Python 3 installation found!")
    endif()
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

if(AURA_BUILD_WIN)
    set(AURA_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})

    # set compiler
    find_win_compiler(AURA_COMPILER)

    # update build type
    if(NOT AURA_RELEASE)
        set(CMAKE_BUILD_TYPE "debug")
    endif()

    if(MSVC_VERSION GREATER_EQUAL 1915 AND MSVC_VERSION LESS 1925) # vc15.8 ~ vc16.4
        list(APPEND AURA_COMPILE_OPTIONS "/experimental:preprocessor")
    elseif(MSVC_VERSION GREATER_EQUAL 1925) # vc16.5 or higher
        list(APPEND AURA_COMPILE_OPTIONS "/Zc:preprocessor")
    endif()

    # set compile arch
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(AURA_COMPILE_ARCH "x64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(AURA_COMPILE_ARCH "x86")
    endif()

    # Windows compile options
    list(APPEND AURA_COMPILE_OPTIONS "-DNOMINMAX -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE")

    # check asan
    if(AURA_ENABLE_ASAN OR AURA_ENABLE_HWASAN)
        message(FATAL_ERROR "platform does not support Asan")
    endif()

    # check nnlite
    if(AURA_ENABLE_NN_LITE)
        message(FATAL_ERROR "nnlite only support Android")
    endif()
elseif(AURA_BUILD_LINUX)
    # update build info
    find_ubuntu_release_version(AURA_UBUNTU_RELEASE_VERSION)
    find_glibc_version(AURA_GLIBC_VERSION)

    set(AURA_SYSTEM_NAME "ubuntu${AURA_UBUNTU_RELEASE_VERSION}-glibc${AURA_GLIBC_VERSION}")

    # Linux compile options
    list(APPEND AURA_COMPILE_OPTIONS "-Wall -Wextra -Werror -Ofast -ffast-math -fvisibility=hidden -fvisibility-inlines-hidden -fPIC")

    # x86 or x64
    if(${AURA_LINUX_ABI} MATCHES "(x86|X86)")
        set(AURA_COMPILE_ARCH            "x86")
        list(APPEND AURA_COMPILE_OPTIONS "-m32")
        list(APPEND AURA_LINK_OPTIONS    "-m32")
    else()
        set(AURA_COMPILE_ARCH            "x64")
        list(APPEND AURA_COMPILE_OPTIONS "-m64")
    endif()

    if(${AURA_LINUX_COMPILER} MATCHES "(clang|Clang|CLANG)")
        find_program(CLANGPP_FOUND clang++)
        if(CLANGPP_FOUND)
            set(CMAKE_CXX_COMPILER ${CLANGPP_FOUND})
        else()
            message(FATAL_ERROR "Clang++ compiler not found")
        endif()

        list(APPEND AURA_COMPILE_OPTIONS "-stdlib=libc++")
        list(APPEND AURA_LINK_OPTIONS    "c++")
    endif()

    # check asan
    if(AURA_ENABLE_ASAN OR AURA_ENABLE_HWASAN)
        message(FATAL_ERROR "platform does not support Asan")
    endif()

elseif(AURA_BUILD_ANDROID)
    set(AURA_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")

    # set compiler
    find_android_ndk_version(AURA_ANDROID_NDK_VERSION)
    set(AURA_COMPILER "ndkr${AURA_ANDROID_NDK_VERSION}")

    # set compile arch
    set(AURA_COMPILE_ARCH ${ANDROID_ABI})

    # Android compile options
    list(APPEND AURA_COMPILE_OPTIONS "-Wall -Wextra -Werror -Ofast -ffast-math -fvisibility=hidden -fvisibility-inlines-hidden -fPIC")

    # Armv82 compile options
    if(${AURA_ENABLE_ARM82})
        set(AURA_ARM82 ON)
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-march=armv8.2a+fp16)
        else()
            message(FATAL_ERROR "do not support arm32 with AURA_ARM82 on")
        endif()
    endif()

    # opencl
    if(${AURA_ENABLE_OPENCL})
        if((<${AURA_OPENCL_VERSION}|string> VERSION_LESS_EQUAL <300|string>)
                AND (<${AURA_OPENCL_VERSION}|string> VERSION_GREATER_EQUAL <100|string>))
            set(CL_TARGET_OPENCL_VERSION     ${AURA_OPENCL_VERSION})
            set(CL_HPP_TARGET_OPENCL_VERSION ${AURA_OPENCL_VERSION})
        else()
            message(FATAL_ERROR "invalid opencl version ${AURA_OPENCL_VERSION}")
        endif()

        set(OPENCL_CODE_GEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/opencl/opencl_code_gen.py")
    endif()

    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    # check asan
    if((AURA_ENABLE_ASAN OR AURA_ENABLE_HWASAN) AND AURA_RELEASE)
        message(FATAL_ERROR "asan/hwsan do not support release mode")
    endif()
elseif(AURA_BUILD_EMBEDDED_LINUX)
    # Embedded-Linux compile options
    list(APPEND AURA_COMPILE_OPTIONS "-Wall -Wextra -Werror -Ofast -ffast-math -fvisibility=hidden -fvisibility-inlines-hidden -fPIC")

    # TODO
    #set(AURA_COMPILE_ARCH "")

    # check asan
    if(AURA_ENABLE_ASAN OR AURA_ENABLE_HWASAN)
        message(FATAL_ERROR "platform does not support Asan")
    endif()

    # check nnlite
    if(AURA_ENABLE_NN_LITE)
        message(FATAL_ERROR "nnlite only support Android")
    endif()
elseif(AURA_BUILD_HEXAGON)
    set(AURA_SYSTEM_NAME "hexagon")

    # update aura lib name
    set(AURA_LIB_NAME "aura_hexagon_skel")

    if(NOT DEFINED AURA_HEXAGON_ARCH)
        message(FATAL_ERROR "AURA_HEXAGON_ARCH is not set")
    endif()
    set(AURA_COMPILE_ARCH ${AURA_HEXAGON_ARCH})

    # enable ASM
    enable_language(ASM)

    # HEXAGON compile options
    list(APPEND AURA_COMPILE_OPTIONS "-Wall -Wextra -Werror -fvisibility=hidden -fvisibility-inlines-hidden -fPIC")

    # check asan
    if(AURA_ENABLE_ASAN OR AURA_ENABLE_HWASAN)
        message(FATAL_ERROR "platform does not support Asan")
    endif()

    # check nnlite
    if(AURA_ENABLE_NN_LITE)
        message(FATAL_ERROR "nnlite only support Android")
    endif()

    # check shared build
    if(AURA_SHARED_LIBRARY AND AURA_BUILD_UNIT_TEST)
        message(FATAL_ERROR "when AURA_BUILD_UNIT_TEST=ON, AURA_LIB_TYPE must be static lib.")
    endif()
elseif(AURA_BUILD_XPLORER OR AURA_BUILD_XTENSA)
    # check shared build
    if(AURA_SHARED_LIBRARY)
        message(FATAL_ERROR "Xtensa only support static lib")
    endif()

    if(AURA_BUILD_UNIT_TEST)
        if(NOT AURA_XTENSA_LIB_DIR)
            message(FATAL_ERROR "AURA_XTENSA_LIB_DIR is not set")
        endif()

        set(AURA_XTENSA_LIB_NAME aura_xtensa)
        link_directories(${AURA_XTENSA_LIB_DIR})
    endif()

    # set build info
    set(AURA_BUILD_INFO "Xtensa")

    # update aura lib name
    if(AURA_BUILD_XPLORER)
        set(AURA_SYSTEM_NAME "xtensa-xplorer")
    elseif(AURA_BUILD_XTENSA)
        set(AURA_SYSTEM_NAME "xtensa")
        set(AURA_LIB_NAME "aura_xtensa")
    endif()

    # set xtensa compiler
    find_program(CLANGPP_FOUND xt-clang++)
    if(CLANGPP_FOUND)
        set(CMAKE_CXX_COMPILER ${CLANGPP_FOUND})
    else()
        message(FATAL_ERROR "Clang++ compiler not found")
    endif()

    # compile options
    list(APPEND AURA_COMPILE_OPTIONS "-Wall -Wextra -Werror -Ofast -ffast-math -fvisibility=hidden -fvisibility-inlines-hidden -fPIC")

    list(APPEND AURA_COMPILE_OPTIONS "-m32")
    list(APPEND AURA_LINK_OPTIONS    "-m32")

    list(APPEND AURA_COMPILE_OPTIONS "-stdlib=libc++")
    list(APPEND AURA_COMPILE_OPTIONS "-std=c++11")

    set(XTENSA_CODE_GEN_PY ${CMAKE_SOURCE_DIR}/scripts/xtensa/xtensa_rpc_gen.py)
endif()

if(AURA_ENABLE_NN_LITE)
    set(AURA_FEATURE_NAME "nnlite")
endif()

# enable asan
if(AURA_ENABLE_HWASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=hwaddress -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=hwaddress -fno-omit-frame-pointer")
    set(AURA_ASAN_TYPE "hwasan")
elseif(AURA_ENABLE_ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(AURA_ASAN_TYPE "asan")
endif()

if(AURA_BUILD_ANDROID OR AURA_BUILD_EMBEDDED_LINUX OR AURA_BUILD_HEXAGON)
    list(APPEND AURA_COMPILE_OPTIONS "-gline-tables-only")
elseif(AURA_BUILD_LINUX)
    list(APPEND AURA_COMPILE_OPTIONS "-g")
endif()

# 3rdparty
include(3rdparty/3rdparty.cmake)

# add compile options
if(AURA_COMPILE_OPTIONS)
    string(REPLACE " " ";" AURA_COMPILE_OPTIONS "${AURA_COMPILE_OPTIONS}")
    add_compile_options(${AURA_COMPILE_OPTIONS})
endif(AURA_COMPILE_OPTIONS)

# add link options
if(AURA_LINK_OPTIONS)
    string(REPLACE " " ";" AURA_LINK_OPTIONS "${AURA_LINK_OPTIONS}")
    add_link_options(${AURA_LINK_OPTIONS})
endif(AURA_LINK_OPTIONS)

# set install prefix
list(APPEND AURA_BUILD_LIST "aura"
                            "${AURA_VERSION}"
                            "${AURA_FEATURE_NAME}"
                            "${AURA_SYSTEM_NAME}"
                            "${AURA_COMPILE_ARCH}"
                            "${AURA_COMPILER}"
                            "${AURA_LIB_TYPE}"
                            "${AURA_BUILD_TYPE}"
                            "${AURA_ASAN_TYPE}")

join_string_list(AURA_BUILD_INFO "_" ${AURA_BUILD_LIST})

set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${AURA_BUILD_INFO}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# configure
include(configure/configure.cmake)

add_subdirectory(src)

if(AURA_BUILD_HOST)
    add_subdirectory(sample)
endif()