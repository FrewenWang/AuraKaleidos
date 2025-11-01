set(TOOLCHAIN AX620A)
set(CMAKE_SYSTEM_NAME AX620A)
set(CMAKE_SYSTEM_VERSION 7.5.0)
set(CMAKE_SYSTEM_PROCESSOR arm7)
set(TARGET_OS AX620A)
set(TARGET_ARCH arm7)
set(TARGET_COMPILER GCC)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(CMAKE_FIND_ROOT_PATH "/home/baiduiov/tools/axsdk_adas/tools/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin")
set(CMAKE_C_COMPILER "${CMAKE_FIND_ROOT_PATH}/arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${CMAKE_FIND_ROOT_PATH}/arm-linux-gnueabihf-g++")

