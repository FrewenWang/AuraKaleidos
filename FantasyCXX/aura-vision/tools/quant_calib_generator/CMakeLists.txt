# add by @wangzhijiang 量化精度对其功能，目前只在Mac、Linux系统上进行编译打包
if (NOT ${ICP_TARGET_OS} MATCHES "osx" AND NOT ${ICP_TARGET_OS} MATCHES "linux")
    message(WARNING "ignored compiling QuantCalibGenerator on ${ICP_TARGET_OS}")
    return()
endif ()

cmake_minimum_required(VERSION 3.10)
project(QuantCalibGenerator)

link_directories(
        ${VISION_LINK_DIR}
        ${OPENCV_LINK_DIR}
        ${OPENCV_3RDPARTY_LINK_DIR})
set(UT_NAME QuantCalibGenerator)
add_executable(${UT_NAME}
        quant_calib_generator.cpp)

target_compile_options(${UT_NAME} PRIVATE -std=c++14 -O3)

#set(TARGET_OS osx)
#set(TARGET_ARCH x86_64)
#set(CMAKE_BUILD_TYPE release)

# vision
set(VISION_DIR ${CMAKE_SOURCE_DIR}/../../../build/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE})
set(VISION_INCLUDE_DIR ${VISION_DIR}/install/include)
set(VISION_LINK_DIR ${VISION_DIR}/install/lib/${ICP_TARGET_ARCH})

set(LIB_DIR ${CMAKE_SOURCE_DIR}/../../../third_party)
# opencv
#set(OPENCV_DIR ${LIB_DIR}/opencv/opencv_2.4.13.4)
#set(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/include/)
#set(OPENCV_LINK_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/lib/)
#set(OPENCV_3RDPARTY_LINK_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/3rdparty/)
#set(OPENCV_LIB opencv_calib3d opencv_highgui opencv_imgproc opencv_core libtiff libpng libjpeg libjasper IlmImf)

# protobuf
set(PROTOBUF_DIR ${LIB_DIR}/protobuf)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_DIR}/include)
set(PROTOBUF_LINK_DIR ${PROTOBUF_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE})
if (NOT EXISTS ${PROTOBUF_LINK_DIR})
    set(PROTOBUF_LINK_DIR ${PROTOBUF_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH})
endif ()
set(PROTOBUF_LIB protobuf)

target_include_directories(${UT_NAME} PUBLIC
        src
        ../../src
        ${VISION_INCLUDE_DIR}
        ${OPENCV_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIR})

if (${ICP_TARGET_OS} MATCHES "linux")
    set(PTHREAD_LIB pthread)
endif()

target_link_libraries(
        ${UT_NAME}
        PRIVATE
        ${PTHREAD_LIB}
        ${OPENCV_LIB}
        vision)

if (USE_SNPE)
    # 虽然支持编译arm平台版本的QuantCalibGenerator工具，但推荐在mac环境下编译并使用QuantCalibGenerator工具
    link_directories(
            ${SNPE_LINK_DIR})
    target_include_directories(${UT_NAME} PUBLIC
            ${SNPE_INCLUDE_DIR})
    target_link_libraries(
            ${UT_NAME}
            ${SNPE_LIB})
endif()


