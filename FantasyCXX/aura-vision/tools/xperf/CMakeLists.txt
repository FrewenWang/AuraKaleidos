cmake_minimum_required(VERSION 3.10)
project(xperf)

if (NOT DEFINED ICP_TARGET_OS)
    set(ICP_TARGET_OS osx)
endif ()
if (NOT DEFINED ICP_TARGET_ARCH)
    set(ICP_TARGET_ARCH x86_64)
endif ()
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif ()

if (${ICP_TARGET_OS} MATCHES android)
    set(ICP_TARGET_ARCH ${ANDROID_ABI})
else ()
    if (NOT DEFINED ICP_TARGET_ARCH)
        set(ICP_TARGET_ARCH "x86_64")
    endif ()
endif ()

message(STATUS "ICP_TARGET_OS=${ICP_TARGET_OS}")
message(STATUS "ICP_TARGET_ARCH=${ICP_TARGET_ARCH}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# ========= dependencies ==========
# vision
set(VISION_DIR ${CMAKE_SOURCE_DIR}/../../../build/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE})
set(VISION_INCLUDE_DIR ${VISION_DIR}/install/include)
set(VISION_LINK_DIR ${VISION_DIR}/install/lib/${ICP_TARGET_ARCH})

set(LIB_DIR ${CMAKE_SOURCE_DIR}/../../../third_party)
# opencv
set(OPENCV_DIR ${LIB_DIR}/opencv/opencv_2.4.13.4)
set(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/include/)
set(OPENCV_LINK_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/lib/)
set(OPENCV_3RDPARTY_LINK_DIR ${OPENCV_DIR}/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/3rdparty/)
set(OPENCV_LIB opencv_calib3d opencv_highgui opencv_imgproc opencv_core libtiff libpng libjpeg libjasper IlmImf)

# protobuf
set(PROTOBUF_DIR ${LIB_DIR}/protobuf)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_DIR}/include)
set(PROTOBUF_LINK_DIR ${PROTOBUF_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE})
if (NOT EXISTS ${PROTOBUF_LINK_DIR})
    set(PROTOBUF_LINK_DIR ${PROTOBUF_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH})
endif ()
set(PROTOBUF_LIB protobuf)

# json
set(JSON_DIR ${LIB_DIR}/json)
set(JSON_INCLUDE_DIR ${JSON_DIR}/include/json)
set(JSON_LINK_DIR ${JSON_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE})
if (NOT EXISTS ${JSON_LINK_DIR})
    set(JSON_LINK_DIR ${JSON_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH})
endif ()
set(JSON_LIB jsoncpp)

# snpe
if (${ENABLE_SNPE})
    message(STATUS "ENABLE_SNPE=${ENABLE_SNPE}")
    SET(SNPE_DIR ${LIB_DIR}/snpe)
    set(SNPE_INCLUDE_DIR ${SNPE_DIR}/include/zdl)
    set(SNPE_LINK_DIR ${SNPE_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE}/lib)
    if (NOT EXISTS ${SNPE_LINK_DIR})
        set(SNPE_LINK_DIR ${SNPE_DIR}/lib/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/lib)
    endif ()
    if (${ICP_TARGET_OS} STREQUAL "android")
        SET(SNPE_LIB SNPE c++_shared PlatformValidatorShared)
    elseif(${ICP_TARGET_OS} STREQUAL "linux")
        SET(SNPE_LIB SNPE)
    endif()
endif ()

# qnn
if (${ENABLE_QNN})
    message(STATUS "ENABLE_QNN=${ENABLE_QNN}")
    SET(QNN_DIR ${LIB_DIR}/qnn)
    set(QNN_INCLUDE_DIR ${QNN_DIR}/include ${QNN_DIR}/include_app)
    set(QNN_LINK_DIR ${QNN_DIR}/prebuilt/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}-${CMAKE_BUILD_TYPE} ${QNN_DIR}/target/${ICP_TARGET_OS}-${ICP_TARGET_ARCH}/lib)
endif ()
# =================================

link_directories(
        ${VISION_LINK_DIR}
        ${OPENCV_LINK_DIR}
        ${OPENCV_3RDPARTY_LINK_DIR}
        ${SNPE_LINK_DIR}
        ${JSON_LINK_DIR}
        ${QNN_LINK_DIR})

add_executable(xperf
        src/xperf_main.cpp
        src/profiler/speed_profiler.cpp
        src/speed_test/detection_speed.cpp
        src/accuracy_test/detection_accuracy.cpp)

target_compile_options(xperf PUBLIC -std=c++11 -O3)

target_include_directories(xperf PUBLIC
        src
        ${VISION_INCLUDE_DIR}
        ${OPENCV_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIR}
        ${SNPE_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
        ${QNN_INCLUDE_DIR})

if (${ICP_TARGET_OS} MATCHES android)
    target_link_libraries(xperf PUBLIC vision ${OPENCV_LIB} ${SNPE_LIB} ${JSON_LIB} z log)
else ()
    target_link_libraries(xperf PUBLIC vision ${OPENCV_LIB} ${JSON_LIB})
endif ()