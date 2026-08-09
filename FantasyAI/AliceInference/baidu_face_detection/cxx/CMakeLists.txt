cmake_minimum_required(VERSION 3.14)
project(AliceBaiduFaceDetection VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Cross-platform ONNX Runtime installation root")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include include/onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime was not found. Set "
        "ONNXRUNTIME_ROOT or pass ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY.")
endif()

set(AURA_OPENCV_ROOT "" CACHE PATH "Optional OpenCV installation root")
if(AURA_OPENCV_ROOT AND NOT OpenCV_DIR)
    foreach(config_suffix IN ITEMS lib/cmake/opencv4 sdk/native/jni share/opencv4)
        if(EXISTS "${AURA_OPENCV_ROOT}/${config_suffix}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${AURA_OPENCV_ROOT}/${config_suffix}")
            break()
        endif()
    endforeach()
endif()
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
endif()
if(OpenCV_DIR)
    get_filename_component(_opencv_prefix "${OpenCV_DIR}/../../.." ABSOLUTE)
endif()
if((AURA_OPENCV_ROOT AND
    (EXISTS "${AURA_OPENCV_ROOT}/lib/libopencv_world.a" OR
     EXISTS "${AURA_OPENCV_ROOT}/lib/opencv_world.lib")) OR
   (OpenCV_DIR AND
    (EXISTS "${_opencv_prefix}/lib/libopencv_world.a" OR
     EXISTS "${_opencv_prefix}/lib/opencv_world.lib")))
    find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS world)
else()
    find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
endif()

add_executable(AliceBaiduFaceDetection main.cpp Helpers.cpp)
target_include_directories(AliceBaiduFaceDetection PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(AliceBaiduFaceDetection PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIBRARY}")
if(MSVC)
    target_compile_options(AliceBaiduFaceDetection PRIVATE /W4 /utf-8 /permissive-)
else()
    target_compile_options(AliceBaiduFaceDetection PRIVATE -Wall -Wextra -Wpedantic)
endif()

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(AliceBaiduFaceDetection PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
if(APPLE)
    set_property(TARGET AliceBaiduFaceDetection PROPERTY INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_property(TARGET AliceBaiduFaceDetection PROPERTY INSTALL_RPATH "\$ORIGIN")
elseif(WIN32)
    find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
        HINTS "${ONNXRUNTIME_ROOT}" PATH_SUFFIXES bin lib)
    if(ONNXRUNTIME_DLL)
        add_custom_command(TARGET AliceBaiduFaceDetection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:AliceBaiduFaceDetection>")
    endif()
endif()

message(STATUS "[AliceBaiduFaceDetection] ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "[AliceBaiduFaceDetection] OpenCV: ${OpenCV_VERSION}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
