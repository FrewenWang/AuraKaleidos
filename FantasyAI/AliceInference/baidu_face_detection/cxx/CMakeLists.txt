cmake_minimum_required(VERSION 3.14)
project(AliceBaiduFaceDetection VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AURAKALEIDOS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
include("${AURAKALEIDOS_ROOT}/cmake/AuraPlatform.cmake")

set(AURA_ONNXRUNTIME_VERSION "1.16.3" CACHE STRING "Preferred ONNX Runtime version")
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Cross-platform ONNX Runtime installation root")
if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT
        "${AURAKALEIDOS_ROOT}/FantasyCXX/3rdparty/onnx/v${AURA_ONNXRUNTIME_VERSION}/${TARGET_OS}-${TARGET_ARCH}-${AURA_DEPENDENCY_VARIANT}")
endif()

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
        "ONNX Runtime was not found for ${AURA_PLATFORM_ID}. Set "
        "ONNXRUNTIME_ROOT or pass ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY.")
endif()

include("${AURAKALEIDOS_ROOT}/cmake/AuraOpenCV.cmake")

add_executable(AliceBaiduFaceDetection main.cpp Helpers.cpp)
target_include_directories(AliceBaiduFaceDetection PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(AliceBaiduFaceDetection PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIBRARY}")
aura_apply_platform(AliceBaiduFaceDetection)

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(AliceBaiduFaceDetection PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
if(AURA_BUILD_MAC)
    set_property(TARGET AliceBaiduFaceDetection PROPERTY INSTALL_RPATH "@loader_path")
elseif(AURA_BUILD_LINUX)
    set_property(TARGET AliceBaiduFaceDetection PROPERTY INSTALL_RPATH "\$ORIGIN")
elseif(AURA_BUILD_WINDOWS)
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
