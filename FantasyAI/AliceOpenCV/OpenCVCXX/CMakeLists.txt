cmake_minimum_required(VERSION 3.14)
project(OpenCVDemo LANGUAGES CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AURAKALEIDOS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include("${AURAKALEIDOS_ROOT}/cmake/AuraPlatform.cmake")
include("${AURAKALEIDOS_ROOT}/cmake/AuraOpenCV.cmake")

add_library(alice_opencv src/utils/image_util.cpp)
target_include_directories(alice_opencv PUBLIC src)
target_link_libraries(alice_opencv PUBLIC ${OpenCV_LIBS})
aura_apply_platform(alice_opencv)

add_executable(OpenCVDemo src/main.cpp)
target_link_libraries(OpenCVDemo PRIVATE alice_opencv)
aura_apply_platform(OpenCVDemo)

option(BUILD_OPENCV_EXAMPLES "Build standalone OpenCV learning examples" OFF)
if(BUILD_OPENCV_EXAMPLES)
    foreach(example_name IN ITEMS channel_exchange image_gray otsu_threshold binary_threshold)
        add_executable(opencv_${example_name} "examples/${example_name}.cpp")
        target_link_libraries(opencv_${example_name} PRIVATE ${OpenCV_LIBS})
        aura_apply_platform(opencv_${example_name})
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(OpenCVDemoTests tests/image_util_test.cpp)
    target_link_libraries(OpenCVDemoTests PRIVATE alice_opencv)
    aura_apply_platform(OpenCVDemoTests)
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME OpenCVDemoTests COMMAND OpenCVDemoTests)
        add_test(
            NAME OpenCVDemoCli
            COMMAND OpenCVDemo
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/images/image1.jpg"
                "${CMAKE_CURRENT_BINARY_DIR}/average_pooling.jpg")
    endif()
endif()
