cmake_minimum_required(VERSION 3.14)
project(OpenCVDemo LANGUAGES CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AURA_OPENCV_ROOT "" CACHE PATH "Optional OpenCV installation root")
if(AURA_OPENCV_ROOT AND NOT OpenCV_DIR)
    foreach(config_suffix IN ITEMS lib/cmake/opencv4 sdk/native/jni share/opencv4)
        if(EXISTS "${AURA_OPENCV_ROOT}/${config_suffix}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${AURA_OPENCV_ROOT}/${config_suffix}")
            break()
        endif()
    endforeach()
endif()

# Some static OpenCV packages reference this target without exporting Eigen.
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

function(alice_apply_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /utf-8 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(alice_opencv src/utils/image_util.cpp)
target_include_directories(alice_opencv PUBLIC src)
target_link_libraries(alice_opencv PUBLIC ${OpenCV_LIBS})
alice_apply_warnings(alice_opencv)

add_executable(OpenCVDemo src/main.cpp)
target_link_libraries(OpenCVDemo PRIVATE alice_opencv)
alice_apply_warnings(OpenCVDemo)

option(BUILD_OPENCV_EXAMPLES "Build standalone OpenCV learning examples" OFF)
if(BUILD_OPENCV_EXAMPLES)
    foreach(example_name IN ITEMS channel_exchange image_gray otsu_threshold binary_threshold)
        add_executable(opencv_${example_name} "examples/${example_name}.cpp")
        target_link_libraries(opencv_${example_name} PRIVATE ${OpenCV_LIBS})
        alice_apply_warnings(opencv_${example_name})
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(OpenCVDemoTests tests/image_util_test.cpp)
    target_link_libraries(OpenCVDemoTests PRIVATE alice_opencv)
    alice_apply_warnings(OpenCVDemoTests)
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME OpenCVDemoTests COMMAND OpenCVDemoTests)
        add_test(
            NAME OpenCVDemoCli
            COMMAND OpenCVDemo
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/images/image1.jpg"
                "${CMAKE_CURRENT_BINARY_DIR}/average_pooling.jpg")
    endif()
endif()
