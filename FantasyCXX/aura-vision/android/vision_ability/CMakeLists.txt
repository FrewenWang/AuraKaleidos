cmake_minimum_required(VERSION 3.5)

set(CMAKE_VERBOSE_MAKEFILE on)
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Vision native lib stuff
set(VISION_NATIVE_DIR ${PROJECT_SOURCE_DIR}/../../..)
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(VISION_NATIVE_INSTALL_DIR ${VISION_NATIVE_DIR}/build/android-${ANDROID_ABI}-debug/install)
elseif(${CMAKE_BUILD_TYPE} MATCHES Release)
    set(VISION_NATIVE_INSTALL_DIR ${VISION_NATIVE_DIR}/build/android-${ANDROID_ABI}-release/install)
endif()
set(VISION_NATIVE_INCLUDE_DIR ${VISION_NATIVE_INSTALL_DIR}/include)
set(VISION_NATIVE_LIBRARY_DIR ${VISION_NATIVE_INSTALL_DIR}/lib/)

message(STATUS "VISION_NATIVE_INCLUDE_DIR=${VISION_NATIVE_INCLUDE_DIR}")
message(STATUS "VISION_NATIVE_LIBRARY_DIR=${VISION_NATIVE_LIBRARY_DIR}")

# copy libvision.so
#message(STATUS "Copy ${vision_native_file} to  ${PROJECT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}")
#file(COPY ${vision_native_file} DESTINATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
# if use external model file, copy vision_model.bin
#include(${VISION_NATIVE_DIR}/cmake/generate_model.cmake)
#if (${USE_EXTERNAL_MODEL} MATCHES "true")
#    file(COPY ${VISION_NATIVE_DIR}/models/generated/vision_model.bin DESTINATION ${CMAKE_SOURCE_DIR}/src/main/assets/model)
#endif()

include_directories(${VISION_NATIVE_INCLUDE_DIR})
set(vision_native_file ${VISION_NATIVE_LIBRARY_DIR}/libvision.so)
add_library(vision_native_lib SHARED IMPORTED)
set_target_properties(vision_native_lib PROPERTIES IMPORTED_LOCATION ${vision_native_file})

file(GLOB_RECURSE native_srcs "${CMAKE_SOURCE_DIR}/src/main/cpp/jni/*.cpp")
file(GLOB_RECURSE native_headers "${CMAKE_SOURCE_DIR}/src/main/cpp/jni/*.h")

# Vision JNI lib
add_library(vision_jni SHARED ${native_srcs})
target_link_libraries(vision_jni vision_native_lib android log)



