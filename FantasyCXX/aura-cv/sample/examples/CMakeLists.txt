cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

project(sample)

if(AuraCV_DIR)
    find_package(AuraCV REQUIRED)
    if(NOT AuraCV_FOUND)
        message(FATAL_ERROR "failed to find AuraCV")
    endif()
else()
    message(FATAL_ERROR "AuraCV path is empty")
endif()

include_directories(include)
add_subdirectory(ops)
add_subdirectory(runtime)