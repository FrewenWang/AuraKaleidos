cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

project(sample)

if(Aura_DIR)
    find_package(Aura REQUIRED)
    if(NOT Aura_FOUND)
        message(FATAL_ERROR "failed to find aura")
    endif()
else()
    message(FATAL_ERROR "aura path is empty")
endif()

include_directories(include)
add_subdirectory(ops)
add_subdirectory(runtime)