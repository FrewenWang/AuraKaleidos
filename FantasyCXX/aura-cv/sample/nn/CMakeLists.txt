cmake_minimum_required(VERSION 3.2)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Android")
    message(FATAL_ERROR "the target system must be Android")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

project(aura_nn)

if(Aura_DIR)
    find_package(Aura REQUIRED)
    if(NOT Aura_FOUND)
        message(FATAL_ERROR "failed to find aura")
    endif()
else()
    message(FATAL_ERROR "aura path is empty")
endif()

add_executable(sample_nn ./main.cpp)

target_link_libraries(sample_nn ${Aura_LIBS})

install(TARGETS sample_nn DESTINATION nn)
