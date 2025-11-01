# build lib
if(AURA_SHARED_LIBRARY)
    add_library(aura_opencl_library STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cl_library.cpp
                                           ${CMAKE_CURRENT_SOURCE_DIR}/../utils/src/logger.cpp)

    target_include_directories(aura_opencl_library PRIVATE ${AURA_CONFIG_INC_DIRS}
                                                           ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/private
                                                           ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include ${CMAKE_CURRENT_SOURCE_DIR}/../utils/private)

    target_link_libraries(aura_opencl_library log)

    add_dependencies(aura_opencl_library OpenCL-Headers OpenCL-Headers-Hpp)

    strip_lib(aura_opencl_library OFF ON)

    install(TARGETS aura_opencl_library DESTINATION lib)
endif()

# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src"  AURA_LIB_HOST_CL_SRCS)

# test
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test" AURA_TEST_HOST_CL_SRCS)

# update
aura_update_group_vars()