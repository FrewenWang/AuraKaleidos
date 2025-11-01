# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src"              AURA_LIB_COMM_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/host/opencl"  AURA_LIB_HOST_CL_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/host/xtensa"  AURA_LIB_HOST_XTENSA_SRCS)

# test
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test"             AURA_TEST_COMM_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/host/opencl" AURA_TEST_HOST_CL_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/host/xtensa" AURA_TEST_HEXAGON_SRCS)

# update
aura_update_group_vars()