# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src"          AURA_LIB_COMM_SRCS)
if(AURA_BUILD_ANDROID)
    aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/host/android" AURA_LIB_HOST_SRCS)
endif()
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/hexagon"  AURA_LIB_HEXAGON_SRCS)

# test
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test"         AURA_TEST_COMM_SRCS)

# update
aura_update_group_vars()