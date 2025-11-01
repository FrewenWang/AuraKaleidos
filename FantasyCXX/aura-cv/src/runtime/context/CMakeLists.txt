# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/host"     AURA_LIB_HOST_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/hexagon"  AURA_LIB_HEXAGON_SRCS)

# test
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/host"    AURA_TEST_HOST_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/hexagon" AURA_TEST_HEXAGON_SRCS)

# update
aura_update_group_vars()