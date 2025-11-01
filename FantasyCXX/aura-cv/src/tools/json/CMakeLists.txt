# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src" AURA_LIB_COMM_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/serialize/runtime" AURA_LIB_COMM_SRCS)

# update
aura_update_group_vars()