# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# add subdirectory
add_subdirectory(json)
add_subdirectory(any_params)
add_subdirectory(unit_test)

# update
aura_update_group_vars()