# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

if (NOT AURA_BUILD_XTENSA)
    # add all module
    add_subdirectory(worker_pool)
    add_subdirectory(memory)
    add_subdirectory(utils)
    add_subdirectory(array)
    add_subdirectory(context)
endif()

if(AURA_ENABLE_OPENCL)
    add_subdirectory(opencl)
endif()

if(AURA_ENABLE_HEXAGON OR AURA_BUILD_HEXAGON)
    add_subdirectory(hexagon)
endif()

if(AURA_ENABLE_XTENSA OR AURA_BUILD_XTENSA)
    add_subdirectory(xtensa)
endif()

if(AURA_ENABLE_NN)
    add_subdirectory(nn)
endif()

# update
aura_update_group_vars()