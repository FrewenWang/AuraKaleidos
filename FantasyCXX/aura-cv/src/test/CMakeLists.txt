if(AURA_BUILD_UNIT_TEST)
    # aura test main inc
    list(APPEND AURA_TEST_INC_DIRS         "${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/include")
    list(APPEND AURA_TEST_HOST_INC_DIRS    "${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/include/host")
    list(APPEND AURA_TEST_HEXAGON_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/include/hexagon")
    if(AURA_BUILD_HOST)
        list(APPEND AURA_TEST_INC_DIRS ${AURA_TEST_HOST_INC_DIRS})
    elseif(AURA_BUILD_HEXAGON)
        list(APPEND AURA_TEST_INC_DIRS ${AURA_TEST_HEXAGON_INC_DIRS})
    endif()

    # aura test main src
    if(AURA_BUILD_XPLORER)
        aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/src/xtensa" AURA_TEST_HOST_SRCS)
    else()
        aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/src/host"   AURA_TEST_HOST_SRCS)
    endif()
    aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/aura_test_main/src/hexagon"    AURA_TEST_HEXAGON_SRCS)
endif()

if(AURA_ENABLE_NN)
    # aura nn run inc
    list(APPEND AURA_NN_RUN_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/aura_nn_run/include")

    # aura nn run srcs
    aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/aura_nn_run/src" AURA_NN_RUN_HOST_SRCS)
endif()

# update
aura_update_group_vars()
