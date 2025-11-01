# idl
execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hexagon/hexagon_idl_gen.py
    ${HEXAGON_SDK_PATH} ${HEXAGON_QAIC_PATH} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} aura_hexagon.idl
    RESULT_VARIABLE ERR_VAR
)
if(${ERR_VAR})
    message(FATAL_ERROR "hexagon idl generate failed")
endif()

# inc
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
list(APPEND AURA_LIB_INC_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
list(APPEND AURA_TEST_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/test")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/host"         AURA_LIB_HOST_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src/device"       AURA_LIB_HEXAGON_SRCS)

list(APPEND AURA_LIB_HOST_SRCS "${CMAKE_CURRENT_BINARY_DIR}/aura_hexagon_stub.c")
list(APPEND AURA_LIB_HEXAGON_SRCS "${CMAKE_CURRENT_BINARY_DIR}/aura_hexagon_skel.c")
list(APPEND AURA_LIB_HEXAGON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/device/vmemcpy_asm.S")

# test
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/host"        AURA_TEST_HOST_SRCS)
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/device/src"  AURA_TEST_HEXAGON_SRCS)

# update
aura_update_group_vars()