set(MODULE_NAME AuraCV-core)
logger_d("[${MODULE_NAME}] ====================Build Start============================")

list(APPEND AURA_LIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_PRIVATE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src"     AURA_LIB_SRCS)

# test
# aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/host"    AURA_TEST_HOST_SRCS)
# aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/test/hexagon" AURA_TEST_HEXAGON_SRCS)

# update
aura_update_group_vars()


logger_d("[${MODULE_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_PRIVATE_DIRS: ${AURA_LIB_PRIVATE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_SRCS: ${AURA_LIB_SRCS}")
logger_d("[${MODULE_NAME}] ====================Build  End =============================")