set(MODULE_NAME AuraCV-ops)
logger_d("[${MODULE_NAME}] ====================Build Start============================")

# 1. add include dir to aura_lib_include_dirs variable
list(APPEND AURA_LIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

# 2. install include files
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION ${SUB_INSTALL_PREFIX}/include)

# 3. add modules
aura_add_module(cvtcolor)
#aura_add_module(resize)

# 4. update all the vars
aura_update_group_vars()

################################ start build lib ################################
logger_d("[${MODULE_NAME}] CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
logger_d("[${MODULE_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_PRIVATE_DIRS: ${AURA_LIB_PRIVATE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_SRCS: ${AURA_LIB_SRCS}")

logger_d("[${MODULE_NAME}] ====================Build  End =============================")