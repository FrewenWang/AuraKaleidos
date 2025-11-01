set(MODULE_NAME AuraUtils-runtime)
logger_d("[${MODULE_NAME}] ====================Build Start============================")
# install include
# 1. add include dir to aura_lib_include_dirs variable
list(APPEND AURA_LIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND AURA_LIB_PRIVATE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/private")
# 2. install include files
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura/" DESTINATION include)

# lib
aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}/src" AURA_LIB_COMMON_SRCS)

# 2. update all the vars
aura_update_group_vars()

logger_d("[${MODULE_NAME}] ====================Build  End =============================")