set(MODULE_NAME AuraCV-src)
set(TARGET_NAME aura_cv)

logger_d("[${MODULE_NAME}] ====================Build Start============================")

add_subdirectory(core)
# add_subdirectory(runtime)
add_subdirectory(ops)


aura_finalize_group_vars()

if (BUILD_STATIC_LIB)
    add_library(${TARGET_NAME} SHARED ${AURA_LIB_SRCS})
else ()
    add_library(${TARGET_NAME} STATIC ${AURA_LIB_SRCS})
endif ()

# include
target_include_directories(${TARGET_NAME} PRIVATE
                           ${AURA_UTILS_INCLUDE_DIRS}
                           ${AURA_LIB_INCLUDE_DIRS}
                           ${AURA_LIB_PRIVATE_DIRS})

install(TARGETS ${TARGET_NAME} DESTINATION ${SUB_INSTALL_PREFIX}/lib)


################################ start build lib ################################
logger_d("[${MODULE_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_PRIVATE_DIRS: ${AURA_LIB_PRIVATE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_UTILS_INCLUDE_DIRS: ${AURA_UTILS_INCLUDE_DIRS}")
logger_d("[${MODULE_NAME}] AURA_LIB_SRCS: ${AURA_LIB_SRCS}")
logger_d("[${MODULE_NAME}] ====================Build End============================")