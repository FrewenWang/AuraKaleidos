set(MODULE_NAME AuraUtils-src)
set(TARGET_NAME aura_utils)
logger_d("[${MODULE_NAME}] ====================Build Start============================")

aura_add_module(core)
aura_add_module(concurrent)
#aura_add_module(handler)
aura_add_module(logger)
aura_add_module(runtime)
#aura_add_module(safe_stl)
#aura_add_module(schedule)
#aura_add_module(utils)

aura_finalize_group_vars()

if (BUILD_STATIC_LIB)
    add_library(${TARGET_NAME} SHARED ${AURA_LIB_SRCS})
else ()
    add_library(${TARGET_NAME} STATIC ${AURA_LIB_SRCS})
endif ()

# include
target_include_directories(${TARGET_NAME} PRIVATE ${AURA_LIB_INCLUDE_DIRS} ${AURA_LIB_PRIVATE_DIRS})

install(TARGETS ${TARGET_NAME} DESTINATION ${SUB_INSTALL_PREFIX}/lib)

################################ start build lib ################################
#logger_d("[${MODULE_NAME}] SUB_INSTALL_PREFIX: ${SUB_INSTALL_PREFIX}")
#logger_d("[${MODULE_NAME}] AURA_LIB_INCLUDE_DIRS: ${AURA_LIB_INCLUDE_DIRS}")
# logger_d("[${MODULE_NAME}] AURA_LIB_PRIVATE_DIRS: ${AURA_LIB_PRIVATE_DIRS}")
#logger_d("[${MODULE_NAME}] AURA_LIB_SRCS: ${AURA_LIB_SRCS}")
logger_d("[${MODULE_NAME}] ====================Build End============================")