set(CONFIG_TAG CmakeConfig)
message(STATUS "[${CONFIG_TAG}] ==================${CONFIG_TAG} Build==============================")

include(utils)
include("${CMAKE_CURRENT_LIST_DIR}/AuraPlatform.cmake")

if(NOT CMAKE_CONFIGURATION_TYPES AND AURA_BUILD_TYPE STREQUAL "debug")
    set(DEBUG TRUE)
endif()
add_compile_definitions($<$<CONFIG:Debug>:AURA_DEBUG=1>)

string(TIMESTAMP AURA_BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")

set(AURA_BUILD_GIT_HASH "unknown")
get_git_hash(AURA_BUILD_GIT_HASH)

set(AURA_BUILD_GIT_BRANCH "unknown")
get_git_branch(AURA_BUILD_GIT_BRANCH)

set(AURA_BUILD_HOST_NAME "unknown")
set(AURA_BUILD_HOST_OS   "unknown")
cmake_host_system_information(RESULT AURA_BUILD_HOST_NAME QUERY HOSTNAME)
cmake_host_system_information(RESULT AURA_BUILD_HOST_OS   QUERY OS_VERSION)
set(AURA_BUILD_HOST_INFO "(${AURA_BUILD_HOST_NAME}@${AURA_BUILD_HOST_OS})")

set(AURA_VERSION_DETAILS_TARGET_LEN 1024)
set(AURA_VERSION_DETAILS "${AURA_BUILD_INFO} "
    "${ARUA_BUILD_GIT_BRANCH}:${ARUA_BUILD_GIT_HASH} "
    "${AURA_BUILD_TIMESTAMP} "
    "${AURA_BUILD_HOST_INFO}")
string(LENGTH "${AURA_VERSION_DETAILS}" AURA_VERSION_DETAILS_ORIGINAL_LEN)
math(EXPR PADDING_LEN "${AURA_VERSION_DETAILS_TARGET_LEN} - ${AURA_VERSION_DETAILS_ORIGINAL_LEN}")
add_padding(PADDING ${PADDING_LEN} " ")

message(STATUS "[${CONFIG_TAG}] [environment] AURA_BUILD_TIMESTAMP=${AURA_BUILD_TIMESTAMP}")
message(STATUS "[${CONFIG_TAG}] [environment] AURA_BUILD_GIT_HASH=${AURA_BUILD_GIT_HASH}")
message(STATUS "[${CONFIG_TAG}] [environment] AURA_BUILD_GIT_BRANCH=${AURA_BUILD_GIT_BRANCH}")
message(STATUS "[${CONFIG_TAG}] [environment] AURA_BUILD_HOST_INFO=${AURA_BUILD_HOST_INFO}")
message(STATUS "[${CONFIG_TAG}] [environment] HOST_OS=${HOST_OS}")
message(STATUS "[${CONFIG_TAG}] [environment] HOST_ARCH=${HOST_ARCH}")
message(STATUS "[${CONFIG_TAG}] [environment] TARGET_OS=${TARGET_OS}")
message(STATUS "[${CONFIG_TAG}] [environment] TARGET_ARCH=${TARGET_ARCH}")
# message(STATUS "[${CONFIG_TAG}] [environment] TARGET_BUILD_TYPE=${TARGET_BUILD_TYPE}")
message(STATUS "[${CONFIG_TAG}] [environment] CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "[${CONFIG_TAG}] [environment] TOOLCHAIN=${CMAKE_TOOLCHAIN_FILE}")

message(STATUS "[${CONFIG_TAG}] ==================${CONFIG_TAG} Build End==============================")
