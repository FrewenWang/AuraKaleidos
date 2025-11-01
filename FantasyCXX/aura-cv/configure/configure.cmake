# get git hash
macro(get_git_hash git_hash)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
            OUTPUT_VARIABLE ${git_hash}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif(GIT_FOUND)
endmacro(get_git_hash)

# get git branch
macro(get_git_branch git_branch)
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} symbolic-ref --short -q HEAD
            OUTPUT_VARIABLE ${git_branch}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif(GIT_FOUND)
endmacro(get_git_branch)

# add padding
macro(add_padding str len character)
    set(output "")
    math(EXPR loop_count "${len} - 1")
    foreach(loop_idx RANGE ${loop_count})
        set(output "${output}${character}")
    endforeach()
    set(${str} "${output}")
endmacro(add_padding)

###############################################################################
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
                         "${AURA_BUILD_GIT_BRANCH}:${AURA_BUILD_GIT_HASH} "
                         "${AURA_BUILD_TIMESTAMP} "
                         "${AURA_BUILD_HOST_INFO}")
string(LENGTH "${AURA_VERSION_DETAILS}" AURA_VERSION_DETAILS_ORIGINAL_LEN)
math(EXPR PADDING_LEN "${AURA_VERSION_DETAILS_TARGET_LEN} - ${AURA_VERSION_DETAILS_ORIGINAL_LEN}")
add_padding(PADDING ${PADDING_LEN} " ")

message(STATUS "Build Version        = ${AURA_VERSION}")
message(STATUS "Build Timestamp      = ${AURA_BUILD_TIMESTAMP}")
message(STATUS "Build Git hash       = ${AURA_BUILD_GIT_HASH}")
message(STATUS "Build Git branch     = ${AURA_BUILD_GIT_BRANCH}")
message(STATUS "Build Host Info      = ${AURA_BUILD_HOST_INFO}")

###############################################################################

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure/headers/version.h.in
    ${CMAKE_BINARY_DIR}/configure_gen/include/aura/version.h
    @ONLY
)

###############################################################################

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure/headers/config.h.in
    ${CMAKE_BINARY_DIR}/configure_gen/include/aura/config.h
    @ONLY
)

###############################################################################

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure/cmake/AuraConfig.cmake.in
    ${CMAKE_BINARY_DIR}/configure_gen/cmake/AuraConfig.cmake
    @ONLY
)

###############################################################################

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure/mk/Aura.mk.in
    ${CMAKE_BINARY_DIR}/configure_gen/mk/Aura.mk
    @ONLY
)

set(AURA_CONFIG_INC_DIRS "${CMAKE_BINARY_DIR}/configure_gen/include")

install(FILES "${AURA_CONFIG_INC_DIRS}/aura/config.h" DESTINATION include/aura)
install(FILES "${CMAKE_BINARY_DIR}/configure_gen/cmake/AuraConfig.cmake" DESTINATION cmake)
if(AURA_BUILD_ANDROID)
    install(FILES "${CMAKE_BINARY_DIR}/configure_gen/mk/Aura.mk" DESTINATION mk)
endif()