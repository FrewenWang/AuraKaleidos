

# Check whether a file is changed
# if changed, var will be set as true
# Usage: va_cache(vision_model.bin vision_model NEED_MERGE)
macro(file_changed file_name cache_name var)
    file(MD5 ${file_name} CONFIG_CHECKSUM)
    if (NOT "${CONFIG_CHECKSUM}" STREQUAL "$CACHE{${cache_name}}")
        set(${cache_name} ${CONFIG_CHECKSUM} CACHE STRING "md5 of ${file_name}" FORCE)
        set(${var} true)
    else ()
        set(${var} false)
    endif ()
endmacro()

# get git hash
macro(get_git_hash git_hash)
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
                OUTPUT_VARIABLE ${git_hash}
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif (GIT_FOUND)
endmacro(get_git_hash)

# get git branch
macro(get_git_branch git_branch)
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} symbolic-ref --short -q HEAD
                OUTPUT_VARIABLE ${git_branch}
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif (GIT_FOUND)
endmacro(get_git_branch)

# add padding
macro(add_padding str len character)
    set(output "")
    math(EXPR loop_count "${len} - 1")
    foreach (loop_idx RANGE ${loop_count})
        set(output "${output}${character}")
    endforeach ()
    set(${str} "${output}")
endmacro(add_padding)