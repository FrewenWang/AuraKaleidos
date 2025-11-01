macro(opencl_code_gen cl_dir cpp_dir)
    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE} ${OPENCL_CODE_GEN_PY} ${AURA_OPENCL_KERNEL_INC_DIRS} ${cl_dir} ${cpp_dir} ${CMAKE_BINARY_DIR}/opencl_gen/opencl_helper ${CMAKE_BINARY_DIR}/opencl_gen/opencl_code_gen.tmp
        RESULT_VARIABLE ERR_VAR
    )
    if(${ERR_VAR})
        message(FATAL_ERROR "opencl code generate failed")
    endif()
endmacro()

macro(find_hexagon_tool hexagon_sdk_path hexagon_toolchain_root hexagon_tool_version hexaogn_sdk_version)
    string(REGEX REPLACE "/$" "" sdk_path ${hexagon_sdk_path})
    file(GLOB_RECURSE heagon_clang_paths "${sdk_path}/*/hexagon-clang++")
    set(max_version "0.0.00")
    foreach(cur_clang_path ${heagon_clang_paths})
        get_filename_component(hexagon_tmp0 "${cur_clang_path}"     DIRECTORY)
        get_filename_component(hexagon_tmp1 "${hexagon_tmp0}"       DIRECTORY)
        get_filename_component(hexagon_tmp2 "${hexagon_tmp1}"       DIRECTORY)
        get_filename_component(hexagon_tmp3 "${hexagon_tmp2}"       NAME)
        get_filename_component(hexagon_tmp4 "${sdk_path}"           NAME)

        set(version "${hexagon_tmp3}")

        string(REPLACE "." ";" version_parts ${version})
        list(GET version_parts 0 major)
        list(GET version_parts 1 minor)
        list(GET version_parts 2 patch)

        string(REPLACE "." ";" max_parts ${max_version})
        list(GET max_parts 0 max_major)
        list(GET max_parts 1 max_minor)
        list(GET max_parts 2 max_patch)

        if(${major} GREATER ${max_major})
            set(max_version ${version})
            set(${hexagon_toolchain_root} "${hexagon_tmp1}")
            set(${hexagon_tool_version}   "${hexagon_tmp3}")
            set(${hexaogn_sdk_version}    "${hexagon_tmp4}")
        elseif(${major} EQUAL ${max_major})
            if(${minor} GREATER ${max_minor})
                set(max_version ${version})
                set(${hexagon_toolchain_root} "${hexagon_tmp1}")
                set(${hexagon_tool_version}   "${hexagon_tmp3}")
                set(${hexaogn_sdk_version}    "${hexagon_tmp4}")
            elseif(${minor} EQUAL ${max_minor})
                if(${patch} GREATER ${max_patch})
                    set(max_version ${version})
                    set(${hexagon_toolchain_root} "${hexagon_tmp1}")
                    set(${hexagon_tool_version}   "${hexagon_tmp3}")
                    set(${hexaogn_sdk_version}    "${hexagon_tmp4}")
                endif()
            endif()
        endif()
    endforeach()
endmacro()

macro(find_win_compiler win_compiler)
    if(MSVC)
        if(MSVC_VERSION EQUAL 1400)
            set(${win_compiler} vc8)
        elseif(MSVC_VERSION EQUAL 1500)
            set(${win_compiler} vc9)
        elseif(MSVC_VERSION EQUAL 1600)
            set(${win_compiler} vc10)
        elseif(MSVC_VERSION EQUAL 1700)
            set(${win_compiler} vc11)
        elseif(MSVC_VERSION EQUAL 1800)
            set(${win_compiler} vc12)
        elseif(MSVC_VERSION EQUAL 1900)
            set(${win_compiler} vc14)
        elseif(MSVC_VERSION MATCHES "^191[0-9]$")
            set(${win_compiler} vc15)
        elseif(MSVC_VERSION MATCHES "^192[0-9]$")
            set(${win_compiler} vc16)
        elseif(MSVC_VERSION MATCHES "^193[0-9]$")
            set(${win_compiler} vc17)
        else()
            message(FATAL_ERROR "failed to recognize msvc version: ${MSVC_VERSION}")
        endif()
    elseif(MINGW)
        set(${win_compiler} mingw)
    endif()
endmacro()

macro(find_ubuntu_release_version ubuntu_release_version)
    cmake_host_system_information(RESULT OS_VERSION_INFO QUERY OS_VERSION)

    if(OS_VERSION_INFO MATCHES "Ubuntu")
        string(REGEX REPLACE ".*~([0-9]+)\\..*" "\\1" UBUNTU_RELEASE_EXPECTED_VERSION ${OS_VERSION_INFO})

        if(NOT UBUNTU_RELEASE_EXPECTED_VERSION)
            message(FATAL_ERROR "failed to determine Ubuntu release version from ${OS_VERSION_INFO}")
        endif()

        set(${ubuntu_release_version} ${UBUNTU_RELEASE_EXPECTED_VERSION})
    else()
        message(FATAL_ERROR "only support Ubuntu")
    endif()
endmacro()

macro(find_glibc_version glibc_version)
    execute_process(
        COMMAND ldd --version
        OUTPUT_VARIABLE LDD_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(LDD_VERSION_OUTPUT MATCHES "GLIBC")
        string(REGEX MATCH "([0-9]\\.[0-9]+)" GLIBC_EXPECTED_VERSION ${LDD_VERSION_OUTPUT})

        if(NOT GLIBC_EXPECTED_VERSION)
            message(FATAL_ERROR "failed to parse glibc version from ldd output")
        endif()

        set(${glibc_version} ${GLIBC_EXPECTED_VERSION})
    else()
        message(FATAL_ERROR "unknown ldd version")
    endif()
endmacro()

macro(find_android_ndk_version android_ndk_version)
    if(NOT DEFINED ANDROID_NDK_REVISION)
        message(FATAL_ERROR "failed to determine ndk version from ANDROID_NDK_REVISION")
    endif()

    if(NOT ANDROID_NDK_REVISION MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        message(FATAL_ERROR "failed to parse ANDROID_NDK_REVISION=${ANDROID_NDK_REVISION}")
    endif()

    if(CMAKE_MATCH_1 LESS 19 OR CMAKE_MATCH_2 GREATER 4)
        message(FATAL_ERROR "only support ndk r19 and later")
    endif()

    if(CMAKE_MATCH_2 EQUAL 1)
        set(${android_ndk_version} "${CMAKE_MATCH_1}b")
    elseif(CMAKE_MATCH_2 EQUAL 2)
        set(${android_ndk_version} "${CMAKE_MATCH_1}c")
    elseif(CMAKE_MATCH_2 EQUAL 3)
        set(${android_ndk_version} "${CMAKE_MATCH_1}d")
    elseif(CMAKE_MATCH_2 EQUAL 4)
        set(${android_ndk_version} "${CMAKE_MATCH_1}e")
    else()
        set(${android_ndk_version} "${CMAKE_MATCH_1}")
    endif()
endmacro()

macro(strip_lib target is_shared_lib is_release)
    if(${is_release})
        if(${is_shared_lib})
            add_custom_command(
                TARGET ${target}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} -s lib${target}.so
                COMMENT "Stripping debug symbols from lib${target}.so"
               )
        else()
            add_custom_command(
                TARGET ${target}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} --strip-debug lib${target}.a
                COMMENT "Stripping debug symbols from lib${target}.a"
               )
        endif()
    endif()
endmacro()

macro(xtensa_build_pil target)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_PKG_LOADLIB} -s ${target} ${target}
        COMMENT "Running xt-pkg-loadlib to generate ${target}"
    )
endmacro()

macro(aura_add_dir dir group)
    if(EXISTS ${dir})
        aux_source_directory(${dir} ${group})
        file(GLOB ASM_FILES "${dir}/*.S")
        list(APPEND ${group} ${ASM_FILES})
    endif()
endmacro()

macro(aura_update_group_vars)
    set(AURA_LIB_INC_DIRS ${AURA_LIB_INC_DIRS} PARENT_SCOPE)
    set(AURA_TEST_INC_DIRS ${AURA_TEST_INC_DIRS} PARENT_SCOPE)

    set(AURA_LIB_COMM_SRCS ${AURA_LIB_COMM_SRCS} PARENT_SCOPE)
    set(AURA_LIB_NONE_SRCS ${AURA_LIB_NONE_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HOST_SRCS ${AURA_LIB_HOST_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HOST_NEON_SRCS ${AURA_LIB_HOST_NEON_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HOST_CL_SRCS ${AURA_LIB_HOST_CL_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HOST_HEXAGON_SRCS ${AURA_LIB_HOST_HEXAGON_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HOST_XTENSA_SRCS ${AURA_LIB_HOST_XTENSA_SRCS} PARENT_SCOPE)
    set(AURA_LIB_HEXAGON_SRCS ${AURA_LIB_HEXAGON_SRCS} PARENT_SCOPE)
    set(AURA_LIB_XTENSA_SRCS ${AURA_LIB_XTENSA_SRCS} PARENT_SCOPE)
    set(AURA_LIB_XTENSA_PIL_SRCS ${AURA_LIB_XTENSA_PIL_SRCS} PARENT_SCOPE)

    set(AURA_TEST_COMM_SRCS ${AURA_TEST_COMM_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_SRCS ${AURA_TEST_HOST_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_NONE_SRCS ${AURA_TEST_HOST_NONE_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_NEON_SRCS ${AURA_TEST_HOST_NEON_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_CL_SRCS ${AURA_TEST_HOST_CL_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_HEXAGON_SRCS ${AURA_TEST_HOST_HEXAGON_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HEXAGON_SRCS ${AURA_TEST_HEXAGON_SRCS} PARENT_SCOPE)
    set(AURA_TEST_HOST_XTENSA_SRCS ${AURA_TEST_HOST_XTENSA_SRCS} PARENT_SCOPE)
    set(AURA_TEST_XTENSA_SRCS ${AURA_TEST_XTENSA_SRCS} PARENT_SCOPE)

    set(AURA_NN_RUN_INC_DIRS ${AURA_NN_RUN_INC_DIRS} PARENT_SCOPE)
    set(AURA_NN_RUN_HOST_SRCS ${AURA_NN_RUN_HOST_SRCS} PARENT_SCOPE)
endmacro()

macro(aura_add_op name)
    set(AURA_CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${name}")

    # inc
    list(APPEND AURA_LIB_INC_DIRS  "${AURA_CURRENT_DIR}/include")
    list(APPEND AURA_LIB_INC_DIRS  "${AURA_CURRENT_DIR}/private")
    list(APPEND AURA_TEST_INC_DIRS "${AURA_CURRENT_DIR}/test")

    install(DIRECTORY "${AURA_CURRENT_DIR}/include/aura" DESTINATION include)

    # lib
    aura_add_dir("${AURA_CURRENT_DIR}/src"                      AURA_LIB_COMM_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/none"                 AURA_LIB_NONE_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/host"                 AURA_LIB_HOST_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/host/neon"            AURA_LIB_HOST_NEON_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/host/opencl"          AURA_LIB_HOST_CL_SRCS)
    if(EXISTS "${AURA_CURRENT_DIR}/src/opencl" AND AURA_ENABLE_OPENCL)
        opencl_code_gen("${AURA_CURRENT_DIR}/src/opencl" "${CMAKE_BINARY_DIR}/opencl_gen/ops/${name}")
        aux_source_directory("${CMAKE_BINARY_DIR}/opencl_gen/ops/${name}" AURA_LIB_HOST_CL_SRCS)
    endif()
    aura_add_dir("${AURA_CURRENT_DIR}/src/host/hexagon"         AURA_LIB_HOST_HEXAGON_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/host/xtensa"          AURA_LIB_HOST_XTENSA_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/hexagon"              AURA_LIB_HEXAGON_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/src/xtensa"               AURA_LIB_XTENSA_SRCS)

    # test
    aura_add_dir("${AURA_CURRENT_DIR}/test"                     AURA_TEST_COMM_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/test/host/src/none"       AURA_TEST_HOST_NONE_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/test/host/src/neon"       AURA_TEST_HOST_NEON_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/test/host/src/opencl"     AURA_TEST_HOST_CL_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/test/host/src/hexagon"    AURA_TEST_HOST_HEXAGON_SRCS)
    aura_add_dir("${AURA_CURRENT_DIR}/test/host/src/xtensa"     AURA_TEST_HOST_XTENSA_SRCS)

    aura_update_group_vars()
endmacro()

macro(aura_finalize_group_vars)
    # lib
    if(AURA_BUILD_HOST)
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_COMM_SRCS})
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_NONE_SRCS})
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_SRCS})

        if(AURA_BUILD_ANDROID OR AURA_BUILD_EMBEDDED_LINUX)
            list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_NEON_SRCS})
        endif()

        if(AURA_ENABLE_OPENCL)
            list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_CL_SRCS})
        endif()

        if(AURA_ENABLE_HEXAGON)
            list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_HEXAGON_SRCS})
        endif()

        if(AURA_ENABLE_XTENSA)
            list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_XTENSA_SRCS})
            list(APPEND AURA_LIB_SRCS ${3RDPARTY_LIB_XTENSA_SRCS})
        endif()
    elseif(AURA_BUILD_HEXAGON)
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_COMM_SRCS})
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_NONE_SRCS})
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_HEXAGON_SRCS})
    elseif(AURA_BUILD_XTENSA)
        list(APPEND AURA_LIB_SRCS ${AURA_LIB_XTENSA_SRCS})
        list(APPEND AURA_LIB_SRCS ${AURA_TEST_XTENSA_SRCS})
        list(APPEND AURA_LIB_SRCS ${3RDPARTY_LIB_XTENSA_SRCS})
    endif()

    # test
    if(AURA_BUILD_HOST)
        list(APPEND AURA_TEST_SRCS ${AURA_TEST_COMM_SRCS})
        list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_SRCS})
        list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_NONE_SRCS})

        if(AURA_BUILD_ANDROID OR AURA_BUILD_EMBEDDED_LINUX)
            list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_NEON_SRCS})
        endif()

        if(AURA_ENABLE_OPENCL)
            list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_CL_SRCS})
        endif()

        if(AURA_ENABLE_HEXAGON)
            list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_HEXAGON_SRCS})
        endif()

        if(AURA_ENABLE_XTENSA)
            list(APPEND AURA_TEST_SRCS ${AURA_TEST_HOST_XTENSA_SRCS})
        endif()
    elseif(AURA_BUILD_HEXAGON)
        list(APPEND AURA_TEST_SRCS ${AURA_TEST_COMM_SRCS})
        list(APPEND AURA_TEST_SRCS ${AURA_TEST_HEXAGON_SRCS})
    endif()

    # aura nn run
    if(AURA_BUILD_HOST)
        list(APPEND AURA_NN_RUN_SRCS ${AURA_NN_RUN_HOST_SRCS})
    endif()
endmacro()

macro(join_string_list output seperator)
    set(first_element TRUE)
    set(result)

    foreach(item ${ARGN})
        if(NOT "${item}" STREQUAL "")
            if(first_element)
                set(first_element FALSE)
            else()
                set(result "${result}${seperator}")
            endif()

            string(TOLOWER "${item}" item)
            set(result "${result}${item}")
        endif()
    endforeach()

    set(${output} ${result})
endmacro()