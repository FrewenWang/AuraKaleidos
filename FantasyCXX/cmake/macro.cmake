
#
macro(aura_update_group_vars)
    # include
    set(AURA_LIB_INCLUDE_DIRS ${AURA_LIB_INCLUDE_DIRS} PARENT_SCOPE)
    set(AURA_TEST_INCLUDE_DIRS ${AURA_TEST_INCLUDE_DIRS} PARENT_SCOPE)

    # private include
    set(AURA_LIB_PRIVATE_DIRS ${AURA_LIB_PRIVATE_DIRS} PARENT_SCOPE)

    # srcs
    set(AURA_LIB_COMMON_SRCS ${AURA_LIB_COMMON_SRCS} PARENT_SCOPE)

    # test srcs
    # set(AURA_TEST_COMMON_SRCS ${AURA_TEST_COMMON_SRCS} PARENT_SCOPE)

    set(AURA_LIB_SRCS ${AURA_LIB_SRCS} PARENT_SCOPE)
endmacro()

macro(aura_add_module name)
    set(AURA_CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    # include
    # add all the include dir to  aura_lib_include_dirs
    list(APPEND AURA_LIB_INCLUDE_DIRS "${AURA_CURRENT_DIR}/include")
    list(APPEND AURA_LIB_PRIVATE_DIRS "${AURA_CURRENT_DIR}/private")
    # list(APPEND AURA_TEST_INC_DIRS "${AURA_CURRENT_DIR}/test")
    # install
    install(DIRECTORY "${AURA_CURRENT_DIR}/include/aura" DESTINATION ${SUB_INSTALL_PREFIX}/include)

    # lib
    aura_add_src_dir("${AURA_CURRENT_DIR}/src" AURA_LIB_COMMON_SRCS)

    # test
    # aura_add_src_dir("${AURA_CURRENT_DIR}/test" AURA_TEST_COMMON_SRCS)

    # update all the variables
    aura_update_group_vars()

    #    message(STATUS "AURA_LIB_INC_DIRS:${AURA_LIB_INC_DIRS}")
    #    message(STATUS "AURA_TEST_INC_DIRS:${AURA_TEST_INC_DIRS}")
    #    message(STATUS "AURA_CURRENT_DIR:${AURA_CURRENT_DIR}")
    #    message(STATUS "AURA_LIB_COMMON_SRCS:${AURA_LIB_COMMON_SRCS}")
    #    message(STATUS "AURA_TEST_COMMON_SRCS:${AURA_TEST_COMMON_SRCS}")
endmacro()

macro(aura_add_src_dir dir group)
    if (EXISTS ${dir})
        aux_source_directory(${dir} ${group})
        file(GLOB ASM_FILES "${dir}/*.S")
        list(APPEND ${group} ${ASM_FILES})
    endif ()
endmacro()



# 将工程中的所有SRC代码合并到AURA_LIB_SRCS变量
macro(aura_finalize_group_vars)
    list(APPEND AURA_LIB_SRCS ${AURA_LIB_COMMON_SRCS})
    list(APPEND AURA_LIB_SRCS ${AURA_LIB_NONE_SRCS})
    list(APPEND AURA_LIB_SRCS ${AURA_LIB_HOST_SRCS})

endmacro()

macro(join_string_list output seperator)
    set(first_element TRUE)
    set(result)

    foreach (item ${ARGN})
        if (NOT "${item}" STREQUAL "")
            if (first_element)
                set(first_element FALSE)
            else ()
                set(result "${result}${seperator}")
            endif ()

            string(TOLOWER "${item}" item)
            set(result "${result}${item}")
        endif ()
    endforeach ()

    set(${output} ${result})
endmacro()

# Collecting sources from globbing and appending to output list variable
# Usage: collect_source_files(dest_src_list GLOB[_RECURSE] <module_name>)
# 注意：需要明确指定SRC_ROOT的目录，才能正常遍历此目录下所有的.c .cc .cpp
# 定义一个名字为collect_source_files(采集遍历所有的src文件的宏定义)。参数是所有的文件的参数命名：
macro(collect_source_files var)
    # 使用 cmake_parse_arguments 解析宏的可选参数。
    # cmake_parse_arguments(<prefix> <options> <one_value_keywords>
    #                      <multi_value_keywords> <args>...)
    # <prefix> MODULE是参数前缀，解析出的变量会变成 MODULE_GLOB 和 MODULE_GLOB_RECURSE。
    # <options> 为空
    # <multi_value_keywords> 多值关键词列表：GLOB;GLOB_RECURSE
    # 参数, 一般传入 ${ARGN} 即可。${ARGN}是定义要求的掺入的参数以外的参数
    cmake_parse_arguments(MODULE "" "" "GLOB;GLOB_RECURSE" ${ARGN})
    # 加上前缀之后 MODULE_GLOB
    if (MODULE_GLOB)
        # GLOB 是正则表示中的GLOB模式
        # GLOB 会产生一个由所有匹配globbing表达式的文件组成的列表，并将其保存到变量中。Globbing 表达式与正则表达式类似，但更简单。
        # 如果指定了RELATIVE 标记，返回的结果将是与指定的路径相对的路径构成的列表。 (通常不推荐使用GLOB命令来从源码树中收集源文件列表。
        # 原因是：如果CMakeLists.txt文件没有改变，即便在该源码树中添加或删除文件，产生的构建系统也不会知道何时该要求CMake重新产生构建文件。
        file(GLOB files LIST_DIRECTORIES FALSE
                ${SRC_ROOT}/${MODULE_GLOB}/*.c
                ${SRC_ROOT}/${MODULE_GLOB}/*.cc
                ${SRC_ROOT}/${MODULE_GLOB}/*.cpp)
        #
        list(APPEND ${var} ${files})
    endif ()
    # OB_RECURSE 与GLOB类似，区别在于它会遍历匹配目录的所有文件以及子目录下面的文件。
    # 对于属于符号链接的子目录，只有FOLLOW_SYMLINKS指定一或者cmake策略CMP0009没有设置为NEW时，才会遍历这些目录。
    if (MODULE_GLOB_RECURSE)
        file(GLOB_RECURSE files LIST_DIRECTORIES FALSE
                ${SRC_ROOT}/${MODULE_GLOB_RECURSE}/*.c
                ${SRC_ROOT}/${MODULE_GLOB_RECURSE}/*.cc
                ${SRC_ROOT}/${MODULE_GLOB_RECURSE}/*.cpp)
        list(APPEND ${var} ${files})
    endif ()
endmacro()

# 用法示例：
# collect_source_files_recurse("${CMAKE_CURRENT_SOURCE_DIR}/src",SRC_LIST)
# collect_source_files_recurse("${CMAKE_CURRENT_SOURCE_DIR}/src" SRC_LIST, DEBUG)
macro(collect_source_files_recurse dir out_var)
    # 解析是否传入了 PRINT_DEBUG 选项（布尔值开关）
    cmake_parse_arguments(PRINT "DEBUG" "" "" ${ARGN})
    file(GLOB_RECURSE files
            LIST_DIRECTORIES false
            "${dir}/*.c"
            "${dir}/*.cc"
            "${dir}/*.cpp"
            "${dir}/*.cxx"
    )
    list(APPEND ${out_var} ${files})

    # 可选：打印文件列表供调试
    if (PRINT_DEBUG)
        message(STATUS "Collected source files into ${out_var}:")
        foreach (f IN LISTS files)
            message(STATUS "  ${f}")
        endforeach ()
    endif ()
endmacro()
