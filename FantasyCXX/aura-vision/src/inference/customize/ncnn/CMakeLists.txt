
macro(ncnn_add_custom_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    message(STATUS "WITH_LAYER_${name} = ${WITH_LAYER_${name}}")


    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS "${NCNN_ROOT_SRC_DIR}/layer/${name}.cpp")

        if((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm")
                OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)"))
            set(arch arm)
            set(LAYER_SRC ${NCNN_ROOT_SRC_DIR}/layer/${arch}/${name}_${arch}.cpp)
        else()
            set(arch iov_x86_64)
            set(LAYER_SRC ${CUSTOM_LAYER_SRC_DIR}/${arch}/${name}_${arch}.cpp)
        endif()

        if(EXISTS ${LAYER_SRC})
            set(WITH_LAYER_${name}_${arch} 1)
            list(APPEND ncnn_SRCS ${LAYER_SRC})
            message(STATUS "NCNN Adding custom layer: ${LAYER_SRC}")
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        if(WITH_LAYER_${name}_${arch})
            string(APPEND layer_declaration
                    "extern Layer* ${class}_${arch}_layer_creator();\n")
            string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_${arch}_layer_creator},\n#else\n{${class}_${arch}_layer_creator},\n#endif\n")
        else()
            string(APPEND layer_declaration
                    "extern Layer* ${class}_layer_creator();\n")
            string(APPEND layer_registry
                    "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
        endif()
    else()
        string(APPEND layer_registry "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    string(APPEND layer_type_enum "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

set(NCNN_ROOT_SRC_DIR ${PROJECT_SOURCE_DIR}/src) # project_source_dir refers to ncnn project!
set(VISION_ROOT_SRC_DIR ${PROJECT_SOURCE_DIR}/../../../vision/src)
set(CUSTOM_LAYER_SRC_DIR ${VISION_ROOT_SRC_DIR}/inference/customize/ncnn)

if(USE_IOV_LAYERS)
    ncnn_add_custom_layer(Convolution)
    ncnn_add_custom_layer(ConvolutionDepthWise)
    message(STATUS "NCNN USE_IOV_LAYERS")
endif()