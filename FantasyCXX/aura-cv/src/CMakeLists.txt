# set opencl kernel inc dir
set(AURA_OPENCL_KERNEL_INC_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/include/aura/runtime/core/opencl")

# add subdirectory
add_subdirectory(runtime)

if(NOT AURA_BUILD_XTENSA)
    add_subdirectory(tools)
    if(NOT AURA_ENABLE_NN_LITE)
        add_subdirectory(algos)
    endif()
endif()

if(NOT AURA_ENABLE_NN_LITE)
    add_subdirectory(ops)
endif()

if(AURA_BUILD_UNIT_TEST OR AURA_ENABLE_NN)
    add_subdirectory(test)
endif()

if(EXISTS "${CMAKE_BINARY_DIR}/opencl_gen/opencl_helper" AND AURA_ENABLE_OPENCL)
    aux_source_directory("${CMAKE_BINARY_DIR}/opencl_gen/opencl_helper" AURA_LIB_HOST_CL_SRCS)
endif()

# finalize group vars
aura_finalize_group_vars()

################################ build lib ################################
# generate
if(AURA_SHARED_LIBRARY)
    add_library(${AURA_LIB_NAME} SHARED ${AURA_LIB_SRCS})
else()
    add_library(${AURA_LIB_NAME} STATIC ${AURA_LIB_SRCS})
endif()

# include
target_include_directories(${AURA_LIB_NAME} PRIVATE ${AURA_CONFIG_INC_DIRS} ${AURA_LIB_INC_DIRS})
if(AURA_ENABLE_OPENCL)
    add_dependencies(${AURA_LIB_NAME} OpenCL-Headers OpenCL-Headers-Hpp)

    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE} ${OPENCL_CODE_GEN_PY} ${CMAKE_BINARY_DIR}/opencl_gen/opencl_code_gen.tmp ${CMAKE_BINARY_DIR}/opencl_gen/cl_program_string_register.hpp
        RESULT_VARIABLE ERR_VAR
    )
    if(${ERR_VAR})
        message(FATAL_ERROR "opencl code generate failed")
    endif()

    target_include_directories(${AURA_LIB_NAME} PRIVATE ${CMAKE_BINARY_DIR}/opencl_gen)
endif()

if(AURA_BUILD_XTENSA)
    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE} ${XTENSA_CODE_GEN_PY} xtensa_rpc_func_register "${AURA_LIB_INC_DIRS}" "${AURA_LIB_SRCS}" ${CMAKE_BINARY_DIR}/xtensa_gen/
        RESULT_VARIABLE ERR_VAR
        )
    if (${ERR_VAR})
        message(FATAL_ERROR "FATAL: xtensa code generate failed")
    endif()

    target_include_directories(${AURA_LIB_NAME} PRIVATE ${CMAKE_BINARY_DIR}/xtensa_gen)
endif()

# link
if(AURA_BUILD_LINUX)
    target_link_libraries(${AURA_LIB_NAME} pthread dl)
elseif(AURA_BUILD_ANDROID)
    target_link_libraries(${AURA_LIB_NAME} log)
endif()

if(NOT AURA_BUILD_WIN  AND NOT AURA_BUILD_MAC)
    strip_lib(${AURA_LIB_NAME} ${AURA_SHARED_LIBRARY} ${AURA_RELEASE})
endif()

if(AURA_BUILD_XTENSA)
    set(AURA_XTENSA_PIL_LIB_NAME "aura_xtensa_pil.so")
    add_executable(${AURA_XTENSA_PIL_LIB_NAME} ${AURA_LIB_XTENSA_PIL_SRCS})

    target_include_directories(${AURA_XTENSA_PIL_LIB_NAME} PRIVATE ${AURA_CONFIG_INC_DIRS} ${AURA_LIB_INC_DIRS})
    target_link_libraries(${AURA_XTENSA_PIL_LIB_NAME} ${AURA_LIB_NAME})
    xtensa_build_pil(${AURA_XTENSA_PIL_LIB_NAME})

    install(TARGETS ${AURA_XTENSA_PIL_LIB_NAME} DESTINATION lib)
endif()

# install
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/aura" DESTINATION include)
install(TARGETS ${AURA_LIB_NAME} DESTINATION lib)

################################ build test ################################
if(AURA_BUILD_UNIT_TEST)
    if(AURA_BUILD_HOST)
        # generate
        add_executable(aura_test_main ${AURA_TEST_SRCS})

        # include
        target_include_directories(aura_test_main PRIVATE ${AURA_CONFIG_INC_DIRS} ${AURA_LIB_INC_DIRS} ${AURA_TEST_INC_DIRS})

        # link
        target_link_libraries(aura_test_main ${AURA_LIB_NAME})

        if(AURA_BUILD_XPLORER)
            # link aura_xtensa for profile
            target_link_libraries(aura_test_main ${AURA_XTENSA_LIB_NAME})
        else()
            if(OPENCV_DIR)
                message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")
                # TODO 暂时还是不能根据OPENCV_DIR的加载OpenCV的环境，后续我们需要解决
                find_package(OpenCV REQUIRED)
                if(NOT OpenCV_FOUND)
                    message(FATAL_ERROR "find_package(OpenCV REQUIRED) failed")
                endif()

                target_link_libraries(aura_test_main ${OpenCV_LIBS})
            else()
                message(FATAL_ERROR "OpenCV path is empty!")
            endif()
        endif()

        if(AURA_BUILD_ANDROID AND AURA_ENABLE_OPENCL AND AURA_SHARED_LIBRARY)
            target_link_libraries(aura_test_main aura_opencl_library)
        endif()

        # install
        install(TARGETS aura_test_main DESTINATION bin)
    elseif(AURA_BUILD_HEXAGON)
        # generate
        add_library(aura_test_main_skel SHARED ${AURA_TEST_SRCS})

        # include
        target_include_directories(aura_test_main_skel PRIVATE ${AURA_CONFIG_INC_DIRS} ${AURA_LIB_INC_DIRS} ${AURA_TEST_INC_DIRS})

        # link
        target_link_libraries(aura_test_main_skel --whole-archive ${AURA_LIB_NAME})

        # install
        install(TARGETS aura_test_main_skel DESTINATION lib)
    endif()
endif()

################################ build aura nn run ################################
if(AURA_ENABLE_NN AND AURA_BUILD_HOST)
    # generate
    add_executable(aura-nn-run ${AURA_NN_RUN_SRCS})

    # include
    target_include_directories(aura-nn-run PRIVATE ${AURA_CONFIG_INC_DIRS} ${AURA_LIB_INC_DIRS} ${AURA_NN_RUN_INC_DIRS})

    # link
    target_link_libraries(aura-nn-run ${AURA_LIB_NAME})

    # install
    install(TARGETS aura-nn-run DESTINATION bin)
endif()