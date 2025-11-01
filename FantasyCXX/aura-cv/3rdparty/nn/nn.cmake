set(3RDPARTY_NN_DIRS ${CMAKE_BINARY_DIR}/3rdparty/nn)

execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/nn/nn_headers_gen.py ${CMAKE_SOURCE_DIR}/3rdparty/nn ${3RDPARTY_NN_DIRS}
    RESULT_VARIABLE ERR_VAR
    )
if(${ERR_VAR})
    message(FATAL_ERROR "FATAL: nn headers generation failed")
endif()

set(3RDPARTY_NN_MNN_INC_DIRS  ${3RDPARTY_NN_DIRS}/mnn)
set(3RDPARTY_NN_NP_INC_DIRS   ${3RDPARTY_NN_DIRS}/np)
set(3RDPARTY_NN_QNN_INC_DIRS  ${3RDPARTY_NN_DIRS}/qnn)
set(3RDPARTY_NN_SNPE_INC_DIRS ${3RDPARTY_NN_DIRS}/snpe)
set(3RDPARTY_NN_XNN_INC_DIRS  ${3RDPARTY_NN_DIRS}/xnn)

include_directories(${3RDPARTY_NN_MNN_INC_DIRS})
include_directories(${3RDPARTY_NN_NP_INC_DIRS})
include_directories(${3RDPARTY_NN_QNN_INC_DIRS})
include_directories(${3RDPARTY_NN_SNPE_INC_DIRS})
include_directories(${3RDPARTY_NN_XNN_INC_DIRS})

if(AURA_BUILD_ANDROID)
    if(ANDROID_ABI MATCHES "arm64-v8a")
        install(FILES "${CMAKE_SOURCE_DIR}/3rdparty/nn/mnn/mnn2.7.1/lib/android/lib64/cpu+gpu/libmnn_wrapper.so" DESTINATION 3rdparty/mnn/mnn2.7.1/cpu+gpu)
        install(FILES "${CMAKE_SOURCE_DIR}/3rdparty/nn/mnn/mnn2.7.1/lib/android/lib64/cpu/libmnn_wrapper.so"     DESTINATION 3rdparty/mnn/mnn2.7.1/cpu)
    else()
        install(FILES "${CMAKE_SOURCE_DIR}/3rdparty/nn/mnn/mnn2.7.1/lib/android/lib32/cpu+gpu/libmnn_wrapper.so" DESTINATION 3rdparty/mnn/mnn2.7.1/cpu+gpu)
        install(FILES "${CMAKE_SOURCE_DIR}/3rdparty/nn/mnn/mnn2.7.1/lib/android/lib32/cpu/libmnn_wrapper.so"     DESTINATION 3rdparty/mnn/mnn2.7.1/cpu)
    endif()
elseif(AURA_BUILD_LINUX)
    install(FILES "${CMAKE_SOURCE_DIR}/3rdparty/nn/mnn/mnn2.7.1/lib/linux/lib64/libmnn_wrapper.so" DESTINATION 3rdparty/mnn/mnn2.7.1)
endif()
