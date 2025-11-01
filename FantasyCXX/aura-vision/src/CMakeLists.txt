set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include)

collect_source_files(lib_srcs GLOB .)
collect_source_files(lib_srcs GLOB core)
collect_source_files(lib_srcs GLOB core/bean)
collect_source_files(lib_srcs GLOB core/common)
collect_source_files(lib_srcs GLOB core/helper)
collect_source_files(lib_srcs GLOB core/request)
collect_source_files(lib_srcs GLOB core/result)
collect_source_files(lib_srcs GLOB config)
#collect_source_files(lib_srcs GLOB config/runtime_config)
collect_source_files(lib_srcs GLOB_RECURSE config/static_config/ref_landmark)
collect_source_files(lib_srcs GLOB_RECURSE config/static_config/schedule_config)
collect_source_files(lib_srcs GLOB_RECURSE initializer)
collect_source_files(lib_srcs GLOB_RECURSE model_ir)
collect_source_files(lib_srcs GLOB_RECURSE util)
collect_source_files(lib_srcs GLOB_RECURSE proto)
collect_source_files(lib_srcs GLOB vacv)
collect_source_files(lib_srcs GLOB ops)
#collect_source_files(lib_srcs GLOB deprecated_detector)
#collect_source_files(lib_srcs GLOB deprecated_detector/iov/ncnn)

include(${SRC_ROOT}/detector/CMakeLists.txt)
include(${SRC_ROOT}/inference/CMakeLists.txt)
include(${SRC_ROOT}/manager/CMakeLists.txt)
include(${SRC_ROOT}/scheduler/CMakeLists.txt)
include(${SRC_ROOT}/config/static_config/model_ports/CMakeLists.txt)
include(${SRC_ROOT}/config/runtime_config/CMakeLists.txt)

link_directories(
        ${OPENCV_LINK_DIR}
        ${OPENCV_3RDPARTY_LINK_DIR}
        #        ${JSON_LINK_DIR}
)

if (${USE_EXT_ENCRYPT} MATCHES "true")
    link_directories(${BD_PROTECT_LINK_DIR})
endif ()

if (BUILD_NCNN)
    link_directories(${NCNN_LINK_DIR})
endif ()

if (USE_TF_LITE)
    link_directories(${TF_LITE_LINK_DIR})
endif ()

if (USE_SNPE)
    link_directories(${SNPE_LINK_DIR})
endif ()

if (USE_QNN)
    link_directories(${QNN_LINK_DIR})
endif ()

if (USE_ONNX)
    link_directories(${ONNX_LINK_DIR})
endif ()

# ----------------------------------------------------------------------------------------------------------------------
# 编译库文件
# ----------------------------------------------------------------------------------------------------------------------
set(LIB_NAME vision)
if (BUILD_STATIC_LIB)
    add_library(${LIB_NAME} STATIC ${lib_srcs})
else ()
    add_library(${LIB_NAME} SHARED ${lib_srcs})
endif ()

if (BUILD_VACV_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vacv/test)
endif ()

if (BUILD_OPS_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ops/atest)
endif ()

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/config/static_config/params)
#add_dependencies(${LIB_NAME} vision_config)
#add_dependencies(${LIB_NAME} configure_vision_model)
#add_dependencies(${LIB_NAME} compile_proto plainbuffer)

# ----------------------------------------------------------------------------------------------------------------------
# 添加编译器配置
# ----------------------------------------------------------------------------------------------------------------------
#if(CMAKE_COMPILER_IS_GNUCXX)
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14") #-std=gnu++0x
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
#endif(CMAKE_COMPILER_IS_GNUCXX)
#add_compile_options(-std=c++11)

if (${CMAKE_BUILD_TYPE} STREQUAL "release")
    if ((${ICP_TARGET_OS} MATCHES "android") OR (${ICP_TARGET_OS} MATCHES "ios"))
        if (BUILD_MOBILE)
            target_compile_options(${LIB_NAME} PRIVATE -Os -std=c++11 -fno-rtti -fexceptions -ffast-math -fPIC -Wextra -Wno-unused-parameter -Wno-format-security -Wno-unused-variable)
        else ()
            target_compile_options(${LIB_NAME} PRIVATE -O3 -std=c++11 -frtti -fexceptions -ffast-math -fPIC -Wextra -Wno-unused-parameter -Wno-format-security -Wno-unused-variable)
        endif ()
    else ()
        target_compile_options(${LIB_NAME} PRIVATE -O3)
    endif ()
else ()
    target_compile_options(${LIB_NAME} PRIVATE -O0 -ffast-math -frtti -fexceptions -Wextra -Werror=return-type -g) # -Wall -Wno-unused-parameter -Wno-format-security -Wno-unused-variable -finstrument-functions  -p
endif ()

if ((${TARGET_ARCH_ABI} MATCHES "x86") OR (${TARGET_ARCH_ABI} MATCHES "x86_64"))
    target_compile_options(${LIB_NAME} PRIVATE -msse4.2)
endif ()

# compile options
#target_compile_options(${LIB_NAME} PRIVATE -std=c++11)
#if (${CMAKE_BUILD_TYPE} MATCHES "release")
#    target_compile_options(${LIB_NAME} PRIVATE -Ofast -ffast-math -Wall -Wextra -fPIC -frtti -fexceptions -Wno-unused-parameter -Wno-format-security -Wno-unused-variable)
#else ()
#    target_compile_options(${LIB_NAME} PRIVATE -O0    -ffast-math -Wall -Wextra -fPIC -frtti -fexceptions -Wno-unused-parameter -Wno-format-security -Wno-unused-variable -p -g -finstrument-functions)
#endif ()
#if ((${TARGET_ARCH_ABI} MATCHES "x86") OR
#    (${TARGET_ARCH_ABI} MATCHES "x86_64"))
#    target_compile_options(${LIB_NAME} PRIVATE -msse4.2)
#endif ()

target_include_directories(
        ${LIB_NAME}
        PUBLIC
        ${SRC_ROOT}
        ${INC_ROOT}
        ${PROJECT_BINARY_DIR}
        ${OPENCV_INCLUDE_DIR}
        #        ${JSON_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/aura-lightbuffer/include
)

target_link_libraries(${LIB_NAME} PUBLIC
                      ${OPENCV_LIB}
                      #        ${JSON_LIB}
                      LightBuffer
                      ${3D_LANDMARK_LIB}
)

#if ((${CMAKE_BUILD_TYPE} MATCHES "Debug") OR (${CMAKE_BUILD_TYPE} MATCHES "debug"))
#    if (${ICP_TARGET_OS} MATCHES "qnx")
#        target_link_libraries(${LIB_NAME} PUBLIC
#                profilingS
#        )
#    endif()
#endif()

if (${ICP_TARGET_OS} MATCHES "android")
    #    find_library(log_lib log)
    #    find_library(z_lib z)
    #    target_link_libraries(${LIB_NAME} PUBLIC ${log_lib} ${z_lib})
    target_link_libraries(${LIB_NAME} PUBLIC log z)

    # encrypt lib
    if (${USE_EXT_ENCRYPT} MATCHES "true")
        message(STATUS "USE_EXT_ENCRYPT=${USE_EXT_ENCRYPT}")
        target_include_directories(${LIB_NAME} PUBLIC ${BD_PROTECT_INCLUDE_DIR})
        target_link_libraries(${LIB_NAME} PUBLIC ${BD_PROTECT_LIB})
    endif ()

    if (DEFINED ANDROID_NDK_MAJOR AND (${ANDROID_NDK_MAJOR} GREATER 20))
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
    endif ()
endif ()

# openmp
if (USE_OPENMP)
    if (${ICP_TARGET_OS} MATCHES "android")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -fopenmp ")
        set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -fopenmp ")
    else ()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -lgomp ")
        set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -lgomp ")
    endif ()
endif ()

if (BUILD_NCNN)
    target_include_directories(${LIB_NAME} PUBLIC ${NCNN_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${NCNN_LIB})
endif ()

if (USE_PADDLE_LITE)
    target_link_libraries(${LIB_NAME} PRIVATE ${PADDLE_LITE_LIB})
    target_link_libraries(${LIB_NAME} PRIVATE ${PADDLE_LITE_THIRD_PARTY_LIB})
    #    find_package(OpenMP)
    #    if(OpenMP_CXX_FOUND)
    #        target_link_libraries(${LIB_NAME} PRIVATE OpenMP::OpenMP_CXX)
    #    endif()
endif ()

if (USE_TF_LITE)
    target_include_directories(${LIB_NAME} PUBLIC ${TF_LITE_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${TF_LITE_LIB})
endif ()

if (USE_SNPE)
    target_include_directories(${LIB_NAME} PUBLIC ${SNPE_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${SNPE_LIB})
endif ()

if (USE_QNN)
    message(STATUS "[Dependency] qnn QNN_INCLUDE_DIR=" ${QNN_INCLUDE_DIR})
    message(STATUS "[Dependency] qnn QNN_LINK_DIR=" ${QNN_LINK_DIR})
    message(STATUS "[Dependency] qnn LIB_NAME=" ${LIB_NAME})
    message(STATUS "[Dependency] qnn QNN_LIB=" ${QNN_LIB})
    target_include_directories(${LIB_NAME} PUBLIC ${QNN_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${QNN_LIB})
endif ()

if (USE_ONNX)
    target_include_directories(${LIB_NAME} PUBLIC ${ONNX_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${ONNX_LIB})
endif ()

if (BUILD_FASTCV)
    target_include_directories(${LIB_NAME} PUBLIC ${FAST_CV_LINK_DIR})
    target_link_libraries(${LIB_NAME} PRIVATE ${FAST_CV_LIB})
endif ()

if (BUILD_SLOG)
    target_include_directories(${LIB_NAME} PUBLIC ${LOG_HELPER_INCLUDE_DIR})
    target_link_libraries(${LIB_NAME} PUBLIC ${LOG_HELPER_LIB})
endif ()

# install
install(TARGETS ${LIB_NAME}
        ARCHIVE DESTINATION ${VALIB_INSTALL_LIB_DIR}
        LIBRARY DESTINATION ${VALIB_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${VALIB_INSTALL_LIB_DIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/vision DESTINATION include)
install(FILES ${CMAKE_BINARY_DIR}/build_config.h DESTINATION include/vision)
if (${ICP_TARGET_OS} MATCHES "ios")
    set(buildType "Debug")
    if (${CMAKE_BUILD_TYPE} MATCHES "release")
        set(buildType "Release")
    endif ()
    set(buildTarget "iphoneos")
    if (${ICP_TARGET_ARCH} MATCHES "x86_64")
        set(buildTarget "iphonesimulator")
    endif ()
    set(buildTypeDir ${buildType}-${buildTarget})
    #    install(FILES ${CMAKE_BINARY_DIR}/vision/src/config/static_config/params/${buildTypeDir}/libvision_config.a DESTINATION lib)
    install(FILES ${CMAKE_BINARY_DIR}/plainbuffer/${buildTypeDir}/libplainbuffer.a DESTINATION lib)
endif ()

if (USE_PADDLE_LITE)
    if (${ICP_TARGET_OS} MATCHES "ios")
        install(FILES ${PADDLE_LITE_LINK_DIR}libpaddle_api_light_bundled.a DESTINATION ${VALIB_INSTALL_LIB_DIR})
    else ()
        install(FILES ${PADDLE_LITE_LINK_DIR}libpaddle_light_api_shared.so DESTINATION ${VALIB_INSTALL_LIB_DIR})
    endif ()
endif ()

if ((${ICP_TARGET_OS} MATCHES "android") OR (${ICP_TARGET_OS} MATCHES "qnx"))
    if (USE_SNPE)
        set(LIBS_TO_COPY ${LIBS_TO_COPY} ${SNPE_LINK_DIR}/*.so)
    endif ()
    if (${USE_EXT_ENCRYPT} MATCHES "true")
        set(LIBS_TO_COPY ${LIBS_TO_COPY} ${BD_PROTECT_LINK_DIR}/*.so)
    endif ()

    list(LENGTH LIBS_TO_COPY DEPS_LEN)
    if (${DEPS_LEN} GREATER 0)
        set(COPY_DEPENDENCY_CMD cp ${LIBS_TO_COPY} ${VALIB_INSTALL_LIB_DIR}/)
    endif ()
endif ()

# if the target platform cannot be detected, it can be manually set here
# set(UNIX true)
if (${ICP_TARGET_OS} MATCHES "ios")
    if (${BUILD_STATIC_LIB})
        if (${ICP_TARGET_ARCH} MATCHES "x86_64")
            set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/Release-iphonesimulator/libvision.a)
        else ()
            set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/Release-iphoneos/libvision.a)
        endif ()
    else ()
        if (${ICP_TARGET_ARCH} MATCHES "x86_64")
            set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/Release-iphonesimulator/libvision.dylib)
        else ()
            set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/Release-iphoneos/libvision.dylib)
        endif ()
    endif ()
elseif (APPLE)
    set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/libvision.dylib)
elseif (UNIX)
    if (${BUILD_STATIC_LIB})
        set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/libvision.a)
    else ()
        set(BUILT_LIB_PATH ${CMAKE_BINARY_DIR}/vision/src/libvision.so)
    endif ()
endif ()

message(STATUS BUILT_LIB_PATH=${BUILT_LIB_PATH})

if (${TARGET_OS} MATCHES "ios" OR ${TARGET_OS} MATCHES "mac")
    set(STRIP_CMD strip -xS ${BUILT_LIB_PATH})
else ()
    if ((${CMAKE_BUILD_TYPE} MATCHES "debug") OR (${CMAKE_BUILD_TYPE} MATCHES "Debug"))
        set(STRIP_CMD echo "Ignore stripping!")
        set(CP_VISION_SYMBOL echo "Ignore cp vision symbol!")
    elseif (${BUILT_LIB_PATH} MATCHES ${CMAKE_BINARY_DIR}/vision/src/libvision.so AND (EXISTS /usr/bin/eu-strip))
        # 将so和调试信息分离
        set(STRIP_CMD eu-strip ${BUILT_LIB_PATH} -f ${BUILT_LIB_PATH}.symbol)
        set(CP_VISION_SYMBOL cp ${BUILT_LIB_PATH}.symbol ${VALIB_INSTALL_LIB_DIR})
    else ()
        set(STRIP_CMD ${CMAKE_STRIP} -s ${BUILT_LIB_PATH})
        set(CP_VISION_SYMBOL cp ${BUILT_LIB_PATH}.symbol ${VALIB_INSTALL_LIB_DIR})
    endif ()
endif ()
message(STATUS CMAKE_STRIP=${CMAKE_STRIP})
message(STATUS STRIP_CMD=${STRIP_CMD})
message(STATUS CP_VISION_SYMBOL=${CP_VISION_SYMBOL})

if (USE_EXTERNAL_MODEL)
    set(COPY_MODEL cp ${CMAKE_SOURCE_DIR}/models/generated/vision_model.bin ${VALIB_INSTALL_LIB_DIR}/libvision_model.bin)
endif ()

add_custom_command(
        TARGET ${LIB_NAME}
        POST_BUILD
        COMMAND echo "Begin embedding model..."
        COMMAND python ${CMAKE_SOURCE_DIR}/models/scripts/embed_model.py
        --model ${CMAKE_SOURCE_DIR}/models/generated/vision_model.bin
        --lib ${BUILT_LIB_PATH}
        COMMAND echo "Begin stripping..."
        COMMAND ${STRIP_CMD}
        COMMAND echo "Begin copying dependencies..."
        COMMAND mkdir -p ${VALIB_INSTALL_LIB_DIR}
        COMMAND cp ${BUILT_LIB_PATH} ${VALIB_INSTALL_LIB_DIR}
        COMMAND ${CP_VISION_SYMBOL}
        COMMAND ${COPY_DEPENDENCY_CMD}
        COMMAND ${COPY_MODEL}
        COMMENT "Embedding model data into lib, and then stripping..."
)


