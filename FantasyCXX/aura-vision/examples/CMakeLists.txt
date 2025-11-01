
set(TARGET_NAME DemoAuraVision)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread ")
if (USE_OPENMP)
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -lgomp ")
    set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -lgomp ")
endif()

if (${TARGET_OS} STREQUAL "mac")
    message(STATUS "build native examples on ${ICP_TARGET_OS}")
else ()
    # vision_demo打开预览画面，绘制特征点。
    # camera权限问题，通过终端获取camera权限(open /Applications/CLion.app/Contents/MacOS/clion)
     list(APPEND demo_srcs vgui.cpp)
    # 保存执行检测的图片(多帧图片会生成视频数据)
    # add_definitions(-DSAVE_IMG)
endif ()

list(APPEND demo_srcs main_demo.cpp InfoPrinter.cpp fileHelper.cpp)

link_directories(
        ${OPENCV_LINK_DIR}
        ${OPENCV_3RDPARTY_LINK_DIR}
)

if (USE_SNPE)
    link_directories(${SNPE_LINK_DIR})
endif ()

if (USE_QNN)
    link_directories(${QNN_LINK_DIR})
endif ()

if (USE_TF_LITE)
    link_directories(${TF_LITE_LINK_DIR})
endif ()

add_executable(
        ${TARGET_NAME}
        ${demo_srcs}
)

if (USE_PADDLE_LITE)
    link_directories(${PADDLE_LITE_LINK_DIR})
    if(${ICP_TARGET_OS} MATCHES "linux")
        target_link_libraries(${TARGET_NAME} PRIVATE ${PADDLE_LITE_LIB})
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE ${PADDLE_LITE_LIB} c++_shared)
    endif()
    #target_link_libraries(${LIB_NAME} PRIVATE ${PADDLE_LITE_THIRD_PARTY_LIB})
    message(STATUS "[Dependency] PADDLE_LITE_LINK_DIR_0715=" ${PADDLE_LITE_LINK_DIR})
    message(STATUS "[Dependency] PADDLE_LITE_LIB_0715=" ${PADDLE_LITE_LIB})
    #    find_package(OpenMP)
    #    if(OpenMP_CXX_FOUND)
    #        target_link_libraries(${DEMO_NAME} PRIVATE OpenMP::OpenMP_CXX)
    #    endif()
endif ()
target_compile_options(${TARGET_NAME} PRIVATE -std=c++11)
target_compile_options(${TARGET_NAME} PRIVATE -O3)

target_include_directories(
        ${TARGET_NAME}
        PRIVATE
        ${OPENCV_INCLUDE_DIR}
        ${QNN_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
)

target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        ${OPENCV_LIB}
        ${JSON_LIB}
        ${LOG_HELPER_LIB_DEMO}
        vision
)

if (USE_QNN)
    target_link_libraries(${TARGET_NAME} PRIVATE ${QNN_LIB})
endif()

if (BUILD_FASTCV)
    #    target_include_directories(${DEMO_NAME} PUBLIC ${FAST_CV_LINK_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE ${FAST_CV_LIB})
endif ()

if (${ICP_TARGET_OS} MATCHES "android")
    target_link_libraries(${TARGET_NAME} PRIVATE log z)

    if (USE_SNPE)
        target_include_directories(${TARGET_NAME} PRIVATE ${SNPE_INCLUDE_DIR})
        target_link_libraries(${TARGET_NAME} PRIVATE ${SNPE_LIB})
    endif ()

    install(FILES ${CMAKE_BINARY_DIR}/vision/examples/${TARGET_NAME} DESTINATION ${CMAKE_BINARY_DIR}/vision_examples/)
    install(FILES ${CMAKE_BINARY_DIR}/vision/src/libvision.so DESTINATION ${CMAKE_BINARY_DIR}/vision_examples/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/test_face.jpg DESTINATION ${CMAKE_BINARY_DIR}/vision_examples/)

    if (USE_SNPE)
        list(APPEND snpe_libs
                ${SNPE_LINK_DIR}/libPlatformValidatorShared.so
                ${SNPE_LINK_DIR}/libSNPE.so
                ${SNPE_LINK_DIR}/libsnpe_dsp_domains_v2_system.so
                ${SNPE_LINK_DIR}/libsnpe_dsp_v66_domains_v2_skel.so
                ${SNPE_LINK_DIR}/libc++_shared.so
                )
        install(FILES ${snpe_libs} DESTINATION ${CMAKE_BINARY_DIR}/vision_examples/)
    endif ()

endif ()

# 在vision_demo进行工程Demo程序的拷贝。需要根据不同的目标产物平台拷贝不同的生成物
message("copy build production ....")
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../res/face.jpg DESTINATION ${VALIB_INSTALL_LIB_DIR})
#install(TARGETS ${DEMO_NAME} RUNTIME DESTINATION ${VALIB_INSTALL_LIB_DIR})
