set(UT_NAME va_cv_ut)


file(GLOB_RECURSE test_srcs LIST_DIRECTORIES FALSE
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE test_heads LIST_DIRECTORIES FALSE
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

link_directories(
                        ${OPENCV_3RDPARTY_LINK_DIR}
                        ${OPENCV_LINK_DIR}
                        )
                        
add_executable(${UT_NAME} ${test_srcs})
target_compile_options(${UT_NAME} PRIVATE -std=c++14)
target_compile_options(${UT_NAME} PRIVATE -O3 )

target_include_directories(${UT_NAME}
                            PRIVATE
                            ${test_heads}
                            ${OPENCV_INCLUDE_DIR}
                            )

target_link_libraries(${UT_NAME}
                        PRIVATE
                        ${OPENCV_LIB}
                        vision
                        )

install(FILES ${CMAKE_BINARY_DIR}/vision/src/vacv/test/${UT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/176x144.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/284x214.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/640x360.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1280x720.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1280x720_grey.jpg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/1920x1080.jpeg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/res/2560x1440.jpeg DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/res)
if (ICP_TARGET_OS MATCHES osx)
    install(FILES ${CMAKE_BINARY_DIR}/vision/src/libvision.dylib DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
else()
    install(FILES ${CMAKE_BINARY_DIR}/vision/src/libvision.so DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
endif()
if (USE_SNPE)
    list(APPEND snpe_libs
            ${SNPE_LINK_DIR}/libPlatformValidatorShared.so
            ${SNPE_LINK_DIR}/libSNPE.so
            ${SNPE_LINK_DIR}/libsnpe_dsp_domains_v2_system.so
            ${SNPE_LINK_DIR}/libsnpe_dsp_v66_domains_v2_skel.so
            ${SNPE_LINK_DIR}/libc++_shared.so
            )
    install(FILES ${snpe_libs} DESTINATION ${CMAKE_BINARY_DIR}/vacv_test/)
endif ()