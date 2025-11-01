
list(APPEND vision_srcs main.cpp)


set(VISION_NAME vision_vast)

add_executable(
        ${VISION_NAME}
        ${vision_srcs}
)

target_compile_options(${VISION_NAME} PRIVATE -std=c++11)
target_compile_options(${VISION_NAME} PRIVATE -O3)

target_include_directories(
        ${VISION_NAME}
        PRIVATE
        ${JSON_INCLUDE_DIR}
        #        ${OPENCV_INCLUDE_DIR}
)

target_link_directories(
        ${VISION_NAME}
        PRIVATE
        ${JSON_LINK_DIR}
)


target_link_libraries(
        ${VISION_NAME}
        PRIVATE
        #        ${OPENCV_LIB}
        ${JSON_LIB}
        #        auralib-vision
)