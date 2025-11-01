cmake_minimum_required(VERSION 3.10)

project(face_detect)

add_executable(${PROJECT_NAME} main.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC -std=c++11)

target_include_directories(${PROJECT_NAME} PUBLIC libs/vision/include)
target_include_directories(${PROJECT_NAME} PUBLIC libs/opencv/include)

target_link_directories(${PROJECT_NAME} PUBLIC libs/vision/lib)
target_link_directories(${PROJECT_NAME} PUBLIC libs/opencv/lib)

target_link_libraries(${PROJECT_NAME} vision opencv_highgui opencv_imgproc opencv_core)