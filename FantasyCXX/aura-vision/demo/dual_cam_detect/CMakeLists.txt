cmake_minimum_required(VERSION 3.10)

project(dual_cam)

add_definitions(-std=c++11)

add_executable(dual_cam src/main.cpp src/util/draw_util.cpp)

target_include_directories(dual_cam PUBLIC ../../../build/osx-x86_64-Release/install/include)
target_include_directories(dual_cam PUBLIC ../../../third_party/opencv/opencv_2.4.13.4/osx-x86_64/include)

target_link_directories(dual_cam PUBLIC 
    ../../../build/osx-x86_64-Release/install/lib/x86_64)
target_link_directories(dual_cam PUBLIC 
    ../../../third_party/opencv/opencv_2.4.13.4/osx-x86_64/lib
    ../../../third_party/opencv/opencv_2.4.13.4/osx-x86_64/3rdparty)
target_link_libraries(dual_cam vision opencv_calib3d opencv_highgui opencv_imgproc opencv_core libtiff libpng libjpeg libjasper IlmImf)