include_guard(GLOBAL)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

if(NOT DEFINED AURAKALEIDOS_ROOT)
    message(FATAL_ERROR "Set AURAKALEIDOS_ROOT before including AuraOpenCV.cmake")
endif()

set(AURA_OPENCV_VERSION "4.11.0" CACHE STRING "Preferred OpenCV version")
set(AURA_OPENCV_ROOT "" CACHE PATH
    "OpenCV package root; empty selects a matching bundled or system package")

if(AURA_OPENCV_ROOT)
    set(_aura_opencv_root "${AURA_OPENCV_ROOT}")
elseif(NOT OpenCV_DIR)
    set(_aura_opencv_root
        "${AURAKALEIDOS_ROOT}/FantasyCXX/3rdparty/opencv/lib/v${AURA_OPENCV_VERSION}/${TARGET_OS}-${TARGET_ARCH}-${AURA_DEPENDENCY_VARIANT}")
endif()

if(_aura_opencv_root AND NOT OpenCV_DIR)
    if(EXISTS "${_aura_opencv_root}/lib/cmake/opencv4/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_aura_opencv_root}/lib/cmake/opencv4")
    elseif(EXISTS "${_aura_opencv_root}/sdk/native/jni/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_aura_opencv_root}/sdk/native/jni")
    elseif(EXISTS "${_aura_opencv_root}/share/opencv4/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_aura_opencv_root}/share/opencv4")
    endif()
endif()

# Some bundled static OpenCV exports reference this header-only target without
# shipping Eigen's CMake package. The consuming projects do not use Eigen APIs.
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
endif()

if(EXISTS "${_aura_opencv_root}/lib/libopencv_world.a" OR
   EXISTS "${_aura_opencv_root}/lib/opencv_world.lib" OR
   EXISTS "${_aura_opencv_root}/sdk/native/staticlibs/${TARGET_ARCH}/libopencv_world.a")
    find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS world)
else()
    find_package(OpenCV 4 CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
endif()

message(STATUS "[AuraOpenCV] version=${OpenCV_VERSION}")
message(STATUS "[AuraOpenCV] config=${OpenCV_DIR}")
