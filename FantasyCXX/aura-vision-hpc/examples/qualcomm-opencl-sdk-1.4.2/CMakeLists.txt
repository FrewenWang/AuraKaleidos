#--------------------------------------------------------------------------------------
# File: CMakeLists.txt
# Desc:
#
# Author:      QUALCOMM
#
#           Copyright (c) 2017, 2020-2021 QUALCOMM Technologies, Inc.
#                         All Rights Reserved.
#                      QUALCOMM Proprietary/GTDR
#--------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)
project(cl_sdk
        DESCRIPTION "Example usage of Qualcomm's OpenCL extensions."
        HOMEPAGE_URL "https://developer.qualcomm.com/download/adrenosdk/adreno-sdk-opencl.zip")

if(NOT ANDROID)
    message(FATAL_ERROR "Only Android is supported.")
endif()


# Import library dependencies.

set(ANDROID_PATH "" CACHE PATH "Path to the Android Open Source Project.")
if(NOT ANDROID_PATH)
    message(FATAL_ERROR "Please set the ANDROID_PATH CMake variable to /path/to/android/open/source/project.")
endif()

# If the EGL library isn't provided by a parent CMake project, ask about and optionally import it.
if(NOT TARGET EGL)
    set(CLSDK_EGL_LIB "" CACHE FILEPATH "Filepath to \"libEGL.so\".")
    if(EXISTS ${CLSDK_EGL_LIB})
        add_library(EGL SHARED IMPORTED)
        set_target_properties(EGL PROPERTIES IMPORTED_LOCATION ${CLSDK_EGL_LIB})

        set(CLSDK_GRAPHICS_HEADERS "${PROJECT_SOURCE_DIR}/inc" CACHE PATH "Path to \"EGL/egl.h\" and other graphics headers.")
        if(NOT EXISTS ${CLSDK_GRAPHICS_HEADERS})
            message(FATAL_ERROR "Please set the CLSDK_GRAPHICS_HEADERS CMake variable to /path/to where it contains \"EGL/egl.h\".")
        endif()
        target_include_directories(EGL INTERFACE ${CLSDK_GRAPHICS_HEADERS})
    endif()
endif()

# If the GLESv2 library isn't provided by a parent CMake project, ask about and optionally import it.
if(NOT TARGET GLESv2)
    set(CLSDK_GLESv2_LIB "" CACHE FILEPATH "Filepath to \"libGLESv2.so\".")
    if(EXISTS ${CLSDK_GLESv2_LIB})
        add_library(GLESv2 SHARED IMPORTED)
        set_target_properties(GLESv2 PROPERTIES IMPORTED_LOCATION ${CLSDK_GLESv2_LIB})

        set(CLSDK_GRAPHICS_HEADERS "${PROJECT_SOURCE_DIR}/inc" CACHE PATH "Path to \"GLES2/gl2.h\" and other graphics headers.")
        if(NOT EXISTS ${CLSDK_GRAPHICS_HEADERS})
            message(FATAL_ERROR "Please set the CLSDK_GRAPHICS_HEADERS CMake variable to /path/to where it contains \"GLES2/gl2.h\".")
        endif()
        target_include_directories(GLESv2 INTERFACE ${CLSDK_GRAPHICS_HEADERS})
    endif()
endif()

# Dmabuf is the standardized memory heap used in SDK samples, if libdmabufheap is not found
# we cannot build a majority of the SDK samples, but we can still continue if ION is available
if(NOT TARGET dmabufheap)
    add_library(dmabufheap SHARED IMPORTED)
    add_library(vmmem SHARED IMPORTED)
    find_library(DMABUFHEAP_LIBRARY dmabufheap
            PATHS
            ${ANDROID_PATH}/out/target/product/msmnile/system/lib # A640
            ${ANDROID_PATH}/out/target/product/kona/system/lib # A650
            ${ANDROID_PATH}/out/target/product/lahaina/system/lib # A660
            ${ANDROID_PATH}/out/target/product/taro/system/lib # A730
            NO_CMAKE_FIND_ROOT_PATH
            NO_DEFAULT_PATH)
    set_target_properties(dmabufheap PROPERTIES IMPORTED_LOCATION ${DMABUFHEAP_LIBRARY})
    find_library(VMMEM_LIBRARY vmmem
            PATHS
            ${ANDROID_PATH}/out/target/product/lahaina/vendor/lib # A660
            ${ANDROID_PATH}/out/target/product/taro/vendor/lib # A730
            NO_CMAKE_FIND_ROOT_PATH
            NO_DEFAULT_PATH)
    set_target_properties(vmmem PROPERTIES IMPORTED_LOCATION ${VMMEM_LIBRARY})
    find_path(VENDOR_ION_HEADERS
            NAMES linux/msm_ion.h
            HINTS
            ${ANDROID_PATH}/out/target/product/msmnile/obj/KERNEL_OBJ/usr/include # A640
            ${ANDROID_PATH}/out/target/product/kona/obj/KERNEL_OBJ/usr/include # A650
            ${ANDROID_PATH}/out/target/product/lahaina/obj/KERNEL_OBJ/usr/include # A660
            ${ANDROID_PATH}/out/target/product/taro/obj/KERNEL_OBJ/usr/include # A730
            ${ANDROID_PATH}/device/qcom/taro-kernel/kernel-headers/  # A730
            NO_CMAKE_FIND_ROOT_PATH
            NO_DEFAULT_PATH)
    target_include_directories(dmabufheap INTERFACE
            # First: General libion memory management.
            ${ANDROID_PATH}/system/memory/libion/include
            ${ANDROID_PATH}/system/memory/libion/kernel-headers

            # Second: Vendor specific Ion heap IDs and flags to fill in the gaps.
            ${VENDOR_ION_HEADERS}

            ${ANDROID_PATH}/system/libbase/include
            ${ANDROID_PATH}/system/memory/libdmabufheap/include
            ${ANDROID_PATH}/vendor/qcom/proprietary/memory-ship/libvmmem/include)
    target_compile_features(dmabufheap INTERFACE cxx_std_17)
else()
    message(WARNING "libdmabufheap not in ANDROID_PATH, most SDK samples will not be available.")
endif()

if(NOT TARGET ion)
    set(CLSDK_ION_LIB "" CACHE FILEPATH "Filepath to \"libion.so\".")
    set(CLSDK_KERNEL_HEADERS "" CACHE PATH "Path to the built public kernel headers. For example, /path/to/aosp/out/target/product/name/obj/KERNEL_OBJ/usr/include.")
    if(EXISTS "${CLSDK_ION_LIB}" AND EXISTS "${CLSDK_KERNEL_HEADERS}")
        add_library(ion SHARED IMPORTED)
        set_target_properties(ion PROPERTIES IMPORTED_LOCATION ${CLSDK_ION_LIB})
        target_include_directories(ion INTERFACE
                # First: General libion memory management.
                ${ANDROID_PATH}/system/memory/libion/include
                ${ANDROID_PATH}/system/memory/libion/kernel-headers

                # Second: Vendor specific Ion heap IDs and flags to fill in the gaps.
                ${CLSDK_KERNEL_HEADERS})
    else()
        message(WARNING "To enable libion, please set the CLSDK_ION_LIB CMake variable to /path/to/libion.so.")
        message(WARNING "To enable libion, please set the CLSDK_KERNEL_HEADERS CMake variable to /path/to/aosp/out/target/product/name/obj/KERNEL_OBJ/usr/include. This will require building the AOSP's \"kernel\".")
    endif()
endif()

# If the nativewindow library isn't provided by a parent CMake project, ask about and optionally import it.
if(NOT TARGET nativewindow)
    set(CLSDK_NATIVEWINDOW_LIB "" CACHE FILEPATH "Filepath to \"libnativewindow.so\".")
    if(EXISTS ${CLSDK_NATIVEWINDOW_LIB})
        add_library(nativewindow SHARED IMPORTED)
        set_target_properties(nativewindow PROPERTIES IMPORTED_LOCATION ${CLSDK_NATIVEWINDOW_LIB})
        target_include_directories(nativewindow INTERFACE ${ANDROID_PATH}/frameworks/native/libs/nativewindow/include)
    endif()
endif()

# Ask about and alias or import either the OpenCL or OpenCL_system library.
set(CLSDK_OPENCL_LIB "" CACHE FILEPATH "Filepath to \"libOpenCL.so\" or \"libOpenCL_system.so\" or the target name to use if it's provided by a parent CMake project.")
if(TARGET ${CLSDK_OPENCL_LIB})
    add_library(cl_sdk_OpenCL ALIAS ${CLSDK_OPENCL_LIB})
else()
    add_library(cl_sdk_OpenCL SHARED IMPORTED)

    if(NOT EXISTS ${CLSDK_OPENCL_LIB})
        message(FATAL_ERROR "Please set the CLSDK_OPENCL_LIB CMake variable to /path/to/libOpenCL.so or /path/to/libOpenCL_system.so.")
    endif()
    set_target_properties(cl_sdk_OpenCL PROPERTIES IMPORTED_LOCATION ${CLSDK_OPENCL_LIB})

    set(CLSDK_GRAPHICS_HEADERS "${PROJECT_SOURCE_DIR}/inc" CACHE PATH "Path to \"CL/cl.h\" and other graphics headers.")
    if(NOT EXISTS ${CLSDK_GRAPHICS_HEADERS})
        message(FATAL_ERROR "Please set the CLSDK_GRAPHICS_HEADERS CMake variable to /path/to where it contains \"CL/cl.h\".")
    endif()
    target_include_directories(cl_sdk_OpenCL INTERFACE ${CLSDK_GRAPHICS_HEADERS})
endif()

# If the utils library isn't provided by a parent CMake project, ask about and optionally import it.
if(NOT TARGET utils)
    set(CLSDK_UTILS_LIB "" CACHE FILEPATH "Filepath to \"libutils.so\".")
    if(EXISTS ${CLSDK_UTILS_LIB})
        add_library(utils SHARED IMPORTED)
        set_target_properties(utils PROPERTIES IMPORTED_LOCATION ${CLSDK_UTILS_LIB})
        target_include_directories(utils INTERFACE ${ANDROID_PATH}/system/core/libutils/include)
    endif()
endif()

# If the ui library isn't provided by a parent CMake project, ask about and optionally import it.
if(NOT TARGET ui)
    if(TARGET utils)
        set(CLSDK_UI_LIB "" CACHE FILEPATH "Filepath to \"libui.so\".")
        if(EXISTS ${CLSDK_UI_LIB})

            find_path(AIDL_NDK_PLATFORM_SOURCES
                    NAMES aidl/android/hardware/graphics/common/BlendMode.h
                    HINTS
                    ${ANDROID_PATH}/out/soong/.intermediates/hardware/interfaces/graphics/common/aidl/android.hardware.graphics.common-ndk_platform-source/gen/include # A660
                    ${ANDROID_PATH}/prebuilts/vndk/v30/arm64/include/out/soong/.intermediates/hardware/interfaces/graphics/common/aidl/android.hardware.graphics.common-ndk_platform-source/gen/include #A730
                    NO_CMAKE_FIND_ROOT_PATH
                    NO_DEFAULT_PATH)

            find_path(LOGGING_INCLUDE_SOURCES
                    NAMES log/log.h
                    HINTS
                    ${ANDROID_PATH}/system/core/liblog/include # A660
                    ${ANDROID_PATH}/system/logging/liblog/include # A730
                    NO_CMAKE_FIND_ROOT_PATH
                    NO_DEFAULT_PATH)

            add_library(ui SHARED IMPORTED)
            set_target_properties(ui PROPERTIES IMPORTED_LOCATION ${CLSDK_UI_LIB})
            target_include_directories(ui INTERFACE
                    ${ANDROID_PATH}/frameworks/native/libs/nativebase/include
                    ${ANDROID_PATH}/frameworks/native/libs/ui/include
                    ${ANDROID_PATH}/hardware/libhardware/include
                    ${ANDROID_PATH}/out/soong/.intermediates/hardware/interfaces/graphics/common/1.0/android.hardware.graphics.common@1.0_genc++_headers/gen
                    ${ANDROID_PATH}/out/soong/.intermediates/hardware/interfaces/graphics/common/1.1/android.hardware.graphics.common@1.1_genc++_headers/gen
                    ${ANDROID_PATH}/out/soong/.intermediates/hardware/interfaces/graphics/common/1.2/android.hardware.graphics.common@1.2_genc++_headers/gen
                    ${AIDL_NDK_PLATFORM_SOURCES}
                    ${ANDROID_PATH}/system/core/libcutils/include
                    ${LOGGING_INCLUDE_SOURCES}
                    ${ANDROID_PATH}/system/core/libsystem/include
                    ${ANDROID_PATH}/system/libhidl/base/include)
            if(EXISTS ${ANDROID_PATH}/system/libfmq/base) # A730
                target_include_directories(ui INTERFACE
                        ${ANDROID_PATH}/system/libfmq/base
                        ${ANDROID_PATH}/frameworks/native/libs/math/include)
            endif()
            target_compile_features(ui INTERFACE cxx_std_17)
            target_link_libraries(ui INTERFACE utils)
        endif()
    else()
        message(WARNING "The ui library could be imported if the CLSDK_UTILS_LIB CMake variable was set.")
    endif()
endif()

add_library(cl_sdk_common INTERFACE)
target_sources(cl_sdk_common INTERFACE
        src/util/util.cpp
        src/util/half_float.cpp
        src/util/cl_wrapper.cpp)
target_include_directories(cl_sdk_common INTERFACE src/util/include)
target_compile_options(cl_sdk_common INTERFACE -Wall -Werror)
target_link_libraries(cl_sdk_common INTERFACE cl_sdk_OpenCL)

if(TARGET dmabufheap)
    target_link_libraries(cl_sdk_common INTERFACE dmabufheap vmmem)
endif()

# Example binaries ordered by path.

if(TARGET dmabufheap)
    add_executable(cl_sdk_accelerated_convolution src/examples/accelerated_image_ops/accelerated_convolution.cpp)
    target_link_libraries(cl_sdk_accelerated_convolution cl_sdk_common)

    add_executable(cl_sdk_hof_filter src/examples/accelerated_image_ops/hof_filter.cpp)
    target_link_libraries(cl_sdk_hof_filter cl_sdk_common)

    add_executable(cl_sdk_qcom_block_match_sad src/examples/accelerated_image_ops/qcom_block_match_sad.cpp)
    target_link_libraries(cl_sdk_qcom_block_match_sad cl_sdk_common)

    add_executable(cl_sdk_qcom_block_match_ssd src/examples/accelerated_image_ops/qcom_block_match_ssd.cpp)
    target_link_libraries(cl_sdk_qcom_block_match_ssd cl_sdk_common)

    add_executable(cl_sdk_qcom_box_filter_image src/examples/accelerated_image_ops/qcom_box_filter_image.cpp)
    target_link_libraries(cl_sdk_qcom_box_filter_image cl_sdk_common)

    add_executable(cl_sdk_qcom_convolve_image src/examples/accelerated_image_ops/qcom_convolve_image.cpp)
    target_link_libraries(cl_sdk_qcom_convolve_image cl_sdk_common)

    add_executable(cl_sdk_qcom_filter_bicubic src/examples/accelerated_image_ops/qcom_filter_bicubic.cpp)
    target_link_libraries(cl_sdk_qcom_filter_bicubic cl_sdk_common)

    if(TARGET nativewindow)
        add_executable(cl_sdk_ahardwarebuffer_buffer src/examples/basic/ahardwarebuffer_buffer.cpp)
        target_link_libraries(cl_sdk_ahardwarebuffer_buffer cl_sdk_common nativewindow)

        add_executable(cl_sdk_ahardwarebuffer_image src/examples/basic/ahardwarebuffer_image.cpp)
        target_link_libraries(cl_sdk_ahardwarebuffer_image cl_sdk_common nativewindow)
    else()
        message(WARNING "cl_sdk_ahardwarebuffer_buffer could be built if the CLSDK_NATIVEWINDOW_LIB CMake variable was set.")
    endif()

    if(TARGET nativewindow AND TARGET GLESv2 AND TARGET EGL)
        add_executable(cl_sdk_cl_egl_sync src/examples/basic/cl_egl_sync.cpp)
        target_link_libraries(cl_sdk_cl_egl_sync cl_sdk_common nativewindow GLESv2 EGL)
    else()
        message(WARNING "cl_sdk_cl_egl_sync could be built if the CLSDK_NATIVEWINDOW_LIB, CLSDK_GLESv2_LIB, and CLSDK_EGL_LIB CMake variables were set.")
    endif()

    add_executable(cl_sdk_hello_world src/examples/basic/hello_world.cpp)
    target_link_libraries(cl_sdk_hello_world cl_sdk_common)

    add_executable(cl_sdk_hello_world_3_0 src/examples/basic/hello_world_3_0.cpp)
    target_link_libraries(cl_sdk_hello_world_3_0 cl_sdk_common)

    add_executable(cl_sdk_qcom_extended_query_image_info src/examples/basic/qcom_extended_query_image_info.cpp)
    target_link_libraries(cl_sdk_qcom_extended_query_image_info cl_sdk_common)

    add_executable(cl_sdk_qcom_perf_hint src/examples/basic/qcom_perf_hint.cpp)
    target_link_libraries(cl_sdk_qcom_perf_hint cl_sdk_common)

    add_executable(cl_sdk_qcom_reqd_sub_group_size src/examples/basic/qcom_reqd_sub_group_size.cpp)
    target_link_libraries(cl_sdk_qcom_reqd_sub_group_size cl_sdk_common)

    add_executable(cl_sdk_tuning_wgs src/examples/basic/tuning_wgs.cpp)
    target_link_libraries(cl_sdk_tuning_wgs cl_sdk_common)

    add_executable(cl_sdk_dmabuf_buffer_and_image src/examples/dmabuf_buffer_and_image/dmabuf_buffer_and_image.cpp)
    target_link_libraries(cl_sdk_dmabuf_buffer_and_image cl_sdk_common)

    add_executable(cl_sdk_bayer_mipi10_to_rgba src/examples/image_conversions/bayer_mipi10_to_rgba.cpp)
    target_link_libraries(cl_sdk_bayer_mipi10_to_rgba cl_sdk_common)

    add_executable(cl_sdk_compressed_image_nv12 src/examples/image_conversions/compressed_image_nv12.cpp)
    target_link_libraries(cl_sdk_compressed_image_nv12 cl_sdk_common)

    add_executable(cl_sdk_compressed_image_rgba src/examples/image_conversions/compressed_image_rgba.cpp)
    target_link_libraries(cl_sdk_compressed_image_rgba cl_sdk_common)

    add_executable(cl_sdk_compressed_nv12_4r_to_p010 src/examples/image_conversions/compressed_nv12_4r_to_p010.cpp)
    target_link_libraries(cl_sdk_compressed_nv12_4r_to_p010 cl_sdk_common)

    add_executable(cl_sdk_compressed_nv12_to_nv21 src/examples/image_conversions/compressed_nv12_to_nv21.cpp)
    target_link_libraries(cl_sdk_compressed_nv12_to_nv21 cl_sdk_common)

    add_executable(cl_sdk_compressed_nv12_to_p010 src/examples/image_conversions/compressed_nv12_to_p010.cpp)
    target_link_libraries(cl_sdk_compressed_nv12_to_p010 cl_sdk_common)

    add_executable(cl_sdk_compressed_tp10_to_rgba src/examples/image_conversions/compressed_tp10_to_rgba.cpp)
    target_link_libraries(cl_sdk_compressed_tp10_to_rgba cl_sdk_common)

    add_executable(cl_sdk_mipi10_to_unpacked src/examples/image_conversions/mipi10_to_unpacked.cpp)
    target_link_libraries(cl_sdk_mipi10_to_unpacked cl_sdk_common)

    add_executable(cl_sdk_nv12_to_p010 src/examples/image_conversions/nv12_to_p010.cpp)
    target_link_libraries(cl_sdk_nv12_to_p010 cl_sdk_common)

    add_executable(cl_sdk_nv12_to_rgba8888 src/examples/image_conversions/nv12_to_rgba8888.cpp)
    target_link_libraries(cl_sdk_nv12_to_rgba8888 cl_sdk_common)

    if(TARGET ui)
        add_executable(cl_sdk_nv21_to_compressed_nv12 src/examples/image_conversions/nv21_to_compressed_nv12.cpp)
        target_link_libraries(cl_sdk_nv21_to_compressed_nv12 cl_sdk_common ui)
    else()
        message(WARNING "cl_sdk_nv21_to_compressed_nv12 could be built if the CLSDK_UI_LIB CMake variable was set")
    endif()

    if(TARGET ui AND TARGET nativewindow)
        add_executable(cl_sdk_nv12_to_rgb565 src/examples/image_conversions/nv12_to_rgb565.cpp)
        target_link_libraries(cl_sdk_nv12_to_rgb565 cl_sdk_common ui nativewindow)

        add_executable(cl_sdk_nv12_to_nv12_compressed src/examples/image_conversions/nv12_to_nv12_compressed.cpp)
        target_link_libraries(cl_sdk_nv12_to_nv12_compressed cl_sdk_common ui nativewindow)

        add_executable(cl_sdk_nv12_to_nv12_compressed_to_rgba src/examples/image_conversions/nv12_to_nv12_compressed_to_rgba.cpp)
        target_link_libraries(cl_sdk_nv12_to_nv12_compressed_to_rgba cl_sdk_common ui nativewindow)

        add_executable(cl_sdk_tp10_to_compressed_tp10_to_p010 src/examples/image_conversions/tp10_to_compressed_tp10_to_p010.cpp)
        target_link_libraries(cl_sdk_tp10_to_compressed_tp10_to_p010 cl_sdk_common ui nativewindow)
    else()

    endif()

    add_executable(cl_sdk_p010_to_compressed_tp10 src/examples/image_conversions/p010_to_compressed_tp10.cpp)
    target_link_libraries(cl_sdk_p010_to_compressed_tp10 cl_sdk_common)

    add_executable(cl_sdk_p010_to_nv12 src/examples/image_conversions/p010_to_nv12.cpp)
    target_link_libraries(cl_sdk_p010_to_nv12 cl_sdk_common)

    add_executable(cl_sdk_p010_to_rgba src/examples/image_conversions/p010_to_rgba.cpp)
    target_link_libraries(cl_sdk_p010_to_rgba cl_sdk_common)

    add_executable(cl_sdk_rgba_to_nv12 src/examples/image_conversions/rgba_to_nv12.cpp)
    target_link_libraries(cl_sdk_rgba_to_nv12 cl_sdk_common)

    add_executable(cl_sdk_rgba_to_p010 src/examples/image_conversions/rgba_to_p010.cpp)
    target_link_libraries(cl_sdk_rgba_to_p010 cl_sdk_common)

    add_executable(cl_sdk_tp10_to_nv12 src/examples/image_conversions/tp10_to_nv12.cpp)
    target_link_libraries(cl_sdk_tp10_to_nv12 cl_sdk_common)

    add_executable(cl_sdk_unpacked_bayer_to_rgba src/examples/image_conversions/unpacked_bayer_to_rgba.cpp)
    target_link_libraries(cl_sdk_unpacked_bayer_to_rgba cl_sdk_common)

    add_executable(cl_sdk_unpacked_to_mipi10 src/examples/image_conversions/unpacked_to_mipi10.cpp)
    target_link_libraries(cl_sdk_unpacked_to_mipi10 cl_sdk_common)

    add_executable(cl_sdk_buffer_matrix_multiplication src/examples/linear_algebra/buffer_matrix_multiplication.cpp)
    target_link_libraries(cl_sdk_buffer_matrix_multiplication cl_sdk_common)

    add_executable(cl_sdk_buffer_matrix_multiplication_half src/examples/linear_algebra/buffer_matrix_multiplication_half.cpp)
    target_link_libraries(cl_sdk_buffer_matrix_multiplication_half cl_sdk_common)

    add_executable(cl_sdk_buffer_matrix_transpose src/examples/linear_algebra/buffer_matrix_transpose.cpp)
    target_link_libraries(cl_sdk_buffer_matrix_transpose cl_sdk_common)

    add_executable(cl_sdk_convolution src/examples/linear_algebra/convolution.cpp)
    target_link_libraries(cl_sdk_convolution cl_sdk_common)

    add_executable(cl_sdk_fft_matrix src/examples/linear_algebra/fft_matrix.cpp)
    target_link_libraries(cl_sdk_fft_matrix cl_sdk_common)

    add_executable(cl_sdk_image_matrix_multiplication src/examples/linear_algebra/image_matrix_multiplication.cpp)
    target_link_libraries(cl_sdk_image_matrix_multiplication cl_sdk_common)

    add_executable(cl_sdk_image_matrix_multiplication_half src/examples/linear_algebra/image_matrix_multiplication_half.cpp)
    target_link_libraries(cl_sdk_image_matrix_multiplication_half cl_sdk_common)

    add_executable(cl_sdk_image_matrix_transpose src/examples/linear_algebra/image_matrix_transpose.cpp)
    target_link_libraries(cl_sdk_image_matrix_transpose cl_sdk_common)

    add_executable(cl_sdk_matrix_addition src/examples/linear_algebra/matrix_addition.cpp)
    target_link_libraries(cl_sdk_matrix_addition cl_sdk_common)

    add_executable(cl_sdk_qcom_bitreverse src/examples/linear_algebra/qcom_bitreverse.cpp)
    target_link_libraries(cl_sdk_qcom_bitreverse cl_sdk_common)

    add_executable(cl_sdk_qcom_dot_product8 src/examples/linear_algebra/qcom_dot_product8.cpp)
    target_link_libraries(cl_sdk_qcom_dot_product8 cl_sdk_common)

    add_executable(cl_sdk_svm_matrix_multiplication src/examples/linear_algebra/svm_matrix_multiplication.cpp)
    target_link_libraries(cl_sdk_svm_matrix_multiplication cl_sdk_common)

    if(TARGET nativewindow AND TARGET ui AND EXISTS ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc)
        add_executable(cl_sdk_protected_android_hardware_buffer src/examples/protected_memory/protected_android_hardware_buffer.cpp)
        target_include_directories(cl_sdk_protected_android_hardware_buffer PRIVATE
                ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc)
        target_link_libraries(cl_sdk_protected_android_hardware_buffer cl_sdk_common ui nativewindow)
    else()
        message(WARNING "cl_sdk_protected_android_hardware_buffer could be built if the CLSDK_NATIVEWINDOW_LIB and CLSDK_UI_LIB CMake variables were set and ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc were obtained from https://www.codeaurora.org.")
    endif()

    if(TARGET ui AND EXISTS ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc)
        add_executable(cl_sdk_protected_android_native_buffer src/examples/protected_memory/protected_android_native_buffer.cpp)
        target_include_directories(cl_sdk_protected_android_native_buffer PRIVATE
                ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc)
        target_link_libraries(cl_sdk_protected_android_native_buffer cl_sdk_common ui)
    else()
        message(WARNING "cl_sdk_protected_android_native_buffer could be built if the CLSDK_UI_LIB CMake variable was set and ${ANDROID_PATH}/vendor/qcom/opensource/commonsys-intf/display/gralloc were obtained from https://www.codeaurora.org.")
    endif()

    add_executable(cl_sdk_recordable_queues src/examples/recordable_queues/recordable_queues.cpp)
    target_link_libraries(cl_sdk_recordable_queues cl_sdk_common)

    add_executable(cl_sdk_recordable_queues_svm src/examples/recordable_queues/recordable_queues_svm.cpp)
    target_link_libraries(cl_sdk_recordable_queues_svm cl_sdk_common)

    add_executable(cl_sdk_compressed_nv12_vector_image_ops src/examples/vector_image_ops/compressed_nv12_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_compressed_nv12_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_compressed_p010_vector_image_ops src/examples/vector_image_ops/compressed_p010_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_compressed_p010_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_compressed_tp10_vector_image_ops src/examples/vector_image_ops/compressed_tp10_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_compressed_tp10_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_nv12_readwrite_image src/examples/vector_image_ops/nv12_readwrite_image.cpp)
    target_link_libraries(cl_sdk_nv12_readwrite_image cl_sdk_common)

    add_executable(cl_sdk_nv12_vector_image_ops src/examples/vector_image_ops/nv12_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_nv12_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_p010_vector_image_ops src/examples/vector_image_ops/p010_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_p010_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_tp10_vector_image_ops src/examples/vector_image_ops/tp10_vector_image_ops.cpp)
    target_link_libraries(cl_sdk_tp10_vector_image_ops cl_sdk_common)

    add_executable(cl_sdk_protected_dmabuf src/examples/protected_memory/protected_dmabuf.cpp)
    target_link_libraries(cl_sdk_protected_dmabuf cl_sdk_common)
endif()

if(TARGET ion)
    add_library(cl_sdk_ion_standalone INTERFACE)
    target_include_directories(cl_sdk_ion_standalone INTERFACE src/util/include)
    target_compile_options(cl_sdk_ion_standalone INTERFACE -Wall -Werror)
    target_link_libraries(cl_sdk_ion_standalone INTERFACE ion cl_sdk_OpenCL)

    add_executable(cl_sdk_ion_buffer_and_image src/examples/ion_buffer_and_image/ion_buffer_and_image.cpp)
    target_link_libraries(cl_sdk_ion_buffer_and_image cl_sdk_ion_standalone)

    add_executable(cl_sdk_protected_ion src/examples/protected_memory/protected_ion.cpp)
    target_link_libraries(cl_sdk_protected_ion cl_sdk_ion_standalone)
else()
    message(WARNING "ION target is not available so the following targets won't be built: cl_sdk_ion_buffer_and_image, cl_sdk_protected_ion")
endif()
