# Qualcomm OpenCL SDK

## What is this?

This SDK includes extension documents, header files and code samples that show how to use OpenCL to implement image processing, linear algebra and Machine Learning use cases.
It also has additional samples that show how to leverage specific Qualcomm OpenCL extensions.

These samples have been tested on an SM8450 device running Android S.

Please note that samples and header files for the cl_qcom_ml_ops extension are distributed as part of the Adreno OpenCL_ML SDK which is available separately.

## Building for Android 12

These steps work on any OS, such as Linux and Windows, that supports the Android
NDK.

1. OpenCL applications can run in vendor or system partitions on Android.
   To run in vendor partition, they should link to libOpenCL.so and when running in the system partition they should link to libOpenCL_system.so.

   Please use the CMake variable CLSDK_OPENCL_LIB to control this linkage.

   libOpenCL.so and libOpenCL_system.so can be obtained from a built AOSP tree or from an Android device.

2. Download the [Android Open Source Project](https://source.android.com) and
   build "kernel" for necessary header files.

3. Setup the Android NDK. The easiest way is to use
   [Android Studio](https://developer.android.com/studio).
   * On the welcome screen, click Configure > SDK Manager.
   * Under SDK Tools, tick "Show Package Details" and install
     NDK (Side by side) > 23.0.7599858. Newer would likely work too.

4. Build the OpenCL SDK with CMake.
   * Open this directory in [CMake GUI](https://cmake.org) as a source directory.
   * Choose an out-of-source CMake build directory.
   * Add these CMake variables before asking CMake to configure:
      * `CMAKE_TOOLCHAIN_FILE=/path/to/android-sdk/ndk/<version>/build/cmake/android.toolchain.cmake`
      * `ANDROID_PLATFORM=30`
      * `ANDROID_ABI=arm64-v8a`
   * Optionally click Configure to make setting the following variables easier.
   * Select a generator capable of building for Linux, such as
       [Ninja](https://ninja-build.org).
   * Set these CMake variables:
      * `CLSDK_OPENCL_LIB=/path/to/libOpenCL_system.so` (or libOpenCL.so)
      * `ANDROID_PATH=/path/to/android/open/source/project`
   * Optionally set some or all of these CMake variables to build extra examples. These requirements are also listed against specific samples that need them.
     * `CLSDK_NATIVEWINDOW_LIB=/path/to/libnativewindow.so`
     * `CLSDK_UI_LIB=/path/to/libui.so`
     * `CLSDK_UTILS_LIB=/path/to/libutils.so`
     * `CLSDK_EGL_LIB=/path/to/libEGL.so`
     * `CLSDK_GLESv2_LIB=/path/to/libGLESv2.so`
   * Optionally set all these CMake variables to build the ION specific samples:
     * `CLSDK_KERNEL_HEADERS=/path/to/aosp/out/target/product/name/obj/KERNEL_OBJ/usr/include`
         (It must contain `linux/msm_ion.h`.)
     * `CLSDK_ION_LIB=/path/to/libion.so`
   * Click Generate.
   * Click Finish.
   * On the command line:
      * `cmake --build /path/to/build/directory`
   * A few examples have additional requirements that if unsatisfied will appear
     as CMake configuration errors. Follow their guidance to continue.

## Usage

Building will produce a set of binaries. Run each one without arguments to see
a help message and description of what it does. Most binaries take an input
image in the format described above -- several sample images are given in the
example_images directory, which contains arbitrary data (e.g. it is not
visually interesting).

## Descriptions

### src/examples/accelerated_image_ops directory

The examples in this directory show usage of various built-in functions that are part of the cl_qcom_accelerated_image_ops extension.

#### accelerated_convolution.cpp

Demonstrates a gaussian blur filter using the qcom_convolve_imagef built-in function. Additionally, demonstrates that images and buffers can use the same underlying dmabuf memory, by writing to an OpenCL buffer and reading results from an image.

#### hof_filter.cpp

Demonstrates the use of higher order separable and non-separable filters with the qcom_convolve_imagef built-in function.

#### qcom_block_match_sad.cpp, qcom_block_match_ssd.cpp, qcom_box_filter_image.cpp, qcom_convolve_image.cpp

These examples all demonstrate basic usage for the named built-in extension functions.
Look here for minimal examples of how to use the extensions.

#### qcom_filter_bicubic.cpp

This example shows how to upscale an image using the Qualcomm qcom_filter_bicubic extension.

### src/examples/basic directory

#### ahardwarebuffer_buffer.cpp, ahardwarebuffer_image.cpp

In order to activate the build, the following CMake variable needs to be set.

`CLSDK_NATIVEWINDOW_LIB=/path/to/libnativewindow.so`

These examples demonstrate the use of cl_qcom_android_ahardwarebuffer_host_ptr using OpenCL buffers and images.

#### cl_egl_sync.cpp

This example demonstrates the use of cl_khr_gl_sharing to sync between OpenCL and GL/EGL

Activating the build for it requires the following CMake variables to be set:

`CLSDK_NATIVEWINDOW_LIB=/path/to/libnativewindow.so`

`CLSDK_UI_LIB=/path/to/libui.so`

`CLSDK_UTILS_LIB=/path/to/libutils.so`

Further, the build also requires GLES2 headers to be downloaded from https://www.khronos.org/registry/OpenGL/api/GLES2/ into `inc/GLES2` within the SDK.

#### hello_world.cpp

A very basic example to test building and running OpenCL. It simply copies one buffer to another.

#### hello_world_3_0.cpp

A very basic example to test building and running OpenCL 3.0. It simply copies one buffer to another.

#### qcom_extended_query_image_info.cpp

Demonstrates the use of the cl_qcom_extended_query_image_info extension.

#### qcom_perf_hint.cpp

Demonstrates the use of cl_qcom_perf_hint.

#### qcom_reqd_sub_group_size.cpp

Demonstrates usage of the cl_qcom_reqd_sub_group_size extension. It can
be used to manually tune the sub-group size for performance.

#### tuning_wgs.cpp

Demonstrates a method to find an optimal local work group size (in terms of least execution time) for a kernel.

### src/examples/dmabuf_buffer_and_image directory

This sample showcases how to use dmabuf-backed buffers and images. Both examples simply copy a buffer or image.
This is a standalone sample and does not require the wrapper.

### src/examples/image_conversions directory

Examples in this directory show conversions to and from various image formats.
In many cases, this involves an additional initial pass to convert a linear image to the desired source format for conversion or to convert the destination image of the conversion to a linear image for dumping to file.

Further explanation of some specific samples below:

#### bayer_mipi10_to_rgba.cpp and unpacked_bayer_to_rgba.cpp

Demonstrates a  scheme for de-mosaicing. The former uses the packed `MIPI10` format, and the latter
uses an unpacked 10-bit format (held in a 16-bit int with 6 bits unused). Both
use Bayer-ordered images to exploit the GPU's interpolation capabilities without
mixing different color channels. The destination format has 8-bits per channel,
so some precision is lost.

#### compressed_image_nv12.cpp and compressed_image_rgba.cpp

Demonstrates use of compressed images using Qualcomm extensions to OpenCL.
The input image is compressed and then decompressed, with the result written
to the specified output file for comparison. (The compression is not lossy so
they are identical.)

Compressed image formats may be saved to disk, however be advised that the format
is specific to each GPU.

The two examples show compression for `NV12` and `RGBA` images.

#### mipi10_to_unpacked.cpp and unpacked_to_mipi10.cpp

Demonstrates using the MIPI10 data format with a single-channel `CL_R` order. The former converts a
packed MIPI10 image into an unpacked 10-bit image. The latter shows the
unpacked-to-packed conversion.

#### nv21_to_compressed_nv12.cpp

Unlike the other image conversion samples that use dmabuf, this sample uses Android Native Buffer.
Activating the build requires the following CMake variables to be defined

`CLSDK_UI_LIB=/path/to/libui.so`

`CLSDK_UTILS_LIB=/path/to/libutils.so`

#### nv12_to_nv12_compressed.cpp, nv12_to_nv12_compressed_to_rgba.cpp, tp10_to_compressed_tp10_to_p010.cpp, nv12_to_rgb565.cpp

These image samples perform image conversions as indicated by the name, however instead of using dmabuf they use
Android Hardware Buffer (AHB). Activating the build requires the following CMake variables to be defined

`CLSDK_NATIVEWINDOW_LIB=/path/to/libnativewindow.so`

`CLSDK_UI_LIB=/path/to/libui.so`

`CLSDK_UTILS_LIB=/path/to/libutils.so`

### src/examples/ion_buffer_and_image directory

This is the ION counterpart for the dmabuf_buffer_and_image sample, it shows how to use the
io-coherent and uncached host cache policies for ION buffers and images. Other than
the parameters used to create the ION buffers, there is no difference in the
host or kernel code for io-coherent or uncached policies. This is a standalone sample and does not require the wrapper.
Activating the build requires the following CMake variables to be defined

`CLSDK_KERNEL_HEADERS=/path/to/aosp/out/target/product/name/obj/KERNEL_OBJ/usr/include`

`CLSDK_ION_LIB=/path/to/libion.so`

### src/examples/linear_algebra

Demonstrates some basic linear algebra operations:

* Matrix addition
* Matrix multiplication
* Matrix transposition

The transposition and multiplication examples come in two flavors, one using
OpenCL buffers and another that packs the matrices into 2D images. It is not a
foregone conclusion that using an image or a buffer will enjoy better
performance in any given use case, so generally one must try and see what works
best.

The image versions of both examples pad irregularly sized matrices, both because
images have per-row alignment requirements and because this permits an efficient
tiled algorithm to be applied uniformly. This approach can use substantially
more memory than the buffer-based version.

In contrast, the buffer versions do not pad the input matrices. They use an
efficient tiled algorithm where possible, and a less efficient algorithm to
calculate the remaining portion of the output not covered by the tiled
algorithm.

The multiplication examples additionally have a "half" variant, that
demonstrates using the 16-bit half-float data type. The input, output and
arithmetic all use half-floats. This can be a significant performance advantage,
although it introduces more error. One may mix use of floats and half-floats to
achieve the desired performance/accuracy trade off.


#### convolution.cpp

Demonstrates efficient convolution without the use of built-in extension functions.

#### fft_matrix.cpp

The example computes the 2-dimensional fast Fourier transform (2D FFT) of a
matrix using the in-place Cooley-Tukey algorithm. First in the
"row pass" each work group calculates the 1D FFT of a row. The intermediate
result is transposed and written to global memory. This procedure is
then repeated in a "column pass" that acts on the rows of the result
of the first pass. The result is transposed, producing the final result.
Calculating the 1D FFTs back-to-back in this way is equal to the 2D FFT.

The sample takes a real-valued matrix as input (specified as
below), and produces two matrices as the output holding the real and imaginary
parts of the FFT.

#### qcom_bitreverse.cpp

Demonstrates the usage of cl_qcom_bitreverse extension by reversing the bits of an array of prime numbers.

#### qcom_dot_product8.cpp

Demonstrates the usage of cl_qcom_dot_product8 extension.

#### svm_matrix_multiplicaton.cpp

Processes data on the host and device without using expensive clFinish
synchronization. Instead locks are used on shared virtual memory allocations.

### src/examples/protected_memory

These examples create and initialize GPU memory that's protected from being read
or written by the CPU. One of several backing stores is used by each example.

#### protected_android_hardware_buffer.cpp
Demonstrates the cl_qcom_protected_context and cl_qcom_android_ahardwarebuffer_host_ptr extensions. Activating the build for cl_sdk_protected_android_hardware_buffer requires the following CMake variables to be set:

`CLSDK_NATIVEWINDOW_LIB=/path/to/libnativewindow.so`

`CLSDK_UI_LIB=/path/to/libui.so`

`CLSDK_UTILS_LIB=/path/to/libutils.so`

#### protected_android_native_buffer.cpp
Demonstrates the cl_qcom_protected_context and cl_qcom_android_native_buffer_host_ptr extensions. Activating the build for cl_sdk_protected_android_native_buffer requires the following CMake variables to be set.

`CLSDK_UI_LIB=/path/to/libui.so`

`CLSDK_UTILS_LIB=/path/to/libutils.so`

#### protected_dmabuf.cpp
Demonstrates the cl_qcom_protected_context and cl_qcom_dmabuf_host_ptr extensions.

#### protected_ion.cpp
Demonstrates the cl_qcom_protected_context and cl_qcom_ion_host_ptr extensions. Activating the build for
cl_sdk_protected_ion requires the following CMake variables to be set.

`CLSDK_KERNEL_HEADERS=/path/to/aosp/out/target/product/name/obj/KERNEL_OBJ/usr/include`

`CLSDK_ION_LIB=/path/to/libion.so`

### src/examples/recordable_queues directory

The recordable queues sample demonstrates recording multiple kernel enqueues.
It multiplies two simple matrices, and then increments every element of the
result many times.  The increment kernel increments by one for every run.
Additionally it shows how a single kernel parameter can be updated without
re-recording the entire enqueue sequence.

### src/examples/vector_image_ops directory

All examples in this directory demonstrate a variety of kernels using vector
read and write operations for the given image formats.

## Image data format

Input and output images have the following format, where multi-byte data types
are written with the least significant byte first:

* 4 bytes: plane width in pixels (unsigned integer)
* 4 bytes: plane height (unsigned integer)
* 4 bytes: OpenCL channel data type.
* 4 bytes: OpenCL channel order.
* N bytes: pixel data, where N is dependent on the preceding four values.

## Matrix data format

Matrices used by the examples in the `linear_algebra` directory have the
following plain text format:

* Two integers separated by whitespace indicating the number of columns and rows
  of the matrix.
* A sequence of whitespace-separated floating point element values in row-major
  order.

For example, the following represents a 3x2 matrix:

```
2 3
1.0 2.0
3.1 4.1
6   0
```
## Bayer-ordered images

Bayer-ordered images have one red, green or blue value per pixel, and the pixels
are interleaved in a mosaic pattern. In order to get an equivalent RGB image
one must "demosaic" the image by interpolating the missing red, green, and blue
values. Bayer-ordered images are addressed by 2x2 blocks of such pixels, where
each block has one red and blue value, and two green values. A Bayer-ordered
image may also be addressed as a single-channel (`CL_R`) image to get one color
channel at a time.
