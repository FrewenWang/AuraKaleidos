#include <algorithm>
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/time.h>
#include "cl_engine.h"
#include "common_util.h"
#include <cmath>

using namespace gaussian;

#define LOOP_COUNT (1)

#define CHECK_ERROR(err)                                                               \
    if (err != CL_SUCCESS)                                                                  \
    {                                                                                       \
        fprintf(stderr, "OpenCL Error: %d @ %s:%d\n", err, __FILE__, __LINE__); exit(1);    \
        exit(err);                                                                              \
    }

bool USE_SVM = false;

void PrintDuration(timeval *start, const char *str, int loop_count);

void TestCpuProcess(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst, size_t ostride);

void TestBuffer(const char *kernel_src, const char *kernel_name);

void TestImage(const char *kernel_src, const char *kernel_name);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <type>\n"
                "\t<type> = buffer | image\n"
                "This sample shows how to use opencl buffer and image", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string arg(argv[1]);
    if (arg != "buffer" && arg != "image")
    {
        fprintf(stderr, "Usage: %s <type>\n"
                "\t<type> = buffer | image\n"
                "This sample shows how to use opencl buffer and image", argv[0]);
        return EXIT_FAILURE;
    }

    const bool is_buffer = arg == "buffer";
    const char *kernel_src = "gaussian.cl";
    if (argc >= 3)
    {
        kernel_src = argv[2];
    }

    if (is_buffer)
    {
        TestBuffer(kernel_src, "Gauss3x3u8c1Buffer");
    } else
    {
        TestImage(kernel_src, "Gauss3x3u8c1Image");
    }
    return 0;
}

void PrintDuration(timeval *begin, const char *function_name, int loop_count)
{
    if (NULL == begin || NULL == function_name)
    {
        printf("begin or function_name NULL \n");
        return;
    }
    timeval current;
    gettimeofday(&current, NULL);
    uint64_t time_in_microseconds = (current.tv_sec - begin->tv_sec) * 1e6 + (current.tv_usec - begin->tv_usec);
    printf("%s consume average time: %ld us\n", function_name, time_in_microseconds / loop_count);
}

void TestBuffer(const char *kernel_src, const char *kernel_name)
{
    if (NULL == kernel_src || NULL == kernel_name)
    {
        printf("kernel_src or kernel_name NULL \n");
        return;
    }
    printf("TestBuffer called with kernel_src:%s, kernel_name: %s\n", kernel_src, kernel_name);
    CLEngine engine;

    cl_int err = engine.QueryPlatforms();
    CHECK_ERROR(err);

    err = engine.QueryDevices();
    CHECK_ERROR(err);

    err = engine.CreateContext();
    CHECK_ERROR(err);

    err = engine.CreateCommandQueue();
    CHECK_ERROR(err);

    std::string source = CommonUtil::ClReadKernelSource(kernel_src);
    engine.CreateProgram(source.data());
    CHECK_ERROR(err);

    engine.CreateKernel(kernel_name);
    CHECK_ERROR(err);

    cl_context context = engine.GetContext();
    cl_command_queue command_queue = engine.GetCommandQueue();
    cl_kernel kernel = engine.GetKernel();
    cl_mem buffer_src;
    cl_mem buffer_dst;

    int width                        = 4095;
    int height                       = 2161;
    int istride                      = width;   // input data stride width 4095
    int ostride                      = width;   // ouput data stride width 4095
    int gauss_kernel[3]              = {1, 2, 1};
    int shift                        = 4;
    cl_uint buffer_size              = width * height * sizeof(cl_uchar);
    cl_uchar *host_src_matrix        = static_cast<cl_uchar *>(malloc(buffer_size));
    cl_uchar *host_gaussian_matrix   = static_cast<cl_uchar *>(malloc(buffer_size));
    cl_uchar *device_gaussian_matrix = static_cast<cl_uchar *>(malloc(buffer_size));
    memset(device_gaussian_matrix, 0, buffer_size);
    CommonUtil::InitGray(host_src_matrix, width, height);
    printf("Matrix Width =%d Height=%d buffer_size:%u \n", width, height, buffer_size);

    size_t local_work_size[3];
    CommonUtil::GenLocalSize(local_work_size);
    // 因为再进行kernel函数进行计算的时候，第一列和最后一列是不进行卷积的处理。
    // 那么除去这两列。剩下的我们分成六组进行训练。
    size_t cl_process_col      = (width - 2) / 6;    // 682  682*6 = 4092
    size_t cl_process_row      = height;
    size_t remain_col_index    = cl_process_col * 6; // 4092
    /// 因为是每个内核处理6个数据，所以我们只需要 682 * 2161 个工作项。
    /// 每个工作组是 32*32个工作项（2.0之后不需要强制global_work_size和local_work_size整数倍对齐）
    size_t global_work_size[2] = {cl_process_col, cl_process_row};
    printf("global_work_size=(%zu,%zu), local_work_size=(%zu,%zu),remain_col_index=%zu \n",
           global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1], remain_col_index);

    // test cpu gaussian blur cost
    TestCpuProcess(host_src_matrix, width, height, width, host_gaussian_matrix, width);

    if (USE_SVM)
    {
        cl_uchar *svm_input  = static_cast<cl_uchar *>(clSVMAlloc(context,CL_MEM_READ_WRITE, buffer_size, 0));
        cl_uchar *svm_output = static_cast<cl_uchar *>(clSVMAlloc(context,CL_MEM_READ_WRITE, buffer_size, 0));
        memcpy(svm_input, host_src_matrix, buffer_size);

        err = clSetKernelArgSVMPointer(kernel, 0, svm_input);
        err |= clSetKernelArg(kernel, 1, sizeof(height), &height);
        err |= clSetKernelArg(kernel, 2, sizeof(width), &width);
        err |= clSetKernelArg(kernel, 3, sizeof(istride), &istride);
        err |= clSetKernelArg(kernel, 4, sizeof(ostride), &ostride);
        err |= clSetKernelArgSVMPointer(kernel, 5, svm_output);
        CHECK_ERROR(err);
        cl_mem buffer_gauss_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 3,
                                                    gauss_kernel, &err);
        CHECK_ERROR(err);
        err |= clSetKernelArg(kernel, 6, sizeof(buffer_gauss_kernel), &buffer_gauss_kernel);
        CHECK_ERROR(err);
        err |= clSetKernelArg(kernel, 7, sizeof(shift), &shift);
        CHECK_ERROR(err);

        timeval start{};
        gettimeofday(&start, NULL);
        for (int i = 0; i < LOOP_COUNT; i++)
        {
            cl_event kernel_event = NULL;
            err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL,
                                             &kernel_event);
            CHECK_ERROR(err);
            err = clWaitForEvents(1, &kernel_event);
            // CHECK_ERROR(err_num, "ClWaitForEvents");
            engine.GetProfilingInfo(kernel_event);
            clReleaseEvent(kernel_event);
        }
        clFinish(command_queue);
        PrintDuration(&start, "OpenCL Gaussian", LOOP_COUNT);

        memcpy(device_gaussian_matrix, svm_output, buffer_size);
        clSVMFree(context, svm_input);
        clSVMFree(context, svm_output);
    } else
    {
        buffer_src = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, host_src_matrix,
                                    &err);
        CHECK_ERROR(err);
        buffer_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, &err);
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel, 0, sizeof(buffer_src), &buffer_src);
        err |= clSetKernelArg(kernel, 1, sizeof(height), &height);
        err |= clSetKernelArg(kernel, 2, sizeof(width), &width);
        err |= clSetKernelArg(kernel, 3, sizeof(istride), &istride);
        err |= clSetKernelArg(kernel, 4, sizeof(ostride), &ostride);
        err |= clSetKernelArg(kernel, 5, sizeof(buffer_dst), &buffer_dst);
        cl_mem buffer_gauss_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 3,
                                                    gauss_kernel, &err);
        err |= clSetKernelArg(kernel, 6, sizeof(buffer_gauss_kernel), &buffer_gauss_kernel);
        err |= clSetKernelArg(kernel, 7, sizeof(shift), &shift);
        CHECK_ERROR(err);

        timeval start{};
        gettimeofday(&start, NULL);
        for (int i = 0; i < LOOP_COUNT; i++)
        {
            cl_event kernel_event = NULL;
            err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size,
                                             local_work_size, 0, NULL, &kernel_event);
            // CHECK_ERROR(err_num, "ClEnqueueNDRangeKernel");
            err = clWaitForEvents(1, &kernel_event);
            // CHECK_ERROR(err_num, "ClWaitForEvents");
            engine.GetProfilingInfo(kernel_event);
            clReleaseEvent(kernel_event);
        }
        PrintDuration(&start, "OpenCL Gaussian", LOOP_COUNT);

        err = clEnqueueReadBuffer(command_queue, buffer_dst, CL_TRUE, 0, buffer_size, device_gaussian_matrix,
                                      0, NULL, NULL);
        CHECK_ERROR(err);

        clReleaseMemObject(buffer_src);
        buffer_src = NULL;
        clReleaseMemObject(buffer_dst);
        buffer_dst = NULL;
    }

    err = CommonUtil::Gauss3x3Sigma0U8C1RemainData(host_src_matrix, remain_col_index, height, width,
                                                       istride, ostride, device_gaussian_matrix);
    CHECK_ERROR(err);

    bool copare_ret = CommonUtil::ImgDataCompare(host_gaussian_matrix, device_gaussian_matrix, width, height);
    if (copare_ret)
    {
        printf("BufferDataCompare A and B matched!!!\n");
    }

    free(host_src_matrix);
    host_gaussian_matrix = NULL;
    free(host_gaussian_matrix);
    host_gaussian_matrix = NULL;
    free(device_gaussian_matrix);
    host_gaussian_matrix = NULL;
    engine.ClRelease();
}

void TestImage(const char *kernel_src, const char *kernel_name)
{
    if (NULL == kernel_src || NULL == kernel_name)
    {
        printf("kernel_src or kernel_name NULL \n");
        return;
    }
    printf("TestImage2D called with kernel_src:%s, kernel_name: %s\n", kernel_src, kernel_name);
    CLEngine engine;

    cl_int err = engine.QueryPlatforms();
    CHECK_ERROR(err);

    err = engine.QueryDevices();
    CHECK_ERROR(err);

    err = engine.CreateContext();
    CHECK_ERROR(err);

    err = engine.CreateCommandQueue();
    CHECK_ERROR(err);

    std::string source = CommonUtil::ClReadKernelSource(kernel_src);
    engine.CreateProgram(source.data());
    CHECK_ERROR(err);

    engine.CreateKernel(kernel_name);
    CHECK_ERROR(err);

    cl_context context             = engine.GetContext();
    cl_command_queue queue         = engine.GetCommandQueue();
    cl_kernel kernel               = engine.GetKernel();
    ClDeviceInfo device_info       = engine.GetDevicesInfo();
    cl_device_id device            = device_info.device_id;
    int width                      = 4095;
    int height                     = 2161;
    int image_pitch                = device_info.image_pitch_alignment;
    int aligned_width              = (width * sizeof(cl_uchar) + image_pitch - 1) / image_pitch * image_pitch;
    int aligned_height             = height;
    int gauss_kernel[2]            = {1, 2};
    int shift                      = 4;
    cl_uint buffer_size            = width * height * sizeof(cl_uchar);
    cl_uint aligned_buffer_size    = aligned_width * aligned_height * sizeof(cl_uchar);
    unsigned char *input           = static_cast<unsigned char *>(malloc(buffer_size));
    unsigned char *output_cpu      = static_cast<unsigned char *>(malloc(buffer_size));
    unsigned char *output_opencl   = static_cast<unsigned char *>(malloc(buffer_size));
    printf("Matrix width=%d,height=%d,aligned_width=%d,aligned_height=%d buffer_size=%u aligned_buffer_size=%d\n",
           width, height, aligned_width, aligned_height, buffer_size, aligned_buffer_size);

    CommonUtil::InitGray(input, width, height);

    TestCpuProcess(input, width, height, width, output_cpu, width);
    // align input image data
    unsigned char *aligned_input = static_cast<unsigned char *>(malloc(aligned_width * aligned_height));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < aligned_width; x++)
        {
            int srcX = CommonUtil::Reflect(x, width);
            aligned_input[y * aligned_width + x] = input[y * width + srcX];
        }
    }

    //local_work_size should not greater than CL_KERNEL_WORK_GROUP_SIZE 1024
    int cl_process_col         = ((width - 2) / 4);
    int cl_process_row         = height;
    int remain_col_index       = cl_process_col * 4;
    size_t global_work_size[2] = {static_cast<size_t>(cl_process_col), static_cast<size_t>(cl_process_row)};
    size_t local_work_size[3];
    CommonUtil::GenLocalSize(local_work_size);

    // create image format & image desc
    cl_image_format image_format{};
    // memset(&image_format, 0, sizeof(cl_image_format));
    image_format.image_channel_order     = CL_RGBA;
    image_format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc image_desc{};
    // memset(&image_desc, 0, sizeof(cl_image_desc));
    image_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width     = aligned_width >> 2;
    image_desc.image_height    = aligned_height;
    image_desc.image_row_pitch = aligned_width;

    printf("cl_process_col = %d, remain_col_index = %d\n", cl_process_col, remain_col_index);
    printf("global_work_size = (%zu,%zu)\n", global_work_size[0], global_work_size[1]);
    printf("local_work_size = (%zu,%zu)\n", local_work_size[0], local_work_size[1]);
    printf("image_desc.image_type = %d\n", image_desc.image_type);
    printf("image_desc.image_width = %lu\n", image_desc.image_width);
    printf("image_desc.image_height = %lu\n", image_desc.image_height);
    printf("image_desc.image_row_pitch =  %lu\n", image_desc.image_row_pitch);
    cl_mem input_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       &image_format, &image_desc, aligned_input, &err);
    CHECK_ERROR(err);

    // create image format & image desc
    cl_image_format out_image_format{};
    out_image_format.image_channel_order     = CL_RGBA;
    out_image_format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc out_image_desc{};
    // memset(&image_desc, 0, sizeof(cl_image_desc));
    out_image_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    out_image_desc.image_width     = aligned_width >> 2;
    out_image_desc.image_height    = aligned_height;
    out_image_desc.image_row_pitch = 0;
    cl_mem output_image = clCreateImage(context, CL_MEM_WRITE_ONLY,
                                   &out_image_format, &out_image_desc, NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 1, sizeof(height), &height);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 2, sizeof(width), &width);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 3, sizeof(width), &width);
    CHECK_ERROR(err);
    cl_mem kernel_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(gauss_kernel), gauss_kernel, &err);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 4, sizeof(kernel_buffer), &kernel_buffer);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 5, sizeof(shift), &shift);
    CHECK_ERROR(err);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &output_image);
    CHECK_ERROR(err);

    timeval start{};
    gettimeofday(&start, NULL);
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        cl_event kernel_event = NULL;
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                     local_work_size, 0, NULL, &kernel_event);
        CHECK_ERROR(err);
        err = clWaitForEvents(1, &kernel_event);
        CHECK_ERROR(err);
        engine.GetProfilingInfo(kernel_event);
        err = clReleaseEvent(kernel_event);
        CHECK_ERROR(err);
    }

    size_t origin[3] = {0,0,0}, region[3] = {(size_t)aligned_width, (size_t)(height), 1};
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin, region, 0, 0, output_opencl, 0, NULL, NULL);
    CHECK_ERROR(err);

    PrintDuration(&start, "OpenCL Gaussian", LOOP_COUNT);


    err = CommonUtil::Gauss3x3Sigma0U8C1RemainData(input, remain_col_index, height, width,
                                                   width, width, output_opencl);
    bool copare_ret = CommonUtil::ImgDataCompare(output_cpu, output_opencl, width, height);
    if (copare_ret)
    {
        printf("testImage ImgDataCompare success!! A and B matched!!!\n");
    }


    free(aligned_input);
    aligned_input = NULL;
    free(input);
    input = NULL;
    free(output_cpu);
    output_cpu = NULL;
    free(output_opencl);
    output_opencl = NULL;
    clReleaseMemObject(input_image);
    input_image = NULL;

    engine.ClRelease();
}

void TestCpuProcess(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst, size_t ostride)
{
    if (NULL == src || NULL == dst)
    {
        printf("src is NULL or dst is NULL!\n");
        return;
    }
    timeval start{};
    gettimeofday(&start, NULL);
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        CommonUtil::Gaussian3x3Sigma0U8C1(src, width, height, istride, dst, ostride);
    }
    PrintDuration(&start, "Cpu Gaussian", LOOP_COUNT);
}
