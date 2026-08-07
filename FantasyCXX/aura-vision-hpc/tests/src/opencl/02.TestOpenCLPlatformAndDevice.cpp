//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLPlatformAndDevice";


class TestOpenCLPlatformAndDevice : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


TEST_F(TestOpenCLPlatformAndDevice, TestGetPlatform) {
    ALOGE(TAG, "==============TestGetPlatform================");

    /// 典型使用场景：
    /// 第一次调用的时候： platforms = NULL，通过 num_platforms 获取实际数量。
    /// clGetPlatformIDs(0, NULL, &num_platforms); // 获取平台总数
    /// 第二次调用的时候：分配内存后再次调用以获取所有平台 ID。
    ///
    /// 第一步：
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        // 处理错误或无平台的情况
        printf("获取平台数量的大小异常 size: %d\n", num_platforms);
    }
    /// 第二步：分配内存后再次调用以获取所有平台 ID。
    /// Clang-Tidy: Use auto when initializing with a cast to avoid duplicating the type name
    /// Clang-Tidy：给出这个提示，是为了让代码更加简洁、易读，同时减少因手动指定类型可能引发的错误。你可以依据这个提示，在使用强制类型转换初始化变量时，采用auto关键字。
    // cl_platform_id *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    auto *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    err = clGetPlatformIDs(num_platforms, platform, nullptr);
    if (err != CL_SUCCESS) {
        printf("获取平台的ID列表异常 size: %d\n", num_platforms);
    }

    printf("获取到的 OpenCL 平台数量 num_platforms：%d \n", num_platforms);
}

TEST_F(TestOpenCLPlatformAndDevice, TestGetPlatformInfo) {
    ALOGE(TAG, "==============TestGetPlatformInfo================");


    /// 典型使用场景：
    /// 第一次调用的时候： platforms = NULL，通过 num_platforms 获取实际数量。
    /// clGetPlatformIDs(0, NULL, &num_platforms); // 获取平台总数
    /// 第二次调用的时候：分配内存后再次调用以获取所有平台 ID。
    ///
    /// 第一步：
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        // 处理错误或无平台的情况
        printf("获取平台数量的大小异常 size: %d\n", num_platforms);
    }
    /// 第二步：分配内存后再次调用以获取所有平台 ID。
    /// Clang-Tidy: Use auto when initializing with a cast to avoid duplicating the type name
    /// Clang-Tidy：给出这个提示，是为了让代码更加简洁、易读，同时减少因手动指定类型可能引发的错误。你可以依据这个提示，在使用强制类型转换初始化变量时，采用auto关键字。
    // cl_platform_id *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    auto *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    err = clGetPlatformIDs(num_platforms, platform, nullptr);
    if (err != CL_SUCCESS) {
        printf("获取平台的ID列表异常 size: %d\n", num_platforms);
    }

    for (int i = 0; i < num_platforms; i++) {

        size_t size;
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
        char *PName = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, size, PName, NULL);
        printf("CL_PLATFORM_NAME 获取平台名称: %s\n", PName);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
        char *PVendor = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, size, PVendor, NULL);
        printf("CL_PLATFORM_VENDOR 获取平台开发商名称: %s\n", PVendor);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        char *PVersion = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, PVersion, NULL);
        printf("CL_PLATFORM_VERSION 获取平台支持的最大的OpenCL版本: %s\n", PVersion);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, 0,NULL, &size);
        char *PProfile = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, size, PProfile, NULL);
        printf("CL_PLATFORM_PROFILE 获取平台支持FULL_PROFILE还是EMBBEDED_PROFILE: %s\n", PProfile);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
        char *PExten = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, size, PExten, NULL);
        printf("CL_PLATFORM_EXTENSIONS 获取平台支持的扩展名列表: %s\n", PExten);
        free(PName);
        free(PVendor);
        free(PVersion);
        free(PProfile);
        free(PExten);
    }
    // 回收platform的开辟的内存
    free(platform);
}

TEST_F(TestOpenCLPlatformAndDevice, TestGetDevice) {
    ALOGE(TAG, "==============TestGetDevice================");
    cl_uint num_platform;
    cl_uint num_device;
    cl_platform_id *platform;
    cl_device_id *devices;
    cl_int err;

    // 第一次调用时，devices参数设置为NULL，num_devices返回指定平台中的设备数；
    // 可以获取num_platform的数量
    err = clGetPlatformIDs(0, 0, &num_platform);

    ///
    platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platform));
    err = clGetPlatformIDs(num_platform, platform, NULL);
    //获得第一个平台上的设备数量
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    devices = static_cast<cl_device_id *>(malloc(sizeof(cl_device_id) * num_device));
    //初始化可用的设备
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, num_device, devices, NULL);
}
