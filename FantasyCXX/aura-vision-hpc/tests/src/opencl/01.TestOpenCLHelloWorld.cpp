//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

#ifdef MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLDemo";
const static char *TAG_CLENG = "OpenCL";


class TestOpenCLHelloWorld : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


//
// 查询当前机器支持的平台ID
//
cl_int clGetPlatformIDs(
    cl_uint num_entries,         // 输出缓冲区大小，可以为0
    cl_platform_id *platforms,   // 输出 platformId 列表buffer，可以为空
    cl_uint *num_platforms);     // 输出实际的platformId数量

//
// 根据 platformId 查询当前平台相应的属性信息
// 这里 param_name 值可以为如下的：
//  CL_PLATFORM_PROFILE: 返回平台支持的profile，相应的 param_value是 char[]
//  CL_PLATFORM_VERSION: 返回平台支持的openCL版本，相应的 param_value是 char[]
//  CL_PLATFORM_NAME:    返回平台名称，相应的 param_value是 char[]
//  CL_PLATFORM_VENDOR:  返回平台Vendor，相应的 param_value是 char[]
//
cl_int clGetPlatformInfo(
    cl_platform_id platform, // 要查询的PlatformId
    cl_platform_info param_name, // 要查询的属性索引
    size_t param_value_size, // 属性值缓冲区大小
    void *param_value, // 属性值输出缓冲区
    size_t *param_value_size_ret); // 实际输出的属性值大小

//  这个clGetDeviceIDs是进行函数声明，声明一个获取clGetDeviceIDs
//  typedef int32_t         cl_int;
//  查询指定平台支持的设备Id列表
//  这里 device_type 可以是如下几种类型：
//     CL_DEVICE_TYPE_CPU
//     CL_DEVICE_TYPE_GPU
//     CL_DEVICE_TYPE_ACCELERATOR
//     CL_DEVICE_TYPE_DEFAULT
//     CL_DEVICE_TYPE_ALL
//
cl_int clGetDeviceIDs(
    cl_platform_id platform, // 平台Id
    cl_device_type device_type, // 要查询的设备类型
    cl_uint num_entries, // 输出设备Id缓冲区大小，可以为0
    cl_device_id *devices, // 输出设备Id列表，可以为0
    cl_uint *num_devices); // 实际输出设备Id数量

TEST_F(TestOpenCLHelloWorld, TestOpenCLHelloWorld) {
    ALOGE(TAG, "==============TestOpenCLHelloWorld================");

    // 定义的32位int数据
    cl_int err;
    // 获取platform平台的个数
    cl_uint num_platforms;
    // 获取所有的平台的ID
    cl_platform_id platform;
    // 获取device设备的的个数
    cl_uint num_devices;
    // 获取所有的设备的ID
    cl_device_id device;


    // 第一步：获取平台数量
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 平台信息\n");
        return;
    }
    printf("获取到的 OpenCL 平台信息 num_platforms：%d \n", num_platforms);

    // 第一步：获取设备数量
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 设备信息\n");
        return ;
    }
    printf("获取到的 OpenCL 设备信息 num_devices：%d \n", num_devices);


    // 输出 OpenCL 版本信息
    char version[1024];
    // 使用 clGetPlatformInfo 函数获取平台的版本信息，并将其存储在 version 变量中，然后输出。
    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 版本信息\n");
        return;
    }
    printf("获取到的 OpenCL 版本号: %s\n", version);

    // 输出设备信息
    char device_name[1024];
    // 使用 clGetDeviceInfo 函数获取设备的名称信息，并将其存储在 device_name 变量中，然后输出。
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        printf("无法获取设备名称\n");
        return ;
    }
    printf("获取到的 OpenCL 设备名称: %s\n", device_name);

}
