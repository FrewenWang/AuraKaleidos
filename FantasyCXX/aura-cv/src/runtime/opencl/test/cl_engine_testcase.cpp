#include <memory>

#include "aura/runtime/opencl.h"
#include "aura/tools/unit_test.h"

using namespace aura;

NEW_TESTCASE(runtime_opencl_engine_test)
{
    // 定义Status的状态变量
    Status ret = Status::OK;
    // 获取上下文
    Context *ctx = UnitTest::GetInstance()->GetContext();
    
    std::shared_ptr<cl::Platform> cl_platform = ctx->GetCLEngine()->GetCLRuntime()->GetPlatform();
    AURA_LOGI(ctx, AURA_TAG, "Platform Name       : %s\n", cl_platform->getInfo<CL_PLATFORM_NAME>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Platform Vendor     : %s\n", cl_platform->getInfo<CL_PLATFORM_VENDOR>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Platform version    : %s\n", cl_platform->getInfo<CL_PLATFORM_VERSION>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Platform extensions : %s\n", cl_platform->getInfo<CL_PLATFORM_EXTENSIONS>().c_str());

    std::shared_ptr<cl::Device> cl_device = ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
    switch (cl_device->getInfo<CL_DEVICE_TYPE>())
    {
        case CL_DEVICE_TYPE_GPU:
        {
            AURA_LOGI(ctx, AURA_TAG, "Device Type    : GPU\n");
            break;
        }

        case CL_DEVICE_TYPE_CPU:
        {
            AURA_LOGI(ctx, AURA_TAG, "Device Type    : CPU\n");
            break;
        }
        
        default:
        {
            AURA_LOGI(ctx, AURA_TAG, "Device Type    : unknown\n");
        }
    }

    AURA_LOGI(ctx, AURA_TAG, "Device Name    : %s\n", cl_device->getInfo<CL_DEVICE_NAME>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Device Vendor  : %s\n", cl_device->getInfo<CL_DEVICE_VENDOR>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Device Version : %s\n", cl_device->getInfo<CL_DEVICE_VERSION>().c_str());
    AURA_LOGI(ctx, AURA_TAG, "Driver Version : %s\n", cl_device->getInfo<CL_DRIVER_VERSION>().c_str());

    AURA_LOGI(ctx, AURA_TAG, "Device Max Compute Units    : %u\n", cl_device->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    AURA_LOGI(ctx, AURA_TAG, "Device Global Memory        : %lu\n", cl_device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    AURA_LOGI(ctx, AURA_TAG, "Device Max Clock Frequency  : %u\n", cl_device->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    AURA_LOGI(ctx, AURA_TAG, "Device Max Memory Allocation: %lu\n", cl_device->getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    AURA_LOGI(ctx, AURA_TAG, "Device Local Memory         : %lu\n", cl_device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    AURA_LOGI(ctx, AURA_TAG, "Device Available            : %u\n", cl_device->getInfo<CL_DEVICE_AVAILABLE>());
    AURA_LOGI(ctx, AURA_TAG, "Device print Buffer Size    : %llu\n", cl_device->getInfo<CL_DEVICE_PRINTF_BUFFER_SIZE>());

#if defined(CL_VERSION_2_0)
    cl_device_svm_capabilities cl_cap = cl_device->getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if ((cl_cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) || (cl_cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER))
    {
        if (cl_cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
        {
            AURA_LOGI(ctx, AURA_TAG, "Device Type    : SVM fine grain\n");
        }
        else if (cl_cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
        {
            AURA_LOGI(ctx, AURA_TAG, "Device Type    : SVM coarse grain\n");
        }
    }
    else
    {
        AURA_LOGI(ctx, AURA_TAG, "Device Type    : SVM unsupport\n");
    }
#endif

    Buffer buffer = ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(ctx, AURA_MEM_SVM, 1024, 0));
    AURA_LOGI(ctx, AURA_TAG, "%s\n", buffer.ToString().c_str());
    ret |= AURA_CHECK_EQ(ctx, buffer.m_type, static_cast<MI_S32>(AURA_MEM_SVM), "check Buffer::m_type failed\n");
    
    MI_U8 *ptr = (MI_U8*)buffer.m_data;
    for (MI_S32 i = 0; i < 1024; ++i)
    {
        ptr[i] = i & 0xFF;
    }

    for (MI_S32 i = 0; i < 1024; ++i)
    {
        ret |= AURA_CHECK_EQ(ctx, static_cast<MI_S32>(ptr[i]), i & 0xFF, "check Buffer::m_data failed\n");
    }

    AURA_FREE(ctx, buffer.m_data);

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}