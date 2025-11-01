//
// Created by Frewen.Wang on 25-4-10.
//
#pragma once
#include <cstdint>
#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @brief The information of CL device
 *
 */
struct XClDeviceInfo {
    cl_device_id id;
    cl_device_type device_type;
    cl_uint vendor_id;

    cl_uint work_item_max_dim;
    std::vector<cl_uint> work_item_max_sizes;
    size_t max_workgroup_size;
    cl_ulong max_mem_size;

    cl_bool image_support;
    size_t image_max_buf_size;
    size_t image_max_array_size;
    cl_uint image_max_samplers;
    cl_uint image_max_read_args;
    cl_uint image_max_write_args;
    size_t img2d_max_width;
    size_t img2d_max_height;
    size_t img3d_max_width;
    size_t img3d_max_height;
    size_t img3d_max_depth;

    cl_uint vector_width_char;
    cl_uint vector_width_short;
    cl_uint vector_width_int;
    cl_uint vector_width_long;
    cl_uint vector_width_float;
    cl_uint vector_width_double;
};

struct XClPlatformInfo {
    cl_platform_id id;
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;
    std::string extension;

    std::vector<XClDeviceInfo> devices;
};


/**
 * OpenCL的标准初始化流程就是：先
 * 1. 查询支持的平台和设备信息，
 * 2. 然后根据这些信息创建相应的 OpenCL上下文对象，
 * 3. 之后所有的 OpenCL 控制操作都以该上下文对象来作为标识，相当于后续的并行计算也都是在该设备上执行。
 * 在实际的硬件产品中，PC上的设备通常都是一些显卡（例如：NIVIDA独立显卡，或者Intel集成显卡），
 * 而手机上的设备通常都是一些显示芯片（例如：Mali等）。
 */
class XClEngine {
public:
    /**
     * 进行平台信息的查询
     * @return
     */
    int32_t QueryPlatforms();

    /**
     * 进行设备信息的查询
     * @param platform_id   传入之前步骤查询出来的platformID
     * @param out_dev_list  输出设备信息列表
     * @return 
     */
    int32_t QueryDevices(cl_platform_id platform_id, std::vector<XClDeviceInfo> &out_dev_list);


    // ===========================   如下是我们定义的业务代码  ================================================

    std::vector<XClPlatformInfo> &getPlatformList() {
        return platforms_;
    }

public:
    cl_context context_ = nullptr; // CL上下文对象

private:
    std::vector<XClPlatformInfo> platforms_; // 查询到的平台列表

    cl_command_queue cmd_queue_ = nullptr; // 命令队列对象
    cl_program program_ = nullptr; // 程序对象
    cl_kernel kernel_filter_ = nullptr; // 内核函数
};
