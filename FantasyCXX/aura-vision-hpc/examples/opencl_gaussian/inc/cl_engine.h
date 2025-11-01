//
// Created by wangzhijiang on 25-5-7.
//

#ifndef GAUSSIAN_CL_ENGINE_H
#define GAUSSIAN_CL_ENGINE_H
#include <string>
#include <CL/cl.h>

namespace gaussian
{
/**
 *
 */
struct ClPlatformInfo
{
    cl_platform_id platform_id;
    std::string profile;
    std::string version;
    std::string name;
    std::string vendor;
    std::string extension;
};

struct ClDeviceInfo
{
    cl_device_id device_id;
    cl_device_type type;
    cl_uint vendor_id;
    cl_uint num_devices;
    size_t max_work_group_size;
    cl_uint image_pitch_alignment;
};

class CLEngine
{
public:
    /**
     * @brief The constructor of the class
     */
    CLEngine();

    /**
     * @brief  The destructor of the class
     */
    ~CLEngine();

    /**
     * @brief  query opencl platform infos
     * @return ret
     */
    cl_int QueryPlatforms();

    /**
     * @brief query opencl device infos
     * @return ret
     */
    cl_int QueryDevices();

    /**
     * @brief create opencl context
     * @return ret
     */
    cl_int CreateContext();

    /**
     * @brief create command queue
     * @return
     */
    cl_int CreateCommandQueue();

    /**
     * @brief create program
     * @param kernel_src
     * @return
     */
    cl_int CreateProgram(const char *kernel_src);

    /**
     * @brief create opencl kernel
     * @param kernel_name
     * @return
     */
    cl_int CreateKernel(const char *kernel_name);

    /**
     * @brief get opencl event profiling info
     * @param event
     */
    void GetProfilingInfo(cl_event &event);

    /**
     * @brief release
     */
    void ClRelease();

    /**
     * @brief  get ClPlatformInfo
     * @return
     */
    ClPlatformInfo &GetPlatformInfo();

    /**
     * @brief  get ClDeviceInfo
     * @return
     */
    ClDeviceInfo &GetDevicesInfo();

    /**
     * @brief  get opencl command queue
     * @return
     */
    cl_command_queue &GetCommandQueue();

    /**
     * @brief get opencl program
     * @return
     */
    cl_program &GetProgram();

    /**
     * @brief  get opencl context
     * @return
     */
    cl_context &GetContext();

    /**
     * @brief  get opencl kernel
     * @return
     */
    cl_kernel &GetKernel();

private:
    cl_context context;
    ClPlatformInfo platform;
    ClDeviceInfo device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
};
}


#endif //GAUSSIAN_CL_ENGINE_H
