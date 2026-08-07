//
// Created by Frewen.Wang on 25-4-10.
//

#include "aura/aura_utils/utils/AuraLog.h"
#include <aura/aura_utils/core/common/macro.h>
#include "XClEngine.h"


#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLDemo";

static constexpr int TMP_BUF_SIZE = 1024;

int32_t XClEngine::QueryPlatforms() {
    // 查询主机上有多少platforms。记录主机上platforms数量
    cl_uint plat_number = 0;

    // 步骤0： 查询主机上有多少个platforms数量
    cl_int err_code = clGetPlatformIDs(0, nullptr, &plat_number); // 查询支持的平台数量
    // 判断获取的主机上面platforms数量的逻辑是否正常启动
    if (err_code != CL_SUCCESS) {
        ALOGE(TAG, "<XClEngine.QueryPlatforms>fail to get number,ret=%d", err_code);
        return ERR_UNSUPPORTED;
    }
    // 如果获取到的主机上面的platforms数量异常，则也返回不支持
    if (plat_number <= 0) {
        ALOGE(TAG, "<XClEngine.QueryPlatforms> plat_number is 0");
        return ERR_UNSUPPORTED;
    }


    // 步骤1: 查询每个平台Id(如果有多个设备platforms，则返回的ID为一个数组)
    cl_platform_id *id_array = new cl_platform_id[plat_number];
    err_code = clGetPlatformIDs(plat_number, id_array, nullptr);
    if (err_code != CL_SUCCESS) {
        ALOGE(TAG, "<XClEngine.QueryPlatforms> fail to get IDs,ret=%d", err_code);
        return ERR_UNSUPPORTED;
    }

    // 根据平台Id，查询每个平台的信息
    for (cl_uint i = 0; i < plat_number; i++) {
        XClPlatformInfo plat_info;
        plat_info.id = id_array[i];

        // get profile
        size_t ret_size = 0;
        char profile[TMP_BUF_SIZE] = {0};
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_PROFILE, TMP_BUF_SIZE,
                                     profile, &ret_size);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryPlatforms> fail to get profile,ret=%d", err_code);
        } else {
            plat_info.profile = profile;
        }

        // get version
        ret_size = 0;
        char version[TMP_BUF_SIZE] = {0};
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_VERSION, TMP_BUF_SIZE, version, &ret_size);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryPlatforms> fail to get version,ret=%d", err_code);
        } else {
            plat_info.version = version;
        }

        // get vendor
        ret_size = 0;
        char vendor[TMP_BUF_SIZE] = {0};
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_VENDOR, TMP_BUF_SIZE, vendor, &ret_size);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryPlatforms> fail to get vendor,ret=%d", err_code);
        } else {
            plat_info.vendor = vendor;
        }

        // get name
        ret_size = 0;
        char name[TMP_BUF_SIZE] = {0};
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_NAME, TMP_BUF_SIZE, name, &ret_size);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryPlatforms> [ERROR] fail to get name, ret=%d", err_code);
        } else {
            plat_info.name = name;
        }

        // get extension
        ret_size = 0;
        size_t buf_size = 0;
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_EXTENSIONS, 0, nullptr, &buf_size);
        char *extentsion = new char[buf_size];
        memset(extentsion, 0, buf_size);
        err_code = clGetPlatformInfo(id_array[i], CL_PLATFORM_EXTENSIONS, buf_size,
                                     extentsion, &ret_size);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryPlatforms> [ERROR] fail to get extend, ret=%d", err_code);
        } else {
            plat_info.extension = extentsion;
        }
        delete[] extentsion;

        // query all devices
        QueryDevices(id_array[i], plat_info.devices);

        // append into list
        platforms_.emplace_back(plat_info);
    }

    return A_OK;
}

int32_t XClEngine::QueryDevices(cl_platform_id platform_id, std::vector<XClDeviceInfo> &out_dev_list) {
    // 步骤0：查询platform上有多少个device数量
    // 获取设备的数量
    cl_uint device_number = 0;
    // 查询支持设备的数量
    // 查询指定平台支持的设备Id列表
    //  几种类型：CL_DEVICE_TYPE_CPU、CL_DEVICE_TYPE_GPU、CL_DEVICE_TYPE_ACCELERATOR、CL_DEVICE_TYPE_DEFAULT、CL_DEVICE_TYPE_ALL
    //  参数列表：平台Id、要查询的设备类型、输出设备Id缓冲区大小，可以为0、输出设备Id列表，可以为0、实际输出设备Id数量
    // cl_int clGetDeviceIDs (cl_platform_id platform,cl_device_type device_type,cl_uint num_entries,cl_device_id *devices,cl_uint *num_devices);
    cl_int err_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_number);
    if (err_code != CL_SUCCESS) {
        ALOGE(TAG, "<XClEngine.QueryDevices> fail to get number, ret=%d", err_code);
        return ERR_UNSUPPORTED;
    }
    if (device_number <= 0) {
        ALOGE(TAG, "<XClEngine.QueryDevices> [ERROR] dev_number is 0");
        return ERR_UNSUPPORTED;
    }

    // 查询支持的 deviceId列表
    cl_device_id *id_array = new cl_device_id[device_number];
    err_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_number, id_array, nullptr);
    if (err_code != CL_SUCCESS) {
        ALOGE(TAG, "<XClEngine.QueryDevices> fail to clGetDeviceIDs, ret=%d", err_code);
        return ERR_UNSUPPORTED;
    }

    // 查询每一个设备的属性值
    for (cl_uint i = 0; i < device_number; i++) {
        XClDeviceInfo device_info = {nullptr};
        device_info.id = id_array[i];

        // get device type
        err_code = clGetDeviceInfo(id_array[i], CL_DEVICE_TYPE, sizeof(cl_device_type),
                                   &(device_info.device_type), nullptr);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryDevices> fail to get type, ret=%d", err_code);
        }


        // get vendor id
        err_code = clGetDeviceInfo(id_array[i], CL_DEVICE_VENDOR_ID,
                                   sizeof(cl_uint), &(device_info.vendor_id), nullptr);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryDevices> fail to get venderId, ret=%d", err_code);
        }

        // get work_item max dimension
        err_code = clGetDeviceInfo(id_array[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
                                   &(device_info.work_item_max_dim), nullptr);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryDevices> fail to get max dim, ret=%d", err_code);
        }

        // get max workitem counter in one group
        err_code = clGetDeviceInfo(id_array[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                   &(device_info.max_workgroup_size), nullptr);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryDevices> fail to max group size,ret=%d", err_code);
        }

        // get max mem size
        err_code = clGetDeviceInfo(id_array[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                                   &(device_info.max_mem_size), nullptr);
        if (err_code != CL_SUCCESS) {
            ALOGE(TAG, "<XClEngine.QueryDevices> fail to max mem size, ret=%d", err_code);
        }

        // append into list
        out_dev_list.emplace_back(device_info);
    }

    delete[] id_array;

    ALOGD(TAG, "<XClEngine.QueryDevices> devCount=%d", device_number);

    return A_OK;
}
