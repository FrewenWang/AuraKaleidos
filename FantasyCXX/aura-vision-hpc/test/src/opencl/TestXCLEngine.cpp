//
// Created by Frewen.Wang on 25-4-10.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include <aura/aura_utils/core/common/macro.h>
#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "XClEngine.h"

#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestXCLEngine";


class TestXCLEngine : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


TEST_F(TestXCLEngine, TestXCLEngineHello) {
    XClEngine engine;

    ALOGE(TAG, "==============TestXCLEngineHello================");

    engine.QueryPlatforms();


    std::vector<XClPlatformInfo> platforms = engine.getPlatformList();

    ALOGE(TAG, "getPlatformList size:%d", platforms.size());
    assert(platforms.size() == 1);

    ALOGE(TAG, "device size:%d", platforms[0].devices.size());

    // 我们进行拷贝出来设备列表的信息
    std::vector<XClDeviceInfo> devices = platforms[0].devices;

    /// 进行context的获取
    cl_device_id dev_ids[1] = {0};
    dev_ids[0] = devices[0].id;
    cl_int err_code;
    engine.context_ = clCreateContext(nullptr, 1, dev_ids, nullptr, nullptr, &err_code);
    if (err_code != CL_SUCCESS) {
        ALOGE(TAG, "<XClEngine.QueryPlatforms>fail to get number,ret=%d",err_code);
        return;
    }




}
