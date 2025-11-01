//
// Created by Frewen.Wang on 2024/9/2.
//
#include "gtest/gtest.h"
#include "aura/utils/core.h"
#include "aura/utils/string_util.h"
#include <unistd.h>
#include <thread>

const static char *TAG = "TestStringUtils";

class TestStringUtils : public testing::Test
{
public:
    static void SetUpTestSuite()
    {
        // ALOGI(TAG, "Test TestStringUtils");
    }

    static void TearDownTestSuite()
    {
        // ALOGI(TAG, "Test TestStringUtils");
    }
};

TEST_F(TestStringUtils, splitStr)
{
    // ALOGD(TAG, "Test hello");
    //
    std::vector<std::string> result;
    aura::utils::StringUtil::splitStr("Hello,How,are,you", ',', result);
    for (auto basic_string: result)
    {
        // ALOGD(TAG, basic_string.c_str());
    }
}
