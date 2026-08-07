//
// Created by Frewen.Wang on 2024/9/2.
//
#include "gtest/gtest.h"
#include "aura/utils/core.h"
#include <unistd.h>
#include <thread>

const static char *TAG = "TestConditionDelayTask";
using namespace aura::utils;

class TestConditionDelayTask : public testing::Test {
public:
    static void SetUpTestSuite() {
        // AURA_PRINTI"Test SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        // AURA_PRINTI(TAG, "Test TearDownTestSuite");
    }
};

TEST_F(TestConditionDelayTask, hello) {
    // AURA_PRINTI(TAG, "Test hello");
    //
    // aura::utils::ConditionDelayTask delay;
    //
    // // 计划5秒后执行任务
    // delay.RunAfterDelay([]() {
    //     ALOGD(TAG, "5秒时间到！！！任务成功执行！");
    // }, 5);
    //
    // ALOGD(TAG, "任务已计划，5秒后执行...");
    // srand(static_cast<unsigned>(time(nullptr))); // 设置种子（只需一次）
    // int random_num = rand() % 10 + 1; // 1-10的随机数
    // ALOGD(TAG, "但是用户将在 %d 秒进行终端任务！", random_num);
    // std::this_thread::sleep_for(std::chrono::seconds(random_num));
    //
    // if (!delay.HasInterrupted()) {
    //     delay.Interrupt();
    //     ALOGD(TAG, "任务还未中断，中断任务！！");
    // } else {
    //     ALOGD(TAG, "任务已经中断，不需要中断任务！！");
    // }
}
