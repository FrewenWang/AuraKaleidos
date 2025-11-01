//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <iostream>                	// std::cout
#include <thread>                	// std::thread
#include <mutex>                	// std::mutex, std::unique_lock
#include <condition_variable>    	// std::condition_variable

const static char *TAG = "TestCXXConditionVariable";

std::mutex mtx; 				// 全局互斥锁.
std::condition_variable cv; 	// 全局条件变量.
bool ready = false; 			// 全局标志位.

using namespace std;

/**
 * 文章参考：
 */
class TestCXXConditionVariable : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};


void doTask(int id) {
	std::unique_lock <std::mutex> lck(mtx);
	while (!ready) {
		// 如果标志位不为 true, 则等待...
		// 当前线程被阻塞, 当全局标志位变为 true 之后,
		// 条件变量获取到这个锁，让当前线程等待。
		std::cout << "thread_" << id << " is waiting!!!!\n";
		cv.wait(lck);
	}
	// 线程被唤醒, 继续往下执行打印线程编号id.
	std::cout << "thread_" << id << " has waked!!!!\n";
}

void go() {
	ALOGD(TAG, "============== testCXXConditionVariable go==============");
	std::unique_lock <std::mutex> lck(mtx);
	ready = true; // 设置全局标志位为 true.
	cv.notify_all(); // 唤醒所有线程.
}

TEST_F(TestCXXConditionVariable, testCXXConditionVariable) {
	ALOGD(TAG, "============== testCXXConditionVariable ==============");
    // 创建包含10个线程的数组
	std::thread threads[10];
	// spawn 10 threads:
	for (int i = 0; i < 10; ++i) {
		threads[i] = std::thread(doTask, i);
	}

	// 主线程休眠2秒钟
	sleep(2);
	// 主线进行进一步执行.
	// 开始将设置全局标志位为 true.
	// 然后唤醒所有的阻塞线程
	go();

	// 等待启动的线程完成，才会继续往下执行。假如前面的代码使用这种方式
	///
	for (auto & th:threads){
		th.join();
	}

}

