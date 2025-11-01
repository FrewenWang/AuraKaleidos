#pragma once

#include "Handler.h"
#include "Looper.h"
#include <condition_variable>
#include <thread>

namespace aura::utils {

/**
 * 包含Handler的Thread对象。
 * 将handler调度的信息放在对应线程里面执行
 */
class HandlerThread {
private:
    /**
     * 设置线程的优先级
     */
    int mPriority;
    std::thread::id mTid;
    const char *mName;
    /**
     * 线程绑定的Looper对象
     */
    Looper *mLooper;
    Handler *mHandler;
    /**
     * HandlerThread线程
     */
    std::thread *mThread;
    std::condition_variable mCondition;
    std::mutex mMutex;

protected:
    void onLooperPrepared() { }

public:
    HandlerThread();

    explicit HandlerThread(const char *name);

    HandlerThread(const char *name, int priority);

    ~HandlerThread();

    void start();

    /**
     * HandlerThread的run方法
     */
    void run();

    Looper *getLooper();

    Handler *getThreadHandler();

    bool quit() const;

    bool quitSafely() const;

    std::thread::id getThreadId() {
        return mTid;
    }
};

} // namespace aura::aura_lib
