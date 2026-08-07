#include "HandlerThread.h"
#include <atomic>

namespace aura::utils {
    static std::atomic<int> gIndex{1};

    using namespace std;

    HandlerThread::HandlerThread() : HandlerThread(("HandlerThread-" + std::to_string(gIndex.fetch_add(1))).c_str(), 0) {
    }

    HandlerThread::HandlerThread(const char *name) : HandlerThread(name, 0) {
    }

    HandlerThread::HandlerThread(const char *name, int priority) {
        mName = name ? name : "HandlerThread";
        mPriority = priority;
        mLooper = nullptr;
        mHandler = nullptr;
    }

    HandlerThread::~HandlerThread() {
        quit();
        if (mThread.joinable() && mThread.get_id() != std::this_thread::get_id()) {
            mThread.join();
        }
        delete mHandler;
        mHandler = nullptr;
    }

    void HandlerThread::start() {
        std::unique_lock<std::mutex> lock(mMutex);
        if (!mThread.joinable()) {
            mThread = std::thread(&HandlerThread::run, this);
            mCondition.wait(lock, [this] { return mLooper != nullptr; });
        }
    }

    void HandlerThread::run() {
#if defined(__APPLE__)
        pthread_setname_np(mName.c_str());
#else
    pthread_setname_np(pthread_self(), mName.c_str());
#endif
        mTid = std::this_thread::get_id();
        Looper::prepare();
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mLooper = Looper::getForThread();
        }
        mCondition.notify_all();

        // set priority
        onLooperPrepared();
        Looper::loop();
        // 如果执行到此处，说明loop方法执行完毕（异常结束)，后续进行资源回收
        Looper::setForThread(nullptr);
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mLooper = nullptr;
        }
    }

    Looper *HandlerThread::getLooper() {
        if (!mThread.joinable()) {
            throw std::domain_error{"must start thread before get looper"};
        }
        std::unique_lock<std::mutex> lck(mMutex);
        while (!mLooper) {
            mCondition.wait(lck);
        }
        return mLooper;
    }

    Handler *HandlerThread::getThreadHandler() {
        if (!mThread.joinable()) {
            throw domain_error{"must start thread before get handler"};
        }
        Looper *looper = getLooper();
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mHandler) {
            mHandler = new Handler(looper);
        }
        return mHandler;
    }

    bool HandlerThread::quit() const {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mLooper) {
            mLooper->quit();
            return true;
        }
        return false;
    }

    bool HandlerThread::quitSafely() const {
        return quit();
    }
} // namespace aura::aura_lib
