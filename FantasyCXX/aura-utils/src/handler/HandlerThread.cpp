#include "HandlerThread.h"
#include <unistd.h>

namespace aura::utils {
    static int gIndex = 1;

    using namespace std;

    HandlerThread::HandlerThread() : HandlerThread(reinterpret_cast<const char *>(gIndex), 0) {
    }

    HandlerThread::HandlerThread(const char *name) : HandlerThread(name, 0) {
    }

    HandlerThread::HandlerThread(const char *name, int priority) {
        mName = name;
        mPriority = priority;
        mLooper = nullptr;
        mHandler = nullptr;
        mThread = nullptr;
    }

    HandlerThread::~HandlerThread() {
        mLooper = nullptr;
        mHandler = nullptr;
        mThread = nullptr;
    }

    void HandlerThread::start() {
        if (mThread == nullptr) {
            mThread = new std::thread(&HandlerThread::run, this);
            while (true) {
                usleep(1000);
                if (mLooper != nullptr) {
                    break;
                }
            }
            // TODO: wait() 没有被 notify，需要确认问题。暂时用 usleep 解决
            // unique_lock<mutex> lck(_m_mutex);
            // std::cout << "11" << std::endl;
            // _m_condition.wait(lck, [this] { return _m_looper != nullptr; });
            //  std::cout << "22" << std::endl;
            //  mVisThread->join();
        }
    }

    void HandlerThread::run() {
#if defined(MAC) or defined(IOS)
        pthread_setname_np(mName);
#else
    pthread_setname_np(pthread_self(), mName);
#endif
        mTid = std::this_thread::get_id();
        Looper::prepare();
        std::unique_lock<std::mutex> lck(mMutex);
        mLooper = Looper::getForThread();
        mCondition.notify_all();
        lck.unlock();

        // set priority
        onLooperPrepared();
        Looper::loop();
        // 如果执行到此处，说明loop方法执行完毕（异常结束)，后续进行资源回收
        mLooper->quit();
        Looper::setForThread(nullptr);
        mLooper = nullptr;
        delete mHandler;
        mHandler = nullptr;
    }

    Looper *HandlerThread::getLooper() {
        if (!mThread) {
            throw std::domain_error{"must start thread before get looper"};
        }
        std::unique_lock<std::mutex> lck(mMutex);
        while (!mLooper) {
            mCondition.wait(lck);
        }
        return mLooper;
    }

    Handler *HandlerThread::getThreadHandler() {
        if (!mThread) {
            throw domain_error{"must start thread before get handler"};
        }
        if (!mHandler) {
            mHandler = new Handler(getLooper());
        }
        return mHandler;
    }

    bool HandlerThread::quit() const {
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
