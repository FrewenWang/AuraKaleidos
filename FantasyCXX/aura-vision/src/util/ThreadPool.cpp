#include "vision/util/ThreadPool.h"
#include <assert.h>
#include <iostream>
#include <sstream>

namespace aura::vision {

int ThreadPool::mInitThreadsSize = 2;

std::unique_ptr<LoggerIface> active_logger = nullptr;

static const char black[] = {0x1b, '[', '1', ';', '3', '0', 'm', 0};
static const char red[] = {0x1b, '[', '1', ';', '3', '1', 'm', 0};
static const char yellow[] = {0x1b, '[', '1', ';', '3', '3', 'm', 0};
static const char blue[] = {0x1b, '[', '1', ';', '3', '4', 'm', 0};
static const char normal[] = {0x1b, '[', '0', ';', '3', '9', 'm', 0};

logger::logger(log_level level)
        : m_level(level) {}

void logger::debug(const std::string &msg, const std::string &file, std::size_t line) {
    if (m_level >= log_level::debug) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << "[" << black << "DEBUG" << normal << "][sola::logger][" << file << ":" << line << "] " << msg
                  << std::endl;
    }
}

void logger::info(const std::string &msg, const std::string &file, std::size_t line) {
    if (m_level >= log_level::info) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << "[" << blue << "INFO " << normal << "][sola::logger][" << file << ":" << line << "] " << msg
                  << std::endl;
    }
}

void logger::warn(const std::string &msg, const std::string &file, std::size_t line) {
    if (m_level >= log_level::warn) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << "[" << yellow << "WARN " << normal << "][sola::logger][" << file << ":" << line << "] " << msg
                  << std::endl;
    }
}

void logger::error(const std::string &msg, const std::string &file, std::size_t line) {
    if (m_level >= log_level::error) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::cerr << "[" << red << "ERROR" << normal << "][sola::logger][" << file << ":" << line << "] " << msg
                  << std::endl;
    }
}

void debug(const std::string &msg, const std::string &file, std::size_t line) {
    if (active_logger)
        active_logger->debug(msg, file, line);
}

void info(const std::string &msg, const std::string &file, std::size_t line) {
    if (active_logger)
        active_logger->info(msg, file, line);
}

void warn(const std::string &msg, const std::string &file, std::size_t line) {
    if (active_logger)
        active_logger->warn(msg, file, line);
}

void error(const std::string &msg, const std::string &file, std::size_t line) {
    if (active_logger)
        active_logger->error(msg, file, line);
}

ThreadPool::ThreadPool(int initSize) :
          mMutex(),
          mCond(),
          mIsStarted(false) {
}

ThreadPool::~ThreadPool() {
    if (mIsStarted) {
        stop();
    }
}

void ThreadPool::start() {
    assert(mThreads.empty());
    mIsStarted = true;
    mThreads.reserve(mInitThreadsSize);
    for (int i = 0; i < mInitThreadsSize; ++i) {
        mThreads.push_back(new std::thread(std::bind(&ThreadPool::threadLoop, this)));
    }

}

void ThreadPool::stop() {
    __SOLA_LOG(debug, "ThreadPool::stop() stop.");
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIsStarted = false;
        mCond.notify_all();
        __SOLA_LOG(debug, "ThreadPool::stop() notifyAll().");
    }

    for (std::vector<std::thread *>::iterator it = mThreads.begin(); it != mThreads.end(); ++it) {
        (*it)->join();
        delete *it;
    }
    mThreads.clear();
}


void ThreadPool::threadLoop() {
    __SOLA_LOG(debug, "ThreadPool::threadLoop() tid : " + get_tid() + " start.");
    while (mIsStarted) {
        Runnable *task = take();
        if (task != nullptr) {
            task->run();
        }
    }
    __SOLA_LOG(debug, "ThreadPool::threadLoop() tid : " + get_tid() + " exit.");
}

void ThreadPool::execute(Runnable *task) {
    std::unique_lock<std::mutex> lock(mMutex);
    /*while(m_tasks.isFull())
      {//when m_tasks have maxsize
        cond2.notify_one();
      }
    */
    mTasks.push_back(task);
    mCond.notify_one();
}

Runnable *ThreadPool::take() {
    std::unique_lock<std::mutex> lock(mMutex);
    //always use a while-loop, due to spurious wakeup
    while (mTasks.empty() && mIsStarted) {
        __SOLA_LOG(debug, "ThreadPool::take() tid : " + get_tid() + " wait.");
        mCond.wait(lock);
    }

    __SOLA_LOG(debug, "ThreadPool::take() tid : " + get_tid() + " wakeup.");

    Runnable *task = nullptr;
    std::deque<Runnable *>::size_type size = mTasks.size();
    if (!mTasks.empty() && mIsStarted) {
        task = mTasks.front();
        mTasks.pop_front();
        assert(size - 1 == mTasks.size());
        /*if (TaskQueueSize_ > 0)
        {
          cond2.notify_one();
        }*/
    }

    return task;

}

} // namespace aura::vision