#include "aura/utils/concurrent/thread_pool.h"
#include "aura/utils/core/runnable.h"
#include <assert.h>
#include <iostream>
#include <sstream>

namespace aura::utils
{

int ThreadPool::mInitThreadsSize = 0;

ThreadPool::ThreadPool(size_t threads) : mMutex(), mCondition(), mIsStarted(false) {
}

ThreadPool::~ThreadPool() {
  if (mIsStarted) {
    stop();
  }
}


void ThreadPool::start() {
  assert(mWorkers.empty());
  mIsStarted = true;
  mWorkers.reserve(mInitThreadsSize);
  for (int i = 0; i < mInitThreadsSize; ++i) {
    mWorkers.push_back(std::thread(std::bind(&ThreadPool::threadLoop, this)));
  }
}

void ThreadPool::stop() { {
    std::unique_lock<std::mutex> lock(mMutex);
    mIsStarted = false;
    mCondition.notify_all();
  }

  for (std::vector<std::thread>::iterator it = mWorkers.begin(); it != mWorkers.end(); ++it) {
    (*it).join();
  }
  mWorkers.clear();
}


void ThreadPool::threadLoop() {
  while (mIsStarted) {
    Runnable *task = take();
    if (task != nullptr) {
      task->run();
    }
  }
}

void ThreadPool::enqueueTask(Runnable *task) {
  std::unique_lock<std::mutex> lock(mMutex);
  mTasks.push_back(task);
  mCondition.notify_one();
}


Runnable *ThreadPool::take() {
  std::unique_lock<std::mutex> lock(mMutex);
  //always use a while-loop, due to spurious wakeup
  while (mTasks.empty() && mIsStarted) {
    mCondition.wait(lock);
  }

  Runnable *task = nullptr;
  std::deque<Runnable *>::size_type size = mTasks.size();
  if (!mTasks.empty() && mIsStarted) {
    task = mTasks.front();
    mTasks.pop_front();
    assert(size - 1 == mTasks.size());
  }

  return task;
}
}
