//
// Created by Frewen.Wang on 2022/6/13.
//

#include "MessageQueue.h"
#include "Message.h"
#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aura_utils/utils/SystemClock.h"

namespace aura::utils {

static const char *TAG = "MessageQueue";

using namespace std;

MessageQueue::MessageQueue() : mMessages(nullptr) {
    // 设置退出变量的为false
    mQuitting.store(false);
}

MessageQueue::~MessageQueue() {
    removeAll();
}

void MessageQueue::enqueueAtTime(Message *msg, const std::int64_t &when) {
    if (msg->target == nullptr) {
        ALOGW(TAG, "enqueueAtTime target is null");
        return;
    }
    // 获取互斥锁。获取当前变量的锁
    unique_lock<mutex> lck(mMutex);
    msg->when = when;
    // 指向消息队列指针的头部
    auto p = mMessages;
    auto needWake = false;
    // 如果消息队列为空，时间为0，或者当前时间已经小于消息队列头部的时间。
    // 则直接将消息插入消息队列头部。唤醒消息队列
    if (p == nullptr || when == 0 || when < p->when) {
        msg->next = p;
        mMessages = msg;
        needWake = true;
    } else {
        // Inserted within the middle of the queue.  Usually we don't have to wake
        // up the event queue unless there is a barrier at the head of the queue
        // and the message is the earliest asynchronous message in the queue.
        Message *prev = p;
        for (;;) {
            prev = p;
            p = p->next;
            if (p == nullptr || when < p->when) {
                break;
            }
        }
        msg->next = p;
        prev->next = msg;
    }
    // 如果需要唤醒，在调用mCondition.notify_one进行唤醒
    if (needWake) {
        mCondition.notify_one();
    }
}

Message *MessageQueue::dequeue() {
    if (mQuitting.load()) {
        return nullptr;
    }
    // 获取当前mMutex信号量的锁
    unique_lock<mutex> lck(mMutex);
    // 如果消息队列为空。这
    if (mMessages == nullptr) {
        // no more messages, wait forever
        // 当 std::condition_variable 对象的某个 wait 函数被调用的时候，
        // 它使用 std::unique_lock(通过 std::mutex) 来锁住当前线程。
        // 当前线程会一直被阻塞，
        // 直到另外一个线程在相同的 std::condition_variable对象上调用了 notification 函数来唤醒当前线程。
        mCondition.wait(lck);
    }

    // // avoid crash in case of the situation notify() is called
    // if (mMessages == nullptr) {
    //     return mMessages;
    // }
    // when the looper is waiting here for the next message (with time delayed),
    // the message could be removed when the caller invokes removeMessage(),
    // therefore, the mMessages might be nullptr in while condition
    while (mMessages && SystemClock::uptimeMillisStartup() < mMessages->when) {
        // next message is not ready, wait until it is ready
        auto waitDuration = std::chrono::milliseconds(mMessages->when
                                                      - SystemClock::uptimeMillisStartup());
        mCondition.wait_for(lck, waitDuration);
    }
    // got a message
    if (mMessages) {
        // there is a next message
        Message *msg = mMessages;
        mMessages = msg->next;
        return msg;
    } else {
        // there is no next message
        return nullptr;
    }
}

bool MessageQueue::hasMessages(int what, Handler *handler) {
    // 获取当前信号量的锁
    unique_lock<mutex> lck(mMutex);
    auto msg = mMessages;
    while (msg != nullptr) {
        if (msg->what == what && msg->target == handler) {
            return true;
        }
        msg = msg->next;
    }
    return false;
}

void MessageQueue::removeMessage(int what, Handler *handler) {
    unique_lock<mutex> lck(mMutex);
    // when a message is removed, the dequeue method might be waiting for it
    // via wait_for(), so the wait process should be notified at this circumstance;
    bool needWake = false;

    // remove head
    auto msg = mMessages;
    ALOGD(TAG, "removeMessage what:%d, handler:%p", what, handler);
    while (msg != nullptr && msg->what == what && msg->target == handler) {
        Message *n = msg->next;
        mMessages = n;
        Message::recycle(msg);
        msg = n;
        needWake = true;
        ALOGD(TAG, "remove msg head");
        mCondition.notify_one();
    }

    // remove messages at other position
    while (msg != nullptr) {
        Message *n = msg->next;
        if (n != nullptr) {
            ALOGD(TAG, "n.what:%d, what:%d, n.handler:%p, handler:%p", n->what, what, n->target, handler);
            if (n->target == handler && n->what == what) {
                Message *nn = n->next;
                Message::recycle(n);
                msg->next = nn;
                needWake = true;
                ALOGD(TAG, "remove msg at other position");
                continue;
            }
        }
        msg = n;
    }
    // 是否唤醒
    if (needWake) {
        mCondition.notify_one();
    }
}

void MessageQueue::notify() {
    unique_lock<mutex> lck(mMutex);
    mCondition.notify_one();
}

void MessageQueue::removeAll() {
    unique_lock<mutex> lck(mMutex);
    // 遍历Message消息队列，依次回收每个消息Message
    while (mMessages != nullptr) {
        auto msg = mMessages;
        mMessages = mMessages->next;
        msg->next = nullptr;
        Message::recycle(msg);
    }
}

void MessageQueue::quit() {
    ALOGD(TAG, "quit()");
    mQuitting.store(true);
    notify();
}

}// namespace auralib
