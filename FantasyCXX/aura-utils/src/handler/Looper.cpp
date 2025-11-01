//
// Created by Frewen.Wang on 2022/6/13.
//

#include "Looper.h"
#include "mq/MessageQueue.h"
#include "mq/Message.h"
#include "Handler.h"
#include <pthread.h>
#include "aura/aura_utils/utils/AuraLog.h"

namespace aura::utils {

static const char *TAG = "Looper";

using namespace std;
static pthread_once_t gTLSOnce = PTHREAD_ONCE_INIT;
/** 线程存储：创建一个类型pthread_key_t的变量 */
static pthread_key_t gTLSKey = 0;
static Looper *threadLooper = nullptr;

Looper::Looper() {
    mMsgQueue = new MessageQueue();
    looping = false;
}

Looper::~Looper() {
    looping = false;
}

void Looper::initTLSKey() {
    // 通过调用pthread_key_create的创建
    // 参数一：上面声明的pthread_key_t变量
    // 参数二：第二个参数是一个清理函数，用来在线程释放该线程存储的时候被调用。
    int error = pthread_key_create(&gTLSKey, threadDestructor);
    if (error) {
        ALOGW(TAG, "Looper initTLSKey Error: %d", error);
    }
}

Looper *Looper::prepare() {
    Looper *looper = Looper::getForThread();
    if (looper == nullptr) {
        looper = new Looper();
        // 我们我们针对当前线程存储特征变量的时候
        // 我们可以调用pthread_setspecific函数
        // 参数一：前面声明的pthread_key_t变量
        // 参数二：void*变量，这样你可以存储任何类型的值。
        Looper::setForThread(looper);
    }
    return looper;
}

void Looper::loop() {
    // 获取当前线程的绑定的Looper对象
    Looper *me = getForThread();
    if (me == nullptr) {
        ALOGE(TAG, "No Looper; Looper.prepare() wasn't called on this thread");
        throw std::domain_error("Looper.prepare() wasn't called on this thread");
    }
    // 获取当前线程的MessageQueue
    MessageQueue *queue = me->mMsgQueue;
    me->looping = true;
    while (me->looping) {
        Message *msg = queue->dequeue();
        if (msg == nullptr) {
            // if all the messages are consumed,
            // then a nullptr is returned when dequeue is revoked,
            // so use "break" here will terminate
            // the handler thread, this is not the expected result.
            // continue;
            break;
        }
        if (msg->callback != nullptr) {
            msg->callback->onMessage(msg);
        } else if (msg->target != nullptr) {
            msg->target->handleMessage(msg);
        }
        Message::recycle(msg);
    }
    // 如果逻辑走到这个地方，说明looper已经停止
    setForThread(nullptr);
}


void Looper::threadDestructor(void *st) {
    delete threadLooper;
}

Looper *Looper::getForThread() {
    int result = pthread_once(&gTLSOnce, initTLSKey);
    // 如果需要取出所存储的值，调用pthread_getspecific()。
    // 该函数的参数为前面提到的pthread_key_t变量，该函数返回void *类型的值。
    return (Looper *) pthread_getspecific(gTLSKey);
}

void Looper::setForThread(Looper *looper) {
    Looper *old = getForThread();
    pthread_setspecific(gTLSKey, looper);
    delete old;
}

void Looper::enqueueAtTime(Message *msg, const std::int64_t &uptimeMillis) {
    if (mMsgQueue == nullptr || msg == nullptr) {
        return;
    }
    mMsgQueue->enqueueAtTime(msg, uptimeMillis);
}

bool Looper::hasMessages(int what, Handler *pHandler) {
    return mMsgQueue->hasMessages(what, pHandler);
}

bool Looper::hasMessage() {
    return false;
}

void Looper::removeMessage(int what, Handler *handler) {
    mMsgQueue->removeMessage(what, handler);
}

Message *Looper::dequeue() {
    return mMsgQueue->dequeue();
}

void Looper::quit() {
    looping = false;
    mMsgQueue->quit();
}

}// namespace auralib