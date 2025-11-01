//
// Created by Frewen.Wang on 2022/6/13.
//

#pragma once

#include <atomic>
#include <condition_variable>

namespace aura::utils {

class Message;

class Handler;

class MessageQueue {
private:
    /**
     * 消息队列
     */
    Message *mMessages;
    /**
     * 条件变量
     */
    std::condition_variable mCondition;
    std::mutex mMutex;
    /**
     * 原子变量
     */
    std::atomic<bool> mQuitting;

public:
    MessageQueue();

    ~MessageQueue();

    /**
     * 将Handler消息插入队列
     * @param msg  消息
     * @param when 插入时间
     */
    void enqueueAtTime(Message *msg, const std::int64_t &when);

    /**
     * MessageQueue出队列
     * @return
     */
    Message *dequeue();

    /**
     * 判断消息队列中是否有消息
     * @param what
     * @param handler
     * @return
     */
    bool hasMessages(int what, Handler *handler);

    /**
     * 消息队列中移除消息
     * @param what
     * @param handler
     */
    void removeMessage(int what, Handler *handler);

    /**
     * 唤醒等待线程
     */
    void notify();

    void removeAll();

    void quit();
};

} // namespace aura::aura_lib
