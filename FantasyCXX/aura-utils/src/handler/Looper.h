//
// Created by Frewen.Wang on 2022/6/13.
//
#pragma once

#include <cstdint>

namespace aura::utils {

class Message;

class Handler;

class MessageQueue;

class Looper {
public:
    /**
     * 构造函数
     */
    Looper();

    /**
     * 析构函数
     */
    ~Looper();

    /**
     * Looper的prepare方法
     * @return
     */
    static Looper *prepare();

    static void setForThread(Looper *looper);

    static Looper *getForThread();

    static void loop();

    /**
     * 将Handler消息加入消息队列中
     * @param msg  Message对象指针
     * @param uptimeMillis  系统运行时间的时间节点
     */
    void enqueueAtTime(Message *msg, const std::int64_t &uptimeMillis);

    bool hasMessages(int i, Handler *pHandler);

    bool hasMessage();

    /**
     * 移除消息队列中的消息
     * @param what  消息标记
     * @param handler
     */
    void removeMessage(int what, Handler *handler);

    Message *dequeue();

    /**
     * 停止looper的循环操作
     */
    void quit();

private:
    MessageQueue *mMsgQueue;
    /**
     * Looper对象的正在loop标志变量
     */
    bool looping;

    static void initTLSKey();

    static void threadDestructor(void *st);
};

}// namespace auralib
