//
// Created by Frewen.Wang on 2022/6/13.
//
#pragma once

#include <cstdint>

namespace aura::utils {

class Handler;

class Message;

/**
 * 消息接收执行
 */
class MessageRunnable {
public:
    virtual void onMessage(Message *msg) { };

    virtual ~MessageRunnable() = default;
};

/**
 *  消息实体
 */
class Message {

public:
    Message();

    explicit Message(int what);

    Message(int what, Handler *handler);

    Message(int what, int arg1, int arg2);

    Message(int what, int arg1, int arg2, Handler *handler);

    ~Message();

    static Message *obtain();

    static Message *obtain(int what);

    static Message *obtain(int what, Handler *handler);

    static Message *obtain(int what, int arg1, int arg2);

    static Message *obtain(int what, int arg1, int arg2, Handler *handler);

    static Message *obtain(Handler *h);

    static Message *obtain(Handler *h, int what);

    static Message *obtain(Handler *h, int what, void *obj);

    static Message *obtain(Handler *h, int what, int arg1, int arg2);

    static Message *obtain(Handler *h, int what, int arg1, int arg2, void *obj);

    static void recycle(Message *msg) {
        delete msg;
        msg = nullptr;
    }

public:
    /**
     * 清除消息
     */
    void clear();

    /**
     * 发送给目标
     */
    void sendToTarget();

public:
    int what;
    int arg1;
    int arg2;
    void *obj;
    std::int64_t when;

    Handler *target;
    Message *next;
    MessageRunnable *callback;
};

}// namespace auralib
