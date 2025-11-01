//
// Created by Frewen.Wang on 2022/6/13.
//
#pragma once

namespace aura::utils {

class Looper;

class Message;

class MessageRunnable;

class Handler {
public:
    Handler();

    explicit Handler(Looper *looper);

    virtual ~Handler() = default;

    virtual void setLooper(Looper *looper);

    virtual Looper *getLooper();

    virtual void handleMessage(Message *msg);

    virtual void sendMessage(Message *msg);

    virtual void sendEmptyMessage(int what);

    virtual void sendMessageDelayed(Message *msg, long delayMillis);

    virtual void sendEmptyMessageDelayed(int what, long delayMillis);

    virtual bool hasMessages(int what, Handler *handler);

    /**
     * 移除Handler的中消息
     * @param what  消息标记
     * @param handler  对应Handler
     */
    virtual void removeMessage(int what, Handler *handler);

    // virtual Message *obtainMessage();
    //
    // virtual Message *obtainMessage(int what);
    //
    // virtual Message *obtainMessage(int what, void *obj);
    //
    // virtual Message *obtainMessage(int what, int arg1, int arg2);
    //
    // virtual Message *obtainMessage(int what, int arg1, int arg2, void *obj);

    virtual void post(MessageRunnable *callback);

    void post(int what, MessageRunnable *callback);

    virtual void postDelayed(MessageRunnable *callback, long delayMillis);

private:
    Looper *mLooper;
};

}// namespace auralib
