//
// Created by Frewen.Wang on 2022/7/22.
//
#include "Handler.h"
#include "Looper.h"
#include "aura/aura_utils/utils/SystemClock.h"
#include "mq/Message.h"
#include <iostream>

namespace aura::utils {

Handler::Handler() {
    mLooper = nullptr;
}

Handler::Handler(Looper *looper) {
    mLooper = looper;
}

void Handler::setLooper(Looper *looper) {
    mLooper = looper;
}

Looper *Handler::getLooper() {
    return mLooper;
}

void Handler::sendMessage(Message *msg) {
    if (msg != nullptr) {
        sendMessageDelayed(msg, 0);
    }
}

void Handler::sendEmptyMessage(int what) {
    sendEmptyMessageDelayed(what, 0);
}

void Handler::sendMessageDelayed(Message *msg, long delayMillis) {
    msg->target = this;
    mLooper->enqueueAtTime(msg, SystemClock::uptimeMillisStartup() + delayMillis);
}

void Handler::sendEmptyMessageDelayed(int what, long delayMillis) {
    Message *msg = Message::obtain(what, this);
    mLooper->enqueueAtTime(msg, SystemClock::uptimeMillisStartup() + delayMillis);
}

void Handler::post(MessageRunnable *callback) {
    Message *msg = Message::obtain();
    msg->callback = callback;
    msg->target = this;
    mLooper->enqueueAtTime(msg, SystemClock::uptimeMillisStartup());
}

void Handler::post(int what, MessageRunnable *callback) {
    Message *msg = Message::obtain();
    msg->what = what;
    msg->callback = callback;
    msg->target = this;
    mLooper->enqueueAtTime(msg, SystemClock::uptimeMillisStartup());
}

void Handler::postDelayed(MessageRunnable *callback, long delayMillis) {
    Message *msg = Message::obtain();
    msg->callback = callback;
    msg->target = this;
    mLooper->enqueueAtTime(msg, SystemClock::uptimeMillisStartup() + delayMillis);
}

bool Handler::hasMessages(int what, Handler *handler) {
    return mLooper->hasMessages(what, handler);
}

void Handler::handleMessage(Message *msg) {
    std::cout << "message : " << msg->what << std::endl;
}

void Handler::removeMessage(int what, Handler *handler) {
    mLooper->removeMessage(what, handler);
}

} // namespace aura::aura_lib