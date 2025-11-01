//
// Created by Frewen.Wang on 2022/6/13.
//

#include "Message.h"
#include "../Handler.h"

namespace aura::utils {

Message::Message() {
    clear();
}

Message::Message(int what) {
    clear();
    this->what = what;
}

Message::Message(int what, Handler *handler) {
    clear();
    this->what = what;
    this->target = handler;
}

Message::Message(int what, int arg1, int arg2) {
    clear();
    this->what = what;
    this->arg1 = arg1;
    this->arg2 = arg2;
}

Message::Message(int what, int arg1, int arg2, Handler *handler) {
    clear();
    this->what = what;
    this->arg1 = arg1;
    this->arg2 = arg2;
    this->target = handler;
}

Message::~Message() {
}

Message *Message::obtain() {
    return new Message();
}

Message *Message::obtain(int what) {
    return new Message(what);
}

Message *Message::obtain(int what, Handler *handler) {
    return new Message(what, handler);
}

Message *Message::obtain(int what, int arg1, int arg2) {
    return new Message(what, arg1, arg2);
}

Message *Message::obtain(int what, int arg1, int arg2, Handler *handler) {
    return new Message(what, arg1, arg2, handler);
}

Message *Message::obtain(Handler *h) {
    Message *m = obtain();
    m->target = h;
    return m;
}

Message *Message::obtain(Handler *h, int what) {
    Message *m = obtain();
    m->target = h;
    m->what = what;
    return m;
}

Message *Message::obtain(Handler *h, int what, void *obj) {
    Message *m = obtain();
    m->target = h;
    m->what = what;
    m->obj = obj;
    return m;
}

Message *Message::obtain(Handler *h, int what, int arg1, int arg2) {
    Message *m = obtain();
    m->target = h;
    m->arg1 = arg1;
    m->arg2 = arg2;
    return m;
}

Message *Message::obtain(Handler *h, int what, int arg1, int arg2, void *obj) {
    Message *m = obtain();
    m->target = h;
    m->what = what;
    m->arg1 = arg1;
    m->arg2 = arg2;
    m->obj = obj;
    return m;
}

void Message::clear() {
    what = 0;
    arg1 = 0;
    arg2 = 0;
    obj = nullptr;
    target = nullptr;
    next = nullptr;
}

void Message::sendToTarget() {
    target->sendMessage(this);
}

}