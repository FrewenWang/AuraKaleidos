//
// Created by Li,Wendong on 12/19/21.
//

#include "CountLatch.h"

namespace aura::vision {

CountLatch::CountLatch() {
    counter = 0;
}

void CountLatch::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return counter == 0; });
}

void CountLatch::countUp() {
    std::unique_lock<std::mutex> lock(mutex);
    counter++;
    if (counter == 0) {
        condition.notify_all();
    }
}

void CountLatch::countDown() {
    std::unique_lock<std::mutex> lock(mutex);
    counter--;
    if (counter == 0) {
        condition.notify_all();
    }
}

int CountLatch::getCount() const {
    std::unique_lock<std::mutex> lock(mutex);
    return counter;
}

} // namespace aura::vision

