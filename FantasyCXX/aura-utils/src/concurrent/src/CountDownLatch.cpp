//
// Created by Li,Wendong on 12/19/21.
//

#include "aura/utils/concurrent/count_down_latch.h"

namespace aura::utils {

CountDownLatch::CountDownLatch() {
    counter = 0;
}

void CountDownLatch::await() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return counter == 0; });
}

void CountDownLatch::countUp() {
    std::unique_lock<std::mutex> lock(mutex);
    counter++;
    if (counter == 0) {
        condition.notify_all();
    }
}

void CountDownLatch::countDown() {
    std::unique_lock<std::mutex> lock(mutex);
    counter--;
    if (counter == 0) {
        condition.notify_all();
    }
}

int CountDownLatch::getCount() const {
    std::unique_lock<std::mutex> lock(mutex);
    return counter;
}

}

