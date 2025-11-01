//
// Created by v_guojinlong on 2019-03-19.
//

#include "timer_exit.h"
#include <time.h>
#include <cstdlib>
#include "iostream"
#include "sys/time.h"

namespace timer {

    TimerExit::TimerExit() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
    }

    void TimerExit::finish() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long end_time = tv.tv_sec;
        if (end_time - _start_time > 20 * 60) {
            std::cout << " 终止程序启动 " << std::endl;
            std::abort();
        }
    }

    void TimerExit::set_start_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        this->_start_time = tv.tv_sec;
    }

}
