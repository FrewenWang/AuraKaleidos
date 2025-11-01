
#ifndef VISION_TIMER_EXIT_H
#define VISION_TIMER_EXIT_H

namespace timer {
    class TimerExit {
    private:
        long _start_time = 0;
    public:
        TimerExit();

        // 终止程序
        void finish();

        // 设定起始时间
        void set_start_time();

    };
}

#endif //VISION_TIMER_EXIT_H
