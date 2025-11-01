//
// Created by Frewen.Wong on 2022/4/23.
//

#pragma once

#include <string>

namespace aura::utils {

class SystemClock {
public:
    /**
     * 获取当前系统时间戳
     * @return 当前时间戳
     */
    static int64_t nowMillis();

    /**
    * 获取当前系统时间戳  与安卓系统接口重名，实际调用结果为系统当前时间，修改无效
    * @return 当前时间戳
    */
    static int64_t uptimeMillis();

    /**
     * 获取系统启动时间 timer计时使用此接口
     * @return
     */
    static int64_t uptimeMillisStartup();

    /**
     * 获取当前系统的时间格式化表示
     * @return
     */
    static std::string currentTimeStr();
};

}// namespace auralib
