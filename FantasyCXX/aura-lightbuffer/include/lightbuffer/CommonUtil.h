//
// Created by Frewen.Wong on 2022/4/23.
//

#ifndef TYCHE_VISION_ABILITY_PLAIN_UTIL_H
#define TYCHE_VISION_ABILITY_PLAIN_UTIL_H

#include <string>

namespace aura ::light_buffer {

class CommonUtil {
public:
    /**
     * 字符串转大写
     * @param str 字符串
     * @return
     */
    static std::string toUpper(const std::string &str);

    /**
     * 字符串转小写
     * @param str 字符串
     * @return
     */
    static std::string toLower(const std::string &str);
};

} // namespace aura::light_buffer

#endif //TYCHE_VISION_ABILITY_PLAIN_UTIL_H
