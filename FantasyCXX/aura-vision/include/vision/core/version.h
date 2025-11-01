#pragma once

// 8-bit unsigned int, min: 0, max: 255
#define VISION_VERSION_MAJOR 2

// 8-bit unsigned int, min: 0, max: 255
#define VISION_VERSION_MINOR 1

// 8-bit unsigned int, min: 0, max: 255
#define VISION_VERSION_PATCH 1

#define VISION_STRING(x) #x
#define VISION_TOSTRING(x) VISION_STRING(x)

// get version string
#define VISION_VERSION_STR                                                                                             \
    "AbilityVersion_V" VISION_TOSTRING(VISION_VERSION_MAJOR) "." VISION_TOSTRING(                                   \
        VISION_VERSION_MINOR) "." VISION_TOSTRING(VISION_VERSION_PATCH)

#define VIS_VISION_VERSION_NUMBER                                                                                      \
    ((VISION_VERSION_MAJOR << 24) | (VISION_VERSION_MINOR << 16) | (VISION_VERSION_PATCH << 8))

class VisionVersion {
public:
    static std::string getVersion() {
        char data[10], time[10];
        VisionVersion::getCompileDate(data);
        VisionVersion::getCompileTime(time);
        std::string version = VISION_VERSION_STR;
        version.append("_").append(data).append("_").append(time);
        return version;
    }

private:
    static void getCompileDate(char *dateStr) {
        std::string date = __DATE__;                          // 取编译时间
        int year = std::stoi(date.substr(date.length() - 4)); // stoi会做范围检查，atoi不会
        std::string monthes[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        int month = 0;
        for (int i = 0; i < 12; i++) {
            if (date.find(monthes[i]) != std::string::npos) {
                month = i + 1;
                break;
            }
        }
        int day = std::stoi(date.substr(4, 2));
        sprintf(dateStr, "%02d%02d%02d", year, month, day); // 任意格式化
    }

    static void getCompileTime(char *timeStr) {
        std::string time = __TIME__;
        int hour = (uint8_t) std::stoi(time.substr(0, 2));
        int minute = (uint8_t) std::stoi(time.substr(3, 5));
        int second = (uint8_t) std::stoi(time.substr(6, 8));
        sprintf(timeStr, "%02d%02d%02d", hour, minute, second); // 任意格式化
    }
};