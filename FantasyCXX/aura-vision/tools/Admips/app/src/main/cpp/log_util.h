#include <android/log.h>

static const char* TAG = "AdmipsJni";

class WLog {
public:
    static void i(const char* msg) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "%s", msg);
    }

    static void e(const char* msg) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg);
    }

    static void i(const std::string& msg) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "%s", msg.c_str());
    }

    static void e(const std::string& msg) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg.c_str());
    }
};