
#ifndef VISION_ID_UTIL_H
#define VISION_ID_UTIL_H

#include <thread>
#include <mutex>

class FaceIdUtil {
public:
    static FaceIdUtil* instance();

    static void destroy();

    int64_t produce();

    FaceIdUtil(const FaceIdUtil& value) = delete;

    FaceIdUtil& operator=(const FaceIdUtil& value) = delete;

    FaceIdUtil(FaceIdUtil&& value) = delete;

    FaceIdUtil& operator=(FaceIdUtil&& value) = delete;

private:
    FaceIdUtil();

    ~FaceIdUtil() = default;

    static void create();

    static void release();

private:
    int64_t _id;
    std::mutex _produce_id_mutex;

    static pthread_once_t _s_create_once_control;
    static pthread_once_t _s_release_once_control;
    static FaceIdUtil* _s_instance;
};

#endif //VISION_ID_UTIL_H
