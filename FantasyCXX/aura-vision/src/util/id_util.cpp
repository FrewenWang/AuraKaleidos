//
// Created by v_zhangtieshuai on 2021/1/27.
//

#include "id_util.h"
#include <limits>

pthread_once_t FaceIdUtil::_s_create_once_control = PTHREAD_ONCE_INIT;
pthread_once_t FaceIdUtil::_s_release_once_control = PTHREAD_ONCE_INIT;
FaceIdUtil* FaceIdUtil::_s_instance = nullptr;

FaceIdUtil *FaceIdUtil::instance() {
    pthread_once(&_s_create_once_control, create);
    return _s_instance;
}

void FaceIdUtil::destroy() {
    pthread_once(&_s_release_once_control, release);
}

void FaceIdUtil::create() {
    if (_s_instance == nullptr) {
        _s_instance = new FaceIdUtil();
    }
}

void FaceIdUtil::release() {
    if (_s_instance != nullptr) {
        delete _s_instance;
        _s_instance = nullptr;
    }
}

FaceIdUtil::FaceIdUtil() : _id(0) {

}

int64_t FaceIdUtil::produce() {
    std::unique_lock<std::mutex> lock{_produce_id_mutex};
    constexpr static int int_limit_max = std::numeric_limits<int>::max();
    if (_id == int_limit_max) {
        return (_id = 1);
    }
    return ++_id;
}
