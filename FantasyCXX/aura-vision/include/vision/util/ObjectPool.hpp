#ifndef VISION_OBJECT_POOL_H
#define VISION_OBJECT_POOL_H

#include <deque>
#include <mutex>

namespace aura::vision {

constexpr size_t POOL_SIZE = 10;

template <typename T, size_t pool_size = POOL_SIZE>
class ObjectPool {

public:
    ObjectPool() = default;
    virtual ~ObjectPool() = default;
    template <typename ...Args>
    static T* obtain(Args... args);
    static void recycle(T* obj);
protected:
    static std::mutex sPoolMutex;
    static std::deque<T*> _free_pool;
};

template <typename T, size_t pool_size>
std::deque<T*> ObjectPool<T, pool_size>::_free_pool;

template <typename T, size_t pool_size>
std::mutex ObjectPool<T, pool_size>::sPoolMutex;

//template <typename T, size_t pool_size>
//ObjectPool<T, pool_size>::~ObjectPool() {
//    for (auto& obj : _free_pool) {
//        delete obj;
//        obj = nullptr;
//    }
//    _free_pool.clear();
//}

template <typename T, size_t pool_size>
template <typename ...Args>
T* ObjectPool<T, pool_size>::obtain(Args... args) {
    std::unique_lock<std::mutex> lck(sPoolMutex);
    if (_free_pool.empty()) {
        return new T(args...);
    }

    auto* obj = _free_pool.front();
    _free_pool.pop_front();
    return obj;
}

template <typename T, size_t pool_size>
void ObjectPool<T, pool_size>::recycle(T* obj) {
    std::unique_lock<std::mutex> lck(sPoolMutex);
    if (obj == nullptr) {
        return;
    }

    if (_free_pool.size() >= pool_size) {
        delete obj;
        return;
    }
    _free_pool.push_back(obj);
}

} // namespace vision

#endif //VISION_OBJECT_POOL_H
