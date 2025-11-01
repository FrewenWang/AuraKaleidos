//
// Created by Frewen.Wang on 2024/7/2.
//
#pragma once

#include <deque>
#include <mutex>

constexpr size_t POOL_SIZE = 10;

namespace aura::utils {

/**
 * 优点：
 * 1.支持参数不同的构造函数
 *
 * 缺点：
 * 1. 对象用完之后需要手动回收，用起来不够方便，更大的问题是存在忘记回收的风险；
 * 2. 继承自ObjectPoolV1的所有子类需要重载recycleClear方法，避免繁琐
 * @tparam T
 * @tparam pool_size
 */
template<typename T, size_t pool_size = POOL_SIZE>
class object_pool_v1 {

public:
    object_pool_v1() = default;
    
    virtual ~object_pool_v1() = default;
    
    template<typename ...Args>
    static T *obtain(Args... args);
    
    static void recycle(T *obj);

protected:
    static std::mutex sPoolMutex;
    static std::deque<T *> sFreePool;
    
    /**
     * 定义recycleClear函数作为对象池的纯虚函数，
     * 继承自ObjectPool的子类需要重载的这个函数
     * 在对象数据回收到对象池之前，需要清空所有的对象属性的值，
     * 避免下次从对象池取出之后造成脏数据污染
     * @return
     */
    virtual bool recycleClear() = 0;
};

/// 定义类的声明中静态变量
template<typename T, size_t pool_size>
std::deque<T *> object_pool_v1<T, pool_size>::sFreePool;

///定义C++中互斥锁变量，
template<typename T, size_t pool_size>
std::mutex object_pool_v1<T, pool_size>::sPoolMutex;


/**
 *
 */
template<typename T, size_t pool_size>
template<typename ...Args>
T *object_pool_v1<T, pool_size>::obtain(Args... args) {
    std::unique_lock<std::mutex> lck(sPoolMutex);
    if (sFreePool.empty()) {
        return new T(args...);
    }
    
    auto *obj = sFreePool.front();
    sFreePool.pop_front();
    // 回收的时候进行数据清除
    obj->recycleClear();
    return obj;
}


template<typename T, size_t pool_size>
void object_pool_v1<T, pool_size>::recycle(T *obj) {
    std::unique_lock<std::mutex> lck(sPoolMutex);
    if (obj == nullptr) {
        return;
    }
    
    if (sFreePool.size() >= pool_size) {
        delete obj;
        return;
    }
    if (obj->recycleClear()) {
        throw "ObjectPool Must Override recycleClear Function!!!";
    }
    sFreePool.push_back(obj);
}


}
