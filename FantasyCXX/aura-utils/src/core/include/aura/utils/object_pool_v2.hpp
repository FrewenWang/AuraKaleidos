//
// Created by Frewen.Wang on 2024/7/2.
//
#include <string>
#include <functional>
#include <tuple>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "any.hpp"

constexpr size_t POOL_SIZE = 10;

namespace aura {
namespace aura_utils {

class ObjectPool {
    template<typename T, typename... Args>
    using Constructor = std::function<std::shared_ptr<T>(Args...)>;

public:
    ObjectPool() : m_storage(std::make_shared<Storage>()) { }

    ObjectPool(const ObjectPool &) = delete;
    ObjectPool &operator=(const ObjectPool &) = delete;
    
    ~ObjectPool() {
        std::vector<CachedObject> objects;
        {
            std::lock_guard<std::mutex> lock(m_storage->mutex);
            m_storage->accepting = false;
            for (const auto &item : m_storage->objects)
                objects.push_back(item.second);
            m_storage->objects.clear();
        }
        for (const auto &object : objects)
            object.destroy(object.pointer);
    }
    
    //默认创建多少个对象
    template<typename T, typename... Args>
    void Create(int num) {
        if (num <= 0 || static_cast<size_t>(num) > POOL_SIZE)
            throw std::logic_error("object num errer");
        
        auto constructName = typeid(Constructor<T, Args...>).name();
        
        /// @note 用函数对象封装
        Constructor<T, Args...> f = [constructName, this](Args... args) {
            return createPtr<T>(std::string(constructName), args...);
        };
        
        m_map.emplace(typeid(T).name(), f); ///< 存储函数对象
        
        std::lock_guard<std::mutex> lock(m_storage->mutex);
        m_storage->capacities[constructName] = static_cast<size_t>(num);
    }
    
    /// @note 返回智能指针
    template<typename T, typename... Args>
    std::shared_ptr<T> createPtr(const std::string &constructName, Args... args) {
        return wrapPtr<T>(constructName, new T(args...));
    }
    
    template<typename T, typename... Args>
    std::shared_ptr<T> Get(Args... args) {
        using ConstructType = Constructor<T, Args...>;
        
        std::string constructName = typeid(ConstructType).name();
        auto range = m_map.equal_range(typeid(T).name()); ///< 取得满足类型名的函数对象范围
        
        for (auto it = range.first; it != range.second; ++it) {
            /// @note 取得范围中满足类型条件的函数对象
            /// 继而利用它获取（或创建）对象指针
            if (it->second.Is<ConstructType>()) {
                auto ptr = GetInstance<T>(constructName, args...);
                
                if (ptr != nullptr)
                    return ptr;
                
                return CreateInstance<T, Args...>(it->second, args...);
            }
        }
        
        return nullptr;
    }

private:
    template<typename T, typename... Args>
    std::shared_ptr<T> CreateInstance(any &any, Args... args) {
        using ConstructType = Constructor<T, Args...>;
        ConstructType f = any.AnyCast<ConstructType>();
        /// @note 返回智能指针
        return f(args...);
    }
    
    /// @note 从对象池中获取对象
    template<typename T, typename... Args>
    std::shared_ptr<T> GetInstance(std::string &constructName, Args... args) {
        /// @note 寻找对象池中是否已经存有该对象
        std::lock_guard<std::mutex> lock(m_storage->mutex);
        auto it = m_storage->objects.find(constructName);
        if (it == m_storage->objects.end())
            return nullptr;
        
        /// @note 取出并转型该指针
        T *ptr = static_cast<T *>(it->second.pointer);
        if (sizeof...(Args) > 0)
            *ptr = T(args...);
        
        m_storage->objects.erase(it); ///< 从对象池中除名该对象
        return wrapPtr<T>(constructName, ptr);
    }

private:
    struct CachedObject {
        void *pointer;
        void (*destroy)(void *);
    };

    struct Storage {
        std::mutex mutex;
        bool accepting = true;
        std::multimap<std::string, CachedObject> objects;
        std::map<std::string, size_t> capacities;
    };

    template<typename T>
    std::shared_ptr<T> wrapPtr(const std::string &constructName, T *pointer) {
        auto storage = m_storage;
        return std::shared_ptr<T>(pointer, [constructName, storage](T *object) noexcept {
            bool cached = false;
            try {
                std::lock_guard<std::mutex> lock(storage->mutex);
                const auto capacity = storage->capacities.find(constructName);
                if (storage->accepting && capacity != storage->capacities.end()
                    && storage->objects.count(constructName) < capacity->second) {
                    storage->objects.emplace(constructName, CachedObject{
                        object, [](void *value) { delete static_cast<T *>(value); }
                    });
                    cached = true;
                }
            } catch (...) {
                // shared_ptr deleters must not throw; fall back to releasing the object.
            }
            if (!cached)
                delete object;
        });
    }

    std::multimap<std::string, any> m_map;
    std::shared_ptr<Storage> m_storage;
};

} }
