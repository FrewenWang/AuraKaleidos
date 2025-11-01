#include <string>
#include <functional>
#include <tuple>
#include <map>

#include "any.hpp"

const int MaxObjectNum = 10;

class ObjectPool {
    template <typename T, typename... Args>
    using Constructor = std::function<std::shared_ptr<T>(Args...)>;

public:
    ObjectPool() : needClear(false){
    }

    ~ObjectPool(){
        needClear = true;
    }

    //默认创建多少个对象
    template <typename T, typename... Args>
    void Create(int num) {
        if (num <= 0 || num > MaxObjectNum)
            throw std::logic_error("object num errer");

        auto constructName = typeid(Constructor<T, Args...>).name();

        /// @note 用函数对象封装
        Constructor<T, Args...> f = [constructName, this](Args... args)
        {
            return createPtr<T>(std::string(constructName), args...);
        };

        m_map.emplace(typeid(T).name(), f); ///< 存储函数对象

        m_counter.emplace(constructName, num);
    }

    /// @note 返回智能指针
    template <typename T, typename... Args>
    std::shared_ptr<T> createPtr(std::string &constructName, Args... args)
    {
        /// 调用构造函数创建指针
        return std::shared_ptr<T>(new T(args...), [constructName, this](T *t)
                                  {///< Deleter 回收器
                                      if (needClear)
                                          delete[] t;
                                      else
                                          m_object_map.emplace(constructName, std::shared_ptr<T>(t)); ///< 放回对象池（存储对象指针）
                                  });
    }

    template <typename T, typename... Args>
    std::shared_ptr<T> Get(Args... args)
    {
        using ConstructType = Constructor<T, Args...>;

        std::string constructName = typeid(ConstructType).name();
        auto range = m_map.equal_range(typeid(T).name()); ///< 取得满足类型名的函数对象范围

        for (auto it = range.first; it != range.second; ++it)
        {
            /// @note 取得范围中满足类型条件的函数对象
            /// 继而利用它获取（或创建）对象指针
            if (it->second.Is<ConstructType>())
            {
                auto ptr = GetInstance<T>(constructName, args...);

                if (ptr != nullptr)
                    return ptr;

                return CreateInstance<T, Args...>(it->second, constructName, args...);
            }
        }

        return nullptr;
    }

private:
    template <typename T, typename... Args>
    std::shared_ptr<T> CreateInstance(any &any,
                                      std::string &constructName, Args... args)
    {
        using ConstructType = Constructor<T, Args...>;
        ConstructType f = any.AnyCast<ConstructType>();
        /// @note 返回智能指针
        return createPtr<T, Args...>(constructName, args...);
    }

    /// @note 初始化对象池
    template <typename T, typename... Args>
    void InitPool(T &f, std::string &constructName, Args... args)
    {
        int num = m_counter[constructName]; ///< 获取该类型需创建的对象个数

        if (num != 0)
        {
            for (int i = 0; i < num - 1; i++)
            {
                m_object_map.emplace(constructName, f(args...)); ///< 直接构造对象并存储
            }
            m_counter[constructName] = 0;
        }
    }

    /// @note 从对象池中获取对象
    template <typename T, typename... Args>
    std::shared_ptr<T> GetInstance(std::string &constructName, Args... args)
    {
        /// @note 寻找对象池中是否已经存有该对象
        auto it = m_object_map.find(constructName);
        if (it == m_object_map.end())
            return nullptr;

        /// @note 取出并转型该指针
        auto ptr = it->second.AnyCast<std::shared_ptr<T>>();
        if (sizeof...(Args) > 0)
            *ptr.get() = std::move(T(args...));

        m_object_map.erase(it); ///< 从对象池中除名该对象
        return ptr;
    }

private:
    std::multimap<std::string, any> m_map;
    std::multimap<std::string, any> m_object_map;
    std::map<std::string, int> m_counter;
    bool needClear;
};
