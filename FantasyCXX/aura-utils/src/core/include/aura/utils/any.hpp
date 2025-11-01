//
// Created by Frewen.Wang on 2024/7/2.
//
#include <iostream>
#include <string>
#include <memory>
#include <typeindex>

/**
 * 代码参考：https://juejin.cn/post/7027147023115616269
 * 代码参考：https://juejin.cn/post/7028642651465318436
 * 是定义一个变量来存放任意类型的数据，它类似于比如像纯面向对象语言java或.net中的Object类型。
 *  Boost::Any的实现比较简单，Any拥有一个模版构造函数，这使他可以接受任何类型的对象。
 *  真正的变量内容被封装在嵌套类类型的成员变量中，并且在嵌套类中使用typeid来记录真正的类型信息。
 */
struct any {
  any() : m_tpIndex(std::type_index(typeid(void))) {
  }

  any(const any &that) : m_ptr(that.Clone()), m_tpIndex(that.m_tpIndex) {
  }

  any(any &&that) : m_ptr(std::move(that.m_ptr)), m_tpIndex(that.m_tpIndex) {
  }

  /**
   * 创建智能指针时，对于一般的类型，通过 std::decay 来移除引用和 cv 符（即 const/volatile），从而获取原始类型
   * @tparam U
   * @param value
   */
  template<typename U, class = typename std::enable_if<!std::is_same<typename std::decay<U>::type, any>::value,
    U>::type>
  any(U &&value) : m_ptr(new Derived<typename std::decay<U>::type>(std::forward<U>(value))),
                   m_tpIndex(std::type_index(typeid(typename std::decay<U>::type))) {
  }

  bool IsNull() const { return !static_cast<bool>(m_ptr); }

  /// @note 类型不相同
  template<class U>
  bool Is() const {
    return m_tpIndex == std::type_index(typeid(U));
  }

  // 将 Any 转换为实际的类型
  template<class U>
  U &AnyCast() {
    if (!Is<U>()) {
      std::cout << "can not cast to " << typeid(U).name() << " from " << m_tpIndex.name() << std::endl;
      throw std::bad_cast();
    }

    /// @note 将基类指针转为实际的派生类型
    auto derived = dynamic_cast<Derived<U> *>(m_ptr.get());
    return derived->m_value; ///< 获取原始数据
  }

  any &operator=(const any &a) {
    if (m_ptr == a.m_ptr)
      return *this;

    m_ptr = a.Clone();
    m_tpIndex = a.m_tpIndex;
    return *this;
  }

private:
  struct Base;
  typedef std::unique_ptr<Base> BasePtr; /// 基类指针类型

  /// @note 基类不含模板参数
  ///
  struct Base {
    virtual ~Base() {
    }

    virtual BasePtr Clone() const = 0;
  };

  /// @note 派生类含有模板参数
  template<typename T>
  struct Derived : Base {
    /// @note
    template<typename U>
    Derived(U &&value) : m_value(std::forward<U>(value)) {
    }

    /// @note 将派生类对象赋值给了基类指针，通过基类擦除了派生类的原始数据类型
    /// < 用 unique_ptr 指针进行管理
    BasePtr Clone() const {
      return BasePtr(new Derived<T>(m_value));
    }

    T m_value;
  };

  BasePtr Clone() const {
    if (m_ptr != nullptr)
      return m_ptr->Clone();

    return nullptr;
  }

  BasePtr m_ptr;
  std::type_index m_tpIndex;
};
