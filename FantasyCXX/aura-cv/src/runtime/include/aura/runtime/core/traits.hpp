#ifndef AURA_RUNTIME_CORE_TRAITS_TRAITS_HPP__
#define AURA_RUNTIME_CORE_TRAITS_TRAITS_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#if defined(AURA_BUILD_HOST)
#  include "aura/runtime/core/types/fp16.hpp"
#endif

#include <type_traits>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup traits Runtime Core Traits
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup traits
 * @{
*/

/**
 * @brief Template meta-programming struct to promote integer types.
 * 
 * @tparam Tp The input type.
 */
/// 这段代码定义了一个名为 Promote 的模板类，并通过模板全特化为不同的基础数据类型（如 MI_U8、MI_S8 等）提供了类型提升规则。
/// 其核心目的是在编译期自动为数值类型选择更高精度的类型，以防止计算过程中的溢出或精度损失。
template <typename Tp> struct Promote;
// 声明一个泛型模板 Promote，但未提供实现。这表示该模板的特化版本必须覆盖所有需要用到的类型，否则编译会失败
template <> struct Promote<MI_U8>  { using Type = MI_U16; };
template <> struct Promote<MI_S8>  { using Type = MI_S16; };
template <> struct Promote<MI_U16> { using Type = MI_U32; };
template <> struct Promote<MI_S16> { using Type = MI_S32; };
template <> struct Promote<MI_U32> { using Type = MI_U64; };
template <> struct Promote<MI_S32> { using Type = MI_S64; };
template <> struct Promote<MI_F32> { using Type = MI_F32; };

/**
 * @brief A helper struct that checks whether a type is a floating-point type.
 *
 * This struct inherits from std::is_floating_point<Tp> and is used to determine
 * whether the type Tp is a floating-point type.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct _is_floating_point : std::is_floating_point<Tp> {};
/**
 * @brief A type trait that identifies whether a template type is a floating-point type.
 *
 * This metafunction first removes any const and volatile qualifiers from the input type Tp using
 * std::remove_cv<Tp>::type. It then checks if the resulting type is a floating-point type by using
 * the _is_floating_point trait.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct is_floating_point : _is_floating_point<typename std::remove_cv<Tp>::type> {};

/**
 * @brief A helper struct that checks whether a type is integral.
 *
 * This struct inherits from std::integral_constant<bool, true> if the type T is integral,
 * otherwise it inherits from std::integral_constant<bool, false>.
 *
 * @tparam T The type to check.
 */
template <typename Tp> struct _is_integral : std::is_integral<Tp> {};
/**
 * @brief A type trait that identifies whether a template type is an integral data type.
 *
 * This trait inherits from _is_integral<typename std::remove_cv<Tp>::type>.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct is_integral : _is_integral<typename std::remove_cv<Tp>::type> {};

/**
 * @brief A helper struct that checks whether a type is an arithmetic type.
 *
 * This struct inherits from std::is_arithmetic<Tp> and is used to determine
 * whether the type Tp is an arithmetic type (i.e., an integer type, floating-point type,
 * or cv-qualified version of these types).
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct _is_arithmetic : std::is_arithmetic<Tp> {};
/**
 * @brief A type trait that identifies whether a template type is an arithmetic type.
 *
 * This metafunction first removes any const and volatile qualifiers from the input type Tp using
 * std::remove_cv<Tp>::type. It then checks if the resulting type is an arithmetic type by using
 * the _is_arithmetic trait.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct is_arithmetic : _is_arithmetic<typename std::remove_cv<Tp>::type> {};

/**
 * @brief A helper struct that checks whether a type is a signed type.
 *
 * This struct inherits from std::is_signed<Tp> and is used to determine
 * whether the type Tp is a signed type.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct _is_signed : std::is_signed<Tp> {};
/**
 * @brief A type trait that identifies whether a template type is a signed type.
 *
 * This metafunction first removes any const and volatile qualifiers from the input type Tp using
 * std::remove_cv<Tp>::type. It then checks if the resulting type is a signed type by using
 * the _is_signed trait.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct is_signed : _is_signed<typename std::remove_cv<Tp>::type> {};

/**
 * @brief A helper struct that checks whether a type is a scalar type.
 *
 * This struct inherits from std::is_scalar<Tp> and is used to determine
 * whether the type Tp is a scalar type.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct _is_scalar : std::is_scalar<Tp> {};
/**
 * @brief A type trait that identifies whether a template type is a scalar type.
 *
 * This metafunction first removes any const and volatile qualifiers from the input type Tp using
 * std::remove_cv<Tp>::type. It then checks if the resulting type is a scalar type by using
 * the _is_scalar trait.
 *
 * @tparam Tp The type to check.
 */
template <typename Tp> struct  is_scalar : _is_scalar<typename std::remove_cv<Tp>::type> {};

#if defined(AURA_BUILD_HOST)
template <> struct Promote<MI_F16> { using Type = MI_F32; };
template <> struct _is_floating_point<MI_F16> : std::true_type {};
template <> struct _is_integral<MI_F16> : std::false_type {};
template <> struct _is_arithmetic<MI_F16> : std::true_type {};
template <> struct _is_signed<MI_F16> : std::true_type {};
template <> struct _is_scalar<MI_F16> : std::true_type {};
#endif // AURA_BUILD_HOST

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_TRAITS_TRAITS_HPP__