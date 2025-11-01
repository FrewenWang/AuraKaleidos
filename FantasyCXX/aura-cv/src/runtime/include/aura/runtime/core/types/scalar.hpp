#ifndef AURA_RUNTIME_CORE_TYPES_SCALAR_HPP__
#define AURA_RUNTIME_CORE_TYPES_SCALAR_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/saturate.hpp"

#if !defined(AURA_BUILD_XTENSA)
#  include <iostream>
#  include <sstream>
#  include <string>
#  include <utility>
#  include <vector>
#endif
/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup types Runtime Core Types
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup types
 * @{
 */

/**
 * 4元素模板类
 * @brief Template class for a 4-element scalar arrary.
 *
 * The Scalar_ class has an array member named 'm_val', which can be just as typical 4-element vectors and can
 * also represent complex numbers.
 *
 * It supports element-wise arithmetic and comparison operations and can be converted into vector and complex
 * number forms. Additionally, it includes a cast operator and printing operations for its members.
 *
 * @tparam Tp0 Type of the scalar values.
 */
template<typename Tp0>
class Scalar_
{
public:
    using value_type = Tp0;

    /**
     * @brief Default constructor initializing all elements to zero.
     */
    Scalar_() : m_val{0, 0, 0, 0}
    {}

    /**
     * @brief Constructor initializing the scalar with given elements.
     *
     * @param v0 The first element.
     * @param v1 The second element.
     * @param v2 The third element (optional, defaults to 0).
     * @param v3 The fourth element (optional, defaults to 0).
     */
    Scalar_(Tp0 v0, Tp0 v1, Tp0 v2 = 0, Tp0 v3 = 0) : m_val{v0, v1, v2, v3}
    {}

    /**
     * @brief Constructor initializing the scalar with a single value, setting other elements to zero.
     *
     * @param v0 The value for the first element.
     */
    Scalar_(Tp0 v0) : m_val{v0, 0, 0, 0}
    {}

    /**
     * @brief Copy constructor from another Scalar_ object.
     *
     * @param sz Scalar object to be copied.
     */
    Scalar_(const Scalar_ &sz) : m_val{sz.m_val[0], sz.m_val[1], sz.m_val[2], sz.m_val[3]}
    {}

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded stream insertion operator for printing the scalar.
     *
     * @param os The output stream.
     * @param sz Scalar object to be printed.
     *
     * @return Reference to the output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Scalar_ &sz)
    {
        os << "[" << sz.m_val[0] << ", " << sz.m_val[1] << ", " << sz.m_val[2] << ", " << sz.m_val[3] << "]";
        return os;
    }

    /**
     * @brief Get a string representation of the scalar.
     *
     * @return The string representation of the scalar.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    /**
     * @brief Convert the scalar to a vector of elements.
     *
     * @tparam Tp1 The type for the output vector elements (defaults to the scalar's type).
     *
     * @param n The size of the output vector (defaults to 4).
     *
     * @return The vector containing scalar elements.
     */
    template<typename Tp1 = Tp0>
    std::vector<Tp1> ToVector(MI_U32 n = 4) const
    {
        if (n <= 4)
        {
            return std::vector<Tp1>{SaturateCast<Tp1>(m_val[0]), SaturateCast<Tp1>(m_val[1]),
                                    SaturateCast<Tp1>(m_val[2]), SaturateCast<Tp1>(m_val[3])};
        }
        else
        {
            return std::vector<Tp1>(n, SaturateCast<Tp1>(m_val[0]));
        }
    }
#endif

    /**
     * @brief Create a Scalar_ object with all elements set to the same value.
     *
     * @param v0 The value to set for all elements.
     *
     * @return A Scalar_ object with all elements set to the given value.
     */
    static Scalar_<Tp0> All(Tp0 v0)
    {
        return Scalar_<Tp0>(v0, v0, v0, v0);
    }

    /**
     * @brief Conversion operator to convert the Scalar_ to a different type.
     *
     * @tparam Tp1 The type to convert the elements to.
     *
     * @return A Scalar_ object with elements casted to the specified type.
     */
    template<typename Tp1>
    operator Scalar_<Tp1>() const
    {
        return Scalar_<Tp1>(SaturateCast<Tp1>(m_val[0]),
                            SaturateCast<Tp1>(m_val[1]),
                            SaturateCast<Tp1>(m_val[2]),
                            SaturateCast<Tp1>(m_val[3]));
    }

    /**
     * @brief Multiply two Scalar_ objects element-wise with an optional scaling factor.
     *
     * @param a The Scalar_ object to multiply.
     * @param scale The scaling factor (default is 1).
     *
     * @return A Scalar_ object with element-wise multiplication and scaling.
     */
    Scalar_<Tp0> Mul(const Scalar_<Tp0> &a, MI_F64 scale = 1) const
    {
        return Scalar_<Tp0>(SaturateCast<Tp0>(m_val[0] * a.m_val[0] * scale),
                            SaturateCast<Tp0>(m_val[1] * a.m_val[1] * scale),
                            SaturateCast<Tp0>(m_val[2] * a.m_val[2] * scale),
                            SaturateCast<Tp0>(m_val[3] * a.m_val[3] * scale));
    }

    /**
     * @brief Get the complex conjugate of the Scalar_ object.
     *
     * @return A Scalar_ object representing the complex conjugate.
     */
    Scalar_<Tp0> Conj() const
    {
        return Scalar_<Tp0>(SaturateCast<Tp0>(m_val[0]),
                            SaturateCast<Tp0>(-m_val[1]),
                            SaturateCast<Tp0>(-m_val[2]),
                            SaturateCast<Tp0>(-m_val[3]));
    }

    /**
     * @brief Check if the Scalar_ object is purely real.
     *
     * @return True if the Scalar_ is purely real, false otherwise.
     */
    MI_BOOL IsReal() const
    {
        return m_val[1] == 0 && m_val[2] == 0 && m_val[3] == 0;
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param sc The Scalar_ object to compare.
     *
     * @return True if all elements are equal, false otherwise.
     */
    MI_BOOL operator==(const Scalar_ &sc) const
    {
        return m_val[0] == sc.m_val[0] && m_val[1] == sc.m_val[1] &&
               m_val[2] == sc.m_val[2] && m_val[3] == sc.m_val[3];
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param sc The Scalar_ object to compare.
     *
     * @return True if any element is not equal, false if all elements are equal.
     */
    MI_BOOL operator!=(const Scalar_ &sc) const
    {
        return m_val[0] != sc.m_val[0] || m_val[1] != sc.m_val[1] ||
               m_val[2] != sc.m_val[2] || m_val[3] != sc.m_val[3];
    }

    /**
     * @brief Assignment operator.
     *
     * @param sz The Scalar_ object to assign from.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator=(const Scalar_ &sz)
    {
        m_val[0] = sz.m_val[0];
        m_val[1] = sz.m_val[1];
        m_val[2] = sz.m_val[2];
        m_val[3] = sz.m_val[3];
        return *this;
    }

    /**
     * @brief Addition assignment operator.
     *
     * @param b The Scalar_ object to add.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator+=(const Scalar_ &b)
    {
        m_val[0] += b.m_val[0];
        m_val[1] += b.m_val[1];
        m_val[2] += b.m_val[2];
        m_val[3] += b.m_val[3];
        return *this;
    }

    /**
     * @brief Subtraction assignment operator.
     *
     * @param b The Scalar_ object to subtract.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator-=(const Scalar_ &b)
    {
        m_val[0] -= b.m_val[0];
        m_val[1] -= b.m_val[1];
        m_val[2] -= b.m_val[2];
        m_val[3] -= b.m_val[3];
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by a scalar.
     *
     * @param scalar The scaling factor to multiply by.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator*=(Tp0 scalar)
    {
        m_val[0] *= scalar;
        m_val[1] *= scalar;
        m_val[2] *= scalar;
        m_val[3] *= scalar;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by another Scalar_ object element-wise.
     *
     * @param b The Scalar_ object to multiply.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator*=(const Scalar_ &b)
    {
        m_val[0] *= b.m_val[0];
        m_val[1] *= b.m_val[1];
        m_val[2] *= b.m_val[2];
        m_val[3] *= b.m_val[3];
        return *this;
    }

    /**
     * @brief Division assignment operator by a scalar.
     *
     * @param scalar The scalar value to divide by.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator/=(const Tp0 scalar)
    {
        const Tp0 tmp = (scalar == (Tp0)(0)) ? (Tp0)(1) : scalar;
        m_val[0] /= tmp;
        m_val[1] /= tmp;
        m_val[2] /= tmp;
        m_val[3] /= tmp;
        return *this;
    }

    /**
     * @brief Division assignment operator by another Scalar_ object element-wise.
     *
     * @param b The Scalar_ object to divide by.
     *
     * @return A reference to the modified Scalar_ object.
     */
    Scalar_& operator/=(const Scalar_ &b)
    {
        m_val[0] /= b.m_val[0];
        m_val[1] /= b.m_val[1];
        m_val[2] /= b.m_val[2];
        m_val[3] /= b.m_val[3];
        return *this;
    }

    Tp0 m_val[4];   /*!< Array to store the scalar values. */
};

/**
 * @brief Multiplication operator for a Scalar_ object by a scalar.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and scaling factor alpha.
 *
 * @param a The Scalar_ object.
 * @param alpha The scaling factor.
 *
 * @return A new Scalar_ object representing the multiplication result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator*(const Scalar_<Tp0> &a, const Tp0 alpha)
{
    return Scalar_<Tp0>(a.m_val[0] * alpha,
                        a.m_val[1] * alpha,
                        a.m_val[2] * alpha,
                        a.m_val[3] * alpha);
}

/**
 * @brief Multiplication operator for a scalar by a Scalar_ object.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and scalar alpha.
 *
 * @param alpha The scaling factor.
 * @param a The Scalar_ object.
 *
 * @return A new Scalar_ object representing the multiplication result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator*(const Tp0 alpha, const Scalar_<Tp0> &a)
{
    return a * alpha;
}

/**
 * @brief Multiplication operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return A new Scalar_ object representing the multiplication result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator*(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return Scalar_<Tp0>(SaturateCast<Tp0>(a.m_val[0] * b.m_val[0] - a.m_val[1] * b.m_val[1] -
                                          a.m_val[2] * b.m_val[2] - a.m_val[3] * b.m_val[3]),
                        SaturateCast<Tp0>(a.m_val[0] * b.m_val[1] + a.m_val[1] * b.m_val[0] +
                                          a.m_val[2] * b.m_val[3] - a.m_val[3] * b.m_val[2]),
                        SaturateCast<Tp0>(a.m_val[0] * b.m_val[2] - a.m_val[1] * b.m_val[3] +
                                          a.m_val[2] * b.m_val[0] + a.m_val[3] * b.m_val[1]),
                        SaturateCast<Tp0>(a.m_val[0] * b.m_val[3] + a.m_val[1] * b.m_val[2] -
                                          a.m_val[2] * b.m_val[1] + a.m_val[3] * b.m_val[0]));
}

/**
 * @brief Negation operator for a Scalar_ object.
 *
 * @tparam Tp0 The data types of the Scalar_ object a.
 *
 * @param a The Scalar_ object.
 *
 * @return A new Scalar_ object representing the negation of the input Scalar_ object.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator-(const Scalar_<Tp0> &a)
{
    return Scalar_<Tp0>(SaturateCast<Tp0>(-a.m_val[0]),
                        SaturateCast<Tp0>(-a.m_val[1]),
                        SaturateCast<Tp0>(-a.m_val[2]),
                        SaturateCast<Tp0>(-a.m_val[3]));
}

/**
 * @brief Addition operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return A new Scalar_ object representing the addition result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator+(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return Scalar_<Tp0>(SaturateCast<Tp0>(a.m_val[0] + b.m_val[0]),
                        SaturateCast<Tp0>(a.m_val[1] + b.m_val[1]),
                        SaturateCast<Tp0>(a.m_val[2] + b.m_val[2]),
                        SaturateCast<Tp0>(a.m_val[3] + b.m_val[3]));
}

/**
 * @brief Subtraction operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return A new Scalar_ object representing the subtraction result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator-(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return Scalar_<Tp0>(SaturateCast<Tp0>(a.m_val[0] - b.m_val[0]),
                        SaturateCast<Tp0>(a.m_val[1] - b.m_val[1]),
                        SaturateCast<Tp0>(a.m_val[2] - b.m_val[2]),
                        SaturateCast<Tp0>(a.m_val[3] - b.m_val[3]));
}

/**
 * @brief Inequality operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return True if the Scalars_ are not equal, false otherwise.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator!=(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return a.m_val[0] != b.m_val[0] || a.m_val[1] != b.m_val[1] ||
           a.m_val[2] != b.m_val[2] || a.m_val[3] != b.m_val[3];
}

/**
 * @brief Equality operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return True if the Scalars_ are equal, false otherwise.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator==(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return a.m_val[0] == b.m_val[0] && a.m_val[1] == b.m_val[1] &&
           a.m_val[2] == b.m_val[2] && a.m_val[3] == b.m_val[3];
}

/**
 * @brief Division operator for a Scalar_ object by a floating-point value.
 *
 * @tparam Tp0 The data types of the Scalar_ object a.
 *
 * @param a The Scalar_ object.
 * @param b The floating-point divisor.
 *
 * @return A new Scalar_ object representing the division result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator/(const Scalar_<Tp0> &a, MI_F32 b)
{
    MI_F32 s = 1 / b;
    return Scalar_<MI_F32>(a.m_val[0] * s, a.m_val[1] * s, a.m_val[2] * s, a.m_val[3] * s);
}

/**
 * @brief Division operator for a Scalar_ object by a double value.
 *
 * @tparam Tp0 The data types of the Scalar_ object a.
 *
 * @param a The Scalar_ object.
 * @param b The double divisor.
 *
 * @return A new Scalar_ object representing the division result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator/(const Scalar_<Tp0> &a, MI_F64 b)
{
    MI_F64 s = 1 / b;
    return Scalar_<MI_F64>(a.m_val[0] * s, a.m_val[1] * s, a.m_val[2] * s, a.m_val[3] * s);
}

/**
 * @brief Division operator for a scalar value by a Scalar_ object.
 *
 * @tparam Tp0 The data types of the Scalar_ object b.
 *
 * @param a The scalar value.
 * @param b The Scalar_ object.
 *
 * @return A new Scalar_ object representing the division result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator/(const Tp0 a, const Scalar_<Tp0> &b)
{
    Tp0 s = a / (b.m_val[0] * b.m_val[0] + b.m_val[1] * b.m_val[1] +
                 b.m_val[2] * b.m_val[2] + b.m_val[3] * b.m_val[3]);
    return b.Conj() * s;
}

/**
 * @brief Division operator for two Scalar_ objects.
 *
 * @tparam Tp0 The data types of the Scalar_ object a and b.
 *
 * @param a The first Scalar_ object.
 * @param b The second Scalar_ object.
 *
 * @return A new Scalar_ object representing the division result.
 */
template<typename Tp0>
AURA_INLINE Scalar_<Tp0> operator/(const Scalar_<Tp0> &a, const Scalar_<Tp0> &b)
{
    return a * ((Tp0)1 / b);
}

typedef Scalar_<MI_F64> Scalar;
typedef Scalar_<MI_S32> Scalari;

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_SCALAR_HPP__