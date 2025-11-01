#ifndef AURA_RUNTIME_CORE_TYPES_POINT_HPP__
#define AURA_RUNTIME_CORE_TYPES_POINT_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/saturate.hpp"

#if !defined(AURA_BUILD_XTENSA)
#  include <iostream>
#  include <sstream>
#  include <string>
#  include <utility>
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
 * @brief Template class for 2D points.
 *
 * The Point2_ is specified by its coordinates `m_x` and `m_y`, representing points in 2D space such as iauras or
 * matrices.
 *
 * It provides arithmetic and comparison operation applied to 2D points, as well as a cast operator to convert
 * point coordinates to the specified type. The users can also use the `ToString()` to print the coordinates of
 * the 2D point.
 *
 * @tparam Tp0 The data type for the coordinates.
 */
template<typename Tp0>
class Point2_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor initializing the coordinates.
     *
     * @param x The x-coordinate of the point (default is 0).
     * @param y The y-coordinate of the point (default is 0).
     */
    Point2_(Tp0 x = 0, Tp0 y = 0) : m_x(x), m_y(y)
    {}

    /**
     * @brief Copy constructor for a Point2_ object.
     *
     * @param pt The Point2_ object whose coordinates are to be copied.
     */
    Point2_(const Point2_ &pt) : m_x(pt.m_x), m_y(pt.m_y)
    {}

    /**
     * @brief Conversion to another data type.
     *
     * @tparam Tp1 The data type to convert the Point2_ object to.
     *
     * @return The Point2_ object with coordinates casted to type `Tp1`.
     */
    template<typename Tp1>
    operator Point2_<Tp1>() const
    {
        return Point2_<Tp1>(SaturateCast<Tp1>(m_x), SaturateCast<Tp1>(m_y));
    }

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded ostream insertion operator.
     *
     * @param os The output stream.
     * @param pt The Point2_ object whose coordinates are printed.
     *
     * @return Reference to the output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Point2_ &pt)
    {
        os << "[" << pt.m_x << ", " << pt.m_y << "]";
        return os;
    }

    /**
     * @brief Convert the Point2_ object to a string.
     *
     * @return A string representing the Point2_ object's coordinates.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Assignment operator.
     *
     * @param pt The Point2_ object whose coordinates are to be assigned.
     *
     *  @return A reference to the modified Point2_ object.
     */
    Point2_& operator=(const Point2_ &pt)
    {
        m_x = pt.m_x;
        m_y = pt.m_y;
        return *this;
    }

    /**
     * @brief Addition-assignment operator.
     *
     * @param b The Point2_ object whose coordinates are to be added.
     *
     *  @return A reference to the modified Point2_ object.
     */
    Point2_& operator+=(const Point2_ &b)
    {
        m_x += b.m_x;
        m_y += b.m_y;
        return *this;
    }

    /**
     * @brief Subtraction-assignment operator.
     *
     * @param b The Point2_ object whose coordinates are to be subtracted.
     *
     *  @return A reference to the modified Point2_ object.
     */
    Point2_& operator-=(const Point2_ &b)
    {
        m_x -= b.m_x;
        m_y -= b.m_y;
        return *this;
    }

    /**
     * @brief Multiplication-assignment operator.
     *
     * @tparam Tp1 The data type of the scaling factor.
     *
     * @param b The scaling factor to multiply by.
     *
     * @return A reference to the modified Point2_ object.
     */
    template<typename Tp1>
    Point2_& operator*=(Tp1 b)
    {
        m_x = SaturateCast<Tp0>(m_x * b);
        m_y = SaturateCast<Tp0>(m_y * b);
        return *this;
    }

    /**
     * @brief Division-assignment operator.
     *
     * @tparam Tp1 The data type of the scalar value.
     *
     * @param b The scalar value to divide by.
     *
     * @return A reference to the modified Point2_ object.
     */
    template<typename Tp1>
    Point2_& operator/=(Tp1 b)
    {
        m_x = SaturateCast<Tp0>(m_x / b);
        m_y = SaturateCast<Tp0>(m_y / b);
        return *this;
    }

    /**
     * @brief Equality operator.
     *
     * @param b The other Point2_ object to compare.
     *
     *  @return true if both Point2_ objects have the same x and y coordinates, false otherwise.
     */
    MI_BOOL operator==(const Point2_ &b) const
    {
        return m_x == b.m_x && m_y == b.m_y;
    }

    /**
     * @brief Inequality operator.
     *
     * @param b The other Point2_ object to compare.
     *
     * @return true if Point2_ objects have different x or y coordinates, false if they are equal.
     */
    MI_BOOL operator!=(const Point2_ &b) const
    {
        return m_x != b.m_x || m_y != b.m_y;
    }

    /**
     * @brief Addition operator.
     *
     * @param b The other Point2_ object to add.
     *
     * @return A new Point2_ object with the sum of x and y coordinates.
     */
    Point2_ operator+(const Point2_ &b) const
    {
        return Point2_(SaturateCast<Tp0>(m_x + b.m_x), SaturateCast<Tp0>(m_y + b.m_y));
    }

    /**
     * @brief Subtraction operator.
     *
     * @param b The other Point2_ object to subtract.
     *
     *  @return A new Point2_ object with the difference of x and y coordinates.
     */
    Point2_ operator-(const Point2_ &b) const
    {
        return Point2_(SaturateCast<Tp0>(m_x - b.m_x), SaturateCast<Tp0>(m_y - b.m_y));
    }

    /**
     * @brief Negation operator.
     *
     * @return A new Point2_ object with negated x and y coordinates.
     */
    Point2_ operator-() const
    {
        return Point2_(SaturateCast<Tp0>(-m_x), SaturateCast<Tp0>(-m_y));
    }

    /**
     * @brief Dot product calculation.
     *
     * @param pt The other Point2_ object to calculate the dot product with.
     *
     * @return The dot product value between this Point2_ and `pt`.
     */
    Tp0 Dot(const Point2_ &pt) const
    {
        return SaturateCast<Tp0>(m_x * pt.m_x + m_y * pt.m_y);
    }

    /**
     * @brief Double precision dot product calculation.
     *
     * @param pt The other Point2_ object to calculate the double precision dot product with.
     *
     * @return The double precision dot product value between this Point2_ and `pt`.
     */
    MI_F64 DDot(const Point2_ &pt) const
    {
        return (MI_F64)m_x * pt.m_x + (MI_F64)m_y * pt.m_y;
    }

    /**
     * @brief Cross product calculation.
     *
     * @param pt The other Point2_ object to calculate the cross product with.
     *
     * @return The cross product value between this Point2_ and `pt`.
     */
    MI_F64 Cross(const Point2_ &pt) const
    {
        return (MI_F64)m_x * pt.m_y + (MI_F64)m_y * pt.m_x;
    }

    /**
     * @brief Norm calculation.
     *
     * @return The Euclidean norm value (magnitude) of this Point2_ object.
     */
    MI_F64 Norm() const
    {
        MI_F64 tmp = m_x * m_x + m_y * m_y;
        return Sqrt(tmp);
    }

    Tp0 m_x;    /*!< x coordinate of the 2D point */
    Tp0 m_y;    /*!< y coordinate of the 2D point */
};

/**
 * @brief Multiplication operator for a Point2_ object and a scalar.
 *
 * @tparam Tp0 The data type of the Point2_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param a The Point2_ object.
 * @param b The scaling factor.
 *
 * @return A new Point2_ object resulting from the scalar multiplication.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point2_<Tp0> operator*(const Point2_<Tp0> &a, Tp1 b)
{
    return Point2_<Tp0>(SaturateCast<Tp0>(a.m_x * b), SaturateCast<Tp0>(a.m_y * b));
}

/**
 * @brief Multiplication operator for a scalar and a Point2_ object.
 *
 * @tparam Tp0 The data type of the Point2_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param b The scaling factor.
 * @param a The Point2_ object.
 *
 * @return A new Point2_ object resulting from the scalar multiplication.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point2_<Tp0> operator*(Tp1 b, const Point2_<Tp0> &a)
{
    return Point2_<Tp0>(SaturateCast<Tp0>(a.m_x * b), SaturateCast<Tp0>(a.m_y * b));
}

/**
 * @brief Division operator for a Point2_ object and a scalar.
 *
 * @tparam Tp0 The data type of the Point2_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param a The Point2_ object.
 * @param b The scalar value.
 *
 * @return A new Point2_ object resulting from the scalar division.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point2_<Tp0> operator/(const Point2_<Tp0> &a, Tp1 b)
{
    return Point2_<Tp0>(SaturateCast<Tp0>(a.m_x / b), SaturateCast<Tp0>(a.m_y / b));
}

/**
 * @brief Division operator for a scalar and a Point2_ object.
 *
 * @tparam Tp0 The data type of the Point2_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param b The scalar value.
 * @param a The Point2_ object.
 *
 * @return A new Point2_ object resulting from the scalar division.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point2_<Tp0> operator/(Tp1 b, const Point2_<Tp0> &a)
{
    return Point2_<Tp0>(SaturateCast<Tp0>(b / a.m_x), SaturateCast<Tp0>(b / a.m_y));
}

/**
 * @brief Computes the square of the L2 norm of a Point2_ object.
 *
 * Calculates the squared L2 norm (Euclidean norm) of a Point2_ object `pt`.
 * The L2 norm is the square root of the sum of squares of the individual elements.
 *
 * @tparam Tp0 The data type of the result.
 * @tparam Tp1 The data type of the Point2_ object.
 *
 * @param pt The Point2_ object for which the squared L2 norm is to be calculated.
 *
 * @return The square of the L2 norm of the Point2_ object.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Tp0 NormL2Sqr(const Point2_<Tp1> &pt);

template<>
inline MI_S32 NormL2Sqr<MI_S32, MI_S32>(const Point2_<MI_S32> &pt)
{
    return pt.Dot(pt);
}

template<>
inline MI_S64 NormL2Sqr<MI_S64, MI_S64>(const Point2_<MI_S64> &pt)
{
    return pt.Dot(pt);
}

template<>
inline MI_F32 NormL2Sqr<MI_F32, MI_F32>(const Point2_<MI_F32> &pt)
{
    return pt.Dot(pt);
}

template<>
inline MI_F64 NormL2Sqr<MI_F64, MI_S32>(const Point2_<MI_S32> &pt)
{
    return pt.Dot(pt);
}

template<>
inline MI_F64 NormL2Sqr<MI_F64, MI_F32>(const Point2_<MI_F32> &pt)
{
    return pt.DDot(pt);
}

template<>
inline MI_F64 NormL2Sqr<MI_F64, MI_F64>(const Point2_<MI_F64> &pt)
{
    return pt.DDot(pt);
}

typedef Point2_<MI_S32> Point2i;
typedef Point2_<MI_F32> Point2f;
typedef Point2_<MI_F64> Point2d;
typedef Point2f Point2;

////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Template class for 3D points.
 *
 * Similarly to Point2_, Point3_ is specified by its coordinates `m_x`, `m_y` and `m_z` and representing points in
 * 3D space.
 *
 * It provides arithmetic and comparison operation applied to 3D points, as well as a cast operator to convert point
 * coordinates to the specified type. The users can also use the `ToString()` to print the coordinates of the 3D point.
 *
 * @tparam Tp0 The data type for the coordinates.
 */
template<typename Tp0>
class Point3_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor initializing the coordinates.
     *
     * @param x The x-coordinate of the point (default is 0).
     * @param y The y-coordinate of the point (default is 0).
     * @param z The z-coordinate of the point (default is 0).
     */
    Point3_(Tp0 x = 0, Tp0 y = 0, Tp0 z = 0) : m_x(x), m_y(y), m_z(z)
    {}

    /**
     * @brief Constructs a Point3_ object from a Point2_ object, setting the z-coordinate to zero.
     *
     * @param pt The Point2_ object from which to construct the 3D point.
     */
    Point3_(const Point2_<Tp0> &pt) : m_x(pt.m_x), m_y(pt.m_y), m_z(0)
    {}

    /**
     * @brief Copy constructor for a Point3_ object.
     *
     * @param pt The Point3_ object whose coordinates are to be copied.
     */
    Point3_(const Point3_ &pt) : m_x(pt.m_x), m_y(pt.m_y), m_z(pt.m_z)
    {}

    /**
     * @brief Conversion to another data type.
     *
     * @tparam Tp1 The data type to convert the Point3_ object to.
     *
     * @return The Point3_ object with coordinates casted to type Tp1.
     */
    template<typename Tp1>
    operator Point3_<Tp1>() const
    {
        return Point3_<Tp1>(SaturateCast<Tp1>(m_x), SaturateCast<Tp1>(m_y),
                            SaturateCast<Tp1>(m_z));
    }

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded ostream insertion operator.
     *
     * @param os The output stream.
     * @param pt The Point3_ object whose coordinates are printed.
     *
     * @return Reference to the output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Point3_ &pt)
    {
        os << "[" << pt.m_x << ", " << pt.m_y << ", " << pt.m_z << "]";
        return os;
    }

    /**
     * @brief Convert the Point3_ object to a string.
     *
     * @return A string representing the Point3_ object's coordinates.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Assignment operator.
     *
     * @param pt The Point3_ object whose coordinates are to be assigned.
     *
     * @return A reference to the modified Point3_ object.
     */
    Point3_& operator=(const Point3_ &pt)
    {
        m_x = pt.m_x;
        m_y = pt.m_y;
        m_z = pt.m_z;
        return *this;
    }

    /**
     * @brief Addition-assignment operator.
     *
     * @param b The Point3_ object whose coordinates are to be added.
     *
     * @return A reference to the modified Point3_ object.
     */
    Point3_& operator+=(const Point3_ &&pt)
    {
        m_x += pt.m_x;
        m_y += pt.m_y;
        m_z += pt.m_z;
        return *this;
    }

    /**
     * @brief Subtraction-assignment operator.
     *
     * @param b The Point3_ object whose coordinates are to be subtracted.
     *
     * @return A reference to the modified Point3_ object.
     */
    Point3_& operator-=(const Point3_ &&pt)
    {
        m_x -= pt.m_x;
        m_y -= pt.m_y;
        m_z -= pt.m_z;
        return *this;
    }

    /**
     * @brief Multiplication-assignment operator.
     *
     * @tparam Tp1 The data type of the scalar value.
     *
     * @param b The scaling factor to multiply by.
     *
     * @return A reference to the modified Point3_ object.
     */
    template<typename Tp1>
    Point3_& operator*=(Tp1 b)
    {
        m_x = SaturateCast<Tp0>(m_x * b);
        m_y = SaturateCast<Tp0>(m_y * b);
        m_z = SaturateCast<Tp0>(m_z * b);
        return *this;
    }

    /**
     * @brief Division-assignment operator.
     *
     * @tparam Tp1 The data type of the scalar value.
     *
     * @param b The scalar value to divide by.
     *
     * @return A reference to the modified Point3_ object.
     */
    template<typename Tp1>
    Point3_& operator/=(Tp1 b)
    {
        m_x = SaturateCast<Tp0>(m_x / b);
        m_y = SaturateCast<Tp0>(m_y / b);
        m_z = SaturateCast<Tp0>(m_z / b);
        return *this;
    }

    /**
     * @brief Equality operator.
     *
     * @param b The other Point3_ object to compare.
     *
     *  @return true if both Point3_ objects have the same x, y and x coordinates, false otherwise.
     */
    MI_BOOL operator==(const Point3_ &b) const
    {
        return m_x == b.m_x && m_y == b.m_y && m_z == b.m_z;
    }

    /**
     * @brief Inequality operator.
     *
     * @param b The other Point3_ object to compare.
     *
     * @return true if Point3_ objects have different x, y or z coordinates, false if they are equal.
     */
    MI_BOOL operator!=(const Point3_ &b) const
    {
        return m_x != b.m_x || m_y != b.m_y || m_z != b.m_z;
    }

    /**
     * @brief Addition operator.
     *
     * @param b The other Point3_ object to add.
     *
     * @return A new Point3_ object with the sum of x, y and z coordinates.
     */
    Point3_ operator+(const Point3_ &b) const
    {
        return Point3_(SaturateCast<Tp0>(m_x + b.m_x), SaturateCast<Tp0>(m_y + b.m_y),
                        SaturateCast<Tp0>(m_z + b.m_z));
    }

    /**
     * @brief Subtraction operator.
     *
     * @param b The other Point3_ object to subtract.
     *
     *  @return A new Point3_ object with the difference of x, y and z coordinates.
     */
    Point3_ operator-(const Point3_ &b) const
    {
        return Point3_(SaturateCast<Tp0>(m_x - b.m_x), SaturateCast<Tp0>(m_y - b.m_y),
                       SaturateCast<Tp0>(m_z - b.m_z));
    }

    /**
     * @brief Negation operator.
     *
     * @return A new Point3_ object with negated x, y and z coordinates.
     */
    Point3_ operator-() const
    {
        return Point3_(SaturateCast<Tp0>(-m_x), SaturateCast<Tp0>(-m_y), SaturateCast<Tp0>(-m_z));
    }

    /**
     * @brief Dot product calculation.
     *
     * @param pt The other Point3_ object to calculate the dot product with.
     *
     * @return The dot product value between this Point3_ and `pt`.
     */
    Tp0 Dot(const Point3_ &pt) const
    {
        return SaturateCast<Tp0>(m_x * pt.m_x + m_y * pt.m_y + m_z * pt.m_z);
    }

    /**
     * @brief Double precision dot product calculation.
     *
     * @param pt The other Point3_ object to calculate the double precision dot product with.
     *
     * @return The double precision dot product value between this Point3_ and `pt`.
     */
    MI_F64 DDot(const Point3_ &pt) const
    {
        return (MI_F64)m_x * pt.m_x + (MI_F64)m_y * pt.m_y + (MI_F64)m_z * pt.m_z;
    }

    /**
     * @brief Cross product calculation.
     *
     * @param pt The other Point3_ object to calculate the cross product with.
     *
     * @return The cross product value between this Point3_ and `pt`.
     */
    Point3_ Cross(const Point3_ &pt) const
    {
        return Point3_(m_y * pt.m_z - m_z * pt.m_y, m_z * pt.m_x - m_x * pt.m_z,
                       m_x * pt.m_y - m_y * pt.m_x);
    }

    /**
     * @brief Norm calculation.
     *
     * @return The Euclidean norm value (magnitude) of this Point3_ object.
     */
    MI_F64 Norm() const
    {
        MI_F64 tmp = m_x * m_x + m_y * m_y + m_z * m_z;
        return Sqrt(tmp);
    }

    Tp0 m_x;    /*!< x coordinate of the 3D point */
    Tp0 m_y;    /*!< y coordinate of the 3D point */
    Tp0 m_z;    /*!< z coordinate of the 3D point */
};

/**
 * @brief Multiplication operator for a Point3_ object and a scalar.
 *
 * @tparam Tp0 The data type of the Point3_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param a The Point3_ object.
 * @param b The scaling factor.
 *
 * @return A new Point3_ object resulting from the scalar multiplication.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point3_<Tp0> operator*(const Point3_<Tp0> &a, Tp1 b)
{
    return Point3_<Tp0>(SaturateCast<Tp0>(a.m_x * b), SaturateCast<Tp0>(a.m_y * b),
                        SaturateCast<Tp0>(a.m_z * b));
}

/**
 * @brief Multiplication operator for a scalar and a Point3_ object.
 *
 * @tparam Tp0 The data type of the Point3_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param b The scaling factor.
 * @param a The Point3_ object.
 *
 * @return A new Point3_ object resulting from the scalar multiplication.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point3_<Tp0> operator*(Tp1 b, const Point3_<Tp0> &a)
{
    return Point3_<Tp0>(SaturateCast<Tp0>(a.m_x * b), SaturateCast<Tp0>(a.m_y * b),
                        SaturateCast<Tp0>(a.m_z * b));
}

/**
 * @brief Division operator for a Point3_ object and a scalar.
 *
 * @tparam Tp0 The data type of the Point3_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param a The Point3_ object.
 * @param b The scalar value.
 *
 * @return A new Point3_ object resulting from the scalar division.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point3_<Tp0> operator/(const Point3_<Tp0> &a, Tp1 b)
{
    return Point3_<Tp0>(SaturateCast<Tp0>(a.m_x / b), SaturateCast<Tp0>(a.m_y / b),
                        SaturateCast<Tp0>(a.m_z / b));
}

/**
 * @brief Division operator for a scalar and a Point3_ object.
 *
 * @tparam Tp0 The data type of the Point3_ object.
 * @tparam Tp1 The data type of the scalar.
 *
 * @param b The scalar value.
 * @param a The Point3_ object.
 *
 * @return A new Point3_ object resulting from the scalar division.
 */
template<typename Tp0, typename Tp1>
AURA_INLINE Point3_<Tp0> operator/(Tp1 b, const Point3_<Tp0> &a)
{
    return Point3_<Tp0>(SaturateCast<Tp0>(b / a.m_x), SaturateCast<Tp0>(b / a.m_y),
                        SaturateCast<Tp0>(b / a.m_z));
}

typedef Point3_<MI_S32> Point3i;
typedef Point3_<MI_F32> Point3f;
typedef Point3_<MI_F64> Point3d;
typedef Point3f Point3;

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_POINT_HPP__