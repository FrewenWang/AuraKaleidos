#ifndef AURA_RUNTIME_CORE_TYPES_SIZES_HPP__
#define AURA_RUNTIME_CORE_TYPES_SIZES_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/types/scalar.hpp"
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

template<typename Tp0> class Sizes3_;

/**
 * @brief Template class for specifying the size of 2D iauras or matrices.
 *
 * The class encapsulates two members called `m_width` and `m_height`, representing the width and height of a
 * 2D size, respectively.
 *
 * A similar set of arithmetic and comparison operations to Point2D_ can be used, and it also provides `Max()`
 * and `Min()` methods to obtain the maximum and minimum values between two instances. It also provides a cast
 * operator and printing operations for members.
 *
 * @tparam Tp0 Type of the size values.
 */
template<typename Tp0>
class Sizes2_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor initializing height and width to zero.
     */
    Sizes2_() : m_height(0), m_width(0)
    {}

    /**
     * @brief Constructor initializing the size with given height and width.
     *
     * @param height The height value.
     * @param width The width value.
     */
    Sizes2_(Tp0 height, Tp0 width) : m_height(height), m_width(width)
    {}

    /**
     * @brief Constructs a Sizes2_ object from a Scalar_ objects.
     *
     * @param scalr The Scalar_ object to convert.
     */
    Sizes2_(const Scalar_<Tp0> &scalr) : m_height(scalr.m_val[0]), m_width(scalr.m_val[1])
    {}

    /**
     * @brief Copy constructor.
     *
     * @param sz Sizes2_ object to be copied.
     */
    Sizes2_(const Sizes2_ &sz)
    {
        m_height = sz.m_height;
        m_width  = sz.m_width;
    }

    /**
     * @brief Constructor converting from a Sizes3_ object.
     *
     * @param sz Sizes3_ object to convert.
     */
    Sizes2_(const Sizes3_<Tp0> &sz)
    {
        m_height = sz.m_height;
        m_width  = sz.m_width;
    }

    /**
     * @brief Conversion to another data type.
     *
     * @tparam Tp1 The type to convert the elements to.
     *
     * @return A Sizes2_ object with elements saturated to the new template type `Tp1.
     */
    template<typename Tp1>
    operator Sizes2_<Tp1>() const
    {
        return Sizes2_<Tp1>(SaturateCast<Tp1>(m_height), SaturateCast<Tp1>(m_width));
    }

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded stream insertion operator for printing the size.
     *
     * @param os The output stream.
     * @param sz Sizes2_ object to be printed.
     *
     *  @return Reference to the output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Sizes2_ &sz)
    {
        os << sz.m_height << "x" << sz.m_width << "(hw)";
        return os;
    }

    /**
     * @brief Convert the Sizes2_ object to a string.
     *
     * @return A string representing the Sizes2_ object.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Check if the size is empty (either width or height is less than or equal to zero).
     *
     * @return True if the size is empty, false otherwise.
     */
    MI_BOOL Empty() const
    {
        return m_width <= 0 || m_height <= 0;
    }

    /**
     * @brief Get the maximum size between two Sizes2_ objects.
     *
     * @param sz Sizes2_ object to compare.
     *
     * @return A Sizes2_ object with dimensions as the maximum of the two Sizes2_ objects.
     */
    Sizes2_ Max(const Sizes2_ &sz) const
    {
        return Sizes2_(aura::Max(m_height, sz.m_height), aura::Max(m_width, sz.m_width));
    }

    /**
     * @brief Get the minimum size between two Sizes2_ objects.
     *
     * @param sz Sizes2_ object to compare.
     *
     * @return A Sizes2_ object with dimensions as the minimum of the two Sizes2_ objects.
     */
    Sizes2_ Min(const Sizes2_ &sz) const
    {
        return Sizes2_(aura::Min(m_height, sz.m_height), aura::Min(m_width, sz.m_width));
    }

    /**
     * @brief Get Sizes2_ object total element count.
     *
     * @return Sizes2_ object total element count.
     */
    Tp0 Total() const
    {
        return m_height * m_width;
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param sz Sizes2_ object to compare.
     *
     * @return True if the sizes are equal, false otherwise.
     */
    MI_BOOL operator==(const Sizes2_ &sz) const
    {
        return m_width == sz.m_width && m_height == sz.m_height;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param sz Sizes2_ object to compare.
     *
     * @return True if the sizes are not equal, false if they are equal.
     */
    MI_BOOL operator!=(const Sizes2_ &sz) const
    {
        return m_width != sz.m_width || m_height != sz.m_height;
    }

    /**
     * @brief Assignment operator.
     *
     * @param sz Sizes2_ object to assign from.
     *
     *  @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator=(const Sizes2_ &sz)
    {
        m_height = sz.m_height;
        m_width  = sz.m_width;
        return *this;
    }

    /**
     * @brief Addition assignment operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to add.
     *
     * @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator+=(const Sizes2_ &sz)
    {
        m_height += sz.m_height;
        m_width  += sz.m_width;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to subtract.
     *
     * @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator-=(const Sizes2_ &sz)
    {
        m_height -= sz.m_height;
        m_width  -= sz.m_width;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by a scaling factor.
     *
     * @param scalar The scaling factor to multiply by.
     *
     * @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator*=(const Tp0 scalar)
    {
        m_height *= scalar;
        m_width  *= scalar;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to multiply.
     *
     * @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator*=(const Sizes2_ &sz)
    {
        m_height *= sz.m_height;
        m_width  *= sz.m_width;
        return *this;
    }

    /**
     * @brief Division assignment operator by a scalar value.
     *
     * @param scalar The scalar value to divide by.
     *
     *  @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator/=(const Tp0 scalar)
    {
        const Tp0 tmp = (scalar == (Tp0)(0)) ? (Tp0)(1) : scalar;
        m_height /= tmp;
        m_width  /= tmp;
        return *this;
    }

    /**
     * @brief Division assignment operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to divide by.
     *
     * @return A reference to the modified Sizes2_ object.
     */
    Sizes2_& operator/=(const Sizes2_ &sz)
    {
        if (!sz.Empty())
        {
            m_height /= sz.m_height;
            m_width  /= sz.m_width;
        }

        return *this;
    }

    /**
     * @brief Addition operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to add.
     *
     * @return A Sizes2_ object representing the sum.
     */
    Sizes2_ operator+(const Sizes2_ &sz) const
    {
        return Sizes2_(m_height + sz.m_height, m_width + sz.m_width);
    }

    /**
     * @brief Addition operator by a scalar value.
     *
     * @param scalar The scalar value to add.
     *
     * @return A Sizes2_ object representing the sum.
     */
    Sizes2_ operator+(const Tp0 scalar) const
    {
        return Sizes2_(m_height + scalar, m_width + scalar);
    }

    /**
     * @brief Subtraction operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to subtract.
     *
     * @return A Sizes2_ object representing the difference.
     */
    Sizes2_ operator-(const Sizes2_ &sz) const
    {
        return Sizes2_(m_height - sz.m_height, m_width - sz.m_width);
    }

    /**
     * @brief Multiplication operator by a scaling factor.
     *
     * @param scalar The scaling factor to multiply by.
     *
     * @return A Sizes2_ object representing the result of multiplication.
     */
    Sizes2_ operator*(const Tp0 scalar) const
    {
        return Sizes2_(m_height * scalar, m_width * scalar);
    }

    /**
     * @brief Subtraction operator by a scalar value.
     *
     * @param scalar The scalar value to subtract.
     *
     * @return A Sizes2_ object representing the result of subtraction.
     */
    Sizes2_ operator-(const Tp0 scalar) const
    {
        return Sizes2_(m_height - scalar, m_width - scalar);
    }

    /**
     * @brief Multiplication operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to multiply.
     *
     * @return A Sizes2_ object representing the result of multiplication.
     */
    Sizes2_ operator*(const Sizes2_ &sz) const
    {
        return Sizes2_(m_height * sz.m_height, m_width * sz.m_width);
    }

    /**
     * @brief Division operator by a scalar value.
     *
     * @param scalar The scalar value to divide by.
     *
     * @return A Sizes2_ object representing the result of division.
     */
    Sizes2_ operator/(const Tp0 scalar) const
    {
        const Tp0 tmp = (scalar == (Tp0)(0)) ? (Tp0)(1) : scalar;
        return Sizes2_(m_height / tmp, m_width / tmp);
    }

    /**
     * @brief Division operator by another Sizes2_ object.
     *
     * @param sz Sizes2_ object to divide by.
     *
     * @return A Sizes2_ object representing the result of division.
     */
    Sizes2_ operator/(const Sizes2_ &sz) const
    {
        if (!sz.Empty())
        {
            Sizes2_ size;
            size.m_height = m_height / sz.m_height;
            size.m_width  = m_width  / sz.m_width;
            return size;
        }
        return *this;
    }

    Tp0 m_height;   /*!< Height value of the size. */
    Tp0 m_width;    /*!< Width value of the size. */
};

typedef Sizes2_<MI_S32> Sizes;
typedef Sizes2_<MI_S64> Sizesl;
typedef Sizes2_<MI_F32> Sizesf;
typedef Sizes2_<MI_F64> Sizesd;

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Template class for specifying the size of 3D iauras or matrices.
 *
 * The class encapsulates three members called `m_width`, `m_height` and `m_channel`, representing the width, height
 * and channel of a 3D size, respectively.
 *
 * A similar set of arithmetic and comparison operations to Point3D_ can be used, and it also provides `Max()` and
 * `Min()` methods to obtain the maximum and minimum values between two instances. It also provides a cast operator
 * and printing operations for members.
 *
 * @tparam Tp0 Type of the size values.
 */
template<typename Tp0>
class Sizes3_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor initializing height and width to zero, and channel to 1.
     */
    Sizes3_() : m_height(0), m_width(0), m_channel(1)
    {}

    /**
     * @brief Constructor initializing the size with given height and width, and sets channel to 1.
     *
     * @param height The height value.
     * @param width The width value.
     */
    Sizes3_(Tp0 height, Tp0 width) : m_height(height), m_width(width), m_channel(1)
    {}

    /**
     * @brief Constructor specifying height, width, and channel.
     *
     * @param height The height dimension.
     * @param width The width dimension.
     * @param channel The channel dimension.
     */
    Sizes3_(Tp0 height, Tp0 width, Tp0 channel) : m_height(height), m_width(width), m_channel(channel)
    {}

    /**
     * @brief Constructor converting from a Scalar_ object to Sizes3_.
     *
     * @param scalar The Scalar_ object containing dimensions.
     */
    Sizes3_(const Scalar_<Tp0> &scalar) : m_height(scalar.m_val[0]), m_width(scalar.m_val[1]), m_channel(scalar.m_val[2])
    {}

    /**
     * @brief Constructor converting from a Sizes2_ object to Sizes3_, and sets channel to 1.
     *
     * @param sz The Sizes2_ object containing height and width.
     */
    Sizes3_(const Sizes2_<Tp0> &sz) : m_height(sz.m_height), m_width(sz.m_width), m_channel(1)
    {}

    /**
     * @brief Constructor converting from a Sizes2_ object and a channel value to Sizes3_ object.
     *
     * @param sz The Sizes2_ object containing height and width.
     * @param channel The channel value.
     */
    Sizes3_(const Sizes2_<Tp0> &sz, Tp0 channel) : m_height(sz.m_height), m_width(sz.m_width), m_channel(channel)
    {}

    /**
     * @brief Copy constructor.
     *
     * @param sz The Sizes3_ object to copy.
     */
    Sizes3_(const Sizes3_ &sz)
    {
        m_height  = sz.m_height;
        m_width   = sz.m_width;
        m_channel = sz.m_channel;
    }

    /**
     * @brief Conversion to another data type.
     *
     * @tparam Tp1 The type to convert the elements to.
     *
     * @return A Sizes3_ object with elements saturated to the new template type `Tp1.
     */
    template<typename Tp1>
    operator Sizes3_<Tp1>() const
    {
        return Sizes3_<Tp1>(SaturateCast<Tp1>(m_height), SaturateCast<Tp1>(m_width),
                            SaturateCast<Tp1>(m_channel));
    }

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded stream insertion operator for printing the size.
     *
     * @param os The output stream.
     * @param sz Sizes3_ object to be printed.
     *
     *  @return Reference to the output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Sizes3_ &sz)
    {
        os << sz.m_height << "x" << sz.m_width << "x" << sz.m_channel << "(hwc)";
        return os;
    }

    /**
     * @brief Convert the Sizes3_ object to a string.
     *
     * @return A string representing the Sizes3_ object.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Check if the size is empty (either width or height is less than or equal to zero).
     *
     * @return True if the size is empty, false otherwise.
     */
    MI_BOOL Empty() const
    {
        return m_width <= 0 || m_height <= 0;
    }

    /**
     * @brief Get the maximum size between two Sizes3_ objects.
     *
     * @param sz Sizes3_ object to compare.
     *
     * @return A Sizes3_ object with dimensions as the maximum of the two Sizes3_ objects.
     */
    Sizes3_ Max(const Sizes3_ &sz) const
    {
        return Sizes3_(aura::Max(m_height, sz.m_height), aura::Max(m_width, sz.m_width),
                       aura::Max(m_channel, sz.m_channel));
    }

    /**
     * @brief Get Sizes3_ object total element count.
     *
     * @return Sizes3_ object total element count.
     */
    Tp0 Total() const
    {
        return m_height * m_width * m_channel;
    }

    /**
     * @brief Get the minimum size between two Sizes3_ objects.
     *
     * @param sz Sizes3_ object to compare.
     *
     * @return A Sizes3_ object with dimensions as the minimum of the two Sizes3_ objects.
     */
    Sizes3_ Min(const Sizes3_ &sz) const
    {
        return Sizes3_(aura::Min(m_height, sz.m_height), aura::Min(m_width, sz.m_width),
                       aura::Min(m_channel, sz.m_channel));
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param sz Sizes3_ object to compare.
     *
     * @return True if the sizes are equal, false otherwise.
     */
    MI_BOOL operator==(const Sizes3_ &sz) const
    {
        return m_height == sz.m_height && m_width == sz.m_width && m_channel == sz.m_channel;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param sz Sizes3_ object to compare.
     *
     * @return True if the sizes are not equal, false if they are equal.
     */
    MI_BOOL operator!=(const Sizes3_ &sz) const
    {
        return m_height != sz.m_height || m_width != sz.m_width || m_channel != sz.m_channel;
    }

    /**
     * @brief Assignment operator.
     *
     * @param sz Sizes3_ object to assign from.
     *
     *  @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator=(const Sizes3_ &sz)
    {
        m_height  = sz.m_height;
        m_width   = sz.m_width;
        m_channel = sz.m_channel;
        return *this;
    }

    /**
     * @brief Addition assignment operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to add.
     *
     * @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator+=(const Sizes3_ &sz)
    {
        m_height  += sz.m_height;
        m_width   += sz.m_width;
        m_channel += sz.m_channel;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to subtract.
     *
     *  @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator-=(const Sizes3_ &sz)
    {
        m_height  -= sz.m_height;
        m_width   -= sz.m_width;
        m_channel -= sz.m_channel;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by a scaling factor.
     *
     * @param scalar The scaling factor to multiply by.
     *
     * @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator*=(const Tp0 scalar)
    {
        m_height  *= scalar;
        m_width   *= scalar;
        m_channel *= scalar;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to multiply.
     *
     * @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator*=(const Sizes3_ &sz)
    {
        m_height  *= sz.m_height;
        m_width   *= sz.m_width;
        m_channel *= sz.m_channel;
        return *this;
    }

    /**
     * @brief Division assignment operator by a scalar value.
     *
     * @param scalar The scalar value to divide by.
     *
     *  @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator/=(const Tp0 scalar)
    {
        const Tp0 tmp = (scalar == (Tp0)(0)) ? (Tp0)(1) : scalar;
        m_height  /= tmp;
        m_width   /= tmp;
        m_channel /= tmp;
        return *this;
    }

    /**
     * @brief Division assignment operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to divide by.
     *
     * @return A reference to the modified Sizes3_ object.
     */
    Sizes3_& operator/=(const Sizes3_ &sz)
    {
        if (!sz.Empty())
        {
            m_height /= sz.m_height;
            m_width  /= sz.m_width;
            if (sz.m_channel > 0)
            {
                m_channel /= sz.m_channel;
            }
        }

        return *this;
    }

    /**
     * @brief Addition operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to add.
     *
     * @return A Sizes3_ object representing the sum.
     */
    Sizes3_ operator+(const Sizes3_ &sz) const
    {
        return Sizes3_(m_height + sz.m_height, m_width + sz.m_width,
                       m_channel + sz.m_channel);
    }

    /**
     * @brief Subtraction operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to subtract.
     *
     * @return A Sizes3_ object representing the difference.
     */
    Sizes3_ operator-(const Sizes3_ &sz) const
    {
        return Sizes3_(m_height - sz.m_height, m_width - sz.m_width,
                       m_channel - sz.m_channel);
    }

    /**
     * @brief Multiplication operator by a scaling factor.
     *
     * @param scalar The scaling factor to multiply by.
     *
     * @return A Sizes3_ object representing the result of multiplication.
     */
    Sizes3_ operator*(const Tp0 scalar) const
    {
        return Sizes3_(m_height * scalar, m_width * scalar, m_channel * scalar);
    }

    /**
     * @brief Multiplication operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to multiply.
     *
     * @return A Sizes3_ object representing the result of multiplication.
     */
    Sizes3_ operator*(const Sizes3_ &sz) const
    {
        return Sizes3_(m_height * sz.m_height, m_width * sz.m_width,
                       m_channel * sz.m_channel);
    }

    /**
     * @brief Division operator by a scalar value.
     *
     * @param scalar The scalar value to divide by.
     *
     * @return A Sizes3_ object representing the result of division.
     */
    Sizes3_ operator/(const Tp0 scalar) const
    {
        const Tp0 tmp = (scalar == (Tp0)(0)) ? (Tp0)(1) : scalar;
        return Sizes3_(m_height / tmp, m_width / tmp, m_channel / tmp);
    }

    /**
     * @brief Division operator by another Sizes3_ object.
     *
     * @param sz Sizes3_ object to divide by.
     *
     * @return A Sizes3_ object representing the result of division.
     */
    Sizes3_ operator/(const Sizes3_ &sz) const
    {
        if (!sz.Empty())
        {
            Sizes3_ size;
            size.m_height = m_height / sz.m_height;
            size.m_width  = m_width  / sz.m_width;
            if (sz.m_channel > 0)
            {
                size.m_channel = m_channel / sz.m_channel;
            }
            return size;
        }
        return *this;
    }

    Tp0 m_height;   /*!< Height value of the size. */
    Tp0 m_width;    /*!< Width value of the size. */
    Tp0 m_channel;  /*!< Channel value of the size. */
};

typedef Sizes3_<MI_S32> Sizes3;
typedef Sizes3_<MI_S64> Sizes3l;
typedef Sizes3_<MI_F32> Sizes3f;
typedef Sizes3_<MI_F64> Sizes3d;

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_SIZES_HPP__