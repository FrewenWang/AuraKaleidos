#ifndef AURA_RUNTIME_CORE_TYPES_RECT_HPP__
#define AURA_RUNTIME_CORE_TYPES_RECT_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"

#if !defined(AURA_BUILD_XTENSA)
#  include <iostream>
#  include <sstream>
#  include <string>
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
 * @brief Template class for 2D rectangles.
 *
 * This class defines a rectangular region in a 2D space with width, height, and top-left corner coordinates.
 *
 * It supports arithmetic operations applied to coordinate points, width and height. Furthermore, it includes
 * logical operations for calculating the intersection and union of two instances. It also provides a cast
 * operator and printing operations for members.
 *
 * @tparam Tp0 The data type used for representing coordinates and sizes.
 */
template<typename Tp0>
class Rect_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor initializing all values to 0.
     */
    Rect_() : m_x(0), m_y(0), m_width(0), m_height(0)
    {}

    /**
     * @brief Constructor setting width and height while keeping top-left corner coordinates at (0, 0).
     *
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     */
    Rect_(Tp0 width, Tp0 height)
                : m_x(0), m_y(0), m_width(width), m_height(height)
    {}

    /**
     * @brief Constructor setting top-left corner coordinates, width and height.
     *
     * @param x The x-coordinate of the top-left corner of the rectangle.
     * @param y The y-coordinate of the top-left corner of the rectangle.
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     */
    Rect_(Tp0 x, Tp0 y, Tp0 width, Tp0 height)
                : m_x(x), m_y(y), m_width(width), m_height(height)
    {}

    /**
     * @brief Constructor setting top-left corner coordinates, width and height.
     *
     * @param pt The top-left corner point of the rectangle.
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     */
    Rect_(const Point2_<Tp0> &pt, Tp0 width, Tp0 height)
                : m_x(pt.m_x), m_y(pt.m_y), m_width(width), m_height(height)
    {}

    /**
     * @brief Constructor setting size with (0, 0) coordinates.
     *
     * @param sz The size of the rectangle.
     */
    Rect_(const Sizes2_<Tp0> &sz)
                : m_x(0), m_y(0), m_width(sz.m_width), m_height(sz.m_height)
    {}

    /**
     * @brief Constructor setting top-left corner coordinates, width and height.
     *
     * @param x The x-coordinate of the top-left corner of the rectangle.
     * @param y The y-coordinate of the top-left corner of the rectangle.
     * @param sz The size of the rectangle.
     */
    Rect_(Tp0 x, Tp0 y, const Sizes2_<Tp0> &sz)
                : m_x(x), m_y(y), m_width(sz.m_width), m_height(sz.m_height)
    {}

    /**
     * @brief Constructor setting top-left corner coordinates, width and height.
     *
     * @param pt_lt The point representing the top-left corner of the rectangle.
     * @param pt_rb The point representing the bottom-right corner of the rectangle.
     */
    Rect_(const Point2_<Tp0> &pt, const Sizes2_<Tp0> &sz)
                : m_x(pt.m_x), m_y(pt.m_y), m_width(sz.m_width), m_height(sz.m_height)
    {}

    /**
     * @brief Constructor setting coordinates and size from two points defining the rectangle's opposite corners.
     *
     * @param pt_lt The point representing the top-left corner of the rectangle.
     * @param pt_rb The point representing the bottom-right corner of the rectangle.
     */
    Rect_(const Point2_<Tp0> &pt_lt, const Point2_<Tp0> &pt_rb)
    {
        m_x = Min(pt_lt.m_x, pt_rb.m_x);
        m_y = Min(pt_lt.m_y, pt_rb.m_y);

        m_width  = Max(pt_rb.m_x, pt_lt.m_x) - m_x;
        m_height = Max(pt_rb.m_y, pt_lt.m_y) - m_y;
    }

    /**
     * @brief Copy constructor.
     *
     * @param rect Another Rect_ object to copy values from.
     */
    Rect_(const Rect_ &rect)
    {
        m_x      = rect.m_x;
        m_y      = rect.m_y;
        m_width  = rect.m_width;
        m_height = rect.m_height;
    }

    /**
     * @brief Operator to convert to a rectangle of another data type.
     *
     * @tparam Tp1 The data type to convert the rectangle to.
     * @return The converted rectangle.
     */
    template<typename Tp1>
    operator Rect_<Tp1>() const
    {
        return Rect_<Tp1>(SaturateCast<Tp1>(m_x), SaturateCast<Tp1>(m_y),
                          SaturateCast<Tp1>(m_width), SaturateCast<Tp1>(m_height));
    }

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded ostream operator to print the rectangle's attributes.
     *
     * @param os The output stream.
     * @param rect The rectangle to print.
     *
     * @return Output stream with the rectangle's information.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Rect_ &rect)
    {
        os << "(" << rect.m_x << ", " << rect.m_y << ", "
           << rect.m_width << ", " << rect.m_height << ")";
        return os;
    }

    /**
     * @brief Converts the rectangle attributes to a string representation.
     *
     * @return The string representation of the rectangle.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Assignment operator to copy the attributes from another rectangle.
     *
     * @param rect The rectangle to copy attributes from.
     *
     * @return The updated rectangle.
     */
    Rect_& operator=(const Rect_ &rect)
    {
        m_x      = rect.m_x;
        m_y      = rect.m_y;
        m_width  = rect.m_width;
        m_height = rect.m_height;
        return *this;
    }

    /**
     * @brief Adds a Point2_ object to the rectangle's top-left corner coordinates.
     *
     * @param pt The point to add.
     *
     * @return The updated rectangle.
     */
    Rect_& operator+=(const Point2_<Tp0> &pt)
    {
        m_x += pt.m_x;
        m_y += pt.m_y;
        return *this;
    }

    /**
     * @brief Adds Sizes2_ object to the rectangle's size.
     *
     * @param sz The sizes to add.
     *
     * @return The updated rectangle.
     */
    Rect_& operator+=(const Sizes2_<Tp0> &sz)
    {
        m_width  += sz.m_width;
        m_height += sz.m_height;
        return *this;
    }

    /**
     * @brief Subtracts a Point2_ object from the rectangle's top-left corner coordinates.
     *
     * @param pt The point to subtract.
     *
     * @return The updated rectangle.
     */
    Rect_& operator-=(const Point2_<Tp0> &pt)
    {
        m_x -= pt.m_x;
        m_y -= pt.m_y;
        return *this;
    }

    /**
     * @brief Subtracts Sizes2_ from the rectangle's size.
     *
     * @param sz The sizes to subtract.
     *
     * @return The updated rectangle.
     */
    Rect_& operator-=(const Sizes2_<Tp0> &sz)
    {
        m_width  -= sz.m_width;
        m_height -= sz.m_height;
        return *this;
    }

    /**
     * @brief Intersection operation with another rectangle.
     *
     * @param rect The rectangle to intersect with.
     *
     * @return The updated rectangle.
     */
    Rect_& operator&=(const Rect_ &rect)
    {
        Tp0 x1 = Max(m_x, rect.m_x);
        Tp0 y1 = Max(m_y, rect.m_y);

        m_width = Min(m_x + m_width , rect.m_x + rect.m_width ) - x1;
        m_width = Min(m_y + m_height, rect.m_y + rect.m_height) - y1;

        m_x = x1;
        m_y = y1;

        if (m_width <= 0 || m_height <= 0)
        {
            *this = Rect_();
        }
        return *this;
    }

    /**
     * @brief Union operation with another rectangle.
     *
     * @param rect The rectangle to perform a union with.
     *
     * @return The updated rectangle.
     */
    Rect_& operator|=(const Rect_ &rect)
    {
        if (Empty())
        {
            *this = rect;
        }
        else if (!rect.Empty())
        {
            Tp0 x1 = Min(m_x, rect.m_x);
            Tp0 y1 = Min(m_y, rect.m_y);

            m_width = Max(m_x + m_width , rect.m_x + rect.m_width ) - x1;
            m_width = Max(m_y + m_height, rect.m_y + rect.m_height) - y1;

            m_x = x1;
            m_y = y1;
        }
        return *this;
    }

    /**
     * @brief Check for equality with another rectangle.
     *
     * @param rect The rectangle to compare.
     *
     * @return Returns true if rectangles are equal, false otherwise.
     */
    DT_BOOL operator==(const Rect_ &rect) const
    {
        return (m_x == rect.m_x) && (rect.m_width == m_width)
                && (m_y == rect.m_y) && (rect.m_height == m_height);
    }

    /**
     * @brief Check for inequality with another rectangle.
     *
     * @param rect The rectangle to compare.
     *
     * @return Returns true if rectangles are not equal, false otherwise.
     */
    DT_BOOL operator!=(const Rect_ &rect) const
    {
        return (m_x != rect.m_x) || (rect.m_width != m_width)
                || (m_y != rect.m_y) || (rect.m_height != m_height);
    }

    /**
     * @brief Returns a new rectangle by adding a point to the current rectangle‘s top-left corner.
     *
     * @param pt The point to be added.
     *
     * @return The resulting rectangle.
     */
    Rect_ operator+(const Point2_<Tp0> &pt) const
    {
        return Rect_(m_x + pt.m_x, m_y + pt.m_y, m_width, m_height);
    }

    /**
     * @brief Returns a new rectangle by adding sizes to the current rectangle's size.
     *
     * @param sz The sizes to be added.
     *
     * @return The resulting rectangle.
     */
    Rect_ operator+(const Sizes2_<Tp0> &sz) const
    {
        return Rect_(m_x, m_y, m_width + sz.m_width, m_height + sz.m_height);
    }

    /**
     * @brief Returns a new rectangle by subtracting a point from the current rectangle‘s top-left corner.
     *
     * @param pt The point to be subtracted.
     *
     * @return The resulting rectangle.
     */
    Rect_ operator-(const Point2_<Tp0> &pt) const
    {
        return Rect_(m_x - pt.m_x, m_y - pt.m_y, m_width, m_height);
    }

    /**
     * @brief Returns a new rectangle by subtracting sizes from the current rectangle's sizes.
     *
     * @param sz The sizes to be subtracted.
     *
     * @return The resulting rectangle.
     */
    Rect_ operator-(const Sizes2_<Tp0> &sz) const
    {
        return Rect_(m_x, m_y, Max(m_width - sz.m_width, (Tp0)0), Max(m_height - sz.m_height, (Tp0)0));
    }

    /**
     * @brief Returns the intersection of two rectangles.
     *
     * @param rect The rectangle to intersect with.
     *
     * @return The resulting intersection rectangle.
     */
    Rect_ operator&(const Rect_ &rect) const
    {
        Rect_ c = *this;
        c &= rect;
        return c;
    }

    /**
     * @brief Returns the union of two rectangles.
     *
     * @param rect The rectangle to perform a union with.
     *
     * @return The resulting union rectangle.
     */
    Rect_ operator|(const Rect_ &rect) const
    {
        Rect_ c = *this;
        c |= rect;
        return c;
    }

    /**
     * @brief Returns the position of the top-left corner of the rectangle.
     *
     * @return The top-left corner position.
     */
    Point2_<Tp0> TopLeftPos() const
    {
        return Point2_<Tp0>(m_x, m_y);
    }

    /**
     * @brief Returns the position of the bottom-right corner of the rectangle.
     *
     * @return The bottom-right corner position.
     */
    Point2_<Tp0> RightBottomPos() const
    {
        return Point2_<Tp0>(m_x + m_width, m_y + m_height);
    }

    /**
     * @brief Returns the size of the rectangle.
     *
     * @return The size of the rectangle.
     */
    Sizes2_<Tp0> Size() const
    {
        return Sizes2_<Tp0>(m_height, m_width);
    }

    /**
     * @brief Calculates and returns the area of the rectangle.
     *
     * @return The area of the rectangle.
     */
    Tp0 Area() const
    {
        return (m_width * m_height);
    }

    /**
     * @brief Checks if the rectangle is empty (has zero size).
     *
     * @return True if the rectangle is empty, otherwise false.
     */
    DT_BOOL Empty() const
    {
        return m_width <= 0 || m_height <= 0;
    }

    /**
     * @brief Checks if the rectangle contains a point.
     *
     * @param pt The point to check.
     *
     * @return True if the point is contained within the rectangle, otherwise false.
     */
    DT_BOOL Contains(const Point2_<Tp0> &pt) const
    {
        return (m_x <= pt.m_x) && (pt.m_x < m_x + m_width)
                && (m_y <= pt.m_y) && (pt.m_y < m_y + m_height);
    }

    Tp0 m_x;       /*!< The x-coordinate of the top-left corner of the rectangle. */
    Tp0 m_y;       /*!< The y-coordinate of the top-left corner of the rectangle. */
    Tp0 m_width;   /*!< The width of the rectangle. */
    Tp0 m_height;  /*!< The height of the rectangle. */
};

typedef Rect_<DT_S32> Rect2i;
typedef Rect_<DT_F32> Rect2f;
typedef Rect_<DT_F64> Rect2d;
typedef Rect2i Rect;

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_RECT_HPP__