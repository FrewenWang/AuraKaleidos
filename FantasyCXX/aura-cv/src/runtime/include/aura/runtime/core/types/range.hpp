#ifndef AURA_RUNTIME_CORE_TYPES_RANGE_HPP__
#define AURA_RUNTIME_CORE_TYPES_RANGE_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/maths.hpp"

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
 * @brief The class specifying a range.
 *
 * The class is used to specify a row or a column span in a matrix and for many other purposes. This class contains
 * two members, `m_start` and `m_end`, where `m_start` is the inclusive left boundary of the range and `m_end` is
 * the only right boundary of the range. Such a half-opened interval is usually denoted as \f$[start,end)\f$.
 *
 * It supports arithmetic operations applied to range boundaries. Furthermore, it includes logical operations for
 * calculating the intersection and union of two instances. The users can also use the `ToString()` to print the range.
 */
class AURA_EXPORTS Range
{
public:
    /**
     * @brief Default constructor.
     */
    Range() : m_start(0), m_end(0)
    {}

    /**
     * @brief Constructor specifying the end of the range.
     *
     * @param end The end of the range.
     */
    Range(DT_S32 end) : m_start(0), m_end(end)
    {}

    /**
     * @brief Constructor specifying the start and end of the range.
     *
     * @param start The start of the range.
     * @param end The end of the range.
     */
    Range(DT_S32 start, DT_S32 end)
                : m_start(start), m_end(end)
    {}

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded stream insertion operator for Range objects.
     *
     * @param os The output stream.
     * @param com The Range object to be output.
     *
     * @return Output stream with the Range object.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Range &com)
    {
        os << "[" << com.m_start << ", " << com.m_end << "]";
        return os;
    }

    /**
     * @brief Converts the Range object to a string.
     *
     * @return The string representation of the Range object.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
#endif

    /**
     * @brief Calculates the size of the range.
     *
     * @return The size of the range (difference between end and start).
     */
    DT_S32 Size() const
    {
        return m_end - m_start;
    }

    /**
     * @brief Checks if the range is empty.
     *
     * @return Returns 1 if the range is empty (start equals end), otherwise 0.
     */
    DT_S32 Empty() const
    {
        return m_end == m_start;
    }

    /**
     * @brief Equality comparison operator for Range objects.
     *
     * @param r The Range object to compare with.
     *
     * @return Returns true if both Range objects are equal, otherwise false.
     */
    DT_BOOL operator==(const Range &r) const
    {
        return (m_start == r.m_start) && (m_end == r.m_end);
    }

    /**
     * @brief Inequality comparison operator for Range objects.
     *
     * @param r The Range object to compare with.
     *
     * @return DT_BOOL Returns true if both Range objects are not equal, otherwise false.
     */
    DT_BOOL operator!=(const Range &r) const
    {
        return (m_start != r.m_start) || (m_end != r.m_end);
    }

    /**
     * @brief Intersection operation between two Range objects.
     *
     * @param r The Range object to intersect with.
     *
     * @return Range The resulting intersected range.
     */
    Range operator&(const Range &r) const
    {
        Range ret(Max(m_start, r.m_start), Min(m_end, r.m_end));
        ret.m_end = Max(ret.m_end, ret.m_start);
        return ret;
    }

    /**
     * @brief Intersection operation between two Range objects.
     *
     * @param r The Range object to intersect with.
     *
     * @return Reference to the modified current range.
     */
    Range& operator&=(const Range &r)
    {
        *this = *this & r;
        return *this;
    }

    DT_S32 m_start; /*!< The start of the range. */
    DT_S32 m_end;   /*!< The end of the range. */
};

/**
 * @brief Addition operator for a Range object and an integer value.
 *
 * @param r The Range object.
 * @param delta The integer value to be added.
 *
 * @return The new Range with start and end values increased by the delta.
 */
AURA_INLINE Range operator+(const Range &r, DT_S32 delta)
{
    return Range(r.m_start + delta, r.m_end + delta);
}

/**
 * @brief Addition operator for an integer value and a Range object.
 *
 * @param delta The integer value to be added.
 * @param r The Range object.
 *
 * @return The new Range with start and end values increased by the delta.
 */
AURA_INLINE Range operator+(DT_S32 delta, const Range &r)
{
    return Range(r.m_start + delta, r.m_end + delta);
}

/**
 * @brief Subtraction operator for a Range object and an integer value.
 *
 * @param r The Range object.
 * @param delta The integer value to be subtracted.
 *
 * @return Range The new Range with start and end values decreased by the delta.
 */
AURA_INLINE Range operator-(const Range &r, DT_S32 delta)
{
    return Range(r.m_start - delta, r.m_end - delta);
}

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_RANGE_HPP__