#ifndef AURA_TOOLS_UNIT_TEST_TEST_TYPES_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_TYPES_HPP__

#include "aura/runtime/mat.h"

#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <utility>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup unit_test Unit Test
 * @}
 */

namespace aura
{
/**
 * @addtogroup unit_test
 * @{
 */

/**
 * @brief Struct representing time-related results for a test.
 *
 * This struct stores information about the time taken for a test, including
 * the minimum, maximum, and average time.
 */
struct AURA_EXPORTS TestTime
{
    /**
     * @brief Default constructor for TestTime.
     */
    TestTime() : min_time(0.0), max_time(0.0), avg_time(0.0)
    {}

    /**
     * @brief Converts the TestTime struct to a string.
     */
    std::string ToString() const
    {
        std::stringstream time_sstream;

        MI_CHAR avg_time_buffer[20];
        MI_CHAR min_time_buffer[20];
        MI_CHAR max_time_buffer[20];

        std::snprintf(avg_time_buffer, 20, "%.3f", avg_time);
        std::snprintf(min_time_buffer, 20, "%.3f", min_time);
        std::snprintf(max_time_buffer, 20, "%.3f", max_time);

        std::string avg_time_str = avg_time_buffer;
        std::string min_time_str = min_time_buffer;
        std::string max_time_str = max_time_buffer;

        MI_S32 max_str_len = std::max(std::max(avg_time_str.size(), min_time_str.size()), max_time_str.size());
        max_str_len = max_str_len > 8 ? max_str_len : 8;

        // make sure every time string have the same length
        avg_time_str = std::string(max_str_len - avg_time_str.size(), ' ') + avg_time_str;
        min_time_str = std::string(max_str_len - min_time_str.size(), ' ') + min_time_str;
        max_time_str = std::string(max_str_len - max_time_str.size(), ' ') + max_time_str;

        time_sstream << "avg:" << avg_time_str << "ms  |  min:" << min_time_str << "ms  |  max:" << max_time_str << "ms";
        return time_sstream.str();
    }

    MI_F64 min_time; /*!< Minimum time taken for the test. */
    MI_F64 max_time; /*!< Maximum time taken for the test. */
    MI_F64 avg_time; /*!< Average time taken for the test. */
};

/**
 * @brief Struct representing the size and strides of a matrix.
 */
struct AURA_EXPORTS MatSize
{
    /**
     * @brief Default constructor for MatSize.
     */
    MatSize() = default;

    /**
     * @brief Constructor for MatSize with scaling.
     *
     * @param mat_size The original MatSize.
     * @param scale The scaling factor.
     *
     * @return The constructed MatSize.
     */
    MatSize(const MatSize &mat_size, const Scalar &scale)
    {
        MI_S32 height = SaturateCast<MI_S32>(mat_size.m_sizes.m_height * scale.m_val[1]);
        MI_S32 width  = SaturateCast<MI_S32>(mat_size.m_sizes.m_width * scale.m_val[0]);
        m_sizes       = Sizes3(height, width, mat_size.m_sizes.m_channel);

        height    = SaturateCast<MI_S32>(mat_size.m_strides.m_height * scale.m_val[1]);
        width     = SaturateCast<MI_S32>(mat_size.m_strides.m_width * scale.m_val[0]);
        m_strides = Sizes(height, width);
    }

    /**
     * @brief Constructor for MatSize with specified sizes and strides.
     *
     * @param sizes The size of the matrix.
     * @param strides The strides of the matrix (default is Sizes()).
     *
     * @return The constructed MatSize.
     */
    MatSize(Sizes3 sizes, Sizes strides = Sizes())
    {
        m_sizes   = sizes;
        m_strides = strides;
    }

    /**
     * @brief Overloaded stream insertion operator for MatSize.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const MatSize &mat_size)
    {
        os << mat_size.m_sizes;
        return os;
    }

    /**
     * @brief Convert MatSize to a string representation.
     */
    std::string ToString()
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    Sizes3 m_sizes;   /*!< The size of the matrix. */
    Sizes  m_strides; /*!< The strides of the matrix. */
};

/**
 * @brief Struct representing the size of a border.
 */
struct AURA_EXPORTS BorderSize
{
    /**
     * @brief Default constructor for BorderSize.
     */
    BorderSize()
    {}

    /**
     * @brief Constructor for BorderSize with specified top, bottom, left, and right values.
     *
     * @param top The top size of the border.
     * @param bottom The bottom size of the border.
     * @param left The left size of the border.
     * @param right The right size of the border.
     *
     * @return The constructed BorderSize.
     */
    BorderSize(MI_S32 top, MI_S32 bottom, MI_S32 left, MI_S32 right) : top(top), bottom(bottom),
               left(left), right(right)
    {}

    /**
     * @brief Overloaded stream insertion operator for BorderSize.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const BorderSize &sz)
    {
        os << "Border(" << sz.top << ", " << sz.bottom << ", " << sz.left << ", " << sz.right << ")";
        return os;
    }

    /**
     * @brief Convert BorderSize to a string representation.
     */
    std::string ToString()
    {
        std::stringstream sstream;
        sstream << "Border(" << top << ", " << bottom << ", " << left << ", " << right << ")";
        return sstream.str();
    }

    MI_S32 top;    /*!< The top size of the border. */
    MI_S32 bottom; /*!< The bottom size of the border. */
    MI_S32 left;   /*!< The left size of the border. */
    MI_S32 right;  /*!< The right size of the border. */
};

/**
 * @brief Struct representing a pair of matrix element types.
 */
struct AURA_EXPORTS MatElemPair
{
    /**
     * @brief Default constructor for MatElemPair.
     */
    MatElemPair()
    {}

    /**
     * @brief Constructor for MatElemPair with specified source and destination element types.
     *
     * @param first The source element type.
     * @param second The destination element type.
     *
     * @return The constructed MatElemPair.
     */
    MatElemPair(ElemType first, ElemType second) : first(first), second(second)
    {}

    /**
     * @brief Overloaded stream insertion operator for MatElemPair.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const MatElemPair &sz)
    {
        os << "src elem type : " << sz.first << " dst elem type : " << sz.second << std::endl;
        return os;
    }

    /**
     * @brief Convert MatElemPair to a string representation.
     */
    std::string ToString()
    {
        std::stringstream sstream;
        sstream << "src elem type : " << first << " dst elem type : " << second;
        return sstream.str();
    }

    ElemType first;  /*!< The source element type. */
    ElemType second; /*!< The destination element type. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_TYPES_HPP__
