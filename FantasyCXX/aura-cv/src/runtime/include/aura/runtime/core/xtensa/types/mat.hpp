#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_MAT_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_MAT_HPP__

#include "aura/runtime/core/xtensa/comm.hpp"
#include "aura/runtime/core/types.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup mat Runtime Core Xtensa Mat
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup mat
 * @{
*/

/**
 * @brief Returns the size of an element type in bytes.
 *
 * @param elem_type Element type for which the size is queried.
 *
 * @return Size of the element type in bytes.
 */
AURA_INLINE MI_S32 ElemTypeSize(ElemType elem_type)
{
    MI_S32 elem_size = 0;

    switch (elem_type)
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            elem_size = sizeof(MI_U8);
            break;
        }
        case ElemType::U16:
        case ElemType::S16:
        {
            elem_size = sizeof(MI_U16);
            break;
        }
        case ElemType::U32:
        case ElemType::S32:
        {
            elem_size = sizeof(MI_U32);
            break;
        }
        case ElemType::F32:
        {
            elem_size = sizeof(MI_F32);
            break;
        }
        case ElemType::F64:
        {
            elem_size = sizeof(MI_F64);
            break;
        }
        default:
        {
            break;
        }
    }

    return elem_size;
}

/**
 * @brief N-dimensional matirx class.
 *
 * For data layout, 2-dimensional matrices are stored in row-major order, and 3-dimensional matrices are stored in
 * a channel-last order (e.g. RGB|RGB|RGB|...).
 *
 * The formula for accessing the k-th channel of the i-th row and j-th column in an RGB iaura matrix is as follows:
 * \f[M[i][j][k] = i \cdot \text{m_strides.m_width} + j \cdot \text{m_sizes.m_channel} + k\f]
 */
class Mat
{
public:
    /**
     * @brief Default constructor for creating an empty matrix.
     */
    Mat()
    {};

    /**
     * @brief Constructor for creating a matrix with specified properties and existing buffer.
     *
     * @param elem_type Element type of the matrix.
     * @param sizes Size of the matrix in three dimensions (height, width, channels).
     * @param buffer Buffer containing the data for the matrix.
     * @param strides Strides for each dimension of the matrix (default is Sizes of 0, which means no padding).
     */
    Mat(ElemType elem_type, Sizes3 &sizes, const Buffer &buffer, const Sizes &strides = Sizes())
    {
        MI_S64 total_bytes = static_cast<MI_S64>(strides.m_width) * strides.m_height;

        m_elem_type   = elem_type;
        m_sizes       = sizes;
        m_strides     = strides;
        m_total_bytes = total_bytes;
        m_buffer      = buffer;
    };

    /**
     * @brief Get a pointer to the raw data of the matrix.
     *
     * @return Pointer to the raw data of the matrix.
     */
    AURA_VOID* GetData()
    {
        return m_buffer.m_data;
    };

    /**
     * @brief Get a const pointer to the raw data of the matrix.
     *
     * @return Const pointer to the raw data of the matrix.
     */
    const AURA_VOID* GetData() const
    {
        return m_buffer.m_data;
    };

    /**
     * @brief Get a pointer to the specified row of the matrix.
     *
     * @tparam Tp Type of the elements in the matrix.
     * 
     * @param row Index of the row to access.
     *
     * @return Pointer to the specified row of the matrix.
     */
    template<typename Tp>
    Tp* Ptr(MI_S32 row)
    {
        const MI_S32 off = row * m_strides.m_width;
        if (off < m_total_bytes)
        {
            return m_buffer.GetData<Tp*>(off);
        }
        return MI_NULL;
    };

    /**
     * @brief Gets the element type of the array.
     *
     * @return Element type of the array.
     */
    ElemType GetElemType() const
    {
        return m_elem_type;
    };

    /**
     * @brief Gets the sizes of the array in three dimensions.
     *
     * @return Sizes of the array.
     */
    Sizes3 GetSizes() const
    {
        return m_sizes;
    };

    /**
     * @brief Gets the strides of the array in two dimensions.
     *
     * @return Strides of the array.
     */
    Sizes GetStrides() const
    {
        return m_strides;
    };

    /**
     * @brief Gets the total number of bytes occupied by the array.
     *
     * @return Total bytes occupied by the array.
     */
    MI_S64 GetTotalBytes() const
    {
        return m_total_bytes;
    };

    /**
     * @brief Gets the row pitch of the array.
     *
     * @return Row pitch of the array.
     */
    MI_S32 GetRowPitch() const
    {
        return m_strides.m_width;
    };

    /**
     * @brief Gets the row step of the array.
     *
     * @return Row step of the array.
     */
    MI_S32 GetRowStep() const
    {
        MI_S32 pixels = ElemTypeSize(m_elem_type) * m_sizes.m_channel;

        if (0 == pixels)
        {
            return 0;
        }
        else
        {
            return m_strides.m_width / pixels;
        }
    };

    /**
     * @brief Check if the matrix is valid (i.e., allocated and properly initialized).
     *
     * @return True if the matrix is valid, false otherwise.
     */
    MI_BOOL IsValid() const
    {
        return (m_total_bytes > 0 && m_buffer.IsValid() && m_elem_type != ElemType::INVALID);
    }

    /**
     * @brief Checks if the current mat is equal to another mat.
     *
     * @param Mat Mat to compare with.
     *
     * @return True if the arrays are equal, false otherwise.
     */
    MI_BOOL IsEqual(const Mat &mat) const
    {
        return ((mat.GetElemType() == m_elem_type) && (mat.GetSizes() == m_sizes));
    }

private:
    ElemType   m_elem_type;        /*!< Element type of the mat. */
    Sizes3     m_sizes;            /*!< Sizes of the mat in three dimensions. */
    Sizes      m_strides;          /*!< Strides of the mat. */
    MI_S64     m_total_bytes;      /*!< Total number of bytes occupied by the mat. */
    Buffer     m_buffer;           /*!< Buffer containing array data. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_MAT_HPP__
