#ifndef AURA_OPS_CORE_OPENCL_HPP__
#define AURA_OPS_CORE_OPENCL_HPP__

#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup core_host Core Host
 * @}
*/

namespace aura
{
/**
 * @addtogroup core_host
 * @{
*/

/**
 * @brief Structure representing a vector of four single-precision floating-point values for OpenCL.
 */
struct AURA_EXPORTS CLScalar
{
    MI_F32 val[4];
};

/**
 * @brief Convert a Scalar to a CLScalar.
 *
 * @param scaler The input Scalar.
 * 
 * @return The converted CLScalar.
 */
AURA_INLINE CLScalar clScalar(const Scalar &scaler)
{
    CLScalar cl_scaler;

    cl_scaler.val[0] = (MI_F32)scaler.m_val[0];
    cl_scaler.val[1] = (MI_F32)scaler.m_val[1];
    cl_scaler.val[2] = (MI_F32)scaler.m_val[2];
    cl_scaler.val[3] = (MI_F32)scaler.m_val[3];

    return cl_scaler;
}

/**
 * @brief Get the OpenCL type string for the specified element type.
 *
 * @param type The element type.
 * 
 * @return Corresponding OpenCL data type string.
 */
AURA_INLINE std::string CLTypeString(ElemType type)
{
    switch (type)
    {
        case ElemType::U8:
        {
            return "uchar";
        }

        case ElemType::S8:
        {
            return "char";
        }

        case ElemType::U16:
        {
            return "ushort";
        }

        case ElemType::S16:
        {
            return "short";
        }

        case ElemType::U32:
        {
            return "uint";
        }

        case ElemType::S32:
        {
            return "int";
        }

        case ElemType::F16:
        {
            return "half";
        }

        case ElemType::F32:
        {
            return "float";
        }

        default:
        {
            return "DefaultType";
        }
    }
}

/**
 * @brief Get the OpenCL type string for the specified type.
 *
 * @tparam Tp Data type.
 *
 * @return Corresponding OpenCL data type string.
 */
template <typename Tp>
AURA_INLINE std::string CLTypeString()
{
    return "DefaultType";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_U8>()
{
    return "uchar";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_S8>()
{
    return "char";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_U16>()
{
    return "ushort";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_S16>()
{
    return "short";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_U32>()
{
    return "uint";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_S32>()
{
    return "int";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_U64>()
{
    return "ulong";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_S64>()
{
    return "long";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_F16>()
{
    return "half";
}

template <> AURA_NO_STATIC_INLINE std::string CLTypeString<MI_F32>()
{
    return "float";
}

/**
 * @brief Check if the width is compatible with OpenCL vector length for an Array.
 *
 * This function checks if the width of the input Array is compatible with the OpenCL vector length.
 *
 * @param array The input Array.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_INLINE Status CheckCLWidth(const Array &array)
{
    MI_S32 width = array.GetSizes().m_width;
    if (width < 64)
    {
        return Status::ERROR;
    }

    return Status::OK;
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_CORE_OPENCL_HPP__