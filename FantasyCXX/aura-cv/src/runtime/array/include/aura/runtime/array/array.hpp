#ifndef AURA_RUNTIME_ARRAY_ARRAY_HPP__
#define AURA_RUNTIME_ARRAY_ARRAY_HPP__

#include "aura/runtime/context.h"
#include "aura/runtime/memory.h"

#include <iostream>
#include <string>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup array Array
 *      @{
 *          @defgroup array_class Array Class
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup array_class
 * @{
*/

/**
 * @brief Enumeration representing different element types.
 */
enum class ElemType
{
    INVALID    = 0,
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    F32,
    F64,
    F16,
};

/**
 * @brief Template specialization for getting the element type.
 */
template <typename Tp> AURA_NO_STATIC_INLINE ElemType GetElemType()           { return ElemType::INVALID; }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_U8>()    { return ElemType::U8;      }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_S8>()    { return ElemType::S8;      }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_U16>()   { return ElemType::U16;     }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_S16>()   { return ElemType::S16;     }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_U32>()   { return ElemType::U32;     }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_S32>()   { return ElemType::S32;     }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_F32>()   { return ElemType::F32;     }
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_F64>()   { return ElemType::F64;     }
#if defined(AURA_BUILD_HOST)
template <>            AURA_NO_STATIC_INLINE ElemType GetElemType<MI_F16>()   { return ElemType::F16;     }
#endif

/**
 * @brief Overloaded stream operator for ElemType.
 *
 * @param os        Output stream.
 * @param elem_type Element type to be streamed.
 * 
 * @return Output stream with element type information.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ElemType elem_type)
{
    switch (elem_type)
    {
        case ElemType::U8:
        {
            os << "U8";
            break;
        }

        case ElemType::S8:
        {
            os << "S8";
            break;
        }

        case ElemType::U16:
        {
            os << "U16";
            break;
        }

        case ElemType::S16:
        {
            os << "S16";
            break;
        }

        case ElemType::U32:
        {
            os << "U32";
            break;
        }

        case ElemType::S32:
        {
            os << "S32";
            break;
        }

        case ElemType::F32:
        {
            os << "F32";
            break;
        }

        case ElemType::F64:
        {
            os << "F64";
            break;
        }

        case ElemType::F16:
        {
            os << "F16";
            break;
        }

        case ElemType::INVALID:
        {
            os << "INVALID";
            break;
        }

        default:
        {
            os << "INVALID elem_type";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts ElemType to a string.
 *
 * @param elem_type Element type to be converted.
 * 
 * @return String representation of the element type.
 */
AURA_INLINE std::string ElemTypesToString(ElemType elem_type)
{
    std::ostringstream oss;
    oss << elem_type;
    return oss.str();
}

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
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            elem_size = sizeof(MI_F16);
            break;
        }
#endif
        default:
        {
            break;
        }
    }

    return elem_size;
}

/**
 * @brief Enumeration representing different array types.
 */
enum class ArrayType
{
    INVALID        = 0,
    MAT,
    CL_MEMORY,
    XTENSA_MAT
};

/**
 * @brief Overloaded stream operator for ArrayType.
 *
 * @param os         Output stream.
 * @param array_type Array type to be streamed.
 * 
 * @return Output stream with array type information
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, ArrayType array_type)
{
    switch (array_type)
    {
        case ArrayType::MAT:
        {
            os << "MAT";
            break;
        }

        case ArrayType::CL_MEMORY:
        {
            os << "XTENSA_MAT";
            break;
        }

        case ArrayType::XTENSA_MAT:
        {
            os << "XTENSA_MAT";
            break;
        }

        case ArrayType::INVALID:
        {
            os << "INVALID";
            break;
        }

        default:
        {
            os << "INVALID array_type";
            break;
        }
    }

    return os;
}

/**
 * @brief Converts ArrayType to a string.
 *
 * @param array_type Array type to be converted.
 * 
 * @return String representation of the array type.
 */
AURA_INLINE std::string ArrayTypesToString(ArrayType array_type)
{
    std::ostringstream oss;
    oss << array_type;
    return oss.str();
}

/**
 * @brief Base class representing an array of elements.
 * 
 * 'Array' is a basic class of dense numerical array, which can be inherited by multiple subclasses, such as `Mat` class and `CLMem` Class.
 */
class AURA_EXPORTS Array
{
public:
    /**
     * @brief Default constructor.
     */
    Array();

    /**
     * @brief Constructor to initialize an array with specified parameters.
     *
     * @param ctx         Pointer to the context.
     * @param elem_type   Element type of the array.
     * @param sizes       Sizes of the array in three dimensions. [h, w, c]
     * @param strides     Strides of the array in three dimensions. [h, w, c], in Bytes.
     * @param buffer      Buffer containing array data.
     */
    Array(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides, const Buffer &buffer = Buffer());

    /**
     * @brief Copy constructor.
     *
     * @param array Array used to initialize the member variables.
     */
    Array(const Array &array);

    /**
     * @brief Assignment operator for initialize an array.
     *
     * @param array Array used to initialize the member variables.
     * 
     * @return Reference to the assigned array.
     */
    Array& operator=(const Array &array);

    /**
     * @brief Virtual destructor to ensure proper cleanup in derived classes.
     */
    virtual ~Array();

    /**
     * @brief Pure virtual function to release resources associated with the array.
     */
    virtual AURA_VOID Release() = 0;

    /**
     * @brief Gets the element type of the array.
     *
     * @return Element type of the array.
     */
    ElemType GetElemType() const;

    /**
     * @brief Gets the array type.
     *
     * @return Type of the array.
     */
    ArrayType GetArrayType() const;

    /**
     * @brief Gets the sizes of the array in three dimensions.
     *
     * @return Sizes of the array.
     */
    Sizes3 GetSizes() const;

    /**
     * @brief Gets the strides of the array in two dimensions.
     *
     * @return Strides of the array.
     */
    Sizes GetStrides() const;

    /**
     * @brief Gets the total number of bytes occupied by the array.
     *
     * @return Total bytes occupied by the array.
     */
    MI_S64 GetTotalBytes() const;

    /**
     * @brief Gets the row pitch of the array.
     *
     * @return Row pitch of the array.
     */
    MI_S32 GetRowPitch() const;

    /**
     * @brief Gets the row step of the array.
     *
     * @return Row step of the array.
     */
    MI_S32 GetRowStep() const;

    /**
     * @brief Gets the reference count of the array.
     *
     * @return Reference count of the array.
     */
    MI_S32 GetRefCount() const;

    /**
     * @brief Gets the buffer containing array data.
     *
     * @return Buffer containing array data.
     */
    const Buffer& GetBuffer() const;

    /**
     * @brief Gets the memory type of the array.
     *
     * @return Memory type of the array.
     */
    MI_S32 GetMemType() const;

    /**
     * @brief Checks if the array is valid.
     *
     * @return True if the array is valid, false otherwise.
     */
    virtual MI_BOOL IsValid() const;

    /**
     * @brief Checks if the current array is equal to another array.
     *
     * @param array Array to compare with.
     * 
     * @return True if the arrays are equal, false otherwise.
     */
    MI_BOOL IsEqual(const Array &array) const;

    /**
     * @brief Checks if the sizes of the current array are equal to another array.
     *
     * @param array Array to compare with.
     * 
     * @return True if the array sizes are equal, false otherwise.
     */
    MI_BOOL IsSizesEqual(const Array &array) const;

    /**
     * @brief Checks if the channels of the current array are equal to another array.
     *
     * @param array Array to compare with.
     * 
     * @return True if the array channels are equal, false otherwise.
     */
    MI_BOOL IsChannelEqual(const Array &array) const;

    /**
     * @brief Pure virtual function to display information about the array.
     */
    virtual AURA_VOID Show() const = 0;

    /**
     * @brief Pure virtual function to dump the array data to a file.
     *
     * @param fname File name for dumping array data.
     */
    virtual AURA_VOID Dump(const std::string &fname) const = 0;

protected:
    /**
     * @brief Initializes the reference count.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status InitRefCount();

    /**
     * @brief Adds to the reference count.
     *
     * @param num Number to add to the reference count.
     * @return Updated reference count.
     */
    MI_S32 AddRefCount(MI_S32 num) const;

    /**
    * @brief Clears the content of the array.
    *
    * This method is used to reset properties of member vaiables.
    */
    AURA_VOID Clear();

    /**
     * @brief Converts the array information to a string.
     *
     * @return String representation of the array.
     */
    std::string ToString() const;

    Context     *m_ctx;             /*!< Pointer to the associated context. */
    ElemType    m_elem_type;        /*!< Element type of the array. */
    ArrayType   m_array_type;       /*!< Type of the array (e.g., Mat or CLMem). */
    Sizes3      m_sizes;            /*!< Sizes of the array in three dimensions. */
    Sizes       m_strides;          /*!< Strides of the array. */
    MI_S64      m_total_bytes;      /*!< Total number of bytes occupied by the array. */
    MI_S32      *m_refcount;        /*!< Reference count for managing array's memory. */
    Buffer      m_buffer;           /*!< Buffer containing array data. */
};

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_ARRAY_ARRAY_HPP__