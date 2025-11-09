#ifndef AURA_OPS_CORE_COMM_HPP__
#define AURA_OPS_CORE_COMM_HPP__

#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

#include <string>
#include <vector>

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup core_commn Core Common
 * @}
 */

#if defined(AURA_BUILD_HOST)
#  define AURA_OPS_TASK_LOAD                (65536)
#endif

/**
 * @addtogroup core_commn
 * @{
 */

/**
 * @brief Generates a numerical pattern with only one parameters.
 *
 * @param tp First input parameter.
 *
 * @return Calculated numerical pattern value.
 */
#define AURA_MAKE_PATTERN1(tp)                                  (static_cast<DT_S32>(tp))

/**
 * @brief Generates a numerical pattern with two parameters.
 *
 * @param tp0 First input parameter.
 * @param tp1 Second input parameter.
 *
 * @return Calculated numerical pattern value.
 */
#define AURA_MAKE_PATTERN2(tp0, tp1)                            (static_cast<DT_S32>(tp0) * 16  + static_cast<DT_S32>(tp1))

/**
 * @brief Generates a pattern numerical with three parameters.
 *
 * @param tp0 First input parameter.
 * @param tp1 Second input parameter.
 * @param tp2 Third input parameter.
 *
 * @return Calculated numerical pattern value.
 */
#define AURA_MAKE_PATTERN3(tp0, tp1, tp2)                       (static_cast<DT_S32>(tp0) * 256 + static_cast<DT_S32>(tp1) * 16  +         \
                                                                 static_cast<DT_S32>(tp2))
/**
 * @brief Generates a numerical pattern with four parameters.
 *
 * @param tp0 First input parameter.
 * @param tp1 Second input parameter.
 * @param tp2 Third input parameter.
 * @param tp3 Fourth input parameter.
 *
 * @return Calculated numerical pattern value.
 */
#define AURA_MAKE_PATTERN4(tp0, tp1, tp2, tp3)                  (static_cast<DT_S32>(tp0) * 4096 + static_cast<DT_S32>(tp1) * 256 +        \
                                                                 static_cast<DT_S32>(tp2) * 16   + static_cast<DT_S32>(tp3))

/**
 * @brief Helper macro used to select the appropriate pattern macro based on the number of arguments passed.
 */
#define AURA_MAKE_PATTERN_HELPER(_1, _2, _3, _4, NAME, ...)     NAME

/**
 * @brief The main macro that users will typically use.
 *
 * It uses the variadic macro technique to pass the arguments to the helper macro.
 * The helper macro then selects the appropriate pattern macro based on the number of arguments.
 * The maximum number of parameters is 4; if four arguments are provided, it selects AURA_MAKE_PATTERN4;
 * if three arguments are provided, it selects AURA_MAKE_PATTERN3; if two arguments are provided, it selects AURA_MAKE_PATTERN2;
 * if one argument is provided, it selects AURA_MAKE_PATTERN1.
 */
#define AURA_MAKE_PATTERN(...)                                  AURA_MAKE_PATTERN_HELPER(__VA_ARGS__, AURA_MAKE_PATTERN4,                  \
                                                                                         AURA_MAKE_PATTERN3, AURA_MAKE_PATTERN2,           \
                                                                                         AURA_MAKE_PATTERN1, ...)(__VA_ARGS__)

/**
 * @}
 */

namespace aura
{
/**
 * @addtogroup core_commn
 * @{
 */
/**
 * @brief Enumeration of interpolation types.
 */
enum class InterpType
{
    /** 最近邻插值 **/
    NEAREST    = 0, /*!< nearest neighbor interpolation */
    /** 双线性插值 **/
    LINEAR,         /*!< bilinear interpolation */
    /** 三次样条插值  **/
    CUBIC,          /*!< bicubic interpolation */
    /** 使用像素区域关系重新采样。这可能是图像抽取的首选方法，因为它可以获得无摩尔纹的结果。但是当图像缩放时，它类似于 INTER_NEAREST 方法。**/
    AREA,           /*!< resampling using pixel area relation. It may be a preferred method for iaura decimation, as it gives moire’-free results. But when the iaura is zoomed, it is similar to the INTER_NEAREST method. */
};

/**
 * @brief Overloaded output stream operator for InterpType enumeration.
 *
 * This operator allows printing InterpType enumerators to an output stream.
 *
 * @param os The output stream.
 * @param interp_type The InterpType enumerator to be printed.
 *
 * @return The modified output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, InterpType interp_type)
{
    switch (interp_type)
    {
        case InterpType::NEAREST:
        {
            os << "Nearest";
            break;
        }

        case InterpType::LINEAR:
        {
            os << "Linear";
            break;
        }

        case InterpType::CUBIC:
        {
            os << "Cubic";
            break;
        }

        case InterpType::AREA:
        {
            os << "AREA";
            break;
        }

        default:
        {
            os << "undefined interp type";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert InterpType to string representation.
 *
 * This function converts an InterpType enumerator to its string representation.
 *
 * @param type The InterpType enumerator.
 *
 * @return The string representation of the InterpType.
 */
AURA_INLINE const std::string InterpTypeToString(InterpType type)
{
    std::ostringstream ss;
    ss << type ;
    return ss.str();
}

/**
 * @brief The four boundaries of an iaura, including top, bottom, left, and right.
 */
enum class BorderArea
{
    TOP    = 0,
    BOTTOM,
    LEFT,
    RIGHT,
};

/**
 * @brief Overloaded output stream operator for BorderArea enumeration.
 *
 * This operator allows printing BorderArea enumerators to an output stream.
 *
 * @param os The output stream.
 * @param border_area The BorderArea enumerator to be printed.
 *
 * @return The modified output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, BorderArea border_area)
{
    switch (border_area)
    {
        case BorderArea::TOP:
        {
            os << "Top";
            break;
        }

        case BorderArea::BOTTOM:
        {
            os << "Bottom";
            break;
        }

        case BorderArea::LEFT:
        {
            os << "Left";
            break;
        }

        case BorderArea::RIGHT:
        {
            os << "Right";
            break;
        }

        default:
        {
            os << "undefined border area";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert BorderArea to string representation.
 *
 * This function converts a BorderArea enumerator to its string representation.
 *
 * @param area The BorderArea enumerator.
 *
 * @return The string representation of the BorderArea.
 */
AURA_INLINE const std::string BorderAreaToString(BorderArea area)
{
    std::ostringstream ss;
    ss << area ;
    return ss.str();
}

/**
 * @brief Enumeration of Target Platform on which the function runs.
 */
enum class TargetType
{
    INVALID     = 0,
    NONE,       /*!< The scalar processing units of CPU or DSP */
    NEON,       /*!< The ARM NEON */
    OPENCL,     /*!< The GPU: MTK-Mali GPU or QCOM GPU */
    HVX,        /*!< The DSP: QCOM-Hexagon */
    VDSP,       /*!< The DSP: Cadence-Xtensa-VDSP */
};

/**
 * @brief Overloaded output stream operator for TargetType enumeration.
 *
 * This operator allows printing TargetType enumerators to an output stream.
 *
 * @param os The output stream.
 * @param target_type The TargetType enumerator to be printed.
 *
 * @return The modified output stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, TargetType target_type)
{
    switch (target_type)
    {
        case TargetType::NONE:
        {
            os << "None";
            break;
        }

        case TargetType::NEON:
        {
            os << "Neon";
            break;
        }

        case TargetType::OPENCL:
        {
            os << "Opencl";
            break;
        }

        case TargetType::HVX:
        {
            os << "Hvx";
            break;
        }

        case TargetType::VDSP:
        {
            os << "Vdsp";
            break;
        }

        case TargetType::INVALID:
        {
            os << "Invalid";
            break;
        }

        default:
        {
            os << "undefined target type";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert TargetType to string representation.
 *
 * This function converts a TargetType enumerator to its string representation.
 *
 * @param type The TargetType enumerator.
 *
 * @return The string representation of the TargetType.
 */
AURA_INLINE std::string TargetTypeToString(const TargetType &type)
{
    std::stringstream sstream;
    sstream << type;
    return sstream.str();
}

/**
 * @brief Class representing the target platform for an operator.
 *
 * OpTarget allows specifying a target type and contains target-specific data if applicable.
 */
class AURA_EXPORTS OpTarget
{
public:
    /**
     * @brief Default constructor for OpTarget.
     */
    OpTarget() : m_type(TargetType::INVALID)
    {}

    /**
     * @brief Constructor with specified target type.
     *
     * @param type The target type.
     */
    OpTarget(TargetType type) : m_type(type)
    {
        switch (m_type)
        {
            case TargetType::NONE:
            {
                m_data.none = Data::None();
                break;
            }
            case TargetType::NEON:
            {
                break;
            }
            case TargetType::OPENCL:
            {
                m_data.opencl = Data::OpenCL(DT_FALSE);
                break;
            }
            case TargetType::HVX:
            {
                m_data.hvx = Data::Hvx(DT_FALSE);
                break;
            }
            case TargetType::VDSP:
            {
                m_data.vdsp = Data::Vdsp(DT_FALSE);
                break;
            }
            default:
            {
                break;
            }
        }
    }

    /**
     * @brief Get the default OpTarget based on the build configuration.
     *
     * @return The default OpTarget. (return HVX for Hexagon, VDSP for Xtensa, otherwise return NONE)
     */
    static OpTarget Default()
    {
#if defined(AURA_BUILD_HEXAGON)
        return OpTarget(TargetType::HVX);
#elif defined(AURA_BUILD_XTENSA)
        return OpTarget(TargetType::VDSP);
#else
        return OpTarget(TargetType::NONE);
#endif
    }

    /**
     * @brief Get an OpTarget with target type set to NONE.
     *
     * @return The OpTarget with target type NONE.
     */
#if defined(AURA_BUILD_HEXAGON)
    static OpTarget None(DT_BOOL enable_mt = DT_TRUE)
#else
    static OpTarget None(DT_BOOL enable_mt = DT_FALSE)
#endif
    {
#if defined(AURA_BUILD_XPLORER)
        enable_mt = DT_FALSE;
#endif
        OpTarget target(TargetType::NONE);
        target.m_data.none = Data::None(enable_mt);
        return target;
    }

    /**
     * @brief Get an OpTarget with target type set to NEON.
     *
     * @return The OpTarget with target type NEON.
     */
    static OpTarget Neon()
    {
        OpTarget target(TargetType::NEON);
        return target;
    }

    /**
     * @brief Get an OpTarget with target type set to OPENCL.
     *
     * @param profiling Flag indicating whether profiling is enabled.
     *
     * @return The OpTarget with target type OPENCL.
     */
    static OpTarget Opencl(DT_BOOL profiling = DT_FALSE)
    {
        OpTarget target(TargetType::OPENCL);
        target.m_data.opencl = Data::OpenCL(profiling);
        return target;
    }

    /**
     * @brief Get an OpTarget with target type set to HVX.
     *
     * @param profiling Flag indicating whether profiling is enabled.
     *
     * @return The OpTarget with target type HVX.
     */
    static OpTarget Hvx(DT_BOOL profiling = DT_FALSE)
    {
        OpTarget target(TargetType::HVX);
        target.m_data.hvx = Data::Hvx(profiling);
        return target;
    }

    /**
     * @brief Get an OpTarget with target type set to VDSP.
     *
     * @param profiling Flag indicating whether profiling is enabled.
     *
     * @return The OpTarget with target type VDSP.
     */
    static OpTarget Vdsp(DT_BOOL profiling = DT_FALSE)
    {
        OpTarget target(TargetType::VDSP);
        target.m_data.vdsp = Data::Vdsp(profiling);
        return target;
    }

    /**
     * @brief Copy constructor.
     *
     * @param target The OpTarget to copy.
     */
    OpTarget(const OpTarget &target)
    {
        m_type = target.m_type;

        switch (m_type)
        {
            case TargetType::NONE:
            {
                m_data.none = target.m_data.none;
                break;
            }
            case TargetType::OPENCL:
            {
                m_data.opencl = target.m_data.opencl;
                break;
            }
            case TargetType::HVX:
            {
                m_data.hvx = target.m_data.hvx;
                break;
            }
            case TargetType::VDSP:
            {
                m_data.vdsp = target.m_data.vdsp;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    /**
     * @brief Copy assignment operator.
     *
     * @param target The OpTarget to copy.
     *
     * @return Reference to the modified OpTarget.
     */
    OpTarget& operator=(const OpTarget &target)
    {
        m_type = target.m_type;

        switch (m_type)
        {
            case TargetType::NONE:
            {
                m_data.none = target.m_data.none;
                break;
            }
            case TargetType::OPENCL:
            {
                m_data.opencl = target.m_data.opencl;
                break;
            }
            case TargetType::HVX:
            {
                m_data.hvx = target.m_data.hvx;
                break;
            }
            case TargetType::VDSP:
            {
                m_data.vdsp = target.m_data.vdsp;
                break;
            }
            default:
            {
                break;
            }
        }

        return *this;
    }

    /**
     * @brief Equality operator.
     *
     * @param target The OpTarget to compare.
     *
     * @return True if OpTargets are equal, false otherwise.
     */
    DT_BOOL operator==(const OpTarget &target)
    {
        if (m_type != target.m_type)
        {
            return DT_FALSE;
        }

        switch (m_type)
        {
            case TargetType::NONE:
            {
                if (m_data.none.enable_mt != target.m_data.none.enable_mt)
                {
                    return DT_FALSE;
                }
                break;
            }

            case TargetType::OPENCL:
            {
                if (m_data.opencl.profiling != target.m_data.opencl.profiling)
                {
                    return DT_FALSE;
                }
                break;
            }

            case TargetType::HVX:
            {
                if (m_data.hvx.profiling != target.m_data.hvx.profiling)
                {
                    return DT_FALSE;
                }
                break;
            }

            case TargetType::VDSP:
            {
                if (m_data.vdsp.profiling != target.m_data.vdsp.profiling)
                {
                    return DT_FALSE;
                }
                break;
            }

            default:
            {
                break;
            }
        }

        return DT_TRUE;
    }

    /**
     * @brief Inequality operator.
     *
     * @param target The OpTarget to compare.
     *
     * @return True if OpTargets are not equal, false otherwise.
     */
    DT_BOOL operator!=(const OpTarget &target)
    {
        return !(*this == target);
    }

    /**
     * @brief Destructor.
     */
    ~OpTarget() = default;

    /**
     * @brief Overloaded output stream operator for OpTarget.
     *
     * @param os The output stream.
     * @param target The OpTarget to output.
     *
     * @return The modified output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const OpTarget &target)
    {
        switch (target.m_type)
        {
            case TargetType::NONE:
            {
                os << "None with multi-thread : " << (target.m_data.none.enable_mt ? "ON" : "OFF");
                break;
            }

            case TargetType::NEON:
            {
                os << "Neon";
                break;
            }

            case TargetType::OPENCL:
            {
                os << "Opencl(profiling:" << target.m_data.opencl.profiling << ")";
                break;
            }

            case TargetType::HVX:
            {
                os << "Hvx(profiling:" << target.m_data.hvx.profiling << ")";
                break;
            }

            case TargetType::VDSP:
            {
                os << "Vdsp(profiling:" << target.m_data.vdsp.profiling << ")";
                break;
            }

            case TargetType::INVALID:
            {
                os << "Invalid";
                break;
            }

            default:
            {
                break;
            }
        }

        return os;
    }

    /**
     * @brief Get the string representation of OpTarget.
     *
     * @return The string representation.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    TargetType m_type;  /*!< The type of the target platform for the operator. */

    /**
     * @brief Union containing target-specific data.
     */
    union Data
    {
        /**
         * @brief Struct for NONE target.
         */
        struct None
        {
            DT_BOOL enable_mt;

#if defined(AURA_BUILD_HEXAGON)
            None(DT_BOOL enable_mt = DT_TRUE) : enable_mt(enable_mt)
            {}
#else
            None(DT_BOOL enable_mt = DT_FALSE) : enable_mt(enable_mt)
            {}
#endif // AURA_BUILD_HEXAGON
        } none;

        /**
         * @brief Struct for OPENCL target.
         */
        struct OpenCL
        {
            DT_BOOL profiling;      /*!< Flag indicating whether profiling is enabled for OpenCL. */
            OpenCL(DT_S32 profiling) : profiling(profiling)
            {}
        } opencl;

        /**
         * @brief Struct for HVX target.
         */
        struct Hvx
        {
            DT_BOOL profiling;      /*!< Flag indicating whether profiling is enabled for HVX. */
            Hvx(DT_S32 profiling) : profiling(profiling)
            {}
        } hvx;

        /**
         * @brief Struct for VDSP target.
         */
        struct Vdsp
        {
            DT_BOOL profiling;      /*!< Flag indicating whether profiling is enabled for VDSP. */
            Vdsp(DT_S32 profiling) : profiling(profiling)
            {}
        } vdsp;

        Data()
        {}
    } m_data;

};

/**
 * @brief Create a buffer with border values.
 *
 * This function creates a buffer with border values for use in iaura processing operators.
 *
 * @tparam Tp The type of buffer elements.
 *
 * @param ctx The pointer to the Context object.
 * @param width The width of the buffer.
 * @param channel The number of channels.
 * @param border_value The vector containing the border values for each channel.
 *
 * @return A pointer to the created buffer. It is the responsibility of the caller to free the allocated memory.
 */
template <typename Tp>
AURA_INLINE Tp* CreateBorderBuffer(Context *ctx, DT_S32 width, DT_S32 channel, const std::vector<Tp> &border_value)
{
    /// 创建buffer. 创建宽度长度的乘以通道数的buffer数据
    Tp *buffer = (Tp *)AURA_ALLOC(ctx, width * channel * sizeof(Tp));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC failed");
        return DT_NULL;
    }

    /// 将这个数据填充满border_value的数据
    Tp *buffer_row = buffer;
    for (DT_S32 x = 0; x < width; x++)
    {
        for (DT_S32 c = 0; c < channel; c++)
        {
            *buffer_row++ = border_value[c];
        }
    }

    return buffer;
}

/**
 * @brief Get border index for the specified border type.
 *
 * The supported border types are REPLICATE, REFLECT_101, CONSTANT. And each type has a corresponding specialized implementation.
 *
 * @tparam BORDER_TYPE The type of border replication.
 *
 * @param index The original index.
 * @param len The length of the array.
 *
 * @return The real index of the array under a specific border type.
 */
template <BorderType BORDER_TYPE, typename std::enable_if<(BorderType::REPLICATE == BORDER_TYPE)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_S32 GetBorderIdx(DT_S32 index, DT_S32 len)
{
    return Max(static_cast<DT_S32>(0), Min(index, len - 1));
}

template <BorderType BORDER_TYPE, typename std::enable_if<(BorderType::REFLECT_101 == BORDER_TYPE)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_S32 GetBorderIdx(DT_S32 index, DT_S32 len)
{
    index           = Abs(index);
    DT_S32 n        = index / (len - 1);
    DT_S32 leftover = index - n * (len - 1);
    return (n & 1) ? (len - 1 - leftover) : leftover;
}

template <BorderType BORDER_TYPE, typename std::enable_if<(BorderType::CONSTANT == BORDER_TYPE)>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_S32 GetBorderIdx(DT_S32 index, DT_S32 len)
{
    return index >= 0 ? (index < len ? index : -1) : -1;
}

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
/**
 * @brief Check if the width is compatible with Hexagon vector length.
 *
 * This function checks if the width of the input array is compatible with the Hexagon vector length.
 *
 * @param array The input array.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_INLINE Status CheckHvxWidth(const Array &array)
{
    // 判断数据的宽度
    DT_S32 width = array.GetSizes().m_width;
    // 判断元素的个数
    DT_S32 elem_counts = AURA_HVLEN / ElemTypeSize(array.GetElemType());
    // 数据的宽度小于元素的个数，也是不允许使用HVX的 TODO 为什么啊？
    if (width < elem_counts)
    {
        return Status::ERROR;
    }

    return Status::OK;
}
#endif

/**
 * @brief Perform saturation conversion on different types of data
 *
 * If the input is a fixed-point number, saturate the fixed-point number to the floating-point number; otherwise saturate to the target type
 *
 * @tparam St The source data type for the value to be cast
 * @tparam Dt The destination data type for the result
 * @tparam Q Fixed point bit width (non fixed point number is 0)
 *
 * @param val The input value of type St to be cast and shifted
 *
 * @return The result of saturation conversion
 */
template <typename St, typename Dt, DT_U32 Q, typename std::enable_if<std::is_same<St, DT_F32>::value && std::is_same<Dt, St>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE Dt ShiftSatCast(const St val)
{
    return val;
}

template <typename St, typename Dt, DT_U32 Q, typename std::enable_if<std::is_same<St, DT_F32>::value && !std::is_same<Dt, St>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE Dt ShiftSatCast(const St val)
{
    return SaturateCast<Dt>(val);
}

template <typename St, typename Dt, DT_U32 Q, typename std::enable_if<!std::is_same<St, DT_F32>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE Dt ShiftSatCast(const St val)
{
    return SaturateCast<Dt>((val + ((1 << Q) >> 1)) >> Q);
}

/**
 * @brief Base implementation class.
 *
 * This class provides a base implementation for operators.
 */
class AURA_EXPORTS OpImpl
{
public:
    /**
     * @brief Constructor.
     *
     * @param ctx The pointer to the Context object.
     * @param name The name of the operator.
     * @param target The target platform for the operator.
     */
    OpImpl(Context *ctx, const std::string &name, const OpTarget &target) : m_ctx(ctx), m_name(name), m_target(target)
    {}

    /**
     * @brief Destructor.
    */
    virtual ~OpImpl()
    {
        m_ctx = DT_NULL;
    }

    /**
     * @brief Initialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Initialize()
    {
        return Status::OK;
    }

    /**
     * @brief Run operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Run() = 0;

    /**
     * @brief Deinitialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status DeInitialize()
    {
        return Status::OK;
    }

    /**
     * @brief Get a string representation of the operator implementation.
     *
     * @return The string representation.
     */
    virtual std::string ToString() const
    {
        return std::string();
    }

    /**
     * @brief Dump information about the operator implementation.
     *
     * @param prefix The prefix in the dump.
     */
    virtual DT_VOID Dump(const std::string &prefix) const
    {
        AURA_UNUSED(prefix);
    }

    /**
     * @brief Get the target platform for the operator implementation.
     *
     * @return The target platform.
     */
    OpTarget GetOpTarget() const
    {
        return m_target;
    }

    /**
     * @brief Get the name of the operator implementation.
     *
     * @return The name of the operator implementation.
     */
    std::string GetName() const
    {
        return m_name;
    }

    /**
     * @brief Get the input arrays.
     *
     * @return A vector of input arrays.
     */
    virtual std::vector<const Array*> GetInputArrays() const
    {
        return {};
    }

    /**
     * @brief Get the output arrays.
     *
     * @return A vector of output arrays.
     */
    virtual std::vector<const Array*> GetOutputArrays() const
    {
        return {};
    }

protected:
    Context        *m_ctx;      /*!< Pointer to the Context object. */
    std::string    m_name;      /*!< Name of the operator implementation. */
    OpTarget       m_target;    /*!< Target platform for the operator implementation. */
};

/**
 * @brief Base class for operator.
 *
 * This class provides a common interface and functionality for operator.
 */
class AURA_EXPORTS Op
{
public:
    /**
     * @brief Constructor.
     *
     * @param ctx The pointer to the Context object.
     * @param target The target platform for the operator.
     */
    Op(Context *ctx, const OpTarget &target = OpTarget::Default()) : m_ctx(ctx), m_target(target), m_ready(DT_FALSE)
    {}

    virtual ~Op()
    {
        m_ctx = DT_NULL;
    }

    /**
     * @brief Initialize operator.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize()
    {
        Status ret = Status::ERROR;

        if (m_ctx && m_impl)
        {
            ret = m_impl->Initialize();
            if (Status::OK == ret)
            {
                m_ready = DT_TRUE;
            }
        }
        return ret;
    }

    /**
     * @brief Run operator.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Run()
    {
        Status ret = Status::ERROR;

        if (m_ctx && m_impl)
        {
            if (DT_TRUE == m_ready)
            {
                ret = m_impl->Run();
            }
        }
        return ret;
    }

    /**
     * @brief Deinitialize operator.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status DeInitialize()
    {
        Status ret = Status::ERROR;

        if (m_ctx && m_impl)
        {
            ret = m_impl->DeInitialize();
            if (Status::OK == ret)
            {
                m_ready = DT_FALSE;
            }
        }
        return ret;
    }

    /**
     * @brief Get a string representation of operator.
     *
     * @return The string representation.
     */
    std::string ToString() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->ToString();
        }
        else
        {
            return std::string();
        }
    }

    /**
     * @brief Dump information about the operator.
     *
     * @param prefix The prefix in the dump.
     */
    DT_VOID Dump(const std::string &prefix) const
    {
        if (m_ctx && m_impl)
        {
            m_impl->Dump(prefix);
        }
    }

    /**
     * @brief Get the target platform for the operator.
     *
     * @return The target platform.
     */
    OpTarget GetOpTarget() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetOpTarget();
        }
        else
        {
            return OpTarget();
        }
    }

    /**
     * @brief Get the name of the operator.
     *
     * @return The name of the operator.
     */
    std::string GetName() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetName();
        }
        else
        {
            return std::string();
        }
    }

    /**
     * @brief Get the input arrays for the operator.
     *
     * @return A vector of input arrays.
     */
    std::vector<const Array*> GetInputArrays() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetInputArrays();
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief Get the output arrays for the operator.
     *
     * @return A vector of output arrays.
     */
    std::vector<const Array*> GetOutputArrays() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetOutputArrays();
        }
        else
        {
            return {};
        }
    }

protected:
    Context *m_ctx;                     /*!< Pointer to the Context object */
    std::shared_ptr<OpImpl> m_impl;     /*!< Shared pointer to the implementation class. */
    OpTarget m_target;                  /*!< Target platform for the operator. */
    DT_BOOL m_ready;                    /*!< Flag indicating whether the operator is ready for execution. */
};

/**
 * @brief Execute an operator.
 *
 * This function simplifies the process of setting arguments, initializing, running, and deinitializing an operator.
 *
 * @tparam Tp The type of the operator.
 * @tparam ArgsType The types of the arguments.
 *
 * @param ctx The pointer to the Context object.
 * @param op The operator to be executed.
 * @param args The arguments for the operator.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename ...ArgsType>
Status OpCall(Context *ctx, Tp &op, ArgsType &&...args)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }
    Status ret = Status::ERROR;
    /// 步骤一： 这个直接调用回，对应op的SetArgs
    if ((ret = op.SetArgs(std::forward<ArgsType>(args)...)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SetArgs failed");
        return ret;
    }

    /// 步骤二：调用对应OP的具体实现类的Initialize
    if ((ret = op.Initialize()) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Initialize failed");
        goto EXIT;
    }
    // 步骤三： 调用对应OP的具体实现类的Run
    if ((ret = op.Run()) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Run failed");
    }

EXIT:
    op.DeInitialize();

    return ret;
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_CORE_COMM_HPP__
