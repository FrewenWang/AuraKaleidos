#ifndef AURA_CV_OPS_CORE_COMMON_HPP__
#define AURA_CV_OPS_CORE_COMMON_HPP__

#include "aura/utils/core.h"
#include "aura/cv/core.h"

#include <string>
#include <vector>


using namespace aura::utils;
namespace aura::cv
{
/**
 * @brief Enumeration of interpolation types.
 */
enum class InterpType
{
    /** 最近邻插值 **/
    NEAREST = 0, /*!< nearest neighbor interpolation */
    /** 双线性插值 **/
    LINEAR, /*!< bilinear interpolation */
    /** 三次样条插值  **/
    CUBIC, /*!< bicubic interpolation */
    /** 像素区域关系重新采样。图像抽取的首选方法，可获得无摩尔纹的结果。当图像缩放时，类似INTER_NEAREST方法。**/
    AREA,
    /*!< resampling using pixel area relation. It may be a preferred method for image decimation, as it gives
       moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method. */
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
AURA_INLINE std::ostream &operator<<(std::ostream &os, InterpType interp_type)
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
    ss << type;
    return ss.str();
}

/**
 * @brief The four boundaries of an image, including top, bottom, left, and right.
 */
enum class BorderArea
{
    TOP = 0,
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
AURA_INLINE std::ostream &operator<<(std::ostream &os, BorderArea border_area)
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
    ss << area;
    return ss.str();
}

/**
 * @brief Enumeration of Target Platform on which the function runs.
 */
enum class TargetType
{
    INVALID = 0,
    NONE, /*!< The scalar processing units of CPU or DSP */
    NEON, /*!< The ARM NEON */
    OPENCL, /*!< The GPU: MTK-Mali GPU or QCOM GPU */
    HVX, /*!< The DSP: QCOM-Hexagon */
    VDSP, /*!< The DSP: Cadence-Xtensa-VDSP */
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
AURA_INLINE std::ostream &operator<<(std::ostream &os, TargetType target_type)
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
    {
    }

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
                m_data.opencl = Data::OpenCL(AURA_FALSE);
                break;
            }
            case TargetType::HVX:
            {
                m_data.hvx = Data::Hvx(AURA_FALSE);
                break;
            }
            case TargetType::VDSP:
            {
                m_data.vdsp = Data::Vdsp(AURA_FALSE);
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
    static OpTarget None(AURA_BOOL enable_mt = AURA_TRUE)
#else
    static OpTarget None(AURA_BOOL enable_mt = AURA_FALSE)
#endif
    {
#if defined(AURA_BUILD_XPLORER)
        enable_mt = AURA_FALSE;
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
    static OpTarget Opencl(AURA_BOOL profiling = AURA_FALSE)
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
    static OpTarget Hvx(AURA_BOOL profiling = AURA_FALSE)
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
    static OpTarget Vdsp(AURA_BOOL profiling = AURA_FALSE)
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
    OpTarget &operator=(const OpTarget &target)
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
    AURA_BOOL operator==(const OpTarget &target)
    {
        if (m_type != target.m_type)
        {
            return AURA_FALSE;
        }

        switch (m_type)
        {
            case TargetType::NONE:
            {
                if (m_data.none.enable_mt != target.m_data.none.enable_mt)
                {
                    return AURA_FALSE;
                }
                break;
            }

            case TargetType::OPENCL:
            {
                if (m_data.opencl.profiling != target.m_data.opencl.profiling)
                {
                    return AURA_FALSE;
                }
                break;
            }

            case TargetType::HVX:
            {
                if (m_data.hvx.profiling != target.m_data.hvx.profiling)
                {
                    return AURA_FALSE;
                }
                break;
            }

            case TargetType::VDSP:
            {
                if (m_data.vdsp.profiling != target.m_data.vdsp.profiling)
                {
                    return AURA_FALSE;
                }
                break;
            }

            default:
            {
                break;
            }
        }

        return AURA_TRUE;
    }

    /**
     * @brief Inequality operator.
     *
     * @param target The OpTarget to compare.
     *
     * @return True if OpTargets are not equal, false otherwise.
     */
    AURA_BOOL operator!=(const OpTarget &target)
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
    AURA_EXPORTS friend std::ostream &operator<<(std::ostream &os, const OpTarget &target)
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

    TargetType m_type; /*!< The type of the target platform for the operator. */

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
            AURA_BOOL enable_mt;

#if defined(AURA_BUILD_HEXAGON)
            None(AURA_BOOL enable_mt = AURA_TRUE) : enable_mt(enable_mt)
            {
            }
#else
            None(AURA_BOOL enable_mt = AURA_FALSE) : enable_mt(enable_mt)
            {
            }
#endif // AURA_BUILD_HEXAGON
        } none;

        /**
         * @brief Struct for OPENCL target.
         */
        struct OpenCL
        {
            AURA_BOOL profiling; /*!< Flag indicating whether profiling is enabled for OpenCL. */
            OpenCL(AURA_S32 profiling) : profiling(profiling)
            {
            }
        } opencl;

        /**
         * @brief Struct for HVX target.
         */
        struct Hvx
        {
            AURA_BOOL profiling; /*!< Flag indicating whether profiling is enabled for HVX. */
            Hvx(AURA_S32 profiling) : profiling(profiling)
            {
            }
        } hvx;

        /**
         * @brief Struct for VDSP target.
         */
        struct Vdsp
        {
            AURA_BOOL profiling; /*!< Flag indicating whether profiling is enabled for VDSP. */
            Vdsp(AURA_S32 profiling) : profiling(profiling)
            {
            }
        } vdsp;

        Data()
        {
        }
    } m_data;
};

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
    {
    }

    /**
     * @brief Destructor.
     */
    virtual ~OpImpl()
    {
        m_ctx = AURA_NULL;
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
    virtual AURA_VOID Dump(const std::string &prefix) const
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
    virtual std::vector<const Array *> GetInputArrays() const
    {
        return {};
    }

    /**
     * @brief Get the output arrays.
     *
     * @return A vector of output arrays.
     */
    virtual std::vector<const Array *> GetOutputArrays() const
    {
        return {};
    }

protected:
    Context *m_ctx; /*!< Pointer to the Context object. */
    std::string m_name; /*!< Name of the operator implementation. */
    OpTarget m_target; /*!< Target platform for the operator implementation. */
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
    Op(Context *ctx, const OpTarget &target = OpTarget::Default()) : m_ctx(ctx), m_target(target), m_ready(AURA_FALSE)
    {
    }

    virtual ~Op()
    {
        m_ctx = AURA_NULL;
    }

    /**
     * @brief Initialize operator.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize()
    {
        Status ret = Status::ERR_UNKNOWN;

        if (m_ctx && m_impl)
        {
            ret = m_impl->Initialize();
            if (Status::OK == ret)
            {
                m_ready = AURA_TRUE;
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
        Status ret = Status::ERR_UNKNOWN;

        if (m_ctx && m_impl)
        {
            if (AURA_TRUE == m_ready)
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
        Status ret = Status::ERR_UNKNOWN;

        if (m_ctx && m_impl)
        {
            ret = m_impl->DeInitialize();
            if (Status::OK == ret)
            {
                m_ready = AURA_FALSE;
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
        } else
        {
            return std::string();
        }
    }

    /**
     * @brief Dump information about the operator.
     *
     * @param prefix The prefix in the dump.
     */
    AURA_VOID Dump(const std::string &prefix) const
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
        } else
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
        } else
        {
            return std::string();
        }
    }

    /**
     * @brief Get the input arrays for the operator.
     *
     * @return A vector of input arrays.
     */
    std::vector<const Array *> GetInputArrays() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetInputArrays();
        } else
        {
            return {};
        }
    }

    /**
     * @brief Get the output arrays for the operator.
     *
     * @return A vector of output arrays.
     */
    std::vector<const Array *> GetOutputArrays() const
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetOutputArrays();
        } else
        {
            return {};
        }
    }

protected:
    Context *m_ctx; /*!< Pointer to the Context object */
    std::shared_ptr<OpImpl> m_impl; /*!< Shared pointer to the implementation class. */
    OpTarget m_target; /*!< Target platform for the operator. */
    AURA_BOOL m_ready; /*!< Flag indicating whether the operator is ready for execution. */
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
template<typename Tp, typename... ArgsType>
Status OpCall(Context *ctx, Tp &op, ArgsType &&...args)
{
}
} // namespace aura::cv

#endif // AURA_CV_OPS_CORE_COMMON_HPP__
