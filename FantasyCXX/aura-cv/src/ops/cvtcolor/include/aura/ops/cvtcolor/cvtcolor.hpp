#ifndef AURA_OPS_CVTCOLOR_CVTCOLOR_HPP__
#define AURA_OPS_CVTCOLOR_CVTCOLOR_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

#include <vector>

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_cvtcolor Color Space Conversions
 * @}
 */

namespace aura
{
/**
 * @addtogroup imgproc_cvtcolor
 * @{
 */

/**
 * @brief Enumerated type for color conversion.
 *
 * This enumeration defines color space conversion types for use in the CvtColor operation.
 */
enum class CvtColorType
{
    INVALID     = 0,

    BGR2BGRA    = 100,
    RGB2RGBA    = BGR2BGRA,
    BGRA2BGR,
    RGBA2RGB    = BGRA2BGR,
    BGR2RGB,
    RGB2BGR     = BGR2RGB,
    BGR2GRAY,
    RGB2GRAY,
    GRAY2BGR,
    GRAY2RGB    = GRAY2BGR,
    GRAY2BGRA,
    GRAY2RGBA   = GRAY2BGRA,
    BGRA2GRAY,
    RGBA2GRAY,

    RGB2YUV_NV12    = 200,
    RGB2YUV_NV21,
    RGB2YUV_YU12,
    RGB2YUV_YV12,
    RGB2YUV_Y444,
    RGB2YUV_NV12_601,
    RGB2YUV_NV21_601,
    RGB2YUV_YU12_601,
    RGB2YUV_YV12_601,
    RGB2YUV_Y444_601,
    RGB2YUV_NV12_P010,
    RGB2YUV_NV21_P010,

    YUV2RGB_NV12    = 300,
    YUV2RGB_NV21,
    YUV2RGB_YU12,
    YUV2RGB_YV12,
    YUV2RGB_Y422,
    YUV2RGB_UYVY    = YUV2RGB_Y422,
    YUV2RGB_YUYV,
    YUV2RGB_YUY2    = YUV2RGB_YUYV,
    YUV2RGB_YVYU,
    YUV2RGB_Y444,
    YUV2RGB_NV12_601,
    YUV2RGB_NV21_601,
    YUV2RGB_YU12_601,
    YUV2RGB_YV12_601,
    YUV2RGB_Y422_601,
    YUV2RGB_UYVY_601 = YUV2RGB_Y422_601,
    YUV2RGB_YUYV_601,
    YUV2RGB_YUY2_601 = YUV2RGB_YUYV_601,
    YUV2RGB_YVYU_601,
    YUV2RGB_Y444_601,
    YUV2RGB_NV12_601_P010,
    YUV2RGB_NV21_601_P010,

    BAYERBG2BGR    = 400,
    BAYERGB2BGR,
    BAYERRG2BGR,
    BAYERGR2BGR,
};

/**
 * @brief Overloaded output stream operator for CvtColorType enumeration.
 *
 * This operator allows printing CvtColorType enumerators to an output stream.
 *
 * @param os The output stream.
 * @param type The CvtColorType enumerator to be printed.
 *
 * @return Color convention out stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, const CvtColorType &type)
{
    switch (type)
    {
        // RGB <-> BGRA
        case CvtColorType::BGR2BGRA:
        {
            os << "BGR2BGRA";
            break;
        }
        case CvtColorType::BGRA2BGR:
        {
            os << "BGRA2BGR";
            break;
        }
        case CvtColorType::BGR2RGB:
        {
            os << "BGR2RGB";
            break;
        }
        case CvtColorType::BGR2GRAY:
        {
            os << "BGR2GRAY";
            break;
        }
        case CvtColorType::RGB2GRAY:
        {
            os << "RGB2GRAY";
            break;
        }
        case CvtColorType::GRAY2BGR:
        {
            os << "GRAY2BGR";
            break;
        }
        case CvtColorType::GRAY2BGRA:
        {
            os << "GRAY2BGRA";
            break;
        }
        case CvtColorType::BGRA2GRAY:
        {
            os << "BGRA2GRAY";
            break;
        }
        case CvtColorType::RGBA2GRAY:
        {
            os << "RGBA2GRAY";
            break;
        }
        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        {
            os << "RGB2YUV_NV12";
            break;
        }
        case CvtColorType::RGB2YUV_NV21:
        {
            os << "RGB2YUV_NV21";
            break;
        }
        case CvtColorType::RGB2YUV_YU12:
        {
            os << "RGB2YUV_YU12";
            break;
        }
        case CvtColorType::RGB2YUV_YV12:
        {
            os << "RGB2YUV_YV12";
            break;
        }
        case CvtColorType::RGB2YUV_Y444:
        {
            os << "RGB2YUV_Y444";
            break;
        }
        case CvtColorType::RGB2YUV_NV12_601:
        {
            os << "RGB2YUV_NV12_601";
            break;
        }
        case CvtColorType::RGB2YUV_NV21_601:
        {
            os << "RGB2YUV_NV21_601";
            break;
        }
        case CvtColorType::RGB2YUV_YU12_601:
        {
            os << "RGB2YUV_YU12_601";
            break;
        }
        case CvtColorType::RGB2YUV_YV12_601:
        {
            os << "RGB2YUV_YV12_601";
            break;
        }
        case CvtColorType::RGB2YUV_Y444_601:
        {
            os << "RGB2YUV_Y444_601";
            break;
        }
        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        {
            os << "YUV2RGB_NV12";
            break;
        }
        case CvtColorType::YUV2RGB_NV21:
        {
            os << "YUV2RGB_NV21";
            break;
        }
        case CvtColorType::YUV2RGB_YU12:
        {
            os << "YUV2RGB_YU12";
            break;
        }
        case CvtColorType::YUV2RGB_YV12:
        {
            os << "YUV2RGB_YV12";
            break;
        }
        case CvtColorType::YUV2RGB_Y422:
        {
            os << "YUV2RGB_Y422";
            break;
        }
        case CvtColorType::YUV2RGB_YUYV:
        {
            os << "YUV2RGB_YUYV";
            break;
        }
        case CvtColorType::YUV2RGB_YVYU:
        {
            os << "YUV2RGB_YVYU";
            break;
        }
        case CvtColorType::YUV2RGB_Y444:
        {
            os << "YUV2RGB_Y444";
            break;
        }
        case CvtColorType::YUV2RGB_NV12_601:
        {
            os << "YUV2RGB_NV12_601";
            break;
        }
        case CvtColorType::YUV2RGB_NV21_601:
        {
            os << "YUV2RGB_NV21_601";
            break;
        }
        case CvtColorType::YUV2RGB_YU12_601:
        {
            os << "YUV2RGB_YU12_601";
            break;
        }
        case CvtColorType::YUV2RGB_YV12_601:
        {
            os << "YUV2RGB_YV12_601";
            break;
        }
        case CvtColorType::YUV2RGB_Y422_601:
        {
            os << "YUV2RGB_Y422_601";
            break;
        }
        case CvtColorType::YUV2RGB_YUYV_601:
        {
            os << "YUV2RGB_YUYV_601";
            break;
        }
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            os << "YUV2RGB_YVYU_601";
            break;
        }
        case CvtColorType::YUV2RGB_Y444_601:
        {
            os << "YUV2RGB_Y444_601";
            break;
        }
        case CvtColorType::RGB2YUV_NV12_P010:
        {
            os << "RGB2YUV_NV12_P010";
            break;
        }
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            os << "RGB2YUV_NV21_P010";
            break;
        }
        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        {
            os << "BAYERBG2BGR";
            break;
        }
        case CvtColorType::BAYERGB2BGR:
        {
            os << "BAYERGB2BGR";
            break;
        }
        case CvtColorType::BAYERRG2BGR:
        {
            os << "BAYERRG2BGR";
            break;
        }
        case CvtColorType::BAYERGR2BGR:
        {
            os << "BAYERGR2BGR";
            break;
        }

        default:
        {
            os << "Invalid";
            break;
        }
    }
    return os;
}

/**
 * @brief Convert CvtColorType to string representation.
 *
 * This function converts a CvtColorType enumerator to its string representation.
 *
 * @param type The CvtColorType enumerator.
 *
 * @return The string representation of the CvtColorType.
 */
AURA_INLINE std::string CvtColorTypeToString(CvtColorType type)
{
    std::ostringstream ss;
    ss << type ;
    return ss.str();
}

/**
 * @brief The CvtColor interface class.
 *
 * The use of this class for color space conversion is not recommended. It is recommended to use the `ICvtColor` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `ICvtColor` function is as follows:
 *
 * @code
 * CvtColor cvtcolor(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, cvtcolor, &src, &dst, type);
 * @endcode
 */
class AURA_EXPORTS CvtColor : public Op
{
public:
    /**
     * @brief Constructor for the CvtColor.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    CvtColor(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the CvtColor operation.
     *
     * For more details, please refer to @ref cvtcolor_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type);

    /**
     * @brief Generate cvtcolor opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param cvtcolor_type The type of the color space conversion.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, CvtColorType cvtcolor_type);
};

/**
 * @brief Performs color space conversion on a batch of input iauras.
 *
 * @anchor cvtcolor_details
 *
 * @param ctx The pointer to the Context object.
 * @param src Vector of input Mat objects representing the source iauras.
 * @param dst Vector of output Mat objects representing the destination iauras.
 * @param type The type of the color space conversion. And see the below for the supported data types.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The suppored data types and platforms
 * Platforms            | Data type |conversion type
 * ---------------------|-----------|-------------------------------------------------------------------------------------------
 * NONE/NEON/HVX        | U8        | BGR2BGRA, BGRA2BGR, BGR2RGB, GRAY2BGR(A)
 * NONE/NEON/HVX/OpenCL | U8        | BGR(A)2GRAY, RGB(A)2GRAY
 * NONE/NEON/HVX/OpenCL | U8        | YUV2RGB_NV12, YUV2RGB_NV21, YUV2RGB_YU12, YUV2RGB_YV12, YUV2RGB_Y422, YUV2RGB_YUYV, YUV2RGB_YVYU, YUV2RGB_Y444
 * NONE/NEON/HVX/OpenCL | U8        | YUV2RGB_NV12_601, YUV2RGB_NV21_601, YUV2RGB_YU12_601, YUV2RGB_YV12_601, YUV2RGB_Y422_601, YUV2RGB_YUYV_601, YUV2RGB_YVYU_601, YUV2RGB_Y444_601
 * NONE/NEON/HVX/OpenCL | U8        | RGB2YUV_NV12, RGB2YUV_NV21, RGB2YUV_YU12, RGB2YUV_YV12, RGB2YUV_Y444
 * NONE/NEON/HVX/OpenCL | U8        | RGB2YUV_NV12_601, RGB2YUV_NV21_601, RGB2YUV_YU12_601, RGB2YUV_YV12_601, RGB2YUV_Y444_601
 * NONE/NEON/HVX/OpenCL | U16       | RGB2YUV_NV12_P010, RGB2YUV_NV21_P010
 * NONE/NEON/HVX/OpenCL | U8/U16    | BAYERBG2BGR, BAYERGB2BGR, BAYERRG2BGR, BAYERGR2BGR
 */
AURA_EXPORTS Status ICvtColor(Context *ctx, const std::vector<Mat> &src, std::vector<Mat> &dst,
                              CvtColorType type, const OpTarget &target = OpTarget::Default());
/**
 * @}
 */

} // namespace aura
#endif // AURA_OPS_CVTCOLOR_CVTCOLOR_HPP__
