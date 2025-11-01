#ifndef AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_HPP__
#define AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/core/xtensa.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_cvtcolor Color Space Conversions
 * @}
 */

namespace aura
{
namespace xtensa
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

    BAYERBG2BGR    = 400,
    BAYERGB2BGR,
    BAYERRG2BGR,
    BAYERGR2BGR,
};

/**
 * @brief The CvtColorVdsp interface class.
 *
 * This class provides functionality for performing color space conversion on iauras.
 */
class CvtColorVdsp : public VdspOp
{
public:
    /**
     * @brief Constructor for the CvtColorVdsp class.
     *
     * @param tm The pointer to the TileManager object
     * @param mode  The execute mode.
     */
    CvtColorVdsp(TileManager tm, ExecuteMode mode = ExecuteMode::FRAME);

    /**
     * @brief Set the arguments for the CvtColorVdsp operation.
     *
     * @param src The vector of source Mat.
     * @param dst  The vector of destination Mat.
     * @param type  The color conversion type.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status SetArgs(const vector<const Mat*> &src, const vector<Mat*> &dst, CvtColorType type);

    /**
     * @brief Set the arguments for the BoxFilterVdsp filter operation.
     *
     * @param src The vector of source TileWrapper.
     * @param dst  The vector of destination TileWrapper.
     * @param type  The color conversion type.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status SetArgs(const vector<TileWrapper> &src, vector<TileWrapper> &dst, CvtColorType type);

    AURA_VDSP_OP_HPP();
};

/**
 * @}
 */
} // namespace xtensa
} // namespace aura

#endif // AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_HPP__