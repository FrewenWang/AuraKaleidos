#ifndef AURA_OPS_MORPH_MORPH_HPP__
#define AURA_OPS_MORPH_MORPH_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup morph Morphology
 * @}
 */

namespace aura
{
/**
 * @addtogroup morph
 * @{
 */

/**
 * @brief Enumeration defining various morphological operations.
 */
enum class MorphType
{
    ERODE    = 0, /*!< an erode operation */
    DILATE,       /*!< a dilate operation */
    OPEN,         /*!< an opening operation */
    CLOSE,        /*!< a closing operation */
    GRADIENT,     /*!< a morphological gradient */
    TOPHAT,       /*!< top hat */
    BLACKHAT      /*!< black hat */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, MorphType morph_type)
{
    switch (morph_type)
    {
        case MorphType::ERODE:
        {
            os << "ERODE";
            break;
        }

        case MorphType::DILATE:
        {
            os << "DILATE";
            break;
        }

        case MorphType::OPEN:
        {
            os << "OPEN";
            break;
        }

        case MorphType::CLOSE:
        {
            os << "CLOSE";
            break;
        }

        case MorphType::GRADIENT:
        {
            os << "GRADIENT";
            break;
        }

        case MorphType::TOPHAT:
        {
            os << "TOPHAT";
            break;
        }

        case MorphType::BLACKHAT:
        {
            os << "BLACKHAT";
            break;
        }

        default:
        {
            os << "undefined morph type";
            break;
        }
    }

    return os;
}

AURA_INLINE const std::string MorphTypeToString(MorphType type)
{
    std::ostringstream ss;
    ss << type;
    return ss.str();
}

/**
 * @brief Enumeration defining various morphological shapes.
 */
enum class MorphShape
{
    RECT    = 0, /*!< a rectangular-shaped */
    CROSS,       /*!< a cross-shaped */
    ELLIPSE      /*!< an ellipse-shaped */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, MorphShape morph_shape)
{
    switch (morph_shape)
    {
        case MorphShape::RECT:
        {
            os << "RECT";
            break;
        }

        case MorphShape::CROSS:
        {
            os << "CROSS";
            break;
        }

        case MorphShape::ELLIPSE:
        {
            os << "ELLIPSE";
            break;
        }

        default:
        {
            os << "undefined morph shape";
            break;
        }
    }

    return os;
}

AURA_INLINE const std::string MorphShapeToString(MorphShape shape)
{
    std::ostringstream ss;
    ss << shape;
    return ss.str();
}

/**
 * @brief Class representing the Dilate operation.
 *
 * This class represents a dilation operation, which is a morphological operation that expands
 * regionsof higher pixel values in an input iaura. It is recommended to use the `IDilate` API,
 * which internally calls this class. The only recommended scenario for using this class is
 * when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `IDilate` function is as follows:
 * 
 * @code
 * Dilate dilate(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, dilate, &src, &dst, ksize, shape, iterations); 
 * @endcode
 */
class AURA_EXPORTS Dilate : public Op
{
public:
    /**
     * @brief Constructor for the dilate operation.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    Dilate(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the dilate operation.
     *
     * For more details, please refer to @ref dilate_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1);

    /**
     * @brief Generate gaussian opencl precompiled cache.
     *
     * @param elem_type The dilate src/dst array element type.
     * @param channel The dilate src/dst array channel.
     * @param ksize The dilate kernel size.
     * @param shape The morphological shape.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, MorphShape shape);
};

/**
 * @brief Class representing the Erode operation.
 *
 * This class represents an erosion operation, which is a morphological operation that shrinks
 * regions of higher pixel values in an input iaura. It is recommended to use the `IErode` API,
 * which internally calls this class. The only recommended scenario for using this class is
 * when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IErode` function is as follows:
 *
 * @code
 * Erode erode(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, erode, &src, &dst, ksize, shape, iterations); 
 * @endcode
 */
class AURA_EXPORTS Erode : public Op
{
public:
    /**
     * @brief Constructor for the erode operation.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    Erode(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the erode operation.
     *
     * For more details, please refer to @ref erode_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the
     * row pitch should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize`
     * function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1);

    /**
     * @brief Generate gaussian opencl precompiled cache.
     *
     * @param elem_type The erode src/dst array element type.
     * @param channel The erode src/dst array channel.
     * @param ksize The erode kernel size.
     * @param shape The morphological shape.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, MorphShape shape);
};

/**
 * @brief Apply a dilate operation to the src matrix.
 *
 * @anchor dilate_details
 * This function applies a dilate operation to the src matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param ksize The kernel size (must be odd and greater than 0).
 * @param shape The morphological shape (default is MorphShape::RECT).
 * @param iterations The number of iterations (default is 1).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * Platforms | Data type(src or dst)                     | ksize
 * ----------|-------------------------------------------|------------
 * NONE      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N       | K
 * NEON      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3 | 3, 5, 7
 * HVX       | U8Cx/U16Cx/S16Cx,             x = 1, 2, 3 | 3, 5, 7
 *
 * @note 1.N is positive integer, K is odd positive integer. <br>
 *       2.The above implements support all MorphShape(RECT/CROSS/ELLIPSE).
 *
 */
AURA_EXPORTS Status IDilate(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT,
                            DT_S32 iterations = 1, const OpTarget &target = OpTarget::Default());

/**
 * @brief Apply an erode operation to the src matrix.
 *
 * @anchor erode_details
 * This function applies an erode operation to the src matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param ksize The kernel size (must be odd and greater than 0).
 * @param shape The morphological shape (default is MorphShape::RECT).
 * @param iterations The number of iterations (default is 1).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * Platforms | Data type(src or dst)                     | ksize
 * ----------|-------------------------------------------|------------
 * NONE      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N       | K
 * NEON      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3 | 3, 5, 7
 * OpenCL    | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1       | 3, 5, 7
 * HVX       | U8Cx/U16Cx/S16Cx,             x = 1, 2, 3 | 3, 5, 7
 *
 * @note 1.N is positive integer, K is odd positive integer. <br>
 *       2.The above implements support all MorphShape(RECT/CROSS/ELLIPSE).
 */
AURA_EXPORTS Status IErode(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT,
                           DT_S32 iterations = 1, const OpTarget &target = OpTarget::Default());

/**
 * @brief Apply a morphology operation to the src matrix.
 *
 * This function applies a morphology operation to the src matrix. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param type The morphology operation type.
 * @param ksize The kernel size (must be odd and greater than 0).
 * @param shape The morphological shape (default is MorphShape::RECT).
 * @param iterations The number of iterations (default is 1).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * Platforms | Data type(src or dst)                     | ksize
 * ----------|-------------------------------------------|------------
 * NONE      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = N       | K
 * NEON      | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3 | 3, 5, 7
 * OpenCL    | U8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1       | 3, 5, 7
 * HVX       | U8Cx/U16Cx/S16Cx,             x = 1, 2, 3 | 3, 5, 7
 *
 * @note 1.N is positive integer, K is odd positive integer. <br>
 *       2.The above implements supported all MorphType(ERODE/DILATE/OPEN/CLOSE/GRADIENTTOPHAT/BLACKHAT). <br>
 *       3.The above implements supported all MorphShape(RECT/CROSS/ELLIPSE).
 */
AURA_EXPORTS Status IMorphologyEx(Context *ctx, const Mat &src, Mat &dst, MorphType type, DT_S32 ksize,
                                  MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_MORPH_MORPH_HPP__