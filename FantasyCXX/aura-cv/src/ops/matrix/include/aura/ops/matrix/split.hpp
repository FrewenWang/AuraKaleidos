#ifndef AURA_OPS_MATRIX_SPLIT_HPP__
#define AURA_OPS_MATRIX_SPLIT_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup merge_split Merge And Split
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup merge_split
 * @{
 */

/**
 * @brief The matrix split operation class.
 *
 * The use of this class for matrix split operations is not recommended.
 * It is recommended to use the `ISplit` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `ISplit` function is as follows:
 *
 * @code
 * Split split(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, split, &src, dst_arrays);
 * @endcode
 */
class AURA_EXPORTS Split : public Op
{
public:
    /**
     * @brief Constructor for the Split class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Split(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for splitting the source array into multiple destination arrays.
     *
     * For more details, please refer to @ref split_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, const std::vector<Array*> &dst);
};

/**
 * @brief Splits the source matrix into multiple destination matrices.
 *
 * @anchor split_details
 * This function implements the functionality of splitting a matrix into multiple single-channel matrices
 * along the channel dimension. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be split.
 * @param dst Vector of destination matrices to store the split parts.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|------------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3, 4
 *
 * @note 1.N is positive integer. <br>
 *       2.Dst mats' total channel count must match src mat's channel count. <br>
 *       3.Src and dst must have same size and data type.
 */
AURA_EXPORTS Status ISplit(Context *ctx, const Mat &src, std::vector<Mat> &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_SPLIT_HPP__
