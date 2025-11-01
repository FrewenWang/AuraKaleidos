#ifndef AURA_OPS_MATRIX_MERGE_HPP__
#define AURA_OPS_MATRIX_MERGE_HPP__

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
 * @brief The matrix merging operation class.
 *
 * The use of this class for matrix merging operations is not recommended.
 * It is recommended to use the `IMerge` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IMerge` function is as follows:
 *
 * @code
 * Merge merge(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, merge, src_arrays, &dst);
 * @endcode
 */
class AURA_EXPORTS Merge : public Op
{
public:
    /**
     * @brief Constructor for the Merge class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Merge(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for merging multiple arrays into a single array.
     *
     * For more details, please refer to @ref merge_details.
     */
    Status SetArgs(const std::vector<const Array*> &src, Array *dst);
};

/**
 * @brief Merges multiple source matrices into a single destination matrix.
 *
 * @anchor merge_details
 * This function implements the functionality of merging multiple input matrices into a single output matrix
 * along the channel dimension. And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src Vector containing the source matrices to be merged.
 * @param dst The destination matrix to store the merged result.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Data type(src)
 * -------------|--------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note 1.N is positive integer; <br>
 *       2.Src mats' total channels must match dst mat's channels; <br>
 *       3.Src and dst mat must have the same width, height and data type.
 */
AURA_EXPORTS Status IMerge(Context *ctx, const std::vector<Mat> &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_MERGE_HPP__
