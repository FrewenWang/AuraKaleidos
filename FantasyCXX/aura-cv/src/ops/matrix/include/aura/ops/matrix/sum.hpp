#ifndef AURA_OPS_MATRIX_SUM_HPP__
#define AURA_OPS_MATRIX_SUM_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup sum_mean_std_dev Sum & Mean & Standard Deviation
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup sum_mean_std_dev
 * @{
 */

/**
 * @brief Interface class for sum of matrices along channels.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `ISum` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `ISum` function is as follows:
 *
 * @code
 * Sum sum(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, sum, &src, result);
 * @endcode
 */
class AURA_EXPORTS Sum : public Op
{
public:
    /**
     * @brief Constructor for the Sum class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Sum(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing the sum of elements in the source array.
     *
     * For more details, please refer to @ref sum_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Scalar &result);

    /**
     * @brief Generate sum opencl precompiled cache.
     *
     * @param src_elem_type The sum src array element type.
     * @param dst_elem_type The sum dst array element type.
     */
    static Status CLPrecompile(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type);
};

class AURA_EXPORTS Mean : public Op
{
public:
    /**
     * @brief Constructor for the Mean class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Mean(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for computing the mean of elements in the source array.
     *
     * For more details, please refer to @ref mean_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Scalar &result);
};

/**
 * @brief Sum operation of matrices along channels
 *
 * @anchor sum_details
 * This function computes the sum of elements in each channel of the source matrix and stores the result in
 * the provided variable (result).
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which the sum is computed.
 * @param result Reference to store the computed sum of elements.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type
 * -------------|------------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx,             x = 1, 2, 3
 * OpenCL       | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1
 * HVX          | U8Cx/S8Cx/U16Cx/S16Cx,                         x = 1, 2, 3
 */
AURA_EXPORTS Status ISum(Context *ctx, const Mat &src, Scalar &result, const OpTarget &target = OpTarget::Default());

/**
 * @brief Matrix operation to compute the mean along channels.
 *
 * @anchor mean_details
 * This function computes the mean of all elements in each channel of the src matrix and stores the
 * result in the provided variable (result).
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix for which the mean is computed.
 * @param result Reference to store the computed mean value.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type
 * -------------|------------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1, 2, 3
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/F32Cx, x = 1, 2, 3
 * OpenCL       | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = 1
 * HVX          | U8Cx/S8Cx/U16Cx/S16Cx, x = 1, 2, 3
 *
 * @note IMean internally invokes the ISum interface to compute the average value.
 */
AURA_EXPORTS Status IMean(Context *ctx, const Mat &src, Scalar &result, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_SUM_HPP__
