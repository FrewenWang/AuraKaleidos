#ifndef AURA_OPS_FILTER_MEDIAN_HPP__
#define AURA_OPS_FILTER_MEDIAN_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup imgproc_filter Iaura Filtering
 *    @{
 *          @defgroup median Median Filter
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup median
 * @{
 */

/**
 * @brief Median filter class.
 *
 * The use of this class for Median filtering is not recommended. It is recommended to use the `IMedian` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IMedian` function is as follows:
 *
 * @code
 * Median median(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, laplacian, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS Median : public Op
{
public:
    /**
     * @brief Constructor for Median class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    Median(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set the arguments for the Median filter operation.
     *
     * For more details, please refer to @ref median_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize);

    /**
     * @brief Generate median opencl precompiled cache.
     *
     * @param elem_type The element type of the src/dst array.
     * @param channel The channel of the src/dst array.
     * @param ksize The size of the median kernel.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize);
};

/**
 * @brief The median filer API.
 *
 * @anchor median_details
 * The function smooths the iaura using the median filter with the @f$ \texttt{kszie} \times \texttt{kszie} @f$ kernel size.
 * And in-place operation is not supported.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura, and see below for the supported data types.
 * @param dst The output iaura, and see below for the supported data types.
 * @param ksize The filter kernel size, and see below for the supported sizes.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)                        | channel(src & dst) | ksize
 * -------------|----------------------------------------------|--------------------|-----------
 * NONE         | U8, S8, U16, S16, F16, U32, S32, F32         | N                  | N
 * NEON         | U8, S8, U16, S16, F16, U32, S32, F32         | 1,2,3              | 3, 5, 7
 * OpenCL       | U8, S8, U16, S16, F16, U32, S32, F32         | 1,2,3              | 3, 5, 7
 * HVX          | -                                            | -                  | -
 *
 * @note 1.N is positive integer(ksize is odd positive integer). <br>
 *       2.The above implementations only supported BorderType(REPLICATE).
 */
AURA_EXPORTS Status IMedian(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize,
                            const OpTarget &target = OpTarget::Default());

} // namespace aura

#endif // AURA_OPS_FILTER_MEDIAN_HPP__