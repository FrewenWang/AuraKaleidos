#ifndef AURA_OPS_HIST_CALCHIST_HPP__
#define AURA_OPS_HIST_CALCHIST_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup hist Histograms
 * @}
 */

namespace aura
{
/**
 * @addtogroup hist
 * @{
 */

/**
 * @brief Histogram calculation class.
 *
 * The use of this class for histogram is not recommended. It is recommended to use the `ICalcHist` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `ICalcHist` function is as follows:
 *
 * @code
 * CalcHist calchist(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, calchist, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS CalcHist : public Op
{
public:
    /**
     * @brief Constructor for the Histogram calculation operation.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    CalcHist(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the Histogram calculation operation.
     *
     * For more details, please refer to @ref hist_details
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, MI_S32 channel, std::vector<MI_U32> &hist,
                   MI_S32 hist_size, const Scalar &ranges, const Array *mask = NULL, MI_BOOL accumulate = MI_FALSE);
};

/**
 * @brief Histogram calculation function.
 *
 * @anchor hist_details
 * This function calculates the histogram of a set of arrays.
 *
 * @param ctx The pointer to the Context object
 * @param src The input iaura. And see the below for the supported data types.
 * @param channel The channel to be measured.
 * @param hist The output vector containing the histogram.
 * @param hist_size The number of bins in the histogram.
 * @param ranges The range of values to be measured.
 * @param mask Optional mask. If the array is not empty, only the pixels in it that are non-zero are counted.
 * @param accumulate Flag indicating whether to accumulate the histogram (default is MI_FALSE).
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The supported data types and platforms
 * Platforms    | Data type(src or dst)
 * -------------|-------------------------------------------------
 * NONE         | U8Cx, U16Cx, x = N
 * HVX          | U8C1
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status ICalcHist(Context *ctx, const Mat &src, MI_S32 channel, std::vector<MI_U32> &hist,
                              MI_S32 hist_size, const Scalar &ranges, const Mat &mask = Mat(),
                              MI_BOOL accumulate = MI_FALSE, const OpTarget &target = OpTarget::Default());

/**
 * @}
*/
}

#endif // AURA_OPS_HIST_CALCHIST_HPP__