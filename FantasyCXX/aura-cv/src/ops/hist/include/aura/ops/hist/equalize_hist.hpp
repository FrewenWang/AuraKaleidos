#ifndef AURA_OPS_HIST_EQUALIZE_HIST_HPP__
#define AURA_OPS_HIST_EQUALIZE_HIST_HPP__

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
 * @brief Histogram equalization class.
 *
 * The use of this class for EqualizeHist is not recommended. It is recommended to use the `IEqualizeHist` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `IEqualizeHist` function is as follows:
 * 
 * @code
 * EqualizeHist equalizehist(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, equalizehist, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS EqualizeHist : public Op
{
public:
    /**
     * @brief Constructor for the histogram equalization operation.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    EqualizeHist(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the histogram equalization operation.
     *
     * For more details, please refer to @ref equalizehist_details
     * 
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst);
};

/**
 * @brief Histogram equalization function.
 *
 * @anchor equalizehist_details
 * This function equalizes the histogram of the input iaura.
 *
 * @param ctx The pointer to the Context object 
 * @param src The input iaura. And see the below for the supported data types.
 * @param dst The output array containing the histogram-equalized iaura. And see the below for the supported data types.
 * @param target The platform on which this function runs.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The suppored data types and platforms
 * Platforms    | Data type(src and dst)                            
 * -------------|------------------------------------------------- 
 * NONE         | U8C1
 */
AURA_EXPORTS Status IEqualizeHist(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
*/
}

#endif // AURA_OPS_HIST_EQUALIZE_HIST_HPP__