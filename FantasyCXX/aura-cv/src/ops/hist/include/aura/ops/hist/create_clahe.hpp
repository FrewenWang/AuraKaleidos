#ifndef AURA_OPS_HIST_CREATE_CLAHE_HPP__
#define AURA_OPS_HIST_CREATE_CLAHE_HPP__

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
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization) creation operation class.
 *
 * The use of this class for CLAHE is not recommended. It is recommended to use the `ICreateClAHE` API,
 * which internally calls this class. The only recommended scenario for using this class is when the input or output type is `CLMem`.
 * 
 * The approximate internal call within the `ICreateClAHE` function is as follows:
 * 
 * @code
 * CreateClAHE createclahe(ctx, target);
 * 
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, createclahe, &src, &dst, ...);
 * @endcode
 */
class AURA_EXPORTS CreateClAHE : public Op
{
public:
    /**
     * @brief Constructor for the CreateClAHE class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    CreateClAHE(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the CLAHE creation operation.
     *
     * For more details, please refer to @ref clahe_details
     * 
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch should be aligned to a stride,
     * which is obtained through the `GetCLLengthAlignSize` function and is platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_F64 clip_limit = 40.0, const Sizes &tile_grid_size = Sizes(8, 8));
};

/**
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization) function.
 *
 * @anchor clahe_details
 * This function applies Contrast Limited Adaptive Histogram Equalization to the input iaura.
 *
 * @param ctx The pointer to the Context object 
 * @param src The input iaura. And see the below for the supported data types.
 * @param dst The output array containing the CLAHE-processed iaura. And see the below for the supported data types.
 * @param clip_limit Threshold for contrast limiting (default is 40.0).
 * @param tile_grid_size Size of the contextual regions for contrast limiting (default is Sizes(8, 8)).
 * @param target The platform on which this function runs.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 * 
 * ### The suppored data types and platforms
 * Platforms    | Data type(src and dst)                            
 * -------------|------------------------------------------------- 
 * NONE         | U8C1, U16C1
 */
AURA_EXPORTS Status ICreateClAHE(Context *ctx, const Mat &src, Mat &dst, DT_F64 clip_limit = 40.0,
                                 const Sizes &tile_grid_size = Sizes(8, 8), const OpTarget &target = OpTarget::Default());

/**
 * @}
*/
}

#endif // AURA_OPS_HIST_CREATE_CLAHE_HPP__