#ifndef AURA_OPS_MATRIX_MIN_MAX_LOC_HPP__
#define AURA_OPS_MATRIX_MIN_MAX_LOC_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup min_max_loc Min And Max Location
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup min_max_loc
 * @{
 */

/**
 * @brief Interface class representing an operation to find the minimum and maximum values and their locations in a matrix.
 *
 * The use of this class for the above-mentioned operation is not recommended.
 * It is recommended to use the `IMinMaxLoc` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IMinMaxLoc` function is as follows:
 *
 * @code
 * MinMaxLoc min_max_loc(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, min_max_loc, &src, min_val, max_val, min_pos, max_pos);
 * @endcode
 */
class AURA_EXPORTS MinMaxLoc : public Op
{
public:
    /**
     * @brief Constructor for the MinMaxLoc class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    MinMaxLoc(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for finding the minimum and maximum values and their locations.
     *
     * For more details, please refer to @ref minmaxloc_details
     */
    Status SetArgs(const Array *src, DT_F64 *min_val, DT_F64 *max_val, Point3i *min_pos, Point3i *max_pos);
};

/**
 * @brief Finds the minimum and maximum values and their positions in the source matrix.
 *
 * @anchor minmaxloc_details
 * This function computes the minimum and maximum values in the src matrix along with their corresponding
 * positions (min_pos and max_pos) and stores them in the provided variables.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to find the minimum and maximum values.
 * @param min_val Pointer to store the computed minimum value.
 * @param max_val Pointer to store the computed maximum value.
 * @param min_pos Pointer to store the position of the minimum value.
 * @param max_pos Pointer to store the position of the maximum value.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type
 * -------------|---------------------------------------------------------------
 * NONE         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 * NEON         | U8Cx/S8Cx/U16Cx/S16Cx/F16Cx/U32Cx/S32Cx/F32Cx, x = N
 *
 * @note N is positive integer.
 */
AURA_EXPORTS Status IMinMaxLoc(Context *ctx, const Mat &src, DT_F64 *min_val, DT_F64 *max_val,
                               Point3i *min_pos, Point3i *max_pos, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif// AURA_OPS_MATRIX_MIN_MAX_LOC_HPP__
