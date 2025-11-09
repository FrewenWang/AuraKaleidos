#ifndef AURA_OPS_MATRIX_GRID_DFT_HPP__
#define AURA_OPS_MATRIX_GRID_DFT_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup matrix Matrix Process
 *      @{
 *          @defgroup grid_dft Grid DFT
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup grid_dft
 * @{
 */

/**
 * @brief Interface class representing a grid-based Discrete Fourier Transform (DFT) operations.
 *
 * The use of this class for grid-based DFT operations is not recommended.
 * It is recommended to use the `IGridDft` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IGridDft` function is as follows:
 *
 * @code
 * GridDft grid_dft(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, grid_dft, &src, &dst, grid_len);
 * @endcode
 */
class AURA_EXPORTS GridDft : public Op
{
public:
    /**
     * @brief Constructor for the GridDft class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    GridDft(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the grid-based DFT operation.
     *
     * For more details, please refer to @ref griddft_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 grid_len);

    /**
     * @brief Generate griddft opencl precompiled cache.
     *
     * @param elem_type The griddft src/dst array element type.
     * @param grid_len The length of the grid for the DFT operation.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 grid_len);
};

/**
 * @brief Interface class representing an grid-based Inverse Discrete Fourier Transform (IDFT) operations.
 *
 * The use of this class for grid-based IDFT operations is not recommended.
 * It is recommended to use the `IGridIDft` API, which internally calls this class.
 * The only recommended scenario for using this class is when the input or output type is 'CLMem'.
 *
 * The approximate internal call within the `IGridIDft` function is as follows:
 *
 * @code
 * GridIDft grid_idft(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize().
 * return OpCall(ctx, grid_idft, &src, &dst, grid_len, with_scale);
 * @endcode
 */
class AURA_EXPORTS GridIDft : public Op
{
public:
    /**
     * @brief Constructor for the GridIDft class.
     *
     * @param ctx The pointer to the Context object
     * @param target The platform on which this function runs
     */
    GridIDft(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Sets the arguments for the grid-based IDFT operation.
     *
     * For more details, please refer to @ref grididft_details.
     *
     * @note If the type of src or dst is `CLMem` and is an iaura2D memory object, the row pitch
     * should be aligned to a stride, which is obtained through the `GetCLLengthAlignSize` function and is
     * platform-dependent.
     */
    Status SetArgs(const Array *src, Array *dst, DT_S32 grid_len, DT_BOOL with_scale);

    /**
     * @brief Generate grididft opencl precompiled cache.
     *
     * @param elem_type The grididft src/dst array element type.
     * @param grid_len The length of the grid for the IDFT operation.
     * @param with_scale A boolean flag indicating whether to perform scaling in the IDFT (defaults to false).
     * @param save_real_only If dst channel is 1.
     */
    static Status CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 grid_len, DT_S32 with_scale, DT_BOOL save_real_only);
};

/**
 * @brief Performs a grid-based Discrete Fourier Transform (DFT) on the source matrix.
 *
 * @anchor griddft_details
 * This function computes the DFT of the src matrix and stores the result in the dst matrix
 * based on the specified grid length. For more details, please refer to IDft.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be transformed using the DFT.
 * @param dst The destination matrix to store the result of the DFT.
 * @param grid_len The length of the grid for the DFT operation.
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src Data type                                           |Dst Data type | grid_len
 * -------------|---------------------------------------------------------|--------------|---------------
 * NONE         | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1           | F32C2        | 4/8/16/32
 * NEON         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1                       | F32C2        | 4/8/16/32
 * OpenCL       | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1                       | F32C2        | 4/8/16/32
 */
AURA_EXPORTS Status IGridDft(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, const OpTarget &target = OpTarget::Default());

/**
 * @brief Performs an inverse grid-based Discrete Fourier Transform (IDFT) on the source matrix.
 *
 * @anchor grididft_details
 * This function computes the inverse DFT of the src matrix and stores the result in the dst matrix
 * based on the specified grid length and scaling option. For more details, please refer to #IInverseDft.
 *
 * @param ctx The pointer to the Context object
 * @param src The source matrix to be transformed using the IDFT.
 * @param dst The destination matrix to store the result of the IDFT.
 * @param grid_len The length of the grid for the IDFT operation.
 * @param with_scale A boolean flag indicating whether to perform scaling in the IDFT (defaults to false).
 * @param target The platform on which this function runs
 *
 * @return Status Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### the supported data types and platforms
 * Platforms    | Src data type | Dst data type                                              | grid_len
 * -------------|---------------|------------------------------------------------------------|---------------
 * NONE         | F32C2         | U8C1/S8C1/U16C1/S16C1/F16C1/U32C1/S32C1/F32C1/F32C2        | 4/8/16/32
 * NEON         | F32C2         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1/F32C2                    | 4/8/16/32
 * OpenCL       | F32C2         | U8C1/S8C1/U16C1/S16C1/F16C1/F32C1/F32C2                    | 4/8/16/32
 */
AURA_EXPORTS Status IGridIDft(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, DT_BOOL with_scale, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */

}// namespace aura

#endif // AURA_OPS_MATRIX_GRID_DFT_HPP__
