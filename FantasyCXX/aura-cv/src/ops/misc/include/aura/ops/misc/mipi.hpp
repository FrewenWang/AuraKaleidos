#ifndef AURA_OPS_MISC_MIPI_HPP__
#define AURA_OPS_MISC_MIPI_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *    @defgroup misc Miscellaneous Iaura Process
 * @}
 */

namespace aura
{
/**
 * @addtogroup misc
 * @{
 */

/**
 * @brief Class representing the MIPI(Mobile Industry Processor Interface) Pack operation.
 *
 * This class inherits from the Op class and provides functionality for the MIPI Pack operation. It performs
 * a specific iaura data format conversion operation, packing 16-bit pixel data into 8-bit pixel data. It is
 * recommended to use the `IMipiPack` API, which internally calls this class. The only recommended scenario
 * for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IMipiPack` function is as follows:
 *
 * @code
 * MipiPack mipi_pack(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, mipi_pack, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS MipiPack : public Op
{
public:
    /**
     * @brief Constructor for MipiPack class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    MipiPack(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the MIPI Pack operation.
     *
     * For more details, please refer to @ref mipi_pack_details
     */
    Status SetArgs(const Array *src, Array *dst);
};

/**
 * @brief Class representing the MIPI(Mobile Industry Processor Interface) Unpack operation.
 *
 * This class inherits from the Op class and provides functionality for the MIPI Unpack operation. It performs
 * a specific iaura data format conversion operation, unpacking 8-bit pixel data into 8-bit/16-bit pixel data.
 * It is recommended to use the `IMipiUnpack` API, which internally calls this class. The only recommended scenario
 * for using this class is when the input or output type is `CLMem`.
 *
 * The approximate internal call within the `IMipiUnpack` function is as follows:
 *
 * @code
 * MipiUnPack mipi_unpack(ctx, target);
 *
 * // The OpCall API includes SetArgs(), Initialize(), Run(), and DeInitialize(),
 * return OpCall(ctx, mipi_unpack, &src, &dst);
 * @endcode
 */
class AURA_EXPORTS MipiUnPack : public Op
{
public:
    /**
     * @brief Constructor for MipiUnPack class.
     *
     * @param ctx The pointer to the Context object.
     * @param target The platform on which this function runs.
     */
    MipiUnPack(Context *ctx, const OpTarget &target = OpTarget::Default());

    /**
     * @brief Set arguments for the MIPI Unpack operation.
     *
     * For more details, please refer to @ref mipi_unpack_details
     */
    Status SetArgs(const Array *src, Array *dst);
};

/**
 * @brief Apply a MIPI(Mobile Industry Processor Interface) Pack operation to the source matrix.
 *
 * @anchor mipi_pack_details
 * This function performs a specific iaura data format conversion operation, packing 16-bit pixel data into 8-bit pixel data.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src) | Data type(dst)
 * ----------|----------------|----------------
 * NONE      | U16C1          | U8C1
 * NEON      | U16C1          | U8C1
 * HVX       | U16C1          | U8C1
 *
 * @note The width of src is 4/5 of dst, and the height is the same.
 */
AURA_EXPORTS Status IMipiPack(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @brief Apply a MIPI(Mobile Industry Processor Interface) Unpack operation to the source matrix.
 *
 * @anchor mipi_unpack_details
 * This function performs a specific iaura data format conversion operation, unpacking 8-bit pixel data into 8-bit/16-bit pixel data.
 *
 * @param ctx The pointer to the Context object.
 * @param src The source matrix.
 * @param dst The destination matrix.
 * @param target The platform on which this function runs.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 *
 * ### The Support data types and platforms
 * Platforms | Data type(src) | Data type(dst)
 * ----------|----------------|-------------------
 * NONE      | U8C1           | U8C1/U16C1
 * NEON      | U8C1           | U8C1/U16C1
 * HVX       | U8C1           | U8C1/U16C1
 *
 * @note The width of dst is 4/5 of src, and the height is the same.
 */
AURA_EXPORTS Status IMipiUnpack(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target = OpTarget::Default());

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_MISC_MIPI_HPP__