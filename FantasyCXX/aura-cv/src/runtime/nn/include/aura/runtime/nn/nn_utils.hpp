#ifndef AURA_RUNTIME_NN_NN_UTILS_HPP__
#define AURA_RUNTIME_NN_NN_UTILS_HPP__

#include "aura/runtime/core.h"
#include "aura/runtime/mat.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup nn NN
 * @}
 */

namespace aura
{
/**
 * @addtogroup utils
 * @{
*/

/**
 * @brief Encryption function, encrypt intput buffer with key and get encrypted output buffer
 *
 * @param ctx The pointer to the Context object.
 * @param src The input buffer, used for encryption.
 * @param dst The output buffer, store encrypted buffer.
 * @param key The input encrypted secret key.
 * @param size The size of encrypted buffer.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 */
AURA_EXPORTS Status NNEncrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, MI_U64 size);

/**
 * @brief Decryption function, decrypt intput buffer with key and get decrypted output buffer
 *
 * @param ctx The pointer to the Context object.
 * @param src The input buffer, used for decryption.
 * @param dst The output buffer, store decrypted buffer.
 * @param key The input decrypted secret key.
 * @param size The size of decrypted buffer.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 */
AURA_EXPORTS Status NNDecrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, MI_U64 size);

/**
 * @brief Quantize function, use scale、zero_point to quantize input buffer
 *
 * @param ctx The pointer to the Context object.
 * @param src The input buffer, only sppose F32 type
 * @param dst The output buffer, only sppose u8/s8/u16/s16 type
 * @param zero_point quantization zero_point value
 * @param scale quantization scale value.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 */
AURA_EXPORTS Status NNQuantize(Context *ctx, const Mat &src, Mat &dst, MI_S32 zero_point, MI_F32 scale);

/**
 * @brief Dequantize function, use scale、zero_point to quantize input buffer
 *
 * @param ctx The pointer to the Context object.
 * @param src The input buffer, only sppose u8/s8/u16/s16 type
 * @param dst The output buffer, only sppose F32 type
 * @param zero_point dequantization zero_point value
 * @param scale dequantization scale value.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status
 */
AURA_EXPORTS Status NNDeQuantize(Context *ctx, const Mat &src, Mat &dst, MI_S32 zero_point, MI_F32 scale);

/**
 * @brief Split function, use separator to split input string
 *
 * @param src The input string
 * @param separator The delimiter
 *
 * @return Delimited list of strings
 */
AURA_EXPORTS std::vector<std::string> NNSplit(const std::string &src, MI_CHAR separator = ';');

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_NN_NN_UTILS_HPP__