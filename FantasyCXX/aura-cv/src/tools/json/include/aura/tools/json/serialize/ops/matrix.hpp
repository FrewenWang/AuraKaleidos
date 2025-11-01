#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_MARTRIX_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_MARTRIX_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/matrix.h"

namespace aura
{

/**
 * @brief Define the json serialize method for arithmetic operation types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ArithmOpType, {
    {ArithmOpType::ADD, "ADD"},
    {ArithmOpType::SUB, "SUB"},
    {ArithmOpType::MUL, "MUL"},
    {ArithmOpType::DIV, "DIV"},
})

/**
 * @brief Define the json serialize method for binary operation types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(BinaryOpType, {
    {BinaryOpType::MIN, "MIN"},
    {BinaryOpType::MAX, "MAX"},
})

/**
 * @brief Define the json serialize method for flips types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(FlipType, {
    {FlipType::HORIZONTAL, "Horizontal"},
    {FlipType::VERTICAL,   "Vertical"},
    {FlipType::BOTH,       "Both"},
})

/**
 * @brief Define the json serialize method for norms types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(NormType, {
    {NormType::NORM_INF,    "NORM_INF"},
    {NormType::NORM_L1,     "NORM_L1"},
    {NormType::NORM_L2,     "NORM_L2"},
    {NormType::NORM_L2SQR,  "NORM_L2SQR"},
    {NormType::NORM_MINMAX, "NORM_MINMAX"},
})

/**
 * @brief Define the json serialize method for rotations types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(RotateType, {
    {RotateType::ROTATE_90,  "ROTATE_90"},
    {RotateType::ROTATE_180, "ROTATE_180"},
    {RotateType::ROTATE_270, "ROTATE_270"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_MARTRIX_HPP__