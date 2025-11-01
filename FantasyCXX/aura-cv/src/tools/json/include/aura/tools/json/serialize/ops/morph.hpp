#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_MORPH_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_MORPH_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/morph.h"

namespace aura
{

/**
 * @brief Define the json serialize method for morphological operations enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(MorphType, {
    {MorphType::ERODE,    "ERODE"},
    {MorphType::DILATE,   "DILATE"},
    {MorphType::OPEN,     "OPEN"},
    {MorphType::CLOSE,    "CLOSE"},
    {MorphType::GRADIENT, "GRADIENT"},
    {MorphType::TOPHAT,   "TOPHAT"},
    {MorphType::BLACKHAT, "BLACKHAT"},
})

/**
 * @brief Define the json serialize method for morphological shapes enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(MorphShape, {
    {MorphShape::RECT,    "RECT"},
    {MorphShape::CROSS,   "CROSS"},
    {MorphShape::ELLIPSE, "ELLIPSE"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_MORPH_HPP__