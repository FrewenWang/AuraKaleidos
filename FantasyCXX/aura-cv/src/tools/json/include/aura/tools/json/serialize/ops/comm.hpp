#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_COMM_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_COMM_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/core.h"

namespace aura
{

/**
 * @brief Define the json serialize method for InterpType types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(InterpType, {
    {InterpType::NEAREST, "Nearest"},
    {InterpType::LINEAR,  "Linear"},
    {InterpType::CUBIC,   "Cubic"},
    {InterpType::AREA,    "AREA"},
})

/**
 * @brief Define the json serialize method for BorderArea types enumeration.
 */
AURA_JSON_SERIALIZE_ENUM(BorderArea, {
    {BorderArea::TOP,    "Top"},
    {BorderArea::BOTTOM, "Bottom"},
    {BorderArea::LEFT,   "Left"},
    {BorderArea::RIGHT,  "Right"},
})

/**
 * @brief Define the json serialize method for TargetType types enumeration.
 */
AURA_JSON_SERIALIZE_ENUM(TargetType, {
    {TargetType::INVALID, "Invalid"},
    {TargetType::NONE,    "None"},
    {TargetType::NEON,    "Neon"},
    {TargetType::OPENCL,  "Opencl"},
    {TargetType::HVX,     "Hvx"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_COMM_HPP__