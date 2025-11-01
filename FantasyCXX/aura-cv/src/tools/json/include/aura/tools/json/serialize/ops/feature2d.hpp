#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_FEATURE2D_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_FEATURE2D_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/feature2d.h"

namespace aura
{

/**
 * @brief Define the json serialize method for FAST corner detector types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(FastDetectorType, {
    {FastDetectorType::FAST_5_8,  "FAST_5_8"},
    {FastDetectorType::FAST_7_12, "FAST_7_12"},
    {FastDetectorType::FAST_9_16, "FAST_9_16"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_FEATURE2D_HPP__