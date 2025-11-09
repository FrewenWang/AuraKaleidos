#ifndef AURA_TOOLS_JSON_SERIALIZE_RUNTIME_TYPES_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_RUNTIME_TYPES_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/runtime/core/types.h"

namespace aura
{

/**
 * @brief Define the json serialize method for KeyPoint class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const KeyPoint &key_point);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, KeyPoint &key_point);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const KeyPointi &key_pointi);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, KeyPointi &key_pointi);

/**
 * @brief Define the json serialize method for 2D points class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point2i &point2i);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point2i &point2i);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point2f &point2f);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point2f &point2f);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point2d &point2d);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point2d &point2d);

/**
 * @brief Define the json serialize method for 3D points class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point3i &point3i);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point3i &point3i);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point3f &point3f);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point3f &point3f);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Point3d &point3d);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Point3d &point3d);

/**
 * @brief Define the json serialize method for 2D rectangles class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Rect2i &rect2i);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Rect2i &rect2i);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Rect2f &rect2f);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Rect2f &rect2f);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Rect2d &rect2d);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Rect2d &rect2d);

/**
 * @brief Define the json serialize method for 4-element scalar arrary class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Scalar &scalar);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Scalar &scalar);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Scalari &scalari);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Scalari &scalari);

/**
 * @brief Define the json serialize method for size of 2D iauras or matrices class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizes &sizes);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizes &sizes);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizesl &sizesl);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizesl &sizesl);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizesf &sizesf);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizesf &sizesf);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizesd &sizesd);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizesd &sizesd);

/**
 * @brief Define the json serialize method for size of 3D iauras or matrices class.
 *
 */
AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizes3 &sizes3);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizes3 &sizes3);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizes3l &sizes3l);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizes3l &sizes3l);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizes3f &sizes3f);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizes3f &sizes3f);

AURA_EXPORTS DT_VOID to_json(aura_json::json &json, const Sizes3d &sizes3d);

AURA_EXPORTS DT_VOID from_json(const aura_json::json &json, Sizes3d &sizes3d);

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_RUNTIME_TYPES_HPP__