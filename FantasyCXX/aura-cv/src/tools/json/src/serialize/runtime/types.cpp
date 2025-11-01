#include "aura/tools/json/serialize/runtime/types.hpp"

namespace aura
{

// KeyPoint
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const KeyPoint &key_point)
{
    json["m_pt"]       = key_point.m_pt;
    json["m_size"]     = key_point.m_size;
    json["m_angle"]    = key_point.m_angle;
    json["m_response"] = key_point.m_response;
    json["m_octave"]   = key_point.m_octave;
    json["m_class_id"] = key_point.m_class_id;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, KeyPoint &key_point)
{
    json.at("m_pt").get_to(key_point.m_pt);
    json.at("m_size").get_to(key_point.m_size);
    json.at("m_angle").get_to(key_point.m_angle);
    json.at("m_response").get_to(key_point.m_response);
    json.at("m_octave").get_to(key_point.m_octave);
    json.at("m_class_id").get_to(key_point.m_class_id);
}

// KeyPointi
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const KeyPointi &key_pointi)
{
    json["m_pt"]       = key_pointi.m_pt;
    json["m_size"]     = key_pointi.m_size;
    json["m_angle"]    = key_pointi.m_angle;
    json["m_response"] = key_pointi.m_response;
    json["m_octave"]   = key_pointi.m_octave;
    json["m_class_id"] = key_pointi.m_class_id;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, KeyPointi &key_pointi)
{
    json.at("m_pt").get_to(key_pointi.m_pt);
    json.at("m_size").get_to(key_pointi.m_size);
    json.at("m_angle").get_to(key_pointi.m_angle);
    json.at("m_response").get_to(key_pointi.m_response);
    json.at("m_octave").get_to(key_pointi.m_octave);
    json.at("m_class_id").get_to(key_pointi.m_class_id);
}

// Point2i
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point2i &point2i)
{
    json["m_x"] = point2i.m_x;
    json["m_y"] = point2i.m_y;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point2i &point2i)
{
    json.at("m_x").get_to(point2i.m_x);
    json.at("m_y").get_to(point2i.m_y);
}

// Point2f
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point2f &point2f)
{
    json["m_x"] = point2f.m_x;
    json["m_y"] = point2f.m_y;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point2f &point2f)
{
    json.at("m_x").get_to(point2f.m_x);
    json.at("m_y").get_to(point2f.m_y);
}

// Point2d
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point2d &point2d)
{
    json["m_x"] = point2d.m_x;
    json["m_y"] = point2d.m_y;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point2d &point2d)
{
    json.at("m_x").get_to(point2d.m_x);
    json.at("m_y").get_to(point2d.m_y);
}

// Point3i
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point3i &point3i)
{
    json["m_x"] = point3i.m_x;
    json["m_y"] = point3i.m_y;
    json["m_z"] = point3i.m_z;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point3i &point3i)
{
    json.at("m_x").get_to(point3i.m_x);
    json.at("m_y").get_to(point3i.m_y);
    json.at("m_z").get_to(point3i.m_z);
}

// Point3f
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point3f &point3f)
{
    json["m_x"] = point3f.m_x;
    json["m_y"] = point3f.m_y;
    json["m_z"] = point3f.m_z;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point3f &point3f)
{
    json.at("m_x").get_to(point3f.m_x);
    json.at("m_y").get_to(point3f.m_y);
    json.at("m_z").get_to(point3f.m_z);
}

// Point3d
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Point3d &point3d)
{
    json["m_x"] = point3d.m_x;
    json["m_y"] = point3d.m_y;
    json["m_z"] = point3d.m_z;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Point3d &point3d)
{
    json.at("m_x").get_to(point3d.m_x);
    json.at("m_y").get_to(point3d.m_y);
    json.at("m_z").get_to(point3d.m_z);
}

// Rect2i
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Rect2i &rect2i)
{
    json["m_x"]      = rect2i.m_x;
    json["m_y"]      = rect2i.m_y;
    json["m_width"]  = rect2i.m_width;
    json["m_height"] = rect2i.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Rect2i &rect2i)
{
    json.at("m_x").get_to(rect2i.m_x);
    json.at("m_y").get_to(rect2i.m_y);
    json.at("m_width").get_to(rect2i.m_width);
    json.at("m_height").get_to(rect2i.m_height);
}

// Rect2f
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Rect2f &rect2f)
{
    json["m_x"]      = rect2f.m_x;
    json["m_y"]      = rect2f.m_y;
    json["m_width"]  = rect2f.m_width;
    json["m_height"] = rect2f.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Rect2f &rect2f)
{
    json.at("m_x").get_to(rect2f.m_x);
    json.at("m_y").get_to(rect2f.m_y);
    json.at("m_width").get_to(rect2f.m_width);
    json.at("m_height").get_to(rect2f.m_height);
}

// Rect2d
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Rect2d &rect2d)
{
    json["m_x"]      = rect2d.m_x;
    json["m_y"]      = rect2d.m_y;
    json["m_width"]  = rect2d.m_width;
    json["m_height"] = rect2d.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Rect2d &rect2d)
{
    json.at("m_x").get_to(rect2d.m_x);
    json.at("m_y").get_to(rect2d.m_y);
    json.at("m_width").get_to(rect2d.m_width);
    json.at("m_height").get_to(rect2d.m_height);
}

// Scalar
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Scalar &scalar)
{
    json["m_val_0"] = scalar.m_val[0];
    json["m_val_1"] = scalar.m_val[1];
    json["m_val_2"] = scalar.m_val[2];
    json["m_val_3"] = scalar.m_val[3];
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Scalar &scalar)
{
    json.at("m_val_0").get_to(scalar.m_val[0]);
    json.at("m_val_1").get_to(scalar.m_val[1]);
    json.at("m_val_2").get_to(scalar.m_val[2]);
    json.at("m_val_3").get_to(scalar.m_val[3]);
}

// Scalari
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Scalari &scalari)
{
    json["m_val_0"] = scalari.m_val[0];
    json["m_val_1"] = scalari.m_val[1];
    json["m_val_2"] = scalari.m_val[2];
    json["m_val_3"] = scalari.m_val[3];
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Scalari &scalari)
{
    json.at("m_val_1").get_to(scalari.m_val[1]);
    json.at("m_val_0").get_to(scalari.m_val[0]);
    json.at("m_val_2").get_to(scalari.m_val[2]);
    json.at("m_val_3").get_to(scalari.m_val[3]);
}

// Sizes
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizes &sizes)
{
    json["m_width"]  = sizes.m_width;
    json["m_height"] = sizes.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizes &sizes)
{
    json.at("m_width").get_to(sizes.m_width);
    json.at("m_height").get_to(sizes.m_height);
}

// Sizesl
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizesl &sizesl)
{
    json["m_width"]  = sizesl.m_width;
    json["m_height"] = sizesl.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizesl &sizesl)
{
    json.at("m_width").get_to(sizesl.m_width);
    json.at("m_height").get_to(sizesl.m_height);
}

// Sizesf
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizesf &sizesf)
{
    json["m_width"]  = sizesf.m_width;
    json["m_height"] = sizesf.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizesf &sizesf)
{
    json.at("m_width").get_to(sizesf.m_width);
    json.at("m_height").get_to(sizesf.m_height);
}

// Sizesd
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizesd &sizesd)
{
    json["m_width"]  = sizesd.m_width;
    json["m_height"] = sizesd.m_height;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizesd &sizesd)
{
    json.at("m_width").get_to(sizesd.m_width);
    json.at("m_height").get_to(sizesd.m_height);
}

// Sizes3
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizes3 &sizes3)
{
    json["m_width"]   = sizes3.m_width;
    json["m_height"]  = sizes3.m_height;
    json["m_channel"] = sizes3.m_channel;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizes3 &sizes3)
{
    json.at("m_width").get_to(sizes3.m_width);
    json.at("m_height").get_to(sizes3.m_height);
    json.at("m_channel").get_to(sizes3.m_channel);
}

// Sizes3l
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizes3l &sizes3l)
{
    json["m_width"]   = sizes3l.m_width;
    json["m_height"]  = sizes3l.m_height;
    json["m_channel"] = sizes3l.m_channel;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizes3l &sizes3l)
{
    json.at("m_width").get_to(sizes3l.m_width);
    json.at("m_height").get_to(sizes3l.m_height);
    json.at("m_channel").get_to(sizes3l.m_channel);
}

// Sizes3f
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizes3f &sizes3f)
{
    json["m_width"]   = sizes3f.m_width;
    json["m_height"]  = sizes3f.m_height;
    json["m_channel"] = sizes3f.m_channel;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizes3f &sizes3f)
{
    json.at("m_width").get_to(sizes3f.m_width);
    json.at("m_height").get_to(sizes3f.m_height);
    json.at("m_channel").get_to(sizes3f.m_channel);
}

// Sizes3d
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Sizes3d &sizes3d)
{
    json["m_width"]   = sizes3d.m_width;
    json["m_height"]  = sizes3d.m_height;
    json["m_channel"] = sizes3d.m_channel;
}

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Sizes3d &sizes3d)
{
    json.at("m_width").get_to(sizes3d.m_width);
    json.at("m_height").get_to(sizes3d.m_height);
    json.at("m_channel").get_to(sizes3d.m_channel);
}

} // namespace aura