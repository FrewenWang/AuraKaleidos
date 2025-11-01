#ifndef AURA_TOOLS_JSON_SERIALIZE_RUNTIME_ARRAY_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_RUNTIME_ARRAY_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/runtime/array.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/cl_mem.h"

namespace aura
{

/**
 * @brief Define the json serialize method for Mat type reference and pointer.
 *
 */
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Mat *mat);

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Mat *&mat);

AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const Mat &mat);

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, Mat &mat);

/**
 * @brief Define the json serialize method for CLMem type reference and pointer.
 *
 */
#if defined(AURA_ENABLE_OPENCL)
AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const CLMem *cl_mem);

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, CLMem *&cl_mem);

AURA_EXPORTS AURA_VOID to_json(aura_json::json &json, const CLMem &cl_mem);

AURA_EXPORTS AURA_VOID from_json(const aura_json::json &json, CLMem &cl_mem);
#endif

/**
 * @brief Define the json serialize method for element types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ElemType, {
    {ElemType::INVALID,  "INVALID"},
    {ElemType::U8,       "U8"},
    {ElemType::S8,       "S8"},
    {ElemType::U16,      "U16"},
    {ElemType::S16,      "S16"},
    {ElemType::U32,      "U32"},
    {ElemType::S32,      "S32"},
    {ElemType::F32,      "F32"},
    {ElemType::F64,      "F64"},
    {ElemType::F16,      "F16"},
})

/**
 * @brief Define the json serialize method for array types enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ArrayType, {
    {ArrayType::INVALID,    "INVALID"},
    {ArrayType::MAT,        "MAT"},
    {ArrayType::CL_MEMORY,  "CL_MEMORY"},
    {ArrayType::XTENSA_MAT, "XTENSA_MAT"},
})

/**
 * @brief Define the json serialize method for border type enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(BorderType, {
    {BorderType::CONSTANT,    "CONSTANT"},
    {BorderType::REPLICATE,   "REPLICATE"},
    {BorderType::REFLECT_101, "REFLECT_101"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_RUNTIME_ARRAY_HPP__