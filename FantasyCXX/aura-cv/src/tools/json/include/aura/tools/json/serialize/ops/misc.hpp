#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_MISC_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_MISC_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/misc.h"

namespace aura
{

/**
 * @brief Define the json serialize method for adaptive threshold enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(AdaptiveThresholdMethod, {
    {AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C,     "ADAPTIVE_THRESH_MEAN_C"},
    {AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C, "ADAPTIVE_THRESH_GAUSSIAN_C"},
})

/**
 * @brief Define the json serialize method for connectivity enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ConnectivityType, {
    {ConnectivityType::CROSS,    "cross"},
    {ConnectivityType::SQUARE,   "square"},
    {ConnectivityType::DIAGONAL, "diagonal"},
    {ConnectivityType::CUBE,     "cube"},
})

/**
 * @brief Define the json serialize method for equivalence solvers enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(EquivalenceSolver, {
    {EquivalenceSolver::UNION_FIND,               "UnionFindSolver"},
    {EquivalenceSolver::UNION_FIND_PATH_COMPRESS, "UFPCSolver"},
    {EquivalenceSolver::REM_SPLICING,             "RemSpliceSolver"},
    {EquivalenceSolver::THREE_TABLE_ARRAYS,       "TTASolver"},
})

/**
 * @brief Define the json serialize method for connected component labeling(CCL) algorithms enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(CCLAlgo, {
    {CCLAlgo::SAUF,      "SAUF"},
    {CCLAlgo::BBDT,      "BBDT"},
    {CCLAlgo::SPAGHETTI, "SPAGHETTI"},
    {CCLAlgo::HA_GPU,    "HA"},
})

/**
 * @brief Define the json serialize method for contours detection methods enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ContoursMethod, {
    {ContoursMethod::CHAIN_APPROX_NONE,   "CHAIN_APPROX_NONE"},
    {ContoursMethod::CHAIN_APPROX_SIMPLE, "CHAIN_APPROX_SIMPLE"},
})

/**
 * @brief Define the json serialize method for contours detection modes enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(ContoursMode, {
    {ContoursMode::RETR_EXTERNAL, "RETR_EXTERNAL"},
    {ContoursMode::RETR_LIST,     "RETR_LIST"},
})

/**
 * @brief Define the json serialize method for hough circles enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(HoughCirclesMethod, {
    {HoughCirclesMethod::HOUGH_GRADIENT, "HOUGH_GRADIENT"},
})

/**
 * @brief Define the json serialize method for hough lines enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(LinesType, {
    {LinesType::VEC2F, "vec2f"},
    {LinesType::VEC3F, "vec3f"},
})


} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_MISC_HPP__