/** @brief      : warp impl for aura
 *  @file       : warp_impl.hpp
 *  @author     : zhangjilong@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Nov. 14, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_WARP_WARP_IMPL_HPP__
#define AURA_OPS_WARP_WARP_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif
#include "aura/tools/json/json.hpp"

namespace aura
{

enum class WarpType
{
    INVALID = 0,
    AFFINE,      /*!< Affine Transformation */
    PERSPECTIVE, /*!< Perspective Transformation */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, WarpType warp_type)
{
    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            os << "Affine";
            break;
        }

        case WarpType::PERSPECTIVE:
        {
            os << "Perspective";
            break;
        }

        default:
        {
            os << "undefined warp type";
            break;
        }
    }

    return os;
}

AURA_INLINE const std::string WarpTypeToString(WarpType type)
{
    std::ostringstream ss;
    ss << type;
    return ss.str();
}

AURA_JSON_SERIALIZE_ENUM(WarpType, {
    {WarpType::INVALID,     "Invalid"},
    {WarpType::AFFINE,      "Affine"},
    {WarpType::PERSPECTIVE, "Perspective"},
})

DT_VOID InverseMatrix2x3(const Mat &src, DT_F64 dst[6]);
DT_VOID InverseMatrix3x3(const Mat &src, DT_F64 dst[9]);

class WarpImpl : public OpImpl
{
public:
    WarpImpl(Context *ctx, WarpType warp_type, const OpTarget &target);

    virtual Status SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type = InterpType::LINEAR,
                           BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    Status InitMapOffset(Context *ctx, const Mat &matrix, Mat &map_x, Mat &map_y, WarpType warp_type);

protected:
    const Array *m_src;
    const Array *m_matrix;
    Array       *m_dst;

    WarpType   m_warp_type;
    InterpType m_interp_type;
    BorderType m_border_type;
    Scalar     m_border_value;
};

class WarpNone : public WarpImpl
{
public:
    WarpNone(Context *ctx, WarpType warp_type, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_map_x;
    Mat m_map_y;
};

#if defined(AURA_ENABLE_NEON)
class WarpNeon : public WarpImpl
{
public:
    WarpNeon(Context *ctx, WarpType warp_type, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_map_x;
    Mat m_map_y;
};
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class WarpCL : public WarpImpl
{
public:
    WarpCL(Context *ctx, WarpType warp_type, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel,
                                              BorderType border_type, WarpType warp_type, InterpType interp_type);
private:
    std::vector<CLKernel> m_cl_kernels;
    Mat      m_map_x;
    Mat      m_map_y;
    CLMem    m_cl_src;
    CLMem    m_cl_dst;
    CLMem    m_cl_map_x;
    CLMem    m_cl_map_y;
    DT_S32   m_elem_counts;
    DT_S32   m_elem_height;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class WarpHvx : public WarpImpl
{
public:
    WarpHvx(Context *ctx, WarpType warp_type, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

Status InitMapGrid(Context *ctx, const Mat &matrix, Mat &grid, DT_S32 grid_pitch);

#  if defined(AURA_BUILD_HEXAGON)
Status WarpAffineHvx(Context *ctx, const Mat &src, const Mat &grid, Mat &dst, InterpType interp_type,
                     BorderType border_type, Scalar &border_value);
#  endif // defined(AURA_BUILD_HEXAGON)

using WarpInParam = HexagonRpcParamType<Mat, Mat, Mat, WarpType, InterpType, BorderType, Scalar>;
#  define AURA_OPS_WARP_OP_NAME      "Warp"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

/* ======================================================== Coord ======================================================== */

AURA_EXPORTS Status WarpCoord(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type, const OpTarget &target = OpTarget::Default());

Status WarpCoordNone(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type);

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  if defined(AURA_ENABLE_HEXAGON)
Status WarpCoordHvx(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type, const OpTarget &target);
#  endif // defined(AURA_ENABLE_HEXAGON)

using WarpCoordInParam = HexagonRpcParamType<Mat, Mat, WarpType>;
#  define AURA_OPS_WARP_COORD_OP_NAME      "WarpCoord"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

#if defined(AURA_BUILD_HEXAGON)
Status WarpAffineCoordHvx(Context *ctx, const Mat &grid, Mat &coord);
#endif // defined(AURA_BUILD_HEXAGON)

} // namespace aura

#endif // AURA_OPS_WARP_REMAP_IMPL_HPP__