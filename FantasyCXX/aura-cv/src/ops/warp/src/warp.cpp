#include "aura/ops/warp/warp.hpp"
#include "warp_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
Status InitMapGrid(Context *ctx, const Mat &matrix, Mat &grid, DT_S32 grid_pitch)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (!grid.IsValid() || !matrix.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "grid matrix is invalid");
        return Status::ERROR;
    }

    DT_S32 height = grid.GetSizes().m_height;
    DT_S32 width  = grid.GetSizes().m_width;

    DT_F64 mivt[6];
    InverseMatrix2x3(matrix, mivt);

    Mat map_x = Mat(ctx, ElemType::F32, aura::Sizes3(1, width, 2));
    Mat map_y = Mat(ctx, ElemType::F32, aura::Sizes3(1, height, 2));
    if (!map_x.IsValid() || !map_y.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "map_x or map_y is invalid");
        return Status::ERROR;
    }

    DT_F32 *map_x_row = map_x.Ptr<DT_F32>(0);
    DT_F32 *map_y_row = map_y.Ptr<DT_F32>(0);

    for (DT_S32 x = 0; x < width; x++)
    {
        map_x_row[(x << 1)]     = (x * grid_pitch) * mivt[0];
        map_x_row[(x << 1) + 1] = (x * grid_pitch) * mivt[3];
    }

    for (DT_S32 y = 0; y < height; y++)
    {
        map_y_row[(y << 1)]     = (y * grid_pitch) * mivt[1] + mivt[2];
        map_y_row[(y << 1) + 1] = (y * grid_pitch) * mivt[4] + mivt[5];
    }

    for (DT_S32 y = 0; y < height; y++)
    {
        DT_S32 *grid_data = grid.Ptr<DT_S32>(y);
        DT_F32  x_y       = map_y_row[(y << 1)];
        DT_F32  y_y       = map_y_row[(y << 1) + 1];

        for (DT_S32 x = 0; x < width; x++)
        {
            grid_data[(x << 1)]     = Round((map_x_row[(x << 1)] + x_y) * 1024.0);
            grid_data[(x << 1) + 1] = Round((map_x_row[(x << 1) + 1] + y_y) * 1024.0);
        }
    }

    return Status::OK;
}
#endif

DT_VOID InverseMatrix2x3(const Mat &src, DT_F64 dst[6])
{
    const DT_F64 *src_0 = src.Ptr<DT_F64>(0);
    const DT_F64 *src_1 = src.Ptr<DT_F64>(1);

    DT_F64 d   = src_0[0] * src_1[1] - src_0[1] * src_1[0];
    d          = NearlyEqual(d, 0) ? 0.f : 1.f / d;
    DT_F64 a11 = src_1[1] * d, a22 = src_0[0] * d;

    dst[0] = a11;
    dst[1] = src_0[1] * (-d);
    dst[3] = src_1[0] * (-d);
    dst[4] = a22;
    dst[2] = -dst[0] * src_0[2] - dst[1] * src_1[2];
    dst[5] = -dst[3] * src_0[2] - dst[4] * src_1[2];
}

DT_VOID InverseMatrix3x3(const Mat &src, DT_F64 dst[9])
{
    const DT_F64 *src_0 = src.Ptr<DT_F64>(0);
    const DT_F64 *src_1 = src.Ptr<DT_F64>(1);
    const DT_F64 *src_2 = src.Ptr<DT_F64>(2);

    DT_F64 d = src_0[0] * ((DT_F64)src_1[1] * src_2[2] - (DT_F64)src_1[2] * src_2[1]) -
               src_0[1] * ((DT_F64)src_1[0] * src_2[2] - (DT_F64)src_1[2] * src_2[0]) +
               src_0[2] * ((DT_F64)src_1[0] * src_2[1] - (DT_F64)src_1[1] * src_2[0]);

    d = NearlyEqual(d, 0) ? 0.f : 1.f / d;

    dst[0] = ((DT_F64)src_1[1] * src_2[2] - (DT_F64)src_1[2] * src_2[1]) * d;
    dst[1] = ((DT_F64)src_0[2] * src_2[1] - (DT_F64)src_0[1] * src_2[2]) * d;
    dst[2] = ((DT_F64)src_0[1] * src_1[2] - (DT_F64)src_0[2] * src_1[1]) * d;

    dst[3] = ((DT_F64)src_1[2] * src_2[0] - (DT_F64)src_1[0] * src_2[2]) * d;
    dst[4] = ((DT_F64)src_0[0] * src_2[2] - (DT_F64)src_0[2] * src_2[0]) * d;
    dst[5] = ((DT_F64)src_0[2] * src_1[0] - (DT_F64)src_0[0] * src_1[2]) * d;

    dst[6] = ((DT_F64)src_1[0] * src_2[1] - (DT_F64)src_1[1] * src_2[0]) * d;
    dst[7] = ((DT_F64)src_0[1] * src_2[0] - (DT_F64)src_0[0] * src_2[1]) * d;
    dst[8] = ((DT_F64)src_0[0] * src_1[1] - (DT_F64)src_0[1] * src_1[0]) * d;
}

static std::shared_ptr<WarpImpl> CreateWarpImpl(Context *ctx, WarpType warp_type, const OpTarget &target)
{
    std::shared_ptr<WarpImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new WarpNone(ctx, warp_type, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new WarpNeon(ctx, warp_type, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new WarpCL(ctx, warp_type, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new WarpHvx(ctx, warp_type, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

WarpAffine::WarpAffine(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status WarpAffine::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                           BorderType border_type, const Scalar &border_value)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst) || (DT_NULL == matrix))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst/matrix is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*dst) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*dst) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateWarpImpl(m_ctx, WarpType::AFFINE, impl_target);
    }

    // run initialize
    WarpImpl *warp_impl = dynamic_cast<WarpImpl*>(m_impl.get());
    if (DT_NULL == warp_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "warp_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = warp_impl->SetArgs(src, matrix, dst, interp_type, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status WarpAffine::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, BorderType border_type, InterpType interp_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = WarpCL::GetCLKernels(ctx, elem_type, channel, border_type, WarpType::AFFINE, interp_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "WarpAffine CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(border_type);
    AURA_UNUSED(interp_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IWarpAffine(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst, InterpType interp_type,
                                BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    WarpAffine warp(ctx, target);

    return OpCall(ctx, warp, &src, &matrix, &dst, interp_type, border_type, border_value);
}

WarpPerspective::WarpPerspective(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status WarpPerspective::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                                BorderType border_type, const Scalar &border_value)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst) || (DT_NULL == matrix))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst/matrix is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*dst) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateWarpImpl(m_ctx, WarpType::PERSPECTIVE, impl_target);
    }

    // run initialize
    WarpImpl *warp_impl = dynamic_cast<WarpImpl*>(m_impl.get());
    if (DT_NULL == warp_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "warp_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = warp_impl->SetArgs(src, matrix, dst, interp_type, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status WarpPerspective::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 channel, BorderType border_type, InterpType interp_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = WarpCL::GetCLKernels(ctx, elem_type, channel, border_type, WarpType::PERSPECTIVE, interp_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "WarpPerspective CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(border_type);
    AURA_UNUSED(interp_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IWarpPerspective(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst, InterpType interp_type,
                                     BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    WarpPerspective warp(ctx, target);

    return OpCall(ctx, warp, &src, &matrix, &dst, interp_type, border_type, border_value);
}

WarpImpl::WarpImpl(Context *ctx, WarpType warp_type, const OpTarget &target) : OpImpl(ctx, "Warp", target),
                                                                               m_src(DT_NULL), m_matrix(DT_NULL), m_dst(DT_NULL),
                                                                               m_warp_type(warp_type), m_interp_type(InterpType::LINEAR),
                                                                               m_border_type(BorderType::REPLICATE)
{}

Status WarpImpl::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                         BorderType border_type, const Scalar &border_value)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src->IsValid() || !matrix->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input mat is invalid.");
        return Status::ERROR;
    }

    if ((src->GetElemType() != dst->GetElemType()) || (!src->IsChannelEqual(*dst)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same elem type/channels");
        return Status::ERROR;
    }

    m_src          = src;
    m_matrix       = matrix;
    m_dst          = dst;
    m_interp_type  = interp_type;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> WarpImpl::GetInputArrays() const
{
    return {m_src, m_matrix};
}

std::vector<const Array*> WarpImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string WarpImpl::ToString() const
{
    std::string str;

    str = "op(Warp)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + WarpTypeToString(m_warp_type) + " | " +
           InterpTypeToString(m_interp_type) + " | " +
           BorderTypeToString(m_border_type) + " | " +
           "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

DT_VOID WarpImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "matrix", "dst"};
    std::vector<const Array*> arrays = {m_src, m_matrix, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_matrix, m_dst, m_warp_type, m_interp_type, m_border_type, m_border_value);
}

Status WarpImpl::InitMapOffset(Context *ctx, const Mat &matrix, Mat &map_x, Mat &map_y, WarpType warp_type)
{
    if (!matrix.IsValid() || !map_x.IsValid() || !map_y.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "matrix map_x map_y is invalid");
        return Status::ERROR;
    }

    DT_F32 *map_x_row = map_x.Ptr<DT_F32>(0);
    DT_F32 *map_y_row = map_y.Ptr<DT_F32>(0);

    if (WarpType::AFFINE == warp_type)
    {
        DT_F64 mivt[6];
        InverseMatrix2x3(matrix, mivt);

        for (DT_S32 x = 0; x < map_x.GetSizes().m_width; x++)
        {
            map_x_row[(x << 1)]     = x * mivt[0];
            map_x_row[(x << 1) + 1] = x * mivt[3];
        }
        for (DT_S32 y = 0; y < map_y.GetSizes().m_width; y++)
        {
            map_y_row[(y << 1)]     = y * mivt[1] + mivt[2];
            map_y_row[(y << 1) + 1] = y * mivt[4] + mivt[5];
        }
    }
    else if (WarpType::PERSPECTIVE == warp_type)
    {
        DT_F64 mivt[9];
        InverseMatrix3x3(matrix, mivt);

        for (DT_S32 x = 0; x < map_x.GetSizes().m_width; x++)
        {
            map_x_row[x * 3]     = x * mivt[0];
            map_x_row[x * 3 + 1] = x * mivt[3];
            map_x_row[x * 3 + 2] = x * mivt[6];
        }
        for (DT_S32 y = 0; y < map_y.GetSizes().m_width; y++)
        {
            map_y_row[y * 3]     = y * mivt[1] + mivt[2];
            map_y_row[y * 3 + 1] = y * mivt[4] + mivt[5];
            map_y_row[y * 3 + 2] = y * mivt[7] + mivt[8];
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported warp type");
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_EXPORTS Mat GetAffineTransform(Context *ctx, const std::vector<Point2> &src,
                                    const std::vector<Point2> &dst)
{
    if (DT_NULL == ctx)
    {
        return Mat();
    }

    if (src.size() != 3 || dst.size() != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "the number of src points and dst points should be 3");
        return Mat();
    }

    DT_F64 det = src[0].m_x * ((DT_F64)src[1].m_y - src[2].m_y) -
                 src[0].m_y * ((DT_F64)src[1].m_x - src[2].m_x) +
                 ((DT_F64)src[1].m_x * src[2].m_y - (DT_F64)src[1].m_y * src[2].m_x);

    if (NearlyEqual(det, (DT_F64)0))
    {
        AURA_ADD_ERROR_STRING(ctx, "determinant of src is 0");
        return Mat();
    }

    Mat matrix(ctx, ElemType::F64, Sizes3(2, 3), AURA_MEM_HEAP);
    if (!matrix.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create Mat failed!");
        return Mat();
    }

    det = 1. / det;

    matrix.At<DT_F64>(0, 0) = (((DT_F64)src[1].m_y - (DT_F64)src[2].m_y) * dst[0].m_x +
                               ((DT_F64)src[2].m_y - (DT_F64)src[0].m_y) * dst[1].m_x +
                               ((DT_F64)src[0].m_y - (DT_F64)src[1].m_y) * dst[2].m_x) * det;

    matrix.At<DT_F64>(0, 1) = (((DT_F64)src[2].m_x - (DT_F64)src[1].m_x) * dst[0].m_x +
                               ((DT_F64)src[0].m_x - (DT_F64)src[2].m_x) * dst[1].m_x +
                               ((DT_F64)src[1].m_x - (DT_F64)src[0].m_x) * dst[2].m_x) * det;

    matrix.At<DT_F64>(0, 2) = (((DT_F64)src[1].m_x * src[2].m_y - (DT_F64)src[1].m_y * src[2].m_x) * dst[0].m_x +
                               ((DT_F64)src[0].m_y * src[2].m_x - (DT_F64)src[0].m_x * src[2].m_y) * dst[1].m_x +
                               ((DT_F64)src[0].m_x * src[1].m_y - (DT_F64)src[0].m_y * src[1].m_x) * dst[2].m_x) * det;

    matrix.At<DT_F64>(1, 0) = (((DT_F64)src[1].m_y - (DT_F64)src[2].m_y) * dst[0].m_y +
                               ((DT_F64)src[2].m_y - (DT_F64)src[0].m_y) * dst[1].m_y +
                               ((DT_F64)src[0].m_y - (DT_F64)src[1].m_y) * dst[2].m_y) * det;

    matrix.At<DT_F64>(1, 1) = (((DT_F64)src[2].m_x - (DT_F64)src[1].m_x) * dst[0].m_y +
                               ((DT_F64)src[0].m_x - (DT_F64)src[2].m_x) * dst[1].m_y +
                               ((DT_F64)src[1].m_x - (DT_F64)src[0].m_x) * dst[2].m_y) * det;

    matrix.At<DT_F64>(1, 2) = (((DT_F64)src[1].m_x * src[2].m_y - (DT_F64)src[1].m_y * src[2].m_x) * dst[0].m_y +
                               ((DT_F64)src[0].m_y * src[2].m_x - (DT_F64)src[0].m_x * src[2].m_y) * dst[1].m_y +
                               ((DT_F64)src[0].m_x * src[1].m_y - (DT_F64)src[0].m_y * src[1].m_x) * dst[2].m_y) * det;

    return matrix;
}

static Status SolveLU(Context *ctx, DT_F64 a[8][8], DT_F64 b[8])
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < 8; i++)
    {
        DT_S32 k = i;
        for (DT_S32 j = i + 1; j < 8; j++)
        {
            if (Abs(a[j][i]) > Abs(a[k][i]))
            {
                k = j;
            }
        }

        if (Abs(a[k][i]) < 1e-6)
        {
            AURA_ADD_ERROR_STRING(ctx, "LU can not be solved");
            return Status::ERROR;
        }

        if (k != i)
        {
            for (DT_S32 j = i; j < 8; j++)
            {
                Swap(a[i][j], a[k][j]);
            }
            Swap(b[i], b[k]);
        }

        DT_F64 d = -1 / a[i][i];
        for (DT_S32 j = i + 1; j < 8; j++)
        {
            DT_F64 alpha = a[j][i] * d;
            for (DT_S32 k = i + 1; k < 8; k++)
            {
                a[j][k] += alpha * a[i][k];
            }
            b[j] += alpha * b[i];
        }
    }

    for (DT_S32 i = 7; i >= 0; i--)
    {
        DT_F64 s = b[i];
        for (DT_S32 k = i + 1; k < 8; k++)
        {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }

    return Status::OK;
}

AURA_EXPORTS Mat GetPerspectiveTransform(Context *ctx, const std::vector<Point2> &src,
                                         const std::vector<Point2> &dst)
{
    if (DT_NULL == ctx)
    {
        return Mat();
    }

    if (src.size() != 4 || dst.size() != 4)
    {
        AURA_ADD_ERROR_STRING(ctx, "the number of src points and dst points should be 4");
        return Mat();
    }

    Mat matrix(ctx, ElemType::F64, Sizes3(3, 3, 1), AURA_MEM_HEAP);
    if (!matrix.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create Mat failed!");
        return Mat();
    }

    DT_F64 a[8][8];
    DT_F64 b[8];

    for (DT_S32 i = 0; i < 4; i++)
    {
        a[i][0] = a[i + 4][3] = src[i].m_x;
        a[i][1] = a[i + 4][4] = src[i].m_y;
        a[i][2] = a[i + 4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;

        a[i][6]     = -src[i].m_x * dst[i].m_x;
        a[i][7]     = -src[i].m_y * dst[i].m_x;
        a[i + 4][6] = -src[i].m_x * dst[i].m_y;
        a[i + 4][7] = -src[i].m_y * dst[i].m_y;
        b[i]        = dst[i].m_x;
        b[i + 4]    = dst[i].m_y;
    }

    Status ret = SolveLU(ctx, a, b);
    if (Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(ctx, "SolveLU error, please shoose another 4 points");
        return Mat();
    }

    matrix.At<DT_F64>(0, 0) = b[0];
    matrix.At<DT_F64>(0, 1) = b[1];
    matrix.At<DT_F64>(0, 2) = b[2];
    matrix.At<DT_F64>(1, 0) = b[3];
    matrix.At<DT_F64>(1, 1) = b[4];
    matrix.At<DT_F64>(1, 2) = b[5];
    matrix.At<DT_F64>(2, 0) = b[6];
    matrix.At<DT_F64>(2, 1) = b[7];
    matrix.At<DT_F64>(2, 2) = 1;

    return matrix;
}

AURA_EXPORTS Mat GetRotationMatrix2D(Context *ctx, const Point2 &center, DT_F64 angle, DT_F64 scale)
{
    if (DT_NULL == ctx)
    {
        return Mat();
    }

    Mat matrix(ctx, ElemType::F64, Sizes3(2, 3, 1), AURA_MEM_HEAP);
    if (!matrix.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create Mat failed!");
        return Mat();
    }

    angle *= AURA_PI / 180;
    DT_F64 alpha = Cos(angle) * scale;
    DT_F64 beta  = Sin(angle) * scale;

    matrix.At<DT_F64>(0, 0) = alpha;
    matrix.At<DT_F64>(0, 1) = beta;
    matrix.At<DT_F64>(0, 2) = (1 - alpha) * center.m_x - beta * center.m_y;
    matrix.At<DT_F64>(1, 0) = -beta;
    matrix.At<DT_F64>(1, 1) = alpha;
    matrix.At<DT_F64>(1, 2) = beta * center.m_x + (1 - alpha) * center.m_y;

    return matrix;
}

AURA_EXPORTS Status WarpCoord(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type, const OpTarget &target)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (!(matrix.IsValid() && map_xy.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "matrix or map_xy is invalid");
        return Status::ERROR;
    }

    if (map_xy.GetElemType() != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(ctx, "map_xy must be S16 elemtype");
        return Status::ERROR;
    }

    if (map_xy.GetSizes().m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(ctx, "map_xy channel must be 2");
        return Status::ERROR;
    }

    if ((warp_type == WarpType::AFFINE && (matrix.GetSizes().m_height != 2 || matrix.GetSizes().m_width != 3)) ||
        (warp_type == WarpType::PERSPECTIVE && (matrix.GetSizes().m_height != 3 || matrix.GetSizes().m_width != 3)))
    {
        AURA_ADD_ERROR_STRING(ctx, "the size of transform matrix is invalid");
        return Status::ERROR;
    }

    if (matrix.GetElemType() != ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(ctx, "the type of transform matrix should be f64");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            ret = WarpCoordNone(ctx, matrix, map_xy, warp_type);
            break;
        }

#if defined(AURA_ENABLE_HEXAGON)
        case TargetType::HVX:
        {
            ret = WarpCoordHvx(ctx, matrix, map_xy, warp_type, target);
            break;
        }
#endif // defined(AURA_ENABLE_HEXAGON)

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "target type error");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura