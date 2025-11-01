#include "warp_impl.hpp"
#include "aura/ops/warp/remap.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/thread_object.h"
#include "aura/runtime/logger.h"

#define BLOCK_SZ                (64)

namespace aura
{

template <typename Tp, typename std::enable_if<(sizeof(MI_S16) == sizeof(Tp))>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorStore(float32x4x2_t v2qf32_map_xy, int16x4x2_t v2ds16_map_xy, Tp *map_row)
{
    v2ds16_map_xy.val[0] = neon::vqmovn(neon::vcvt<MI_S32>(neon::vrndn(v2qf32_map_xy.val[0])));
    v2ds16_map_xy.val[1] = neon::vqmovn(neon::vcvt<MI_S32>(neon::vrndn(v2qf32_map_xy.val[1])));
    neon::vstore(map_row, v2ds16_map_xy);
}

template <typename Tp, typename std::enable_if<(sizeof(MI_F32) == sizeof(Tp))>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorStore(float32x4x2_t v2qf32_map_xy, int16x4x2_t v2ds16_map_xy, Tp *map_row)
{
    AURA_UNUSED(v2ds16_map_xy);
    neon::vstore(map_row, v2qf32_map_xy);
}

template <typename Tp, WarpType WARP_TYPE, typename std::enable_if<WarpType::AFFINE == WARP_TYPE>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID WarpBlockCore(MI_F32 *map_x_row, MI_F32 *map_y_row, Mat &map, MI_S32 bh, MI_S32 bw)
{
    float32x4_t   vqf32_x_y, vqf32_y_y;
    float32x4x2_t v2qf32_map_xy;
    int16x4x2_t   v2ds16_map_xy;

    MI_S32 bw_align = bw & (-4);

    for (MI_S32 y1 = 0; y1 < bh; y1++)
    {
        Tp *map_row = map.Ptr<Tp>(y1);
        neon::vdup(vqf32_x_y, map_y_row[(y1 << 1)]);
        neon::vdup(vqf32_y_y, map_y_row[(y1 << 1) + 1]);

        // middle
        for (MI_S32 x1 = 0; x1 < bw_align; x1 += 4)
        {
            neon::vload(map_x_row + (x1 << 1), v2qf32_map_xy);
            v2qf32_map_xy.val[0] = neon::vadd(v2qf32_map_xy.val[0], vqf32_x_y);
            v2qf32_map_xy.val[1] = neon::vadd(v2qf32_map_xy.val[1], vqf32_y_y);

            VectorStore<Tp>(v2qf32_map_xy, v2ds16_map_xy, map_row + (x1 << 1));
        }

        // right border
        for (MI_S32 x1 = bw_align; x1 < bw; x1++)
        {
            map_row[(x1 << 1)]     = SaturateCast<Tp>(map_x_row[(x1 << 1)] + map_y_row[(y1 << 1)]);
            map_row[(x1 << 1) + 1] = SaturateCast<Tp>(map_x_row[(x1 << 1) + 1] + map_y_row[(y1 << 1) + 1]);
        }
    }
}

template <typename Tp, WarpType WARP_TYPE, typename std::enable_if<WarpType::PERSPECTIVE == WARP_TYPE>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID WarpBlockCore(MI_F32 *map_x_row, MI_F32 *map_y_row, Mat &map, MI_S32 bh, MI_S32 bw)
{
    float32x4_t   vqf32_x_y, vqf32_y_y, vqf32_w_y;
    float32x4_t   vqf32_w, vqf32_w_recp, vqf32_zero;
    float32x4x3_t v3qf32_x;
    uint32x4_t    vqu32_is_zero;
    float32x4x2_t v2qf32_map_xy;
    int16x4x2_t   v2ds16_map_xy;

    neon::vdup(vqf32_zero, (MI_F32)0);
    MI_S32 bw_align = bw & (-4);

    for (MI_S32 y1 = 0; y1 < bh; y1++)
    {
        Tp *map_row = map.Ptr<Tp>(y1);
        neon::vdup(vqf32_x_y, map_y_row[y1 * 3 + 0]);
        neon::vdup(vqf32_y_y, map_y_row[y1 * 3 + 1]);
        neon::vdup(vqf32_w_y, map_y_row[y1 * 3 + 2]);

        // middle
        for (MI_S32 x1 = 0; x1 < bw_align; x1 += 4)
        {
            neon::vload(map_x_row + x1 * 3, v3qf32_x);
            vqf32_w       = neon::vadd(v3qf32_x.val[2], vqf32_w_y);
            vqf32_w_recp  = neon::vreciprocal_newton(vqf32_w);
            vqu32_is_zero = neon::vceq(vqf32_w, vqf32_zero);
            vqf32_w_recp  = neon::vbsl(vqu32_is_zero, vqf32_zero, vqf32_w_recp);

            v2qf32_map_xy.val[0] = neon::vmul(neon::vadd(v3qf32_x.val[0], vqf32_x_y), vqf32_w_recp);
            v2qf32_map_xy.val[1] = neon::vmul(neon::vadd(v3qf32_x.val[1], vqf32_y_y), vqf32_w_recp);

            VectorStore<Tp>(v2qf32_map_xy, v2ds16_map_xy, map_row + (x1 << 1));
        }

        // right border
        for (MI_S32 x1 = bw_align; x1 < bw; x1++)
        {
            MI_F32 w = map_x_row[x1 * 3 + 2] + map_y_row[y1 * 3 + 2];
            w        = NearlyEqual(w, (MI_F32)0.f) ? 0 : 1. / w;

            map_row[(x1 << 1)]     = SaturateCast<Tp>((map_x_row[x1 * 3] + map_y_row[y1 * 3]) * w);
            map_row[(x1 << 1) + 1] = SaturateCast<Tp>((map_x_row[x1 * 3 + 1] + map_y_row[y1 * 3 + 1]) * w);
        }
    }
}

template <typename Tp, WarpType WARP_TYPE>
static Status WarpNeonBlock(Context *ctx, const Mat &src, Mat &dst, Mat &map_x, Mat &map_y,
                            InterpType interp_type, BorderType border_type, const Scalar &border_value,
                            ThreadObject<Remap> &share_remap, ThreadBuffer &thread_buffer,
                            MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 bh0   = Min(BLOCK_SZ / 2, end_row - start_row);
    MI_S32 bw0   = Min(BLOCK_SZ * BLOCK_SZ / bh0, width);
    bh0          = Min(BLOCK_SZ * BLOCK_SZ / bw0, end_row - start_row);

    MI_S32  xy_channel = map_x.GetSizes().m_channel;
    MI_F32 *map_x_row  = map_x.Ptr<MI_F32>(0);
    MI_F32 *map_y_row  = map_y.Ptr<MI_F32>(0);

    Remap *remap = share_remap.GetObject();
    if (MI_NULL == remap)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get remap failed");
        return Status::ERROR;
    }

    Buffer map_buffer = thread_buffer.GetThreadBuffer();

    if (!map_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }
    Mat map(ctx, GetElemType<Tp>(), Sizes3(bh0, bw0, 2), map_buffer);

    Status ret = Status::ERROR;

    for (MI_S32 y = start_row; y < end_row; y += bh0)
    {
        for (MI_S32 x = 0; x < width; x += bw0)
        {
            MI_S32 bh = Min(bh0, end_row - y);
            MI_S32 bw = Min(bw0, width - x);

            WarpBlockCore<Tp, WARP_TYPE>(map_x_row + (x * xy_channel), map_y_row + (y * xy_channel), map, bh, bw);

            Mat blk_map = map.Roi(Rect(0, 0, bw, bh));
            Mat blk_dst = dst.Roi(Rect(x, y, bw, bh));

            if ((ret = OpCall(ctx, *remap, &src, &blk_map, &blk_dst, interp_type, border_type, border_value)) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "OpCall remap failed");
                goto EXIT;
            }
        }
    }

EXIT:

    AURA_RETURN(ctx, ret);
}

WarpNeon::WarpNeon(Context *ctx, WarpType warp_type, const OpTarget &target) : WarpImpl(ctx, warp_type, target)
{}

Status WarpNeon::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                         BorderType border_type, const Scalar &border_value)
{
    if (WarpImpl::SetArgs(src, matrix, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (matrix->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src matrix dst must be mat type");
        return Status::ERROR;
    }

    MI_S32 channel = src->GetSizes().m_channel;
    if (channel != 1 && channel != 2 && channel != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    if ((m_warp_type == WarpType::AFFINE && (matrix->GetSizes().m_height != 2 || matrix->GetSizes().m_width != 3)) ||
        (m_warp_type == WarpType::PERSPECTIVE && (matrix->GetSizes().m_height != 3 || matrix->GetSizes().m_width != 3)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the size of transform matrix is invalid");
        return Status::ERROR;
    }

    if (matrix->GetElemType() != ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the type of transform matrix should be f64");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpNeon::Initialize()
{
    if (WarpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::Initialize failed");
        return Status::ERROR;
    }

    const Mat *matrix = dynamic_cast<const Mat*>(m_matrix);
    if (MI_NULL == matrix)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "matrix is not mat");
        return Status::ERROR;
    }

    MI_S32 height  = m_dst->GetSizes().m_height;
    MI_S32 width   = m_dst->GetSizes().m_width;
    MI_S32 channel = (WarpType::AFFINE == m_warp_type) ? 2 : 3;

    m_map_x = Mat(m_ctx, ElemType::F32, Sizes3(1, width, channel));
    m_map_y = Mat(m_ctx, ElemType::F32, Sizes3(1, height, channel));

    return InitMapOffset(m_ctx, *matrix, m_map_x, m_map_y, m_warp_type);
}

Status WarpNeon::DeInitialize()
{
    m_map_x.Release();
    m_map_y.Release();

    if (WarpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    MI_S32 height  = dst->GetSizes().m_height;

    WorkerPool *wp = m_ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    auto   thread_ids = wp->GetComputeThreadIDs();
    MI_S32 elem_size  = (m_interp_type == InterpType::NEAREST) ? sizeof(MI_S16) : sizeof(MI_F32);

    ThreadObject<Remap> share_remap(m_ctx, thread_ids, OpTarget::None());
    ThreadBuffer thread_buffer(m_ctx, 2 * elem_size * BLOCK_SZ * BLOCK_SZ);

    Status ret     = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(elem_size, m_warp_type);

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(sizeof(MI_S16), WarpType::AFFINE):
        {
            ret = wp->ParallelFor(0, height, WarpNeonBlock<MI_S16, WarpType::AFFINE>, m_ctx, std::cref(*src), std::ref(*dst), std::ref(m_map_x), std::ref(m_map_y),
                                  m_interp_type, m_border_type, std::cref(m_border_value), std::ref(share_remap), std::ref(thread_buffer));
            break;
        }

        case AURA_MAKE_PATTERN(sizeof(MI_F32), WarpType::AFFINE):
        {
            ret = wp->ParallelFor(0, height, WarpNeonBlock<MI_F32, WarpType::AFFINE>, m_ctx, std::cref(*src), std::ref(*dst), std::ref(m_map_x), std::ref(m_map_y),
                                  m_interp_type, m_border_type, std::cref(m_border_value), std::ref(share_remap), std::ref(thread_buffer));
            break;
        }

        case AURA_MAKE_PATTERN(sizeof(MI_S16), WarpType::PERSPECTIVE):
        {
            ret = wp->ParallelFor(0, height, WarpNeonBlock<MI_S16, WarpType::PERSPECTIVE>, m_ctx, std::cref(*src), std::ref(*dst), std::ref(m_map_x), std::ref(m_map_y),
                                  m_interp_type, m_border_type, std::cref(m_border_value), std::ref(share_remap), std::ref(thread_buffer));
            break;
        }

        case AURA_MAKE_PATTERN(sizeof(MI_F32), WarpType::PERSPECTIVE):
        {
            ret = wp->ParallelFor(0, height, WarpNeonBlock<MI_F32, WarpType::PERSPECTIVE>, m_ctx, std::cref(*src), std::ref(*dst), std::ref(m_map_x), std::ref(m_map_y),
                                  m_interp_type, m_border_type, std::cref(m_border_value), std::ref(share_remap), std::ref(thread_buffer));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported BorderType");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura