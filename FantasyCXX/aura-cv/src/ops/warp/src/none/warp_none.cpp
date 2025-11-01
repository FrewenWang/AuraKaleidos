#include "warp_impl.hpp"
#include "aura/ops/warp/remap.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/thread_object.h"
#include "aura/runtime/logger.h"

#define BLOCK_SZ                (64)

namespace aura
{

template <typename Tp, WarpType WARP_TYPE, typename std::enable_if<WarpType::AFFINE == WARP_TYPE>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID WarpBlockCore(MI_F32 *map_x_row, MI_F32 *map_y_row, Mat &map, MI_S32 bh, MI_S32 bw)
{
    for (MI_S32 y1 = 0; y1 < bh; y1++)
    {
        Tp *map_row = map.Ptr<Tp>(y1);
        for (MI_S32 x1 = 0; x1 < bw; x1++)
        {
            map_row[(x1 << 1)]     = SaturateCast<Tp>(map_x_row[(x1 << 1)] + map_y_row[(y1 << 1)]);
            map_row[(x1 << 1) + 1] = SaturateCast<Tp>(map_x_row[(x1 << 1) + 1] + map_y_row[(y1 << 1) + 1]);
        }
    }
}

template <typename Tp, WarpType WARP_TYPE, typename std::enable_if<WarpType::PERSPECTIVE == WARP_TYPE>::type * = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID WarpBlockCore(MI_F32 *map_x_row, MI_F32 *map_y_row, Mat &map, MI_S32 bh, MI_S32 bw)
{
    for (MI_S32 y1 = 0; y1 < bh; y1++)
    {
        Tp *map_row = map.Ptr<Tp>(y1);
        for (MI_S32 x1 = 0; x1 < bw; x1++)
        {
            MI_F32 w = map_x_row[x1 * 3 + 2] + map_y_row[y1 * 3 + 2];
            w        = NearlyEqual(w, (MI_F32)0.f) ? 0 : 1. / w;

            map_row[(x1 << 1)]     = SaturateCast<Tp>((map_x_row[x1 * 3] + map_y_row[y1 * 3]) * w);
            map_row[(x1 << 1) + 1] = SaturateCast<Tp>((map_x_row[x1 * 3 + 1] + map_y_row[y1 * 3 + 1]) * w);
        }
    }
}

template <typename Tp, WarpType WARP_TYPE>
static Status WarpNoneBlock(Context *ctx, const Mat &src, Mat &dst, Mat &map_x, Mat &map_y,
                            InterpType interp_type, BorderType border_type, const Scalar &border_value,
                            ThreadObject<Remap> &share_remap, ThreadBuffer &thread_buffer,
                            MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 bh0   = Min((MI_S32)(BLOCK_SZ / 2), end_row - start_row);
    MI_S32 bw0   = Min((MI_S32)(BLOCK_SZ * BLOCK_SZ / bh0), width);
    bh0          = Min((MI_S32)(BLOCK_SZ * BLOCK_SZ / bw0), end_row - start_row);

    MI_S32 xy_channel = map_x.GetSizes().m_channel;
    MI_F32 *map_x_row = map_x.Ptr<MI_F32>(0);
    MI_F32 *map_y_row = map_y.Ptr<MI_F32>(0);

    Remap *remap = share_remap.GetObject();
    if (MI_NULL == remap)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get remap failed");
        return ret;
    }

    Buffer map_buffer = thread_buffer.GetThreadBuffer();

    if (!map_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return ret;
    }
    Mat map(ctx, GetElemType<Tp>(), Sizes3(bh0, bw0, 2), map_buffer);

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

WarpNone::WarpNone(Context *ctx, WarpType warp_type, const OpTarget &target) : WarpImpl(ctx, warp_type, target)
{}

Status WarpNone::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                         BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    if (WarpImpl::SetArgs(src, matrix, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::SetArgs failed");
        return ret;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (matrix->GetArrayType() != ArrayType::MAT) ||
        (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src matrix dst must be mat type");
        return ret;
    }

    MI_S32 channel = src->GetSizes().m_channel;
    if (channel != 1 && channel != 2 && channel != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return ret;
    }

    if ((m_warp_type == WarpType::AFFINE && (matrix->GetSizes().m_height != 2 || matrix->GetSizes().m_width != 3)) ||
        (m_warp_type == WarpType::PERSPECTIVE && (matrix->GetSizes().m_height != 3 || matrix->GetSizes().m_width != 3)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the size of transform matrix is invalid");
        return ret;
    }

    if (matrix->GetElemType() != ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the type of transform matrix should be f64");
        return ret;
    }

    return Status::OK;
}

Status WarpNone::Initialize()
{
    Status ret = Status::ERROR;

    if (WarpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::Initialize failed");
        return ret;
    }

    const Mat *matrix = dynamic_cast<const Mat*>(m_matrix);
    if (MI_NULL == matrix)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "matrix is not mat");
        return ret;
    }

    MI_S32 height  = m_dst->GetSizes().m_height;
    MI_S32 width   = m_dst->GetSizes().m_width;
    MI_S32 channel = (WarpType::AFFINE == m_warp_type) ? 2 : 3;

    m_map_x = Mat(m_ctx, ElemType::F32, Sizes3(1, width, channel));
    m_map_y = Mat(m_ctx, ElemType::F32, Sizes3(1, height, channel));

    return InitMapOffset(m_ctx, *matrix, m_map_x, m_map_y, m_warp_type);
}

Status WarpNone::DeInitialize()
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

Status WarpNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    MI_S32 height = dst->GetSizes().m_height;

    MI_S32 elem_size = (m_interp_type == InterpType::NEAREST) ? sizeof(MI_S16) : sizeof(MI_F32);

    MI_S32 pattern = AURA_MAKE_PATTERN(elem_size, m_warp_type);

#define WARP_NONE_IMPL(data_type, warp_type)                                                                                                               \
    if (m_target.m_data.none.enable_mt)                                                                                                                    \
    {                                                                                                                                                      \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                                                           \
        if (MI_NULL == wp)                                                                                                                                 \
        {                                                                                                                                                  \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                                                          \
            return Status::ERROR;                                                                                                                          \
        }                                                                                                                                                  \
                                                                                                                                                           \
        auto thread_ids = wp->GetComputeThreadIDs();                                                                                                       \
        ThreadObject<Remap> share_remap(m_ctx, thread_ids, m_target);                                                                                      \
        ThreadBuffer thread_buffer(m_ctx, 2 * elem_size * BLOCK_SZ * BLOCK_SZ);                                                                            \
                                                                                                                                                           \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), height, WarpNoneBlock<data_type, warp_type>, m_ctx, std::cref(*src), std::ref(*dst),                 \
                              std::ref(m_map_x), std::ref(m_map_y), m_interp_type, m_border_type, std::cref(m_border_value), std::ref(share_remap),        \
                              std::ref(thread_buffer));                                                                                                    \
    }                                                                                                                                                      \
    else                                                                                                                                                   \
    {                                                                                                                                                      \
        auto thread_ids = std::vector<THREAD_ID>(1, THIS_THREAD_ID);                                                                                       \
        ThreadObject<Remap> share_remap(m_ctx, thread_ids, m_target);                                                                                      \
        ThreadBuffer thread_buffer(m_ctx, 2 * elem_size * BLOCK_SZ * BLOCK_SZ);                                                                            \
        ret = WarpNoneBlock<data_type, warp_type>(m_ctx, *src, *dst, m_map_x, m_map_y, m_interp_type, m_border_type, m_border_value,                       \
                                                  share_remap, thread_buffer, 0, height);                                                                  \
    }                                                                                                                                                      \
    if (ret != Status::OK)                                                                                                                                 \
    {                                                                                                                                                      \
        MI_CHAR error_msg[128];                                                                                                                            \
        std::snprintf(error_msg, sizeof(error_msg), "WarpNoneBlock<%s, %s> failed", #data_type, #warp_type);                                               \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                                                           \
    }

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(sizeof(MI_S16), WarpType::AFFINE):
        {
            WARP_NONE_IMPL(MI_S16, WarpType::AFFINE);
            break;
        }

        case AURA_MAKE_PATTERN(sizeof(MI_F32), WarpType::AFFINE):
        {
            WARP_NONE_IMPL(MI_F32, WarpType::AFFINE);
            break;
        }
        case AURA_MAKE_PATTERN(sizeof(MI_S16), WarpType::PERSPECTIVE):
        {
            WARP_NONE_IMPL(MI_S16, WarpType::PERSPECTIVE);
            break;
        }

        case AURA_MAKE_PATTERN(sizeof(MI_F32), WarpType::PERSPECTIVE):
        {
            WARP_NONE_IMPL(MI_F32, WarpType::PERSPECTIVE);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported InterType");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

static Status WarpAffineCoordNone(Context *ctx, const Mat &matrix, Mat &map_xy)
{
    if (!matrix.IsValid() || !map_xy.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "matrix or map_xy is invalid");
        return Status::ERROR;
    }

    MI_S32 height = map_xy.GetSizes().m_height;
    MI_S32 width  = map_xy.GetSizes().m_width;

    MI_F64 mivt[6];
    InverseMatrix2x3(matrix, mivt);

    Mat map_x = Mat(ctx, ElemType::F32, aura::Sizes3(1, width, 2));
    Mat map_y = Mat(ctx, ElemType::F32, aura::Sizes3(1, height, 2));
    if (!map_x.IsValid() || !map_y.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "map_x or map_y is invalid");
        return Status::ERROR;
    }

    MI_F32 *map_x_row = map_x.Ptr<MI_F32>(0);
    MI_F32 *map_y_row = map_y.Ptr<MI_F32>(0);

    for (MI_S32 x = 0; x < width; x++)
    {
        map_x_row[(x << 1)]     = x * mivt[0];
        map_x_row[(x << 1) + 1] = x * mivt[3];
    }

    for (MI_S32 y = 0; y < height; y++)
    {
        map_y_row[(y << 1)]     = y * mivt[1] + mivt[2];
        map_y_row[(y << 1) + 1] = y * mivt[4] + mivt[5];
    }

    for (MI_S32 y = 0; y < height; y++)
    {
        MI_S16 *map_xy_row = map_xy.Ptr<MI_S16>(y);
        MI_F32  x_y        = map_y_row[(y << 1)];
        MI_F32  y_y        = map_y_row[(y << 1) + 1];

        for (MI_S32 x = 0; x < width; x++)
        {
            map_xy_row[(x << 1)]     = SaturateCast<MI_S16>(map_x_row[(x << 1)] + x_y);
            map_xy_row[(x << 1) + 1] = SaturateCast<MI_S16>(map_x_row[(x << 1) + 1] + y_y);
        }
    }

    return Status::OK;
}

Status WarpCoordNone(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type)
{
    Status ret = Status::ERROR;

    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            ret = WarpAffineCoordNone(ctx, matrix, map_xy);
            break;
        }

        case WarpType::PERSPECTIVE:
        {
            /* code */
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported warp type");
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
