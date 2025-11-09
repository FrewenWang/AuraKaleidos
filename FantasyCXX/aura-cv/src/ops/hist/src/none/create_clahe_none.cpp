#include "create_clahe_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
AURA_INLINE DT_VOID CalcLutOneRowNoneImpl(std::vector<DT_S32> &tile_hist, const Mat &src, DT_S32 width,
                                          DT_S32 ox, DT_S32 ox_n, DT_S32 oy, DT_S32 bor_idx_x)
{
    const Tp *src_row = src.Ptr<Tp>(oy);
    DT_S32 ox_align = ox_n - 4;
    DT_S32 x        = ox;

    if (ox_n <= width)
    {
        for (; x <= ox_align; x += 4)
        {
            tile_hist[src_row[x]]++;
            tile_hist[src_row[x + 1]]++;
            tile_hist[src_row[x + 2]]++;
            tile_hist[src_row[x + 3]]++;
        }
        for (; x < ox_n; x++)
        {
            tile_hist[src_row[x]]++;
        }
    }
    else
    {
        for (; x < width; x++)
        {
            tile_hist[src_row[x]]++;
        }

        for (x = bor_idx_x; x < width - 1; x++)
        {
            tile_hist[src_row[x]]++;
        }
    }
}

template <typename Tp>
static Status CLAHECalcLutNoneImpl(Context *ctx, const Mat &src, Mat &lut, const Sizes &tile_size, DT_S32 gwidth, DT_S32 gheight,
                                   DT_S32 clip_limit, DT_F32 lut_scale, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);
    const DT_S32 hist_size = (ElemType::U8 == src.GetElemType()) ? 256 : 65536;
    std::vector<DT_S32> tile_hist(hist_size, 0);

    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 height   = src.GetSizes().m_height;
    DT_S32 bor_idx_x = width - (gwidth - (width % gwidth)) - 1;
    DT_S32 bor_idx_y = height - (gheight - (height % gheight)) - 1;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_S32 ty = y / gwidth;
        const DT_S32 tx = y % gwidth;
        Tp *lut_row     = lut.Ptr<Tp>(y);
        std::fill(tile_hist.begin(), tile_hist.end(), 0);

        DT_S32 ox   = tx * tile_size.m_width;
        DT_S32 oy   = ty * tile_size.m_height;
        DT_S32 ox_n = ox + tile_size.m_width;
        DT_S32 oy_n = oy + tile_size.m_height;

        if (oy_n <= height)
        {
            for (; oy < oy_n; oy++)
            {
                CalcLutOneRowNoneImpl<Tp>(tile_hist, src, width, ox, ox_n, oy, bor_idx_x);
            }
        }
        else
        {
            for (; oy < height; oy++)
            {
                CalcLutOneRowNoneImpl<Tp>(tile_hist, src, width, ox, ox_n, oy, bor_idx_x);
            }
            for (oy = bor_idx_y; oy < height - 1; oy++)
            {
                CalcLutOneRowNoneImpl<Tp>(tile_hist, src, width, ox, ox_n, oy, bor_idx_x);
            }
        }

        if (clip_limit > 0)
        {
            DT_S32 clipped = 0;
            for (DT_S32 i = 0; i < hist_size; i++)
            {
                if (tile_hist[i] > clip_limit)
                {
                    clipped += tile_hist[i] - clip_limit;
                    tile_hist[i] = clip_limit;
                }
            }

            DT_S32 redist_batch = clipped / hist_size;
            DT_S32 residual     = clipped - redist_batch * hist_size;

            for (DT_S32 i = 0; i < hist_size; i++)
            {
                tile_hist[i] += redist_batch;
            }

            if (residual != 0)
            {
                DT_S32 residual_step = Max(hist_size / residual, static_cast<DT_S32>(1));
                for (DT_S32 i = 0; i < hist_size && residual > 0; i += residual_step, residual--)
                {
                    tile_hist[i]++;
                }
            }
        }

        DT_S32 sum = 0;
        for (DT_S32 i = 0; i < hist_size; i++)
        {
            sum += tile_hist[i];
            lut_row[i] = SaturateCast<Tp>(sum * lut_scale);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status CLAHEInterpNoneImpl(Context *ctx, const Mat &src, Mat &lut, Mat &dst, DT_S32 *coe_buffer, DT_S32 m_height,
                                  DT_S32 gwidth, DT_S32 gheight, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);
    DT_S32 width  = src.GetSizes().m_width;
    DT_S32 *ind0_p = coe_buffer;
    DT_S32 *ind1_p = coe_buffer + width;
    DT_F32 *xa0_p  = (DT_F32 *)(ind1_p + width);
    DT_F32 *xa1_p  = xa0_p + width;
    DT_F32 inv_th  = 1.0f / m_height;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        DT_F32 tyf = y * inv_th - 0.5f;
        DT_S32 ty0 = Floor(tyf);
        DT_S32 ty1 = ty0 + 1;

        DT_F32 ya0 = tyf - ty0;
        DT_F32 ya1 = 1.0f - ya0;

        ty0 = Max(ty0, static_cast<DT_S32>(0));
        ty1 = Min(ty1, gheight - 1);

        const Tp* lut_c = lut.Ptr<Tp>(ty0 * gwidth);
        const Tp* lut_n = lut.Ptr<Tp>(ty1 * gwidth);

        for (DT_S32 x = 0; x < width; x++)
        {
            DT_S32 ind0   = ind0_p[x] + src_row[x];
            DT_S32 ind1   = ind1_p[x] + src_row[x];
            DT_F32 result = (lut_c[ind0] * xa1_p[x] + lut_c[ind1] * xa0_p[x]) * ya1 +
                            (lut_n[ind0] * xa1_p[x] + lut_n[ind1] * xa0_p[x]) * ya0;

            dst_row[x] = SaturateCast<Tp>(result);
        }
    }

    return Status::OK;
}

CreateClAHENone::CreateClAHENone(Context *ctx, const OpTarget &target) : CreateClAHEImpl(ctx, target)
{}

Status CreateClAHENone::SetArgs(const Array *src, Array *dst, DT_F64 clip_limit, const Sizes &tile_grid_size)
{
    if (CreateClAHEImpl::SetArgs(src, dst, clip_limit, tile_grid_size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateClAHEImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CreateClAHENone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat *dst = dynamic_cast<Mat *>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return ret;
    }

    const DT_S32 hist_size = (ElemType::U8 == src->GetElemType()) ? 256 : 65536;
    DT_S32 height  = src->GetSizes().m_height;
    DT_S32 width   = src->GetSizes().m_width;
    DT_S32 gheight = m_tile_grid_size.m_height;
    DT_S32 gwidth  = m_tile_grid_size.m_width;
    DT_S32 gsize   = gheight * gwidth;

    Sizes tile_size;
    if ((height % gheight == 0) && (width % gwidth == 0))
    {
        tile_size = Sizes(height / gheight, width / gwidth);
    }
    else
    {
        tile_size = Sizes((height + gheight - (height % gheight)) / gheight, (width + gwidth - (width % gwidth)) / gwidth);
    }
    const DT_S32 tile_size_total = tile_size.m_height * tile_size.m_width;
    const DT_F32 lut_scale       = static_cast<DT_F32>(hist_size - 1) / tile_size_total;

    DT_S32 pix_limit = 0;
    if (m_clip_limit > 0.0)
    {
        pix_limit = static_cast<DT_S32>(m_clip_limit * tile_size_total / hist_size);
        pix_limit = Max(pix_limit, static_cast<DT_S32>(1));
    }

    Mat lut = Mat(m_ctx, src->GetElemType(), Sizes3(gsize, hist_size));
    if (!lut.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "lut Get Buffer failed");
        return ret;
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            if (m_target.m_data.none.enable_mt)
            {
                WorkerPool *wp = m_ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
                    return ret;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), gsize, CLAHECalcLutNoneImpl<DT_U8>, m_ctx, std::cref(*src), std::ref(lut),
                                      std::cref(tile_size), gwidth, gheight, pix_limit, lut_scale);
            }
            else
            {
                ret = CLAHECalcLutNoneImpl<DT_U8>(m_ctx, *src, lut, tile_size, gwidth, gheight, pix_limit, lut_scale, static_cast<DT_S32>(0), gsize);
            }
            break;
        }

        case ElemType::U16:
        {
            if (m_target.m_data.none.enable_mt)
            {
                WorkerPool *wp = m_ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
                    return ret;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), gsize, CLAHECalcLutNoneImpl<DT_U16>, m_ctx, std::cref(*src), std::ref(lut),
                                      std::cref(tile_size), gwidth, gheight, pix_limit, lut_scale);
            }
            else
            {
                ret = CLAHECalcLutNoneImpl<DT_U16>(m_ctx, *src, lut, tile_size, gwidth, gheight, pix_limit,
                                                   lut_scale, static_cast<DT_S32>(0), gsize);
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CLAHECalcLutNoneImpl failed");
        return ret;
    }

    ret = Status::ERROR;

    DT_S32 *coe_buffer = static_cast<DT_S32*>(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, (width << 2) * sizeof(DT_S32), 0));
    if (DT_NULL == coe_buffer)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AURA_ALLOC_PARAM failed");
        return ret;
    }

    DT_S32 lut_step = lut.GetStrides().m_width / ElemTypeSize(src->GetElemType());
    DT_S32 *ind0_p  = coe_buffer;
    DT_S32 *ind1_p  = coe_buffer + width;
    DT_F32 *xa0_p   = (DT_F32 *)(ind1_p + width);
    DT_F32 *xa1_p   = xa0_p + width;
    DT_F32 inv_tw   = 1.0f / tile_size.m_width;

    for (DT_S32 x = 0; x < width; x++)
    {
        DT_F32 txf = x * inv_tw - 0.5f;
        DT_S32 tx0 = Floor(txf);
        DT_S32 tx1 = tx0 + 1;

        xa0_p[x] = txf - tx0;
        xa1_p[x] = 1.0f - xa0_p[x];

        tx0 = Max(tx0, static_cast<DT_S32>(0));
        tx1 = Min(tx1, gwidth - 1);

        ind0_p[x] = tx0 * lut_step;
        ind1_p[x] = tx1 * lut_step;
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            if (m_target.m_data.none.enable_mt)
            {
                WorkerPool *wp = m_ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
                    return ret;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, CLAHEInterpNoneImpl<DT_U8>, m_ctx, std::cref(*src), std::ref(lut), std::ref(*dst),
                                      std::cref(coe_buffer), tile_size.m_height, gwidth, gheight);
            }
            else
            {
                ret = CLAHEInterpNoneImpl<DT_U8>(m_ctx, *src, lut, *dst, coe_buffer, tile_size.m_height, gwidth, gheight, static_cast<DT_S32>(0), height);
            }
            break;
        }

        case ElemType::U16:
        {
            if (m_target.m_data.none.enable_mt)
            {
                WorkerPool *wp = m_ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
                    return ret;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, CLAHEInterpNoneImpl<DT_U16>, m_ctx, std::cref(*src), std::ref(lut), std::ref(*dst),
                                      std::cref(coe_buffer), tile_size.m_height, gwidth, gheight);
            }
            else
            {
                ret = CLAHEInterpNoneImpl<DT_U16>(m_ctx, *src, lut, *dst, coe_buffer, tile_size.m_height, gwidth, gheight, static_cast<DT_S32>(0), height);
            }

            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_FREE(m_ctx, coe_buffer);
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura