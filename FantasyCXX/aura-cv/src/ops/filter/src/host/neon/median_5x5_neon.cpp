#include "median_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VqType>
AURA_ALWAYS_INLINE DT_VOID Median5x5Core(VqType *vqs)
{
    MinMaxOp<VqType>(vqs[0],  vqs[1]);  MinMaxOp<VqType>(vqs[3],  vqs[4]);  MinMaxOp<VqType>(vqs[2],  vqs[4]);
    MinMaxOp<VqType>(vqs[2],  vqs[3]);  MinMaxOp<VqType>(vqs[6],  vqs[7]);  MinMaxOp<VqType>(vqs[5],  vqs[7]);
    MinMaxOp<VqType>(vqs[5],  vqs[6]);  MinMaxOp<VqType>(vqs[9],  vqs[10]); MinMaxOp<VqType>(vqs[8],  vqs[10]);
    MinMaxOp<VqType>(vqs[8],  vqs[9]);  MinMaxOp<VqType>(vqs[12], vqs[13]); MinMaxOp<VqType>(vqs[11], vqs[13]);
    MinMaxOp<VqType>(vqs[11], vqs[12]); MinMaxOp<VqType>(vqs[15], vqs[16]); MinMaxOp<VqType>(vqs[14], vqs[16]);
    MinMaxOp<VqType>(vqs[14], vqs[15]); MinMaxOp<VqType>(vqs[18], vqs[19]); MinMaxOp<VqType>(vqs[17], vqs[19]);
    MinMaxOp<VqType>(vqs[17], vqs[18]); MinMaxOp<VqType>(vqs[21], vqs[22]); MinMaxOp<VqType>(vqs[20], vqs[22]);
    MinMaxOp<VqType>(vqs[20], vqs[21]); MinMaxOp<VqType>(vqs[23], vqs[24]); MinMaxOp<VqType>(vqs[2],  vqs[5]);
    MinMaxOp<VqType>(vqs[3],  vqs[6]);  MinMaxOp<VqType>(vqs[0],  vqs[6]);  MinMaxOp<VqType>(vqs[0],  vqs[3]);
    MinMaxOp<VqType>(vqs[4],  vqs[7]);  MinMaxOp<VqType>(vqs[1],  vqs[7]);  MinMaxOp<VqType>(vqs[1],  vqs[4]);
    MinMaxOp<VqType>(vqs[11], vqs[14]); MinMaxOp<VqType>(vqs[8],  vqs[14]); MinMaxOp<VqType>(vqs[8],  vqs[11]);
    MinMaxOp<VqType>(vqs[12], vqs[15]); MinMaxOp<VqType>(vqs[9],  vqs[15]); MinMaxOp<VqType>(vqs[9],  vqs[12]);
    MinMaxOp<VqType>(vqs[13], vqs[16]); MinMaxOp<VqType>(vqs[10], vqs[16]); MinMaxOp<VqType>(vqs[10], vqs[13]);
    MinMaxOp<VqType>(vqs[20], vqs[23]); MinMaxOp<VqType>(vqs[17], vqs[23]); MinMaxOp<VqType>(vqs[17], vqs[20]);
    MinMaxOp<VqType>(vqs[21], vqs[24]); MinMaxOp<VqType>(vqs[18], vqs[24]); MinMaxOp<VqType>(vqs[18], vqs[21]);
    MinMaxOp<VqType>(vqs[19], vqs[22]); MinMaxOp<VqType>(vqs[9],  vqs[18]); MinMaxOp<VqType>(vqs[0],  vqs[18]);

    vqs[17] = neon::vmax(vqs[8], vqs[17]);
    vqs[9]  = neon::vmax(vqs[0], vqs[9]);
    MinMaxOp<VqType>(vqs[10], vqs[19]); MinMaxOp<VqType>(vqs[1], vqs[19]);  MinMaxOp<VqType>(vqs[1], vqs[10]);
    MinMaxOp<VqType>(vqs[11], vqs[20]); MinMaxOp<VqType>(vqs[2], vqs[20]);  MinMaxOp<VqType>(vqs[12], vqs[21]);
    vqs[11] = neon::vmax(vqs[2], vqs[11]);
    MinMaxOp<VqType>(vqs[3], vqs[21]);  MinMaxOp<VqType>(vqs[3], vqs[12]);  MinMaxOp<VqType>(vqs[13], vqs[22]);
    vqs[4] = neon::vmin(vqs[4], vqs[22]);
    MinMaxOp<VqType>(vqs[4], vqs[13]);  MinMaxOp<VqType>(vqs[14], vqs[23]);
    MinMaxOp<VqType>(vqs[5], vqs[23]);  MinMaxOp<VqType>(vqs[5], vqs[14]);  MinMaxOp<VqType>(vqs[15], vqs[24]);
    vqs[6] = neon::vmin(vqs[6], vqs[24]);
    MinMaxOp<VqType>(vqs[6], vqs[15]);

    vqs[7]  = neon::vmin(vqs[7],  vqs[16]);
    vqs[7]  = neon::vmin(vqs[7],  vqs[19]);
    vqs[13] = neon::vmin(vqs[13], vqs[21]);
    vqs[15] = neon::vmin(vqs[15], vqs[23]);
    vqs[7]  = neon::vmin(vqs[7],  vqs[13]);
    vqs[7]  = neon::vmin(vqs[7],  vqs[15]);
    vqs[9]  = neon::vmax(vqs[1],  vqs[9]);
    vqs[11] = neon::vmax(vqs[3],  vqs[11]);
    vqs[17] = neon::vmax(vqs[5],  vqs[17]);
    vqs[17] = neon::vmax(vqs[11], vqs[17]);
    vqs[17] = neon::vmax(vqs[9],  vqs[17]);

    MinMaxOp<VqType>(vqs[4], vqs[10]);
    MinMaxOp<VqType>(vqs[6], vqs[12]); MinMaxOp<VqType>(vqs[7], vqs[14]);  MinMaxOp<VqType>(vqs[4], vqs[6]);
    vqs[7] = neon::vmax(vqs[4], vqs[7]);
    MinMaxOp<VqType>(vqs[12], vqs[14]);
    vqs[10] = neon::vmin(vqs[10], vqs[14]);
    MinMaxOp<VqType>(vqs[6], vqs[7]);  MinMaxOp<VqType>(vqs[10], vqs[12]); MinMaxOp<VqType>(vqs[6], vqs[10]);
    vqs[17] = neon::vmax(vqs[6], vqs[17]);
    MinMaxOp<VqType>(vqs[12], vqs[17]);
    vqs[7] = neon::vmin(vqs[7], vqs[17]);
    MinMaxOp<VqType>(vqs[7], vqs[10]); MinMaxOp<VqType>(vqs[12], vqs[18]);
    vqs[12] = neon::vmax(vqs[7], vqs[12]);
    vqs[10] = neon::vmin(vqs[10], vqs[18]);
    MinMaxOp<VqType>(vqs[12], vqs[20]);
    vqs[10] = neon::vmin(vqs[10], vqs[20]);
    vqs[12] = neon::vmax(vqs[10], vqs[12]);
}

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VqType>
AURA_ALWAYS_INLINE DT_VOID Median5x5Vector(VqType &vq_src_p1x0, VqType &vq_src_p1x1, VqType &vq_src_p1x2,
                                           VqType &vq_src_p0x0, VqType &vq_src_p0x1, VqType &vq_src_p0x2,
                                           VqType &vq_src_cx0,  VqType &vq_src_cx1,  VqType &vq_src_cx2,
                                           VqType &vq_src_n0x0, VqType &vq_src_n0x1, VqType &vq_src_n0x2,
                                           VqType &vq_src_n1x0, VqType &vq_src_n1x1, VqType &vq_src_n1x2,
                                           VqType &vq_result)
{
    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(16 / sizeof(Tp));

    VqType vqs[25];

    vqs[0] = neon::vext<ELEM_COUNTS - 2>(vq_src_p1x0, vq_src_p1x1);
    vqs[1] = neon::vext<ELEM_COUNTS - 1>(vq_src_p1x0, vq_src_p1x1);
    vqs[2] = vq_src_p1x1;
    vqs[3] = neon::vext<1>(vq_src_p1x1, vq_src_p1x2);
    vqs[4] = neon::vext<2>(vq_src_p1x1, vq_src_p1x2);

    vqs[0 + 5] = neon::vext<ELEM_COUNTS - 2>(vq_src_p0x0, vq_src_p0x1);
    vqs[1 + 5] = neon::vext<ELEM_COUNTS - 1>(vq_src_p0x0, vq_src_p0x1);
    vqs[2 + 5] = vq_src_p0x1;
    vqs[3 + 5] = neon::vext<1>(vq_src_p0x1, vq_src_p0x2);
    vqs[4 + 5] = neon::vext<2>(vq_src_p0x1, vq_src_p0x2);

    vqs[0 + 10] = neon::vext<ELEM_COUNTS - 2>(vq_src_cx0, vq_src_cx1);
    vqs[1 + 10] = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    vqs[2 + 10] = vq_src_cx1;
    vqs[3 + 10] = neon::vext<1>(vq_src_cx1, vq_src_cx2);
    vqs[4 + 10] = neon::vext<2>(vq_src_cx1, vq_src_cx2);

    vqs[0 + 15] = neon::vext<ELEM_COUNTS - 2>(vq_src_n0x0, vq_src_n0x1);
    vqs[1 + 15] = neon::vext<ELEM_COUNTS - 1>(vq_src_n0x0, vq_src_n0x1);
    vqs[2 + 15] = vq_src_n0x1;
    vqs[3 + 15] = neon::vext<1>(vq_src_n0x1, vq_src_n0x2);
    vqs[4 + 15] = neon::vext<2>(vq_src_n0x1, vq_src_n0x2);

    vqs[0 + 20] = neon::vext<ELEM_COUNTS - 2>(vq_src_n1x0, vq_src_n1x1);
    vqs[1 + 20] = neon::vext<ELEM_COUNTS - 1>(vq_src_n1x0, vq_src_n1x1);
    vqs[2 + 20] = vq_src_n1x1;
    vqs[3 + 20] = neon::vext<1>(vq_src_n1x1, vq_src_n1x2);
    vqs[4 + 20] = neon::vext<2>(vq_src_n1x1, vq_src_n1x2);

    Median5x5Core<VqType>(vqs);

    vq_result = vqs[12];

    vq_src_p1x0 = vq_src_p1x1;
    vq_src_p0x0 = vq_src_p0x1;
    vq_src_cx0  = vq_src_cx1;
    vq_src_n0x0 = vq_src_n0x1;
    vq_src_n1x0 = vq_src_n1x1;

    vq_src_p1x1 = vq_src_p1x2;
    vq_src_p0x1 = vq_src_p0x2;
    vq_src_cx1  = vq_src_cx2;
    vq_src_n0x1 = vq_src_n0x2;
    vq_src_n1x1 = vq_src_n1x2;
}

template<typename Tp, DT_S32 C>
static DT_VOID Median5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0, const Tp *src_n1,
                            Tp *dst_c, DT_S32 width)
{
    using MqVType = typename neon::MQVector<Tp, C>::MVType;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(16 / sizeof(Tp));
    constexpr DT_S32 VOFFSET = ELEM_COUNTS * C;
    const DT_S32 width_align = (width & -ELEM_COUNTS) * C;

    MqVType mvq_src_p1[3], mvq_src_p0[3], mvq_src_c[3], mvq_src_n0[3], mvq_src_n1[3];
    MqVType mvq_result;

    // left border
    {
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_c,            mvq_src_c[1]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c  + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p1[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p1[1].val[ch], src_p1[ch], src_p1[ch]);
            mvq_src_p0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], src_p0[ch]);
            mvq_src_c[0].val[ch]  = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_c[1].val[ch],  src_c[ch],  src_c[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], src_n0[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], src_n1[ch]);

            Median5x5Vector<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                mvq_result.val[ch]);
        }
        neon::vstore(dst_c, mvq_result);
    }

    //middle
    {
        for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Median5x5Vector<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                    mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                    mvq_result.val[ch]);
            }
            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_c  + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_c  + x,           mvq_src_c[1]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Median5x5Vector<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                    mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                    mvq_result.val[ch]);
            }
            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // right border
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p1[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p1[1].val[ch], src_p1[last + ch], src_p1[last + ch]);
            mvq_src_p0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last + ch], src_p0[last + ch]);
            mvq_src_c[2].val[ch]  = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_c[1].val[ch],  src_c[last + ch],  src_c[last + ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last + ch], src_n0[last + ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last + ch], src_n1[last + ch]);

            Median5x5Vector<Tp>(mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                mvq_result.val[ch]);
        }
        neon::vstore(dst_c + x, mvq_result);
    }
}

template <typename Tp, DT_S32 C>
static Status Median5x5NeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;
    DT_S32 y = start_row;

    const Tp *src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(y - 2);
    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(y - 1);
    const Tp *src_c  = src.Ptr<Tp>(y);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 1);
    const Tp *src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 2);

    for (; y < end_row; y++)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        Median5x5Row<Tp, C>(src_p1, src_p0, src_c, src_n0, src_n1, dst_c, width);

        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<Tp>(y + 3, BorderType::REPLICATE);
    }

    return Status::OK;
}

template <typename Tp>
static Status Median5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Median5x5NeonImpl<Tp, 1>, std::cref(src), std::ref(dst));
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Median5x5NeonImpl<Tp, 2>, std::cref(src), std::ref(dst));
            break;
        }
        
        case 3:
        {
            ret = wp->ParallelFor(0, height, Median5x5NeonImpl<Tp, 3>, std::cref(src), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Median5x5Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median5x5NeonHelper<DT_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_U8> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = Median5x5NeonHelper<DT_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_S8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = Median5x5NeonHelper<DT_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = Median5x5NeonHelper<DT_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_S16> failed");
            }
            break;
        }

        case ElemType::U32:
        {
            ret = Median5x5NeonHelper<DT_U32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_U32> failed");
            }
            break;
        }

        case ElemType::S32:
        {
            ret = Median5x5NeonHelper<DT_S32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_S32> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Median5x5NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = Median5x5NeonHelper<DT_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median5x5NeonHelper<DT_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura