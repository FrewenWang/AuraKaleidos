#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#include <vector>

namespace aura
{

template <typename Tp>
static Status CvtBayer2BgrNeonImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, DT_S32 start_row, DT_S32 end_row)
{
    using VdType  = typename neon::DVector<Tp>::VType;
    using VqType  = typename neon::WVectorBits<VdType>::VType;
    using V2dType = typename neon::MDVector<Tp, 2>::MVType;
    using V3qType = typename neon::MQVector<Tp, 3>::MVType;

    const Tp *src_p  = DT_NULL;
    const Tp *src_c  = DT_NULL;
    const Tp *src_n0 = DT_NULL;
    const Tp *src_n1 = DT_NULL;

    Tp *dst_c = DT_NULL;
    Tp *dst_n = DT_NULL;

    DT_S32 width       = dst.GetSizes().m_width;
    DT_S32 elem_counts = 16 / ElemTypeSize(GetElemType<Tp>()) - 2;
    DT_S32 width_align = width - (elem_counts + 2);

    DT_S32 offset  = 0;
    DT_S32 b_idx_s = swapb ? -1 : 1;
    DT_S32 b_idx_v = swapb ? 0 : 2;
    DT_S32 r_idx_v = swapb ? 2 : 0;

    V2dType v2d_src_p, v2d_src_c, v2d_src_n0, v2d_src_n1;
    V3qType v3q_dst;

    for (DT_S32 y = (start_row * 2); y < (end_row * 2); y += 2)
    {
        if (swapg)
        {
            src_p  = src.Ptr<Tp>(y + 3);
            src_c  = src.Ptr<Tp>(y + 2);
            src_n0 = src.Ptr<Tp>(y + 1);
            src_n1 = src.Ptr<Tp>(y);
            dst_c  = dst.Ptr<Tp>(y + 2);
            dst_n  = dst.Ptr<Tp>(y + 1);
        }
        else
        {
            src_p  = src.Ptr<Tp>(y);
            src_c  = src.Ptr<Tp>(y + 1);
            src_n0 = src.Ptr<Tp>(y + 2);
            src_n1 = src.Ptr<Tp>(y + 3);
            dst_c  = dst.Ptr<Tp>(y + 1);
            dst_n  = dst.Ptr<Tp>(y + 2);
        }

        DT_S32 x = 1;
        for (; x < width_align; x += elem_counts)
        {
            v2d_src_p  = neon::vload2(src_p + x - 1);
            v2d_src_c  = neon::vload2(src_c + x - 1);
            v2d_src_n0 = neon::vload2(src_n0 + x - 1);
            v2d_src_n1 = neon::vload2(src_n1 + x - 1);

            VdType vd_g0 = neon::vext<1>(v2d_src_c.val[1], v2d_src_c.val[1]);
            VdType vd_r0 = neon::vext<1>(v2d_src_c.val[0], v2d_src_c.val[0]);

            VqType vq_b0 = neon::vaddl(v2d_src_p.val[1], v2d_src_n0.val[1]);
            VqType vq_g0 = neon::vaddl(v2d_src_p.val[0], v2d_src_n0.val[0]);
            VqType vq_r0 = neon::vaddl(v2d_src_c.val[0], vd_r0);

            VqType vq_b1 = neon::vadd(vq_b0, neon::vext<1>(vq_b0, vq_b0));
            VqType vq_g1 = neon::vaddl(v2d_src_c.val[1], vd_g0);

            vq_g1 = neon::vadd(vq_g1, neon::vext<1>(vq_g0, vq_g0));

            V2dType v2d_bb = neon::vzip(neon::vrshrn_n<1>(vq_b0), neon::vrshrn_n<2>(vq_b1));
            V2dType v2d_gg = neon::vzip(v2d_src_c.val[1], neon::vrshrn_n<2>(vq_g1));
            V2dType v2d_rr = neon::vzip(neon::vrshrn_n<1>(vq_r0), vd_r0);

            v3q_dst.val[b_idx_v] = neon::vcombine(v2d_bb.val[0], v2d_bb.val[1]);
            v3q_dst.val[1]       = neon::vcombine(v2d_gg.val[0], v2d_gg.val[1]);
            v3q_dst.val[r_idx_v] = neon::vcombine(v2d_rr.val[0], v2d_rr.val[1]);

            neon::vstore(dst_c + (3 * x), v3q_dst);

            vd_g0 = neon::vext<1>(v2d_src_n0.val[0], v2d_src_n0.val[0]);

            vq_b1 = neon::vaddl(v2d_src_n0.val[1], neon::vext<1>(v2d_src_n0.val[1], v2d_src_n0.val[1]));
            vq_g1 = neon::vaddl(v2d_src_n0.val[0], vd_g0);

            VqType vq_r1 = neon::vaddl(v2d_src_c.val[0], v2d_src_n1.val[0]);

            vq_g0 = neon::vaddl(v2d_src_c.val[1], v2d_src_n1.val[1]);
            vq_r0 = neon::vext<1>(vq_r1, vq_r1);
            vq_g1 = neon::vadd(vq_g1, vq_g0);
            vq_r1 = neon::vadd(vq_r1, vq_r0);

            v2d_bb = neon::vzip(v2d_src_n0.val[1], neon::vrshrn_n<1>(vq_b1));
            v2d_gg = neon::vzip(neon::vrshrn_n<2>(vq_g1), vd_g0);
            v2d_rr = neon::vzip(neon::vrshrn_n<2>(vq_r1), neon::vrshrn_n<1>(vq_r0));

            v3q_dst.val[b_idx_v] = neon::vcombine(v2d_bb.val[0], v2d_bb.val[1]);
            v3q_dst.val[1]       = neon::vcombine(v2d_gg.val[0], v2d_gg.val[1]);
            v3q_dst.val[r_idx_v] = neon::vcombine(v2d_rr.val[0], v2d_rr.val[1]);

            neon::vstore(dst_n + (3 * x), v3q_dst);
        }

        for (; x < width - 1; x += 2)
        {
            offset                  = 3 * x + 1;
            dst_c[offset - b_idx_s] = (src_c[x - 1] + src_c[x + 1] + 1) >> 1;
            dst_c[offset]           = src_c[x];
            dst_c[offset + b_idx_s] = (src_p[x] + src_n0[x] + 1) >> 1;

            offset                  = 3 * x + 4;
            dst_c[offset - b_idx_s] = src_c[x + 1];
            dst_c[offset]           = (src_p[x + 1] + src_c[x] + src_c[x + 2] + src_n0[x + 1] + 2) >> 2;
            dst_c[offset + b_idx_s] = (src_p[x] + src_p[x + 2] + src_n0[x] + src_n0[x + 2] + 2) >> 2;

            offset                  = 3 * x + 1;
            dst_n[offset - b_idx_s] = (src_c[x - 1] + src_c[x + 1] + src_n1[x - 1] + src_n1[x + 1] + 2) >> 2;
            dst_n[offset]           = (src_c[x] + src_n0[x - 1] + src_n0[x + 1] + src_n1[x] + 2) >> 2;
            dst_n[offset + b_idx_s] = src_n0[x];

            offset                  = 3 * x + 4;
            dst_n[offset - b_idx_s] = (src_c[x + 1] + src_n1[x + 1] + 1) >> 1;
            dst_n[offset]           = src_n0[x + 1];
            dst_n[offset + b_idx_s] = (src_n0[x] + src_n0[x + 2] + 1) >> 1;
        }

        offset            = 3 * width;
        dst_c[0]          = dst_c[3];
        dst_c[1]          = dst_c[4];
        dst_c[2]          = dst_c[5];
        dst_c[offset - 3] = dst_c[offset - 6];
        dst_c[offset - 2] = dst_c[offset - 5];
        dst_c[offset - 1] = dst_c[offset - 4];

        dst_n[0]          = dst_n[3];
        dst_n[1]          = dst_n[4];
        dst_n[2]          = dst_n[5];
        dst_n[offset - 3] = dst_n[offset - 6];
        dst_n[offset - 2] = dst_n[offset - 5];
        dst_n[offset - 1] = dst_n[offset - 4];
    }

    return Status::OK;
}

static Status CvtBayer2BgrRemainNeonImpl(Mat &dst)
{
    DT_S32 height = dst.GetSizes().m_height;
    DT_S32 width  = dst.GetSizes().m_width;

    DT_VOID *dst_c = dst.Ptr<DT_VOID>(height - 1);
    DT_VOID *dst_n = dst.Ptr<DT_VOID>(height - 2);
    memcpy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    dst_c = dst.Ptr<DT_VOID>(0);
    dst_n = dst.Ptr<DT_VOID>(1);
    memcpy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    return Status::OK;
}

Status CvtBayer2BgrNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height & 1 || dst.GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst size only support even");
        return ret;
    }

    if (dst.GetSizes().m_height != src.GetSizes().m_height || dst.GetSizes().m_width != src.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if (src.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 1 and dst channel must be 3");
        return ret;
    }

    DT_S32 height = (src.GetSizes().m_height - 2) / 2;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            ret = wp->ParallelFor(0, height, CvtBayer2BgrNeonImpl<DT_U8>, std::cref(src), std::ref(dst), swapb, swapg);
            break;
        }

        case ElemType::U16:
        {
            ret = wp->ParallelFor(0, height, CvtBayer2BgrNeonImpl<DT_U16>, std::cref(src), std::ref(dst), swapb, swapg);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    ret |= CvtBayer2BgrRemainNeonImpl(dst);

    AURA_RETURN(ctx, ret);
}

} // namespace aura