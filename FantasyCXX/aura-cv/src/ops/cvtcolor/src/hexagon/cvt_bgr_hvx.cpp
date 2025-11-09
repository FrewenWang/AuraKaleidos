#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <DT_S32 IC, DT_S32 OC>
static Status CvtBgr2BgraHvxImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_S32 start_row, DT_S32 end_row)
{
    using MVSt = typename MVHvxVector<IC>::Type;
    using MVDt = typename MVHvxVector<OC>::Type;

    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 istride = src.GetStrides().m_width;

    DT_S32 width_align = width & (-AURA_HVLEN);

    MVSt mvu8_src;
    MVDt mvu8_dst;

    DT_S32 blue_idx = swapb ? 2 : 0;
    DT_S32 red_idx  = swapb ? 0 : 2;

    HVX_Vector vu8_const_255 = Q6_Vb_vsplat_R(255);

    DT_U64 L2fetch_param = L2PfParam(istride, width * IC * ElemTypeSize(src.GetElemType()), 1, 0);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y < (end_row - 1))
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<DT_U8>(y + 1)), L2fetch_param);
        }

        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8       *dst_row = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_row + x * IC, mvu8_src);

            mvu8_dst.val[blue_idx] = mvu8_src.val[0];
            mvu8_dst.val[1]        = mvu8_src.val[1];
            mvu8_dst.val[red_idx]  = mvu8_src.val[2];
            if (4 == OC)
            {
                mvu8_dst.val[3] = vu8_const_255;
            }

            vstore(dst_row + x * OC, mvu8_dst);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgr2BgraHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if ((src.GetSizes().m_channel != 3 && src.GetSizes().m_channel != 4) ||
        (dst.GetSizes().m_channel != 3 && dst.GetSizes().m_channel != 4))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst channel should be 3 or 4");
        return ret;
    }

    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetSizes().m_channel, dst.GetSizes().m_channel);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(3, 3):
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2BgraHvxImpl<3, 3>, std::cref(src), std::ref(dst), swapb);
            break;
        }

        case AURA_MAKE_PATTERN(4, 3):
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2BgraHvxImpl<4, 3>, std::cref(src), std::ref(dst), swapb);
            break;
        }

        case AURA_MAKE_PATTERN(3, 4):
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2BgraHvxImpl<3, 4>, std::cref(src), std::ref(dst), swapb);
            break;
        }

        case AURA_MAKE_PATTERN(4, 4):
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2BgraHvxImpl<4, 4>, std::cref(src), std::ref(dst), swapb);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

/**
 *
 * @tparam IC  构造模板函数的模板变量。 IC是input channel
 * @param src
 * @param dst
 * @param swapb
 * @param start_row
 * @param end_row
 * @return
 */
template <DT_S32 IC>
static Status CvtBgr2GrayHvxImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_S32 start_row, DT_S32 end_row)
{
    /// 构建HVX的向量
    using MVType = typename MVHvxVector<IC>::Type;

    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 istride = src.GetStrides().m_width;

    /// 设置宽度对齐
    DT_S32 width_align  = width & (-AURA_HVLEN);

    //// TODO 这个地为什么和NONE不一样
    DT_S32 b_coeff = (Bgr2GrayParam::BC << 16) | (Bgr2GrayParam::BC); // 3735;
    DT_S32 g_coeff = (Bgr2GrayParam::GC << 16) | (Bgr2GrayParam::GC); // 19235;
    DT_S32 r_coeff = (Bgr2GrayParam::RC << 16) | (Bgr2GrayParam::RC); // 9798;

    if (swapb)
    {
        Swap(b_coeff, r_coeff);
    }

    /// 根据通道
    MVType           mvu8_src;
    HVX_VectorPair   wu16_src;
    HVX_Vector       vu8_dst;
    HVX_Vector       vs16_sum0, vs16_sum1;
    HVX_VectorPairX2 w2s32_sum;

    HVX_VectorPair ws32_const_half = Q6_W_vcombine_VV(Q6_V_vsplat_R(1 << 14), Q6_V_vsplat_R(1 << 14));

    DT_U64 L2fetch_param = L2PfParam(istride, width * IC * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        if (y < (end_row - 1))
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<DT_U8>(y + 1)), L2fetch_param);
        }

        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8       *dst_row = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_row + x * IC, mvu8_src);
            // b
            wu16_src         = Q6_Wuh_vzxt_Vub(mvu8_src.val[0]);
            w2s32_sum.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_const_half, Q6_V_lo_W(wu16_src), b_coeff);
            w2s32_sum.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_const_half, Q6_V_hi_W(wu16_src), b_coeff);
            // g
            wu16_src         = Q6_Wuh_vzxt_Vub(mvu8_src.val[1]);
            w2s32_sum.val[0] = Q6_Ww_vmpyacc_WwVhRh(w2s32_sum.val[0], Q6_V_lo_W(wu16_src), g_coeff);
            w2s32_sum.val[1] = Q6_Ww_vmpyacc_WwVhRh(w2s32_sum.val[1], Q6_V_hi_W(wu16_src), g_coeff);
             // r
            wu16_src         = Q6_Wuh_vzxt_Vub(mvu8_src.val[2]);
            w2s32_sum.val[0] = Q6_Ww_vmpyacc_WwVhRh(w2s32_sum.val[0], Q6_V_lo_W(wu16_src), r_coeff);
            w2s32_sum.val[1] = Q6_Ww_vmpyacc_WwVhRh(w2s32_sum.val[1], Q6_V_hi_W(wu16_src), r_coeff);

            vs16_sum0 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w2s32_sum.val[0]), Q6_V_lo_W(w2s32_sum.val[0]), 15);
            vs16_sum1 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w2s32_sum.val[1]), Q6_V_lo_W(w2s32_sum.val[1]), 15);

            vu8_dst = Q6_Vub_vsat_VhVh(vs16_sum1, vs16_sum0);
            vstore(dst_row + x, vu8_dst);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgr2GrayHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if ((src.GetSizes().m_channel != 3 && src.GetSizes().m_channel != 4) || dst.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 3 or 4 and dst channel must be 1");
        return ret;
    }

    DT_S32 height   = src.GetSizes().m_height;
    DT_S32 ichannel = src.GetSizes().m_channel;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (ichannel)
    {
        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2GrayHvxImpl<3>, std::cref(src), std::ref(dst), swapb);
            break;
        }
        case 4:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBgr2GrayHvxImpl<4>, std::cref(src), std::ref(dst), swapb);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <DT_S32 OC>
static Status CvtGray2BgrHvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    /// 定义一个三通道的1024bits的向量
    using MVType = typename MVHvxVector<OC>::Type;

    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 istride  = src.GetStrides().m_width;

    /// 4096 & ~(-128)
    // AURA_HVLEN = 128; TODO 所以咱们HVX必须要进行128位对齐
    DT_S32 width_align  = width & (-AURA_HVLEN);
    /// 使用一个HVX的向量。1024位的数据，如果是U8数据。也就是一次性加载128个元素(128*8 = 1024)
    HVX_Vector vu8_src;
    MVType     mvu8_dst;
    // L2PfParam 是用于配置硬件级缓存预取（L2 Prefetch）参数的语句，常见于高性能计算（如AI推理、图像处理）的底层优化中。以下是逐部分解析：
    // L2PfParam 是一个构造硬件预取指令参数的辅助函数，其参数含义如下：
    // istride	输入数据的行跨度（Stride）	相邻行数据的内存地址偏移（字节单位），反映数据在内存中的连续性
    // width * ElemTypeSize(src.GetElemType())	​单行数据的总字节宽度​	- src.GetElemType()：获取数据源元素类型（如 float16、int8）
    // -ElemTypeSize() ：计算该类型的字节大小（如 float16 为2字节）-width：数据行宽度（元素数量）乘积结果​：单行数据占用的总字节数
    DT_U64 L2fetch_param = L2PfParam(istride, width * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++) {
        if (y < (end_row - 1)) {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<DT_U8>(y + 1)), L2fetch_param);
        }

        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8       *dst_row = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_row + x, vu8_src);

            mvu8_dst.val[0] = vu8_src;
            mvu8_dst.val[1] = vu8_src;
            mvu8_dst.val[2] = vu8_src;
            if (4 == OC)
            {
                mvu8_dst.val[3] = Q6_Vb_vsplat_R(255);
            }

            vstore(dst_row + x * OC, mvu8_dst);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtGray2BgrHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if (src.GetSizes().m_channel != 1 || (dst.GetSizes().m_channel != 3 && dst.GetSizes().m_channel != 4))
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 1 and dst channel must be 3 or 4");
        return ret;
    }

    DT_S32 height   = src.GetSizes().m_height;
    DT_S32 ochannel = dst.GetSizes().m_channel;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (ochannel)
    {
        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtGray2BgrHvxImpl<3>, std::cref(src), std::ref(dst));
            break;
        }

        case 4:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtGray2BgrHvxImpl<4>, std::cref(src), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura