#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status CvtBgr2BgraNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width       = src.GetSizes().m_width;
    DT_S32 width_align = width & (-16);

    uint8x16x4_t v4qu8_dst;
    neon::vdup(v4qu8_dst.val[3], 255);
    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0, ix = 0, ox = 0;
        for (; x < width_align; x += 16, ix += 48, ox += 64)
        {
LOOP_BODY:
            uint8x16x3_t v3qu8_src = neon::vload3q(src_row + ix);
            v4qu8_dst.val[0] = v3qu8_src.val[0];
            v4qu8_dst.val[1] = v3qu8_src.val[1];
            v4qu8_dst.val[2] = v3qu8_src.val[2];
            neon::vstore(dst_row + ox, v4qu8_dst);
        }

        if (x < width)
        {
            x = width - 16;
            ix = x * 3;
            ox = x * 4;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgr2BgraNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return Status::ERROR;
    }

    if (src.GetSizes().m_channel != 3 || dst.GetSizes().m_channel != 4)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel should be 3 and dst channel should be 4");
        return Status::ERROR;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 height = src.GetSizes().m_height;
    if (wp->ParallelFor(0, height, CvtBgr2BgraNeonImpl, std::cref(src), std::ref(dst)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    return Status::OK;
}

static Status CvtBgra2BgrNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width        = src.GetSizes().m_width;
    DT_S32 width_align  = width & (-16);

    uint8x16x3_t v3qu8_dst;
    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0, ix = 0, ox = 0;
        for (; x < width_align; x += 16, ix += 64, ox += 48)
        {
LOOP_BODY:
            uint8x16x4_t v4qu8_src = neon::vload4q(src_row + ix);
            v3qu8_dst.val[0] = v4qu8_src.val[0];
            v3qu8_dst.val[1] = v4qu8_src.val[1];
            v3qu8_dst.val[2] = v4qu8_src.val[2];
            neon::vstore(dst_row + ox, v3qu8_dst);
        }

        if (x < width)
        {
            x = width - 16;
            ix = x * 4;
            ox = x * 3;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgra2BgrNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return Status::ERROR;
    }

    if (src.GetSizes().m_channel != 4 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel should be 4 and dst channel should be 3");
        return Status::ERROR;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 height = src.GetSizes().m_height;
    if (wp->ParallelFor(0, height, CvtBgra2BgrNeonImpl, std::cref(src), std::ref(dst)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    return Status::OK;
}

static Status CvtBgr2RgbNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width        = src.GetSizes().m_width;
    DT_S32 width_align  = width & (-16);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align * 3; x += 48)
        {
LOOP_BODY:
            uint8x16x3_t v3qu8_src = neon::vload3q(src_row + x);
            Swap(v3qu8_src.val[0], v3qu8_src.val[2]);
            neon::vstore(dst_row + x, v3qu8_src);
        }

        if (x < width * 3)
        {
            x = (width - 16) * 3;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgr2RgbNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if (src.GetSizes().m_channel != 3 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst channel should be 3");
        return ret;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height = src.GetSizes().m_height;
    if (wp->ParallelFor(0, height, CvtBgr2RgbNeonImpl, std::cref(src), std::ref(dst)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    return Status::OK;
}

template <DT_S32 IC>
static Status CvtBgr2GrayNeonImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_S32 start_row, DT_S32 end_row)
{
    using MVqType = typename neon::MQVector<DT_U8, IC>::MVType;

    DT_S32 width       = src.GetSizes().m_width;
    DT_S32 ichannel    = src.GetSizes().m_channel;
    DT_S32 width_align = width & (-16);
    DT_S32 ioffset     = ichannel * 16;

    DT_U16 b_coeff = Bgr2GrayParam::BC;
    DT_U16 g_coeff = Bgr2GrayParam::GC;
    DT_U16 r_coeff = Bgr2GrayParam::RC;
    if (swapb)
    {
        Swap(b_coeff, r_coeff);
    }

    MVqType mvqu8_src;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 ix = 0, ox = 0;
        for (; ox < width_align; ix += ioffset, ox += 16)
        {
LOOP_BODY:
            neon::vload(src_row + ix, mvqu8_src);

            uint16x8_t vqu16_b_lo = neon::vmovl(neon::vgetlow(mvqu8_src.val[0]));
            uint16x8_t vqu16_b_hi = neon::vmovl(neon::vgethigh(mvqu8_src.val[0]));
            uint16x8_t vqu16_g_lo = neon::vmovl(neon::vgetlow(mvqu8_src.val[1]));
            uint16x8_t vqu16_g_hi = neon::vmovl(neon::vgethigh(mvqu8_src.val[1]));
            uint16x8_t vqu16_r_lo = neon::vmovl(neon::vgetlow(mvqu8_src.val[2]));
            uint16x8_t vqu16_r_hi = neon::vmovl(neon::vgethigh(mvqu8_src.val[2]));

            uint32x4_t vqu32_gray_lo_lo = neon::vmull(neon::vgetlow(vqu16_b_lo), b_coeff);
            uint32x4_t vqu32_gray_lo_hi = neon::vmull(neon::vgethigh(vqu16_b_lo), b_coeff);
            uint32x4_t vqu32_gray_hi_lo = neon::vmull(neon::vgetlow(vqu16_b_hi), b_coeff);
            uint32x4_t vqu32_gray_hi_hi = neon::vmull(neon::vgethigh(vqu16_b_hi), b_coeff);

            vqu32_gray_lo_lo = neon::vmlal(vqu32_gray_lo_lo, neon::vgetlow(vqu16_g_lo), g_coeff);
            vqu32_gray_lo_hi = neon::vmlal(vqu32_gray_lo_hi, neon::vgethigh(vqu16_g_lo), g_coeff);
            vqu32_gray_hi_lo = neon::vmlal(vqu32_gray_hi_lo, neon::vgetlow(vqu16_g_hi), g_coeff);
            vqu32_gray_hi_hi = neon::vmlal(vqu32_gray_hi_hi, neon::vgethigh(vqu16_g_hi), g_coeff);

            vqu32_gray_lo_lo = neon::vmlal(vqu32_gray_lo_lo, neon::vgetlow(vqu16_r_lo), r_coeff);
            vqu32_gray_lo_hi = neon::vmlal(vqu32_gray_lo_hi, neon::vgethigh(vqu16_r_lo), r_coeff);
            vqu32_gray_hi_lo = neon::vmlal(vqu32_gray_hi_lo, neon::vgetlow(vqu16_r_hi), r_coeff);
            vqu32_gray_hi_hi = neon::vmlal(vqu32_gray_hi_hi, neon::vgethigh(vqu16_r_hi), r_coeff);

            uint16x4_t vdu16_gray_lo_lo = neon::vrshrn_n<15>(vqu32_gray_lo_lo);
            uint16x4_t vdu16_gray_lo_hi = neon::vrshrn_n<15>(vqu32_gray_lo_hi);
            uint16x4_t vdu16_gray_hi_lo = neon::vrshrn_n<15>(vqu32_gray_hi_lo);
            uint16x4_t vdu16_gray_hi_hi = neon::vrshrn_n<15>(vqu32_gray_hi_hi);

            uint8x8_t vdu8_gray0 = neon::vmovn(neon::vcombine(vdu16_gray_lo_lo, vdu16_gray_lo_hi));
            uint8x8_t vdu8_gray1 = neon::vmovn(neon::vcombine(vdu16_gray_hi_lo, vdu16_gray_hi_hi));
            uint8x16_t vqu8_gray = neon::vcombine(vdu8_gray0, vdu8_gray1);
            neon::vstore(dst_row + ox, vqu8_gray);
        }

        if (ox < width)
        {
            ox = width - 16;
            ix = ox * ichannel;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtBgr2GrayNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height != dst.GetSizes().m_height || src.GetSizes().m_width != dst.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if ((src.GetSizes().m_channel != 3 && src.GetSizes().m_channel != 4) || dst.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel should be 3 or 4 and dst channel should be 1");
        return ret;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 ichannel = src.GetSizes().m_channel;
    DT_S32 height   = src.GetSizes().m_height;

    switch (ichannel)
    {
        case 3:
        {
            ret = wp->ParallelFor(0, height, CvtBgr2GrayNeonImpl<3>, std::cref(src), std::ref(dst), swapb);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CvtBgr2GrayNeonImpl<3> failed");
            }
            break;
        }

        case 4:
        {
            ret = wp->ParallelFor(0, height, CvtBgr2GrayNeonImpl<4>, std::cref(src), std::ref(dst), swapb);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CvtBgr2GrayNeonImpl<4> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "src channel should be 3 or 4");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
    }

    AURA_RETURN(ctx, ret);
}

static Status CvtGray2BgrNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width       = src.GetSizes().m_width;
    DT_S32 width_align = width & (-16);

    uint8x16x3_t v3qu8_dst;
    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 ix = 0, ox = 0;
        for (; ix < width_align; ix += 16, ox += 48)
        {
LOOP_BODY:
            v3qu8_dst.val[0] = neon::vload1q(src_row + ix);
            v3qu8_dst.val[1] = v3qu8_dst.val[0];
            v3qu8_dst.val[2] = v3qu8_dst.val[0];
            neon::vstore(dst_row + ox, v3qu8_dst);
        }

        if (ix < width)
        {
            ix = width - width_align;
            ox = ix * 3;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

static Status CvtGray2BgraNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width       = src.GetSizes().m_width;
    DT_S32 width_align = width & (-16);

    uint8x16x4_t v4qu8_dst;
    neon::vdup(v4qu8_dst.val[3], 255);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8 *dst_row       = dst.Ptr<DT_U8>(y);

        DT_S32 ix = 0, ox = 0;
        for (; ix < width_align; ix += 16, ox += 64)
        {
LOOP_BODY:
            v4qu8_dst.val[0] = neon::vload1q(src_row + ix);
            v4qu8_dst.val[1] = v4qu8_dst.val[0];
            v4qu8_dst.val[2] = v4qu8_dst.val[0];
            neon::vstore(dst_row + ox, v4qu8_dst);
        }

        if (ix < width)
        {
            ix = width - width_align;
            ox = ix * 4;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtGray2BgrNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
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

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 ochannel = dst.GetSizes().m_channel;
    DT_S32 height   = src.GetSizes().m_height;
    switch (ochannel)
    {
        case 3:
        {
            ret = wp->ParallelFor(0, height, CvtGray2BgrNeonImpl, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CvtGray2BgrNeonImpl failed");
            }
            break;
        }

        case 4:
        {
            ret = wp->ParallelFor(0, height, CvtGray2BgraNeonImpl, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CvtGray2BgraNeonImpl failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "dst channel should be 3 or 4");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura