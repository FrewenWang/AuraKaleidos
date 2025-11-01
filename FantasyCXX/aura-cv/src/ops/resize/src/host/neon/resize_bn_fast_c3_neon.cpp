#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeBnC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth  = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 owidth_align8 = owidth & (-8);

        MI_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            auto v3q8_c  = neon::vload3q(src_row0);
            auto v3q8_n0 = neon::vload3q(src_row1);

            MVType v3d8_result;
            auto v2q8_ch = neon::vuzp(v3q8_c.val[0], v3q8_n0.val[0]);
            auto vq16_ch = neon::vaddl(neon::vgetlow(v2q8_ch.val[0]), neon::vgethigh(v2q8_ch.val[0]));
            vq16_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q8_ch.val[1])), vq16_ch);
            vq16_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q8_ch.val[1])), vq16_ch);
            v3d8_result.val[0] = neon::vrshrn_n<2>(vq16_ch);

            v2q8_ch = neon::vuzp(v3q8_c.val[1], v3q8_n0.val[1]);
            vq16_ch = neon::vaddl(neon::vgetlow(v2q8_ch.val[0]), neon::vgethigh(v2q8_ch.val[0]));
            vq16_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q8_ch.val[1])), vq16_ch);
            vq16_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q8_ch.val[1])), vq16_ch);
            v3d8_result.val[1] = neon::vrshrn_n<2>(vq16_ch);

            v2q8_ch = neon::vuzp(v3q8_c.val[2], v3q8_n0.val[2]);
            vq16_ch = neon::vaddl(neon::vgetlow(v2q8_ch.val[0]), neon::vgethigh(v2q8_ch.val[0]));
            vq16_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q8_ch.val[1])), vq16_ch);
            vq16_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q8_ch.val[1])), vq16_ch);
            v3d8_result.val[2] = neon::vrshrn_n<2>(vq16_ch);

            neon::vstore(dst_row, v3d8_result);

            src_row0 += 48;
            src_row1 += 48;
            dst_row += 24;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[3];
            MovlType r1 = src_row1[0] + src_row1[3];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[1] + src_row0[4];
            r1 = src_row1[1] + src_row1[4];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[2] + src_row0[5];
            r1 = src_row1[2] + src_row1[5];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 6;
            src_row1 += 6;
        }
    }

    return Status::OK;
}

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeBnC3DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 width_align8 = owidth & (-8);

        MI_S32 x = 0;
        for (; x < width_align8; x += 8)
        {
            auto v3q8_cx0  = neon::vload3q(src_row0);
            auto v3q8_cx1  = neon::vload3q(src_row0 + 48);
            auto v3q8_n0x0 = neon::vload3q(src_row1);
            auto v3q8_n0x1 = neon::vload3q(src_row1 + 48);

            MVType v3d8_result;

            auto v2q8_ch = neon::vuzp(v3q8_cx0.val[0], v3q8_n0x0.val[0]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            auto vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            auto vd16_low = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v2q8_ch = neon::vuzp(v3q8_cx1.val[0], v3q8_n0x1.val[0]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            auto vd16_high = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v3d8_result.val[0] = neon::vrshrn_n<2>(neon::vcombine(vd16_low, vd16_high));

            v2q8_ch = neon::vuzp(v3q8_cx0.val[1], v3q8_n0x0.val[1]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            vd16_low = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v2q8_ch = neon::vuzp(v3q8_cx1.val[1], v3q8_n0x1.val[1]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            vd16_high = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v3d8_result.val[1] = neon::vrshrn_n<2>(neon::vcombine(vd16_low, vd16_high));

            v2q8_ch = neon::vuzp(v3q8_cx0.val[2], v3q8_n0x0.val[2]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            vd16_low = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v2q8_ch = neon::vuzp(v3q8_cx1.val[2], v3q8_n0x1.val[2]);
            v2q8_ch = neon::vuzp(v2q8_ch.val[0], v2q8_ch.val[1]);
            vq16_ch = neon::vaddl(neon::vgethigh(v2q8_ch.val[0]), neon::vgetlow(v2q8_ch.val[1]));
            vd16_high = neon::vadd(neon::vgetlow(vq16_ch), neon::vgethigh(vq16_ch));
            v3d8_result.val[2] = neon::vrshrn_n<2>(neon::vcombine(vd16_low, vd16_high));

            neon::vstore(dst_row, v3d8_result);

            src_row0 += 96;
            src_row1 += 96;
            dst_row  += 24;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[3] + src_row0[6];
            MovlType r1 = src_row1[3] + src_row1[6];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[4] + src_row0[7];
            r1 = src_row1[4] + src_row1[7];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[5] + src_row0[8];
            r1 = src_row1[5] + src_row1[8];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 12;
            src_row1 += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeBnC3UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 3>::MVType;
    using VType    = typename neon::DVector<Tp>::VType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    VType vd8_const_3;
    neon::vdup(vd8_const_3, static_cast<Tp>(3));
    MI_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) * 4;
    *rows1_tmp++ = (*(src_row0 + 1)) * 4;
    *rows1_tmp++ = (*(src_row0 + 2)) * 4;
    MI_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto v3d8_x0 = neon::vload3(src_row0);
        auto v3d8_x1 = neon::vload3(src_row0 + 3);

        auto vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[0]), v3d8_x0.val[0], vd8_const_3);
        auto vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[0]), v3d8_x1.val[0], vd8_const_3);
        auto v2q16_ch0 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

        vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[1]), v3d8_x0.val[1], vd8_const_3);
        vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[1]), v3d8_x1.val[1], vd8_const_3);
        auto v2q16_ch1 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

        vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[2]), v3d8_x0.val[2], vd8_const_3);
        vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[2]), v3d8_x1.val[2], vd8_const_3);
        auto v2q16_ch2 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

        MVType v3q16_result;
        v3q16_result.val[0] = v2q16_ch0.val[0];
        v3q16_result.val[1] = v2q16_ch1.val[0];
        v3q16_result.val[2] = v2q16_ch2.val[0];
        neon::vstore(rows1_tmp, v3q16_result);

        v3q16_result.val[0] = v2q16_ch0.val[1];
        v3q16_result.val[1] = v2q16_ch1.val[1];
        v3q16_result.val[2] = v2q16_ch2.val[1];
        neon::vstore(rows1_tmp + 24, v3q16_result);

        src_row0 += 24;
        rows1_tmp += 48;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[3];
        *rows1_tmp++ = src_row0[1] * 3 + src_row0[4];
        *rows1_tmp++ = src_row0[2] * 3 + src_row0[5];
        *rows1_tmp++ = src_row0[0]     + src_row0[3] * 3;
        *rows1_tmp++ = src_row0[1]     + src_row0[4] * 3;
        *rows1_tmp++ = src_row0[2]     + src_row0[5] * 3;
        src_row0 += 3;
    }
    *rows1_tmp++ = (*src_row0) * 4;
    *rows1_tmp++ = (*(src_row0 + 1)) * 4;
    *rows1_tmp   = (*(src_row0 + 2)) * 4;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp++ = (*(src_r_1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_r_1);
            auto v3d8_x1 = neon::vload3(src_r_1 + 3);

            auto vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[0]), v3d8_x0.val[0], vd8_const_3);
            auto vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[0]), v3d8_x1.val[0], vd8_const_3);
            auto v2q16_ch0  = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[1]), v3d8_x0.val[1], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[1]), v3d8_x1.val[1], vd8_const_3);
            auto v2q16_ch1 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[2]), v3d8_x0.val[2], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[2]), v3d8_x1.val[2], vd8_const_3);
            auto v2q16_ch2 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            MVType v3q16_result;
            v3q16_result.val[0] = v2q16_ch0.val[0];
            v3q16_result.val[1] = v2q16_ch1.val[0];
            v3q16_result.val[2] = v2q16_ch2.val[0];
            neon::vstore(rows0_tmp, v3q16_result);

            v3q16_result.val[0] = v2q16_ch0.val[1];
            v3q16_result.val[1] = v2q16_ch1.val[1];
            v3q16_result.val[2] = v2q16_ch2.val[1];
            neon::vstore(rows0_tmp + 24, v3q16_result);

            src_r_1 += 24;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 3;
            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp   = (*(src_r_1 + 2)) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_result = neon::vmla(vq16_c, vq16_n0, static_cast<MovlType>(3));
            auto vd8_result1 = neon::vrshrn_n<4>(vq16_n0_result);

            neon::vstore(dst_row, vd8_result1);

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (MI_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        MI_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = (*src_row1) * 4;
        *rows1_tmp++ = (*(src_row1 + 1)) * 4;
        *rows1_tmp++ = (*(src_row1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_row1);
            auto v3d8_x1 = neon::vload3(src_row1 + 3);

            auto vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[0]), v3d8_x0.val[0], vd8_const_3);
            auto vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[0]), v3d8_x1.val[0], vd8_const_3);
            auto v2q16_ch0  = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[1]), v3d8_x0.val[1], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[1]), v3d8_x1.val[1], vd8_const_3);
            auto v2q16_ch1 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[2]), v3d8_x0.val[2], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[2]), v3d8_x1.val[2], vd8_const_3);
            auto v2q16_ch2 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            MVType v3q16_result;
            v3q16_result.val[0] = v2q16_ch0.val[0];
            v3q16_result.val[1] = v2q16_ch1.val[0];
            v3q16_result.val[2] = v2q16_ch2.val[0];
            neon::vstore(rows1_tmp, v3q16_result);

            v3q16_result.val[0] = v2q16_ch0.val[1];
            v3q16_result.val[1] = v2q16_ch1.val[1];
            v3q16_result.val[2] = v2q16_ch2.val[1];
            neon::vstore(rows1_tmp + 24, v3q16_result);

            src_row1 += 24;
            rows1_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[3];
            *rows1_tmp++ = src_row1[1] * 3 + src_row1[4];
            *rows1_tmp++ = src_row1[2] * 3 + src_row1[5];
            *rows1_tmp++ = src_row1[0]     + src_row1[3] * 3;
            *rows1_tmp++ = src_row1[1]     + src_row1[4] * 3;
            *rows1_tmp++ = src_row1[2]     + src_row1[5] * 3;
            src_row1 += 3;
        }
        *rows1_tmp++ = (*src_row1) * 4;
        *rows1_tmp++ = (*(src_row1 + 1)) * 4;
        *rows1_tmp   = (*(src_row1 + 2)) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_c_result  = neon::vmla(vq16_n0, vq16_c, static_cast<MovlType>(3));
            auto vq16_n0_result = neon::vmla(vq16_c, vq16_n0, static_cast<MovlType>(3));

            auto vd8_result0 = neon::vrshrn_n<4>(vq16_c_result);
            auto vd8_result1 = neon::vrshrn_n<4>(vq16_n0_result);

            neon::vstore(dst_row0, vd8_result0);
            neon::vstore(dst_row1, vd8_result1);

            dst_row0 += 8;
            dst_row1 += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0]     + 8) >> 4);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp++ = (*(src_r_1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_r_1);
            auto v3d8_x1 = neon::vload3(src_r_1 + 3);

            auto vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[0]), v3d8_x0.val[0], vd8_const_3);
            auto vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[0]), v3d8_x1.val[0], vd8_const_3);
            auto v2q16_ch0  = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[1]), v3d8_x0.val[1], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[1]), v3d8_x1.val[1], vd8_const_3);
            auto v2q16_ch1 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            vq16_x0_ch = neon::vmlal(neon::vmovl(v3d8_x1.val[2]), v3d8_x0.val[2], vd8_const_3);
            vq16_x1_ch = neon::vmlal(neon::vmovl(v3d8_x0.val[2]), v3d8_x1.val[2], vd8_const_3);
            auto v2q16_ch2 = neon::vzip(vq16_x0_ch, vq16_x1_ch);

            MVType v3q16_result;
            v3q16_result.val[0] = v2q16_ch0.val[0];
            v3q16_result.val[1] = v2q16_ch1.val[0];
            v3q16_result.val[2] = v2q16_ch2.val[0];
            neon::vstore(rows0_tmp, v3q16_result);

            v3q16_result.val[0] = v2q16_ch0.val[1];
            v3q16_result.val[1] = v2q16_ch1.val[1];
            v3q16_result.val[2] = v2q16_ch2.val[1];
            neon::vstore(rows0_tmp + 24, v3q16_result);

            src_r_1 += 24;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 3;
            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp   = (*(src_r_1 + 2)) * 4;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_c_result = neon::vmla(vq16_n0, vq16_c, static_cast<MovlType>(3));
            auto vd8_result0 = neon::vrshrn_n<4>(vq16_c_result);

            neon::vstore(dst_row, vd8_result0);

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0]     + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeBnC3UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 3>::MVType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    MI_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto v3d8_x0 = neon::vload3(src_row0);
        auto v3d8_x1 = neon::vload3(src_row0 + 3);

        auto vq16_x0 = neon::vmovl(v3d8_x0.val[0]);
        auto vq16_x1 = neon::vmovl(v3d8_x1.val[0]);
        auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
        auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
        auto vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
        auto vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
        auto vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
        auto vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
        auto v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
        auto v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
        auto v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
        auto vq16_n0_ch0 = v2q16_tmp.val[0];
        auto vq16_n1_ch0 = v2q16_tmp.val[1];
        v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
        auto vq16_n2_ch0 = v2q16_tmp.val[0];
        auto vq16_n3_ch0 = v2q16_tmp.val[1];

        vq16_x0 = neon::vmovl(v3d8_x0.val[1]);
        vq16_x1 = neon::vmovl(v3d8_x1.val[1]);
        vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
        vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
        vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
        vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
        vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
        vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
        v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
        v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
        v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
        auto vq16_n0_ch1 = v2q16_tmp.val[0];
        auto vq16_n1_ch1 = v2q16_tmp.val[1];
        v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
        auto vq16_n2_ch1 = v2q16_tmp.val[0];
        auto vq16_n3_ch1 = v2q16_tmp.val[1];

        vq16_x0 = neon::vmovl(v3d8_x0.val[2]);
        vq16_x1 = neon::vmovl(v3d8_x1.val[2]);
        vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
        vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
        vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
        vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
        vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
        vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
        v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
        v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
        v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
        auto vq16_n0_ch2 = v2q16_tmp.val[0];
        auto vq16_n1_ch2 = v2q16_tmp.val[1];
        v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
        auto vq16_n2_ch2 = v2q16_tmp.val[0];
        auto vq16_n3_ch2 = v2q16_tmp.val[1];

        MVType v3q16_result;
        v3q16_result.val[0] = vq16_n0_ch0;
        v3q16_result.val[1] = vq16_n0_ch1;
        v3q16_result.val[2] = vq16_n0_ch2;
        neon::vstore(rows1_tmp, v3q16_result);

        v3q16_result.val[0] = vq16_n1_ch0;
        v3q16_result.val[1] = vq16_n1_ch1;
        v3q16_result.val[2] = vq16_n1_ch2;
        neon::vstore(rows1_tmp + 24, v3q16_result);

        v3q16_result.val[0] = vq16_n2_ch0;
        v3q16_result.val[1] = vq16_n2_ch1;
        v3q16_result.val[2] = vq16_n2_ch2;
        neon::vstore(rows1_tmp + 48, v3q16_result);

        v3q16_result.val[0] = vq16_n3_ch0;
        v3q16_result.val[1] = vq16_n3_ch1;
        v3q16_result.val[2] = vq16_n3_ch2;
        neon::vstore(rows1_tmp + 72, v3q16_result);

        src_row0 += 24;
        rows1_tmp += 96;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 7 + src_row0[3];
        *rows1_tmp++ = src_row0[1] * 7 + src_row0[4];
        *rows1_tmp++ = src_row0[2] * 7 + src_row0[5];
        *rows1_tmp++ = src_row0[0] * 5 + src_row0[3] * 3;
        *rows1_tmp++ = src_row0[1] * 5 + src_row0[4] * 3;
        *rows1_tmp++ = src_row0[2] * 5 + src_row0[5] * 3;
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[3] * 5;
        *rows1_tmp++ = src_row0[1] * 3 + src_row0[4] * 5;
        *rows1_tmp++ = src_row0[2] * 3 + src_row0[5] * 5;
        *rows1_tmp++ = src_row0[0]     + src_row0[3] * 7;
        *rows1_tmp++ = src_row0[1]     + src_row0[4] * 7;
        *rows1_tmp++ = src_row0[2]     + src_row0[5] * 7;

        src_row0 += 3;
    }
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp   = (*(src_row0 + 2)) << 3;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);

            rows1_tmp += 3;
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_r_1);
            auto v3d8_x1 = neon::vload3(src_r_1 + 3);

            auto vq16_x0 = neon::vmovl(v3d8_x0.val[0]);
            auto vq16_x1 = neon::vmovl(v3d8_x1.val[0]);
            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            auto vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            auto vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            auto vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            auto vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            auto v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            auto v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            auto v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch0 = v2q16_tmp.val[0];
            auto vq16_n1_ch0 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch0 = v2q16_tmp.val[0];
            auto vq16_n3_ch0 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[1]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[1]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch1 = v2q16_tmp.val[0];
            auto vq16_n1_ch1 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch1 = v2q16_tmp.val[0];
            auto vq16_n3_ch1 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[2]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[2]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch2 = v2q16_tmp.val[0];
            auto vq16_n1_ch2 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch2 = v2q16_tmp.val[0];
            auto vq16_n3_ch2 = v2q16_tmp.val[1];

            MVType v3q16_result;
            v3q16_result.val[0] = vq16_n0_ch0;
            v3q16_result.val[1] = vq16_n0_ch1;
            v3q16_result.val[2] = vq16_n0_ch2;
            neon::vstore(rows0_tmp, v3q16_result);

            v3q16_result.val[0] = vq16_n1_ch0;
            v3q16_result.val[1] = vq16_n1_ch1;
            v3q16_result.val[2] = vq16_n1_ch2;
            neon::vstore(rows0_tmp + 24, v3q16_result);

            v3q16_result.val[0] = vq16_n2_ch0;
            v3q16_result.val[1] = vq16_n2_ch1;
            v3q16_result.val[2] = vq16_n2_ch2;
            neon::vstore(rows0_tmp + 48, v3q16_result);

            v3q16_result.val[0] = vq16_n3_ch0;
            v3q16_result.val[1] = vq16_n3_ch1;
            v3q16_result.val[2] = vq16_n3_ch2;
            neon::vstore(rows0_tmp + 72, v3q16_result);

            src_r_1 += 24;
            rows0_tmp += 96;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 7 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 7 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1] * 5 + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2] * 5 + src_r_1[5] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3] * 5;
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4] * 5;
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 7;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 7;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 7;

            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp   = (*(src_r_1 + 2)) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_x5   = neon::vmul(vq16_n0, static_cast<MovlType>(5));
            auto vq16_result2 = neon::vmla(vq16_n0_x5, vq16_c, static_cast<MovlType>(3));
            auto vq16_result3 = neon::vmla(vq16_c,    vq16_n0, static_cast<MovlType>(7));

            auto vd8_result2 = neon::vrshrn_n<6>(vq16_result2);
            auto vd8_result3 = neon::vrshrn_n<6>(vq16_result3);

            neon::vstore(dst_row0, vd8_result2);
            neon::vstore(dst_row1, vd8_result3);

            rows0_y += 8;
            rows1_y += 8;

            dst_row0 += 8;
            dst_row1 += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (MI_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        MI_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_row1);
            auto v3d8_x1 = neon::vload3(src_row1 + 3);

            auto vq16_x0 = neon::vmovl(v3d8_x0.val[0]);
            auto vq16_x1 = neon::vmovl(v3d8_x1.val[0]);
            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            auto vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            auto vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            auto vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            auto vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            auto v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            auto v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            auto v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch0 = v2q16_tmp.val[0];
            auto vq16_n1_ch0 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch0 = v2q16_tmp.val[0];
            auto vq16_n3_ch0 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[1]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[1]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch1 = v2q16_tmp.val[0];
            auto vq16_n1_ch1 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch1 = v2q16_tmp.val[0];
            auto vq16_n3_ch1 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[2]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[2]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch2 = v2q16_tmp.val[0];
            auto vq16_n1_ch2 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch2 = v2q16_tmp.val[0];
            auto vq16_n3_ch2 = v2q16_tmp.val[1];

            MVType v3q16_result;
            v3q16_result.val[0] = vq16_n0_ch0;
            v3q16_result.val[1] = vq16_n0_ch1;
            v3q16_result.val[2] = vq16_n0_ch2;
            neon::vstore(rows1_tmp, v3q16_result);

            v3q16_result.val[0] = vq16_n1_ch0;
            v3q16_result.val[1] = vq16_n1_ch1;
            v3q16_result.val[2] = vq16_n1_ch2;
            neon::vstore(rows1_tmp + 24, v3q16_result);

            v3q16_result.val[0] = vq16_n2_ch0;
            v3q16_result.val[1] = vq16_n2_ch1;
            v3q16_result.val[2] = vq16_n2_ch2;
            neon::vstore(rows1_tmp + 48, v3q16_result);

            v3q16_result.val[0] = vq16_n3_ch0;
            v3q16_result.val[1] = vq16_n3_ch1;
            v3q16_result.val[2] = vq16_n3_ch2;
            neon::vstore(rows1_tmp + 72, v3q16_result);

            src_row1 += 24;
            rows1_tmp += 96;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 7 + src_row1[3];
            *rows1_tmp++ = src_row1[1] * 7 + src_row1[4];
            *rows1_tmp++ = src_row1[2] * 7 + src_row1[5];
            *rows1_tmp++ = src_row1[0] * 5 + src_row1[3] * 3;
            *rows1_tmp++ = src_row1[1] * 5 + src_row1[4] * 3;
            *rows1_tmp++ = src_row1[2] * 5 + src_row1[5] * 3;
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[3] * 5;
            *rows1_tmp++ = src_row1[1] * 3 + src_row1[4] * 5;
            *rows1_tmp++ = src_row1[2] * 3 + src_row1[5] * 5;
            *rows1_tmp++ = src_row1[0]     + src_row1[3] * 7;
            *rows1_tmp++ = src_row1[1]     + src_row1[4] * 7;
            *rows1_tmp++ = src_row1[2]     + src_row1[5] * 7;

            src_row1 += 3;
        }
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp   = (*(src_row1 + 2)) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_x3 = neon::vmul(vq16_n0, static_cast<MovlType>(3));
            auto vq16_n0_x5 = neon::vmul(vq16_n0, static_cast<MovlType>(5));

            auto vq16_result0 = neon::vmla(vq16_n0,    vq16_c, static_cast<MovlType>(7));
            auto vq16_result1 = neon::vmla(vq16_n0_x3, vq16_c, static_cast<MovlType>(5));
            auto vq16_result2 = neon::vmla(vq16_n0_x5, vq16_c, static_cast<MovlType>(3));
            auto vq16_result3 = neon::vmla(vq16_c,    vq16_n0, static_cast<MovlType>(7));

            auto vd8_result0 = neon::vrshrn_n<6>(vq16_result0);
            auto vd8_result1 = neon::vrshrn_n<6>(vq16_result1);
            auto vd8_result2 = neon::vrshrn_n<6>(vq16_result2);
            auto vd8_result3 = neon::vrshrn_n<6>(vq16_result3);

            neon::vstore(dst_row0, vd8_result0);
            neon::vstore(dst_row1, vd8_result1);
            neon::vstore(dst_row2, vd8_result2);
            neon::vstore(dst_row3, vd8_result3);

            rows0_y += 8;
            rows1_y += 8;

            dst_row0 += 8;
            dst_row1 += 8;
            dst_row2 += 8;
            dst_row3 += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);
            *dst_row2++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row3++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row0 = dst.Ptr<Tp>(end_row - 2);
    dst_row1 = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            rows1_tmp += 3;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto v3d8_x0 = neon::vload3(src_r_1);
            auto v3d8_x1 = neon::vload3(src_r_1 + 3);

            auto vq16_x0 = neon::vmovl(v3d8_x0.val[0]);
            auto vq16_x1 = neon::vmovl(v3d8_x1.val[0]);
            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            auto vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            auto vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            auto vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            auto vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            auto v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            auto v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            auto v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch0 = v2q16_tmp.val[0];
            auto vq16_n1_ch0 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch0 = v2q16_tmp.val[0];
            auto vq16_n3_ch0 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[1]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[1]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch1 = v2q16_tmp.val[0];
            auto vq16_n1_ch1 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch1 = v2q16_tmp.val[0];
            auto vq16_n3_ch1 = v2q16_tmp.val[1];

            vq16_x0 = neon::vmovl(v3d8_x0.val[2]);
            vq16_x1 = neon::vmovl(v3d8_x1.val[2]);
            vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));
            vq16_n0 = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            vq16_n1 = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            vq16_n2 = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));
            vq16_n3 = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v2q16_n0n2 = neon::vzip(vq16_n0, vq16_n2);
            v2q16_n1n3 = neon::vzip(vq16_n1, vq16_n3);
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[0], v2q16_n1n3.val[0]);
            auto vq16_n0_ch2 = v2q16_tmp.val[0];
            auto vq16_n1_ch2 = v2q16_tmp.val[1];
            v2q16_tmp = neon::vzip(v2q16_n0n2.val[1], v2q16_n1n3.val[1]);
            auto vq16_n2_ch2 = v2q16_tmp.val[0];
            auto vq16_n3_ch2 = v2q16_tmp.val[1];

            MVType v3q16_result;
            v3q16_result.val[0] = vq16_n0_ch0;
            v3q16_result.val[1] = vq16_n0_ch1;
            v3q16_result.val[2] = vq16_n0_ch2;
            neon::vstore(rows0_tmp, v3q16_result);

            v3q16_result.val[0] = vq16_n1_ch0;
            v3q16_result.val[1] = vq16_n1_ch1;
            v3q16_result.val[2] = vq16_n1_ch2;
            neon::vstore(rows0_tmp + 24, v3q16_result);

            v3q16_result.val[0] = vq16_n2_ch0;
            v3q16_result.val[1] = vq16_n2_ch1;
            v3q16_result.val[2] = vq16_n2_ch2;
            neon::vstore(rows0_tmp + 48, v3q16_result);

            v3q16_result.val[0] = vq16_n3_ch0;
            v3q16_result.val[1] = vq16_n3_ch1;
            v3q16_result.val[2] = vq16_n3_ch2;
            neon::vstore(rows0_tmp + 72, v3q16_result);

            src_r_1 += 24;
            rows0_tmp += 96;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 7 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 7 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1] * 5 + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2] * 5 + src_r_1[5] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3] * 5;
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4] * 5;
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 7;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 7;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 7;

            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp   = (*(src_r_1 + 2)) << 3;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align8 = owidth_x3 & (-8);
        x = 0;
        for(; x < owidth_x3_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_x3   = neon::vmul(vq16_n0, static_cast<MovlType>(3));
            auto vq16_result0 = neon::vmla(vq16_n0,    vq16_c, static_cast<MovlType>(7));
            auto vq16_result1 = neon::vmla(vq16_n0_x3, vq16_c, static_cast<MovlType>(5));

            auto vd8_result0 = neon::vrshrn_n<6>(vq16_result0);
            auto vd8_result1 = neon::vrshrn_n<6>(vq16_result1);

            neon::vstore(dst_row0, vd8_result0);
            neon::vstore(dst_row1, vd8_result1);

            rows0_y += 8;
            rows1_y += 8;

            dst_row0 += 8;
            dst_row1 += 8;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeBnC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 owidth_align4 = owidth & (-4);
        MI_S32 x = 0;
        for (; x < owidth_align4; x += 4)
        {
            auto v3q16_c  = neon::vload3q(src_row0);
            auto v3q16_n0 = neon::vload3q(src_row1);

            MVType v3d16_result;
            auto v2q16_ch = neon::vuzp(v3q16_c.val[0], v3q16_n0.val[0]);
            auto vq32_ch = neon::vaddl(neon::vgetlow(v2q16_ch.val[0]), neon::vgethigh(v2q16_ch.val[0]));
            vq32_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q16_ch.val[1])), vq32_ch);
            vq32_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q16_ch.val[1])), vq32_ch);
            v3d16_result.val[0] = neon::vrshrn_n<2>(vq32_ch);

            v2q16_ch = neon::vuzp(v3q16_c.val[1], v3q16_n0.val[1]);
            vq32_ch = neon::vaddl(neon::vgetlow(v2q16_ch.val[0]), neon::vgethigh(v2q16_ch.val[0]));
            vq32_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q16_ch.val[1])), vq32_ch);
            vq32_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q16_ch.val[1])), vq32_ch);
            v3d16_result.val[1] = neon::vrshrn_n<2>(vq32_ch);

            v2q16_ch = neon::vuzp(v3q16_c.val[2], v3q16_n0.val[2]);
            vq32_ch = neon::vaddl(neon::vgetlow(v2q16_ch.val[0]), neon::vgethigh(v2q16_ch.val[0]));
            vq32_ch = neon::vadd(neon::vmovl(neon::vgetlow(v2q16_ch.val[1])), vq32_ch);
            vq32_ch = neon::vadd(neon::vmovl(neon::vgethigh(v2q16_ch.val[1])), vq32_ch);
            v3d16_result.val[2] = neon::vrshrn_n<2>(vq32_ch);

            neon::vstore(dst_row, v3d16_result);

            src_row0 += 24;
            src_row1 += 24;
            dst_row  += 12;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[3];
            MovlType r1 = src_row1[0] + src_row1[3];
            *dst_row++  = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[1] + src_row0[4];
            r1 = src_row1[1] + src_row1[4];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[2] + src_row0[5];
            r1 = src_row1[2] + src_row1[5];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 6;
            src_row1 += 6;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeBnC3DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 width_align4 = owidth & (-4);
        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            auto v3q16_cx0  = neon::vload3q(src_row0);
            auto v3q16_cx1  = neon::vload3q(src_row0 + 24);
            auto v3q16_n0x0 = neon::vload3q(src_row1);
            auto v3q16_n0x1 = neon::vload3q(src_row1 + 24);

            MVType v3d16_result;

            auto v2q16_ch = neon::vuzp(v3q16_cx0.val[0], v3q16_n0x0.val[0]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            auto vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            auto vd32_low = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v2q16_ch = neon::vuzp(v3q16_cx1.val[0], v3q16_n0x1.val[0]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            auto vd32_high = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v3d16_result.val[0] = neon::vrshrn_n<2>(neon::vcombine(vd32_low, vd32_high));

            v2q16_ch = neon::vuzp(v3q16_cx0.val[1], v3q16_n0x0.val[1]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            vd32_low = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v2q16_ch = neon::vuzp(v3q16_cx1.val[1], v3q16_n0x1.val[1]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            vd32_high = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v3d16_result.val[1] = neon::vrshrn_n<2>(neon::vcombine(vd32_low, vd32_high));

            v2q16_ch = neon::vuzp(v3q16_cx0.val[2], v3q16_n0x0.val[2]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            vd32_low = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v2q16_ch = neon::vuzp(v3q16_cx1.val[2], v3q16_n0x1.val[2]);
            v2q16_ch = neon::vuzp(v2q16_ch.val[0], v2q16_ch.val[1]);
            vq32_ch = neon::vaddl(neon::vgethigh(v2q16_ch.val[0]), neon::vgetlow(v2q16_ch.val[1]));
            vd32_high = neon::vadd(neon::vgetlow(vq32_ch), neon::vgethigh(vq32_ch));
            v3d16_result.val[2] = neon::vrshrn_n<2>(neon::vcombine(vd32_low, vd32_high));
            neon::vstore(dst_row, v3d16_result);

            src_row0 += 48;
            src_row1 += 48;
            dst_row += 12;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[3] + src_row0[6];
            MovlType r1 = src_row1[3] + src_row1[6];
            *dst_row++  = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[4] + src_row0[7];
            r1 = src_row1[4] + src_row1[7];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);
            r0 = src_row0[5] + src_row0[8];
            r1 = src_row1[5] + src_row1[8];
            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 12;
            src_row1 += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeBnC3UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 3>::MVType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = (*src_row0) * 4;
    *rows1_tmp++ = (*(src_row0 + 1)) * 4;
    *rows1_tmp++ = (*(src_row0 + 2)) * 4;
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        auto v3d16_x0 = neon::vload3(src_row0);
        auto v3d16_x1 = neon::vload3(src_row0 + 3);

        auto vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[0]), v3d16_x0.val[0], static_cast<Tp>(3));
        auto vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[0]), v3d16_x1.val[0], static_cast<Tp>(3));
        auto v2q32_ch0 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

        vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[1]), v3d16_x0.val[1], static_cast<Tp>(3));
        vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[1]), v3d16_x1.val[1], static_cast<Tp>(3));
        auto v2q32_ch1 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

        vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[2]), v3d16_x0.val[2], static_cast<Tp>(3));
        vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[2]), v3d16_x1.val[2], static_cast<Tp>(3));
        auto v2q32_ch2 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

        MVType v3q32_result;
        v3q32_result.val[0] = v2q32_ch0.val[0];
        v3q32_result.val[1] = v2q32_ch1.val[0];
        v3q32_result.val[2] = v2q32_ch2.val[0];
        neon::vstore(rows1_tmp, v3q32_result);

        v3q32_result.val[0] = v2q32_ch0.val[1];
        v3q32_result.val[1] = v2q32_ch1.val[1];
        v3q32_result.val[2] = v2q32_ch2.val[1];
        neon::vstore(rows1_tmp + 12, v3q32_result);

        src_row0 += 12;
        rows1_tmp += 24;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[3];
        *rows1_tmp++ = src_row0[1] * 3 + src_row0[4];
        *rows1_tmp++ = src_row0[2] * 3 + src_row0[5];
        *rows1_tmp++ = src_row0[0]     + src_row0[3] * 3;
        *rows1_tmp++ = src_row0[1]     + src_row0[4] * 3;
        *rows1_tmp++ = src_row0[2]     + src_row0[5] * 3;
        src_row0 += 3;
    }
    *rows1_tmp++ = (*src_row0) * 4;
    *rows1_tmp++ = (*(src_row0 + 1)) * 4;
    *rows1_tmp   = (*(src_row0 + 2)) * 4;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp++ = (*(src_r_1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_r_1);
            auto v3d16_x1 = neon::vload3(src_r_1 + 3);

            auto vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[0]), v3d16_x0.val[0], static_cast<Tp>(3));
            auto vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[0]), v3d16_x1.val[0], static_cast<Tp>(3));
            auto v2q32_ch0 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[1]), v3d16_x0.val[1], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[1]), v3d16_x1.val[1], static_cast<Tp>(3));
            auto v2q32_ch1 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[2]), v3d16_x0.val[2], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[2]), v3d16_x1.val[2], static_cast<Tp>(3));
            auto v2q32_ch2 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            MVType v3q32_result;
            v3q32_result.val[0] = v2q32_ch0.val[0];
            v3q32_result.val[1] = v2q32_ch1.val[0];
            v3q32_result.val[2] = v2q32_ch2.val[0];
            neon::vstore(rows0_tmp, v3q32_result);

            v3q32_result.val[0] = v2q32_ch0.val[1];
            v3q32_result.val[1] = v2q32_ch1.val[1];
            v3q32_result.val[2] = v2q32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3q32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 3;
            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp   = (*(src_r_1 + 2)) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_n0_result = neon::vmla(vq32_c, vq32_n0, static_cast<MovlType>(3));
            auto vd16_result1 = neon::vrshrn_n<4>(vq32_n0_result);

            neon::vstore(dst_row, vd16_result1);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (MI_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        MI_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = (*src_row1) * 4;
        *rows1_tmp++ = (*(src_row1 + 1)) * 4;
        *rows1_tmp++ = (*(src_row1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_row1);
            auto v3d16_x1 = neon::vload3(src_row1 + 3);

            auto vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[0]), v3d16_x0.val[0], static_cast<Tp>(3));
            auto vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[0]), v3d16_x1.val[0], static_cast<Tp>(3));
            auto v2q32_ch0 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[1]), v3d16_x0.val[1], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[1]), v3d16_x1.val[1], static_cast<Tp>(3));
            auto v2q32_ch1 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[2]), v3d16_x0.val[2], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[2]), v3d16_x1.val[2], static_cast<Tp>(3));
            auto v2q32_ch2 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            MVType v3q32_result;
            v3q32_result.val[0] = v2q32_ch0.val[0];
            v3q32_result.val[1] = v2q32_ch1.val[0];
            v3q32_result.val[2] = v2q32_ch2.val[0];
            neon::vstore(rows1_tmp, v3q32_result);

            v3q32_result.val[0] = v2q32_ch0.val[1];
            v3q32_result.val[1] = v2q32_ch1.val[1];
            v3q32_result.val[2] = v2q32_ch2.val[1];
            neon::vstore(rows1_tmp + 12, v3q32_result);

            src_row1 += 12;
            rows1_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[3];
            *rows1_tmp++ = src_row1[1] * 3 + src_row1[4];
            *rows1_tmp++ = src_row1[2] * 3 + src_row1[5];
            *rows1_tmp++ = src_row1[0]     + src_row1[3] * 3;
            *rows1_tmp++ = src_row1[1]     + src_row1[4] * 3;
            *rows1_tmp++ = src_row1[2]     + src_row1[5] * 3;
            src_row1 += 3;
        }
        *rows1_tmp++ = (*src_row1) * 4;
        *rows1_tmp++ = (*(src_row1 + 1)) * 4;
        *rows1_tmp   = (*(src_row1 + 2)) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_c_result  = neon::vmla(vq32_n0, vq32_c, static_cast<MovlType>(3));
            auto vq32_n0_result = neon::vmla(vq32_c, vq32_n0, static_cast<MovlType>(3));

            auto vd16_result0 = neon::vrshrn_n<4>(vq32_c_result);
            auto vd16_result1 = neon::vrshrn_n<4>(vq32_n0_result);

            neon::vstore(dst_row0, vd16_result0);
            neon::vstore(dst_row1, vd16_result1);

            dst_row0 += 4;
            dst_row1 += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0]     + 8) >> 4);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp++ = (*(src_r_1 + 2)) * 4;

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_r_1);
            auto v3d16_x1 = neon::vload3(src_r_1 + 3);

            auto vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[0]), v3d16_x0.val[0], static_cast<Tp>(3));
            auto vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[0]), v3d16_x1.val[0], static_cast<Tp>(3));
            auto v2q32_ch0  = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[1]), v3d16_x0.val[1], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[1]), v3d16_x1.val[1], static_cast<Tp>(3));
            auto v2q32_ch1 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            vq32_x0_ch = neon::vmlal(neon::vmovl(v3d16_x1.val[2]), v3d16_x0.val[2], static_cast<Tp>(3));
            vq32_x1_ch = neon::vmlal(neon::vmovl(v3d16_x0.val[2]), v3d16_x1.val[2], static_cast<Tp>(3));
            auto v2q32_ch2 = neon::vzip(vq32_x0_ch, vq32_x1_ch);

            MVType v3q32_result;
            v3q32_result.val[0] = v2q32_ch0.val[0];
            v3q32_result.val[1] = v2q32_ch1.val[0];
            v3q32_result.val[2] = v2q32_ch2.val[0];
            neon::vstore(rows0_tmp, v3q32_result);

            v3q32_result.val[0] = v2q32_ch0.val[1];
            v3q32_result.val[1] = v2q32_ch1.val[1];
            v3q32_result.val[2] = v2q32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3q32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 3;
            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) * 4;
        *rows0_tmp++ = (*(src_r_1 + 1)) * 4;
        *rows0_tmp   = (*(src_r_1 + 2)) * 4;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_c_result = neon::vmla(vq32_n0, vq32_c, static_cast<MovlType>(3));
            auto vd16_result0 = neon::vrshrn_n<4>(vq32_c_result);

            neon::vstore(dst_row, vd16_result0);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0]     + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeBnC3UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 3>::MVType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        auto v3d16_x0 = neon::vload3(src_row0);
        auto v3d16_x1 = neon::vload3(src_row0 + 3);

        auto vq32_x0 = neon::vmovl(v3d16_x0.val[0]);
        auto vq32_x1 = neon::vmovl(v3d16_x1.val[0]);
        auto vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
        auto vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
        auto vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
        auto vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
        auto vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
        auto vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
        auto v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
        auto v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
        auto v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
        auto vq32_n0_ch0 = v2q32_tmp.val[0];
        auto vq32_n1_ch0 = v2q32_tmp.val[1];
        v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
        auto vq32_n2_ch0 = v2q32_tmp.val[0];
        auto vq32_n3_ch0 = v2q32_tmp.val[1];

        vq32_x0 = neon::vmovl(v3d16_x0.val[1]);
        vq32_x1 = neon::vmovl(v3d16_x1.val[1]);
        vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
        vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
        vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
        vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
        vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
        vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
        v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
        v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
        v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
        auto vq32_n0_ch1 = v2q32_tmp.val[0];
        auto vq32_n1_ch1 = v2q32_tmp.val[1];
        v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
        auto vq32_n2_ch1 = v2q32_tmp.val[0];
        auto vq32_n3_ch1 = v2q32_tmp.val[1];

        vq32_x0 = neon::vmovl(v3d16_x0.val[2]);
        vq32_x1 = neon::vmovl(v3d16_x1.val[2]);
        vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
        vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
        vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
        vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
        vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
        vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
        v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
        v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
        v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
        auto vq32_n0_ch2 = v2q32_tmp.val[0];
        auto vq32_n1_ch2 = v2q32_tmp.val[1];
        v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
        auto vq32_n2_ch2 = v2q32_tmp.val[0];
        auto vq32_n3_ch2 = v2q32_tmp.val[1];

        MVType v3q32_result;
        v3q32_result.val[0] = vq32_n0_ch0;
        v3q32_result.val[1] = vq32_n0_ch1;
        v3q32_result.val[2] = vq32_n0_ch2;
        neon::vstore(rows1_tmp, v3q32_result);

        v3q32_result.val[0] = vq32_n1_ch0;
        v3q32_result.val[1] = vq32_n1_ch1;
        v3q32_result.val[2] = vq32_n1_ch2;
        neon::vstore(rows1_tmp + 12, v3q32_result);

        v3q32_result.val[0] = vq32_n2_ch0;
        v3q32_result.val[1] = vq32_n2_ch1;
        v3q32_result.val[2] = vq32_n2_ch2;
        neon::vstore(rows1_tmp + 24, v3q32_result);

        v3q32_result.val[0] = vq32_n3_ch0;
        v3q32_result.val[1] = vq32_n3_ch1;
        v3q32_result.val[2] = vq32_n3_ch2;
        neon::vstore(rows1_tmp + 36, v3q32_result);

        src_row0 += 12;
        rows1_tmp += 48;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 7 + src_row0[3];
        *rows1_tmp++ = src_row0[1] * 7 + src_row0[4];
        *rows1_tmp++ = src_row0[2] * 7 + src_row0[5];
        *rows1_tmp++ = src_row0[0] * 5 + src_row0[3] * 3;
        *rows1_tmp++ = src_row0[1] * 5 + src_row0[4] * 3;
        *rows1_tmp++ = src_row0[2] * 5 + src_row0[5] * 3;
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[3] * 5;
        *rows1_tmp++ = src_row0[1] * 3 + src_row0[4] * 5;
        *rows1_tmp++ = src_row0[2] * 3 + src_row0[5] * 5;
        *rows1_tmp++ = src_row0[0]     + src_row0[3] * 7;
        *rows1_tmp++ = src_row0[1]     + src_row0[4] * 7;
        *rows1_tmp++ = src_row0[2]     + src_row0[5] * 7;

        src_row0 += 3;
    }
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp++ = (*(src_row0 + 2)) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*(src_row0 + 1)) << 3;
    *rows1_tmp   = (*(src_row0 + 2)) << 3;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);

            rows1_tmp += 3;
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_r_1);
            auto v3d16_x1 = neon::vload3(src_r_1 + 3);

            auto vq32_x0 = neon::vmovl(v3d16_x0.val[0]);
            auto vq32_x1 = neon::vmovl(v3d16_x1.val[0]);
            auto vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            auto vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            auto vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            auto vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            auto vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            auto vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            auto v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            auto v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            auto v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch0 = v2q32_tmp.val[0];
            auto vq32_n1_ch0 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch0 = v2q32_tmp.val[0];
            auto vq32_n3_ch0 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[1]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[1]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch1 = v2q32_tmp.val[0];
            auto vq32_n1_ch1 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch1 = v2q32_tmp.val[0];
            auto vq32_n3_ch1 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[2]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[2]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch2 = v2q32_tmp.val[0];
            auto vq32_n1_ch2 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch2 = v2q32_tmp.val[0];
            auto vq32_n3_ch2 = v2q32_tmp.val[1];

            MVType v3q32_result;
            v3q32_result.val[0] = vq32_n0_ch0;
            v3q32_result.val[1] = vq32_n0_ch1;
            v3q32_result.val[2] = vq32_n0_ch2;
            neon::vstore(rows0_tmp, v3q32_result);

            v3q32_result.val[0] = vq32_n1_ch0;
            v3q32_result.val[1] = vq32_n1_ch1;
            v3q32_result.val[2] = vq32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3q32_result);

            v3q32_result.val[0] = vq32_n2_ch0;
            v3q32_result.val[1] = vq32_n2_ch1;
            v3q32_result.val[2] = vq32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3q32_result);

            v3q32_result.val[0] = vq32_n3_ch0;
            v3q32_result.val[1] = vq32_n3_ch1;
            v3q32_result.val[2] = vq32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3q32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 7 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 7 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1] * 5 + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2] * 5 + src_r_1[5] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3] * 5;
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4] * 5;
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 7;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 7;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 7;

            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp   = (*(src_r_1 + 2)) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_n0_x5   = neon::vmul(vq32_n0, static_cast<MovlType>(5));
            auto vq32_result2 = neon::vmla(vq32_n0_x5, vq32_c, static_cast<MovlType>(3));
            auto vq32_result3 = neon::vmla(vq32_c,     vq32_n0, static_cast<MovlType>(7));

            auto vd16_result2 = neon::vrshrn_n<6>(vq32_result2);
            auto vd16_result3 = neon::vrshrn_n<6>(vq32_result3);

            neon::vstore(dst_row0, vd16_result2);
            neon::vstore(dst_row1, vd16_result3);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (MI_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        MI_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_row1);
            auto v3d16_x1 = neon::vload3(src_row1 + 3);

            auto vq32_x0 = neon::vmovl(v3d16_x0.val[0]);
            auto vq32_x1 = neon::vmovl(v3d16_x1.val[0]);
            auto vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            auto vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            auto vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            auto vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            auto vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            auto vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            auto v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            auto v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            auto v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch0 = v2q32_tmp.val[0];
            auto vq32_n1_ch0 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch0 = v2q32_tmp.val[0];
            auto vq32_n3_ch0 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[1]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[1]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch1 = v2q32_tmp.val[0];
            auto vq32_n1_ch1 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch1 = v2q32_tmp.val[0];
            auto vq32_n3_ch1 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[2]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[2]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch2 = v2q32_tmp.val[0];
            auto vq32_n1_ch2 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch2 = v2q32_tmp.val[0];
            auto vq32_n3_ch2 = v2q32_tmp.val[1];

            MVType v3q32_result;
            v3q32_result.val[0] = vq32_n0_ch0;
            v3q32_result.val[1] = vq32_n0_ch1;
            v3q32_result.val[2] = vq32_n0_ch2;
            neon::vstore(rows1_tmp, v3q32_result);

            v3q32_result.val[0] = vq32_n1_ch0;
            v3q32_result.val[1] = vq32_n1_ch1;
            v3q32_result.val[2] = vq32_n1_ch2;
            neon::vstore(rows1_tmp + 12, v3q32_result);

            v3q32_result.val[0] = vq32_n2_ch0;
            v3q32_result.val[1] = vq32_n2_ch1;
            v3q32_result.val[2] = vq32_n2_ch2;
            neon::vstore(rows1_tmp + 24, v3q32_result);

            v3q32_result.val[0] = vq32_n3_ch0;
            v3q32_result.val[1] = vq32_n3_ch1;
            v3q32_result.val[2] = vq32_n3_ch2;
            neon::vstore(rows1_tmp + 36, v3q32_result);

            src_row1 += 12;
            rows1_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 7 + src_row1[3];
            *rows1_tmp++ = src_row1[1] * 7 + src_row1[4];
            *rows1_tmp++ = src_row1[2] * 7 + src_row1[5];
            *rows1_tmp++ = src_row1[0] * 5 + src_row1[3] * 3;
            *rows1_tmp++ = src_row1[1] * 5 + src_row1[4] * 3;
            *rows1_tmp++ = src_row1[2] * 5 + src_row1[5] * 3;
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[3] * 5;
            *rows1_tmp++ = src_row1[1] * 3 + src_row1[4] * 5;
            *rows1_tmp++ = src_row1[2] * 3 + src_row1[5] * 5;
            *rows1_tmp++ = src_row1[0]     + src_row1[3] * 7;
            *rows1_tmp++ = src_row1[1]     + src_row1[4] * 7;
            *rows1_tmp++ = src_row1[2]     + src_row1[5] * 7;

            src_row1 += 3;
        }
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp++ = (*(src_row1 + 2)) << 3;
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*(src_row1 + 1)) << 3;
        *rows1_tmp   = (*(src_row1 + 2)) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_n0_x3 = neon::vmul(vq32_n0, static_cast<MovlType>(3));
            auto vq32_n0_x5 = neon::vmul(vq32_n0, static_cast<MovlType>(5));

            auto vq32_result0 = neon::vmla(vq32_n0,    vq32_c, static_cast<MovlType>(7));
            auto vq32_result1 = neon::vmla(vq32_n0_x3, vq32_c, static_cast<MovlType>(5));
            auto vq32_result2 = neon::vmla(vq32_n0_x5, vq32_c, static_cast<MovlType>(3));
            auto vq32_result3 = neon::vmla(vq32_c,     vq32_n0, static_cast<MovlType>(7));

            auto vd16_result0 = neon::vrshrn_n<6>(vq32_result0);
            auto vd16_result1 = neon::vrshrn_n<6>(vq32_result1);
            auto vd16_result2 = neon::vrshrn_n<6>(vq32_result2);
            auto vd16_result3 = neon::vrshrn_n<6>(vq32_result3);

            neon::vstore(dst_row0, vd16_result0);
            neon::vstore(dst_row1, vd16_result1);
            neon::vstore(dst_row2, vd16_result2);
            neon::vstore(dst_row3, vd16_result3);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
            dst_row2 += 4;
            dst_row3 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);
            *dst_row2++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row3++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row0 = dst.Ptr<Tp>(end_row - 2);
    dst_row1 = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row0++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[0] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[1] + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((rows1_tmp[2] + 4) >> 3);
            rows1_tmp += 3;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            auto v3d16_x0 = neon::vload3(src_r_1);
            auto v3d16_x1 = neon::vload3(src_r_1 + 3);

            auto vq32_x0 = neon::vmovl(v3d16_x0.val[0]);
            auto vq32_x1 = neon::vmovl(v3d16_x1.val[0]);
            auto vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            auto vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            auto vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            auto vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            auto vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            auto vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            auto v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            auto v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            auto v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch0 = v2q32_tmp.val[0];
            auto vq32_n1_ch0 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch0 = v2q32_tmp.val[0];
            auto vq32_n3_ch0 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[1]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[1]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch1 = v2q32_tmp.val[0];
            auto vq32_n1_ch1 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch1 = v2q32_tmp.val[0];
            auto vq32_n3_ch1 = v2q32_tmp.val[1];

            vq32_x0 = neon::vmovl(v3d16_x0.val[2]);
            vq32_x1 = neon::vmovl(v3d16_x1.val[2]);
            vq32_x1_x3 = neon::vmul(vq32_x1, static_cast<MovlType>(3));
            vq32_x1_x5 = neon::vmul(vq32_x1, static_cast<MovlType>(5));
            vq32_n0 = neon::vmla(vq32_x1, vq32_x0, static_cast<MovlType>(7));
            vq32_n1 = neon::vmla(vq32_x1_x3, vq32_x0, static_cast<MovlType>(5));
            vq32_n2 = neon::vmla(vq32_x1_x5, vq32_x0, static_cast<MovlType>(3));
            vq32_n3 = neon::vmla(vq32_x0, vq32_x1, static_cast<MovlType>(7));
            v2q32_n0n2 = neon::vzip(vq32_n0, vq32_n2);
            v2q32_n1n3 = neon::vzip(vq32_n1, vq32_n3);
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[0], v2q32_n1n3.val[0]);
            auto vq32_n0_ch2 = v2q32_tmp.val[0];
            auto vq32_n1_ch2 = v2q32_tmp.val[1];
            v2q32_tmp = neon::vzip(v2q32_n0n2.val[1], v2q32_n1n3.val[1]);
            auto vq32_n2_ch2 = v2q32_tmp.val[0];
            auto vq32_n3_ch2 = v2q32_tmp.val[1];

            MVType v3q32_result;
            v3q32_result.val[0] = vq32_n0_ch0;
            v3q32_result.val[1] = vq32_n0_ch1;
            v3q32_result.val[2] = vq32_n0_ch2;
            neon::vstore(rows0_tmp, v3q32_result);

            v3q32_result.val[0] = vq32_n1_ch0;
            v3q32_result.val[1] = vq32_n1_ch1;
            v3q32_result.val[2] = vq32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3q32_result);

            v3q32_result.val[0] = vq32_n2_ch0;
            v3q32_result.val[1] = vq32_n2_ch1;
            v3q32_result.val[2] = vq32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3q32_result);

            v3q32_result.val[0] = vq32_n3_ch0;
            v3q32_result.val[1] = vq32_n3_ch1;
            v3q32_result.val[2] = vq32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3q32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[3];
            *rows0_tmp++ = src_r_1[1] * 7 + src_r_1[4];
            *rows0_tmp++ = src_r_1[2] * 7 + src_r_1[5];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[3] * 3;
            *rows0_tmp++ = src_r_1[1] * 5 + src_r_1[4] * 3;
            *rows0_tmp++ = src_r_1[2] * 5 + src_r_1[5] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[3] * 5;
            *rows0_tmp++ = src_r_1[1] * 3 + src_r_1[4] * 5;
            *rows0_tmp++ = src_r_1[2] * 3 + src_r_1[5] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[3] * 7;
            *rows0_tmp++ = src_r_1[1]     + src_r_1[4] * 7;
            *rows0_tmp++ = src_r_1[2]     + src_r_1[5] * 7;

            src_r_1 += 3;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp++ = (*(src_r_1 + 2)) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*(src_r_1 + 1)) << 3;
        *rows0_tmp   = (*(src_r_1 + 2)) << 3;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_n0_x3   = neon::vmul(vq32_n0, static_cast<MovlType>(3));
            auto vq32_result0 = neon::vmla(vq32_n0,    vq32_c, static_cast<MovlType>(7));
            auto vq32_result1 = neon::vmla(vq32_n0_x3, vq32_c, static_cast<MovlType>(5));

            auto vd16_result0 = neon::vrshrn_n<6>(vq32_result0);
            auto vd16_result1 = neon::vrshrn_n<6>(vq32_result1);

            neon::vstore(dst_row0, vd16_result0);
            neon::vstore(dst_row1, vd16_result1);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeBnC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth = dst.GetSizes().m_width;

    float32x2_t vdf32_const_1_4;
    neon::vdup(vdf32_const_1_4, 0.25f);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 owidth_align2 = owidth & (-2);
        MI_S32 x = 0;
        for (; x < owidth_align2; x += 2)
        {
            float32x4x3_t v3qf32_c  = neon::vload3q(src_row0);
            float32x4x3_t v3qf32_n0 = neon::vload3q(src_row1);

            MVType v3df32_result;
            float32x4x2_t v2qf32_ch = neon::vuzp(v3qf32_c.val[0], v3qf32_n0.val[0]);
            float32x2_t vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[0]), neon::vgethigh(v2qf32_ch.val[0]));
            vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[1]), vdf32_ch);
            vdf32_ch = neon::vadd(neon::vgethigh(v2qf32_ch.val[1]), vdf32_ch);
            v3df32_result.val[0] = neon::vmul(vdf32_ch, vdf32_const_1_4);

            v2qf32_ch = neon::vuzp(v3qf32_c.val[1], v3qf32_n0.val[1]);
            vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[0]), neon::vgethigh(v2qf32_ch.val[0]));
            vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[1]), vdf32_ch);
            vdf32_ch = neon::vadd(neon::vgethigh(v2qf32_ch.val[1]), vdf32_ch);
            v3df32_result.val[1] = neon::vmul(vdf32_ch, vdf32_const_1_4);

            v2qf32_ch = neon::vuzp(v3qf32_c.val[2], v3qf32_n0.val[2]);
            vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[0]), neon::vgethigh(v2qf32_ch.val[0]));
            vdf32_ch = neon::vadd(neon::vgetlow(v2qf32_ch.val[1]), vdf32_ch);
            vdf32_ch = neon::vadd(neon::vgethigh(v2qf32_ch.val[1]), vdf32_ch);
            v3df32_result.val[2] = neon::vmul(vdf32_ch, vdf32_const_1_4);

            neon::vstore(dst_row, v3df32_result);

            src_row0 += 12;
            src_row1 += 12;
            dst_row  += 6;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[3];
            MovlType r1 = src_row1[0] + src_row1[3];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[1] + src_row0[4];
            r1 = src_row1[1] + src_row1[4];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[2] + src_row0[5];
            r1 = src_row1[2] + src_row1[5];
            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 6;
            src_row1 += 6;
        }
    }

    return Status::OK;
}

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeBnC3DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;

    MI_S32 owidth = dst.GetSizes().m_width;

    float32x2_t vdf32_const_1_4;
    neon::vdup(vdf32_const_1_4, 0.25f);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 width_align2 = owidth & (-2);
        MI_S32 x = 0;
        for (; x < width_align2; x += 2)
        {
            float32x4x3_t v3qf32_cx0  = neon::vload3q(src_row0);
            float32x4x3_t v3qf32_cx1  = neon::vload3q(src_row0 + 12);
            float32x4x3_t v3qf32_n0x0 = neon::vload3q(src_row1);
            float32x4x3_t v3qf32_n0x1 = neon::vload3q(src_row1 + 12);

            MVType v3df32_result;

            float32x4x2_t v2qf32_ch = neon::vuzp(v3qf32_cx0.val[0], v3qf32_n0x0.val[0]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            float32x2_t vdf32_low = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v2qf32_ch = neon::vuzp(v3qf32_cx1.val[0], v3qf32_n0x1.val[0]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            float32x2_t vdf32_high = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v3df32_result.val[0] = neon::vmul(neon::vpadd(vdf32_low, vdf32_high), vdf32_const_1_4);

            v2qf32_ch = neon::vuzp(v3qf32_cx0.val[1], v3qf32_n0x0.val[1]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            vdf32_low = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v2qf32_ch = neon::vuzp(v3qf32_cx1.val[1], v3qf32_n0x1.val[1]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            vdf32_high = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v3df32_result.val[1] = neon::vmul(neon::vpadd(vdf32_low, vdf32_high), vdf32_const_1_4);

            v2qf32_ch = neon::vuzp(v3qf32_cx0.val[2], v3qf32_n0x0.val[2]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            vdf32_low = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v2qf32_ch = neon::vuzp(v3qf32_cx1.val[2], v3qf32_n0x1.val[2]);
            v2qf32_ch = neon::vuzp(v2qf32_ch.val[0], v2qf32_ch.val[1]);
            vdf32_high = neon::vadd(neon::vgethigh(v2qf32_ch.val[0]), neon::vgetlow(v2qf32_ch.val[1]));
            v3df32_result.val[2] = neon::vmul(neon::vpadd(vdf32_low, vdf32_high), vdf32_const_1_4);
            neon::vstore(dst_row, v3df32_result);

            src_row0 += 24;
            src_row1 += 24;
            dst_row  += 6;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[3] + src_row0[6];
            MovlType r1 = src_row1[3] + src_row1[6];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[4] + src_row0[7];
            r1 = src_row1[4] + src_row1[7];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[5] + src_row0[8];
            r1 = src_row1[5] + src_row1[8];
            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 12;
            src_row1 += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeBnC3UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<Tp, 3>::MVType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4x3_t v3qf32_x0 = neon::vload3q(src_row0);
        float32x4x3_t v3qf32_x1 = neon::vload3q(src_row0 + 3);

        float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[0], 0.25f), v3qf32_x0.val[0], 0.75f);
        float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[0], 0.25f), v3qf32_x1.val[0], 0.75f);
        float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[1], 0.25f), v3qf32_x0.val[1], 0.75f);
        vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[1], 0.25f), v3qf32_x1.val[1], 0.75f);
        float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[2], 0.25f), v3qf32_x0.val[2], 0.75f);
        vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[2], 0.25f), v3qf32_x1.val[2], 0.75f);
        float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        MVType v3qf32_result;
        v3qf32_result.val[0] = v2qf32_ch0.val[0];
        v3qf32_result.val[1] = v2qf32_ch1.val[0];
        v3qf32_result.val[2] = v2qf32_ch2.val[0];
        neon::vstore(rows1_tmp, v3qf32_result);

        v3qf32_result.val[0] = v2qf32_ch0.val[1];
        v3qf32_result.val[1] = v2qf32_ch1.val[1];
        v3qf32_result.val[2] = v2qf32_ch2.val[1];
        neon::vstore(rows1_tmp + 12, v3qf32_result);

        src_row0 += 12;
        rows1_tmp += 24;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.75f + src_row0[3] * 0.25f;
        *rows1_tmp++ = src_row0[1] * 0.75f + src_row0[4] * 0.25f;
        *rows1_tmp++ = src_row0[2] * 0.75f + src_row0[5] * 0.25f;
        *rows1_tmp++ = src_row0[0] * 0.25f + src_row0[3] * 0.75f;
        *rows1_tmp++ = src_row0[1] * 0.25f + src_row0[4] * 0.75f;
        *rows1_tmp++ = src_row0[2] * 0.25f + src_row0[5] * 0.75f;
        src_row0 += 3;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp   = *(src_row0 + 2);

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_r_1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_r_1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[0], 0.25f), v3qf32_x0.val[0], 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[0], 0.25f), v3qf32_x1.val[0], 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[1], 0.25f), v3qf32_x0.val[1], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[1], 0.25f), v3qf32_x1.val[1], 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[2], 0.25f), v3qf32_x0.val[2], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[2], 0.25f), v3qf32_x1.val[2], 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            MVType v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75f + src_r_1[3] * 0.25f;
            *rows0_tmp++ = src_r_1[1] * 0.75f + src_r_1[4] * 0.25f;
            *rows0_tmp++ = src_r_1[2] * 0.75f + src_r_1[5] * 0.25f;
            *rows0_tmp++ = src_r_1[0] * 0.25f + src_r_1[3] * 0.75f;
            *rows0_tmp++ = src_r_1[1] * 0.25f + src_r_1[4] * 0.75f;
            *rows0_tmp++ = src_r_1[2] * 0.25f + src_r_1[5] * 0.75f;
            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_n0_result = neon::vmla(neon::vmul(vqf32_c, 0.25f), vqf32_n0, 0.75f);
            neon::vstore(dst_row, vqf32_n0_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (MI_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        MI_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_row1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_row1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[0], 0.25f), v3qf32_x0.val[0], 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[0], 0.25f), v3qf32_x1.val[0], 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[1], 0.25f), v3qf32_x0.val[1], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[1], 0.25f), v3qf32_x1.val[1], 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[2], 0.25f), v3qf32_x0.val[2], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[2], 0.25f), v3qf32_x1.val[2], 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            MVType v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows1_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows1_tmp + 12, v3qf32_result);

            src_row1 += 12;
            rows1_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.75f + src_row1[3] * 0.25f;
            *rows1_tmp++ = src_row1[1] * 0.75f + src_row1[4] * 0.25f;
            *rows1_tmp++ = src_row1[2] * 0.75f + src_row1[5] * 0.25f;
            *rows1_tmp++ = src_row1[0] * 0.25f + src_row1[3] * 0.75f;
            *rows1_tmp++ = src_row1[1] * 0.25f + src_row1[4] * 0.75f;
            *rows1_tmp++ = src_row1[2] * 0.25f + src_row1[5] * 0.75f;
            src_row1 += 3;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp   = *(src_row1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmla(neon::vmul(vqf32_n0, 0.25f), vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmla(neon::vmul(vqf32_c, 0.25f), vqf32_n0, 0.75f);

            neon::vstore(dst_row0, vqf32_c_result);
            neon::vstore(dst_row1, vqf32_n0_result);

            dst_row0 += 4;
            dst_row1 += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.75f + rows1_y[0] * 0.25f;
            *dst_row1++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_r_1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_r_1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[0], 0.25f), v3qf32_x0.val[0], 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[0], 0.25f), v3qf32_x1.val[0], 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[1], 0.25f), v3qf32_x0.val[1], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[1], 0.25f), v3qf32_x1.val[1], 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(v3qf32_x1.val[2], 0.25f), v3qf32_x0.val[2], 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(v3qf32_x0.val[2], 0.25f), v3qf32_x1.val[2], 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            MVType v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75f + src_r_1[3] * 0.25f;
            *rows0_tmp++ = src_r_1[1] * 0.75f + src_r_1[4] * 0.25f;
            *rows0_tmp++ = src_r_1[2] * 0.75f + src_r_1[5] * 0.25f;
            *rows0_tmp++ = src_r_1[0] * 0.25f + src_r_1[3] * 0.75f;
            *rows0_tmp++ = src_r_1[1] * 0.25f + src_r_1[4] * 0.75f;
            *rows0_tmp++ = src_r_1[2] * 0.25f + src_r_1[5] * 0.75f;
            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmla(neon::vmul(vqf32_n0, 0.25f), vqf32_c, 0.75f);
            neon::vstore(dst_row, vqf32_c_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = rows0_y[0] * 0.75f + rows1_y[0] * 0.25f;

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeBnC3UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 3>::MVType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4x3_t v3qf32_x0 = neon::vload3q(src_row0);
        float32x4x3_t v3qf32_x1 = neon::vload3q(src_row0 + 3);

        float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
        float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
        float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
        float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
        float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
        float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
        float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
        float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
        float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

        vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
        vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
        vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
        vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
        vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
        vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
        vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
        vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
        v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

        vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
        vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
        vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
        vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
        vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
        vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
        vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
        vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
        v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

        MVType v3qf32_result;
        v3qf32_result.val[0] = vqf32_n0_ch0;
        v3qf32_result.val[1] = vqf32_n0_ch1;
        v3qf32_result.val[2] = vqf32_n0_ch2;
        neon::vstore(rows1_tmp, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n1_ch0;
        v3qf32_result.val[1] = vqf32_n1_ch1;
        v3qf32_result.val[2] = vqf32_n1_ch2;
        neon::vstore(rows1_tmp + 12, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n2_ch0;
        v3qf32_result.val[1] = vqf32_n2_ch1;
        v3qf32_result.val[2] = vqf32_n2_ch2;
        neon::vstore(rows1_tmp + 24, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n3_ch0;
        v3qf32_result.val[1] = vqf32_n3_ch1;
        v3qf32_result.val[2] = vqf32_n3_ch2;
        neon::vstore(rows1_tmp + 36, v3qf32_result);

        src_row0 += 12;
        rows1_tmp += 48;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.875f + src_row0[3] * 0.125f;
        *rows1_tmp++ = src_row0[1] * 0.875f + src_row0[4] * 0.125f;
        *rows1_tmp++ = src_row0[2] * 0.875f + src_row0[5] * 0.125f;
        *rows1_tmp++ = src_row0[0] * 0.625f + src_row0[3] * 0.375f;
        *rows1_tmp++ = src_row0[1] * 0.625f + src_row0[4] * 0.375f;
        *rows1_tmp++ = src_row0[2] * 0.625f + src_row0[5] * 0.375f;
        *rows1_tmp++ = src_row0[0] * 0.375f + src_row0[3] * 0.625f;
        *rows1_tmp++ = src_row0[1] * 0.375f + src_row0[4] * 0.625f;
        *rows1_tmp++ = src_row0[2] * 0.375f + src_row0[5] * 0.625f;
        *rows1_tmp++ = src_row0[0] * 0.125f + src_row0[3] * 0.875f;
        *rows1_tmp++ = src_row0[1] * 0.125f + src_row0[4] * 0.875f;
        *rows1_tmp++ = src_row0[2] * 0.125f + src_row0[5] * 0.875f;

        src_row0 += 3;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp   = *(src_row0 + 2);

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = rows1_tmp[0];
            *dst_row0++ = rows1_tmp[1];
            *dst_row0++ = rows1_tmp[2];
            *dst_row1++ = rows1_tmp[0];
            *dst_row1++ = rows1_tmp[1];
            *dst_row1++ = rows1_tmp[2];

            rows1_tmp += 3;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_r_1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_r_1 + 3);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            MVType v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[3] * 0.125f;
            *rows0_tmp++ = src_r_1[1] * 0.875f + src_r_1[4] * 0.125f;
            *rows0_tmp++ = src_r_1[2] * 0.875f + src_r_1[5] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[3] * 0.375f;
            *rows0_tmp++ = src_r_1[1] * 0.625f + src_r_1[4] * 0.375f;
            *rows0_tmp++ = src_r_1[2] * 0.625f + src_r_1[5] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[3] * 0.625f;
            *rows0_tmp++ = src_r_1[1] * 0.375f + src_r_1[4] * 0.625f;
            *rows0_tmp++ = src_r_1[2] * 0.375f + src_r_1[5] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[3] * 0.875f;
            *rows0_tmp++ = src_r_1[1] * 0.125f + src_r_1[4] * 0.875f;
            *rows0_tmp++ = src_r_1[2] * 0.125f + src_r_1[5] * 0.875f;

            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_n2_result = neon::vmla(vqf32_c_x3, vqf32_n0, 0.625f);
            float32x4_t vqf32_n3_result = neon::vmla(vqf32_c_x1, vqf32_n0, 0.875f);

            neon::vstore(dst_row0, vqf32_n2_result);
            neon::vstore(dst_row1, vqf32_n3_result);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row1++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (MI_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        MI_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_row1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_row1 + 3);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            MVType v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows1_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows1_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows1_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows1_tmp + 36, v3qf32_result);

            src_row1 += 12;
            rows1_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.875f + src_row1[3] * 0.125f;
            *rows1_tmp++ = src_row1[1] * 0.875f + src_row1[4] * 0.125f;
            *rows1_tmp++ = src_row1[2] * 0.875f + src_row1[5] * 0.125f;
            *rows1_tmp++ = src_row1[0] * 0.625f + src_row1[3] * 0.375f;
            *rows1_tmp++ = src_row1[1] * 0.625f + src_row1[4] * 0.375f;
            *rows1_tmp++ = src_row1[2] * 0.625f + src_row1[5] * 0.375f;
            *rows1_tmp++ = src_row1[0] * 0.375f + src_row1[3] * 0.625f;
            *rows1_tmp++ = src_row1[1] * 0.375f + src_row1[4] * 0.625f;
            *rows1_tmp++ = src_row1[2] * 0.375f + src_row1[5] * 0.625f;
            *rows1_tmp++ = src_row1[0] * 0.125f + src_row1[3] * 0.875f;
            *rows1_tmp++ = src_row1[1] * 0.125f + src_row1[4] * 0.875f;
            *rows1_tmp++ = src_row1[2] * 0.125f + src_row1[5] * 0.875f;

            src_row1 += 3;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp   = *(src_row1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_result = neon::vmla(vqf32_c_x7, vqf32_n0, 0.125f);
            float32x4_t vqf32_n1_result = neon::vmla(vqf32_c_x5, vqf32_n0, 0.375f);
            float32x4_t vqf32_n2_result = neon::vmla(vqf32_c_x3, vqf32_n0, 0.625f);
            float32x4_t vqf32_n3_result = neon::vmla(vqf32_c_x1, vqf32_n0, 0.875f);

            neon::vstore(dst_row0, vqf32_n0_result);
            neon::vstore(dst_row1, vqf32_n1_result);
            neon::vstore(dst_row2, vqf32_n2_result);
            neon::vstore(dst_row3, vqf32_n3_result);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
            dst_row2 += 4;
            dst_row3 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.875f + rows1_y[0] * 0.125f;
            *dst_row1++ = rows0_y[0] * 0.625f + rows1_y[0] * 0.375f;
            *dst_row2++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row3++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row0 = dst.Ptr<Tp>(end_row - 2);
    dst_row1 = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = rows1_tmp[0];
            *dst_row0++ = rows1_tmp[1];
            *dst_row0++ = rows1_tmp[2];
            *dst_row1++ = rows1_tmp[0];
            *dst_row1++ = rows1_tmp[1];
            *dst_row1++ = rows1_tmp[2];
            rows1_tmp += 3;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4x3_t v3qf32_x0 = neon::vload3q(src_r_1);
            float32x4x3_t v3qf32_x1 = neon::vload3q(src_r_1 + 3);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            MVType v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[3] * 0.125f;
            *rows0_tmp++ = src_r_1[1] * 0.875f + src_r_1[4] * 0.125f;
            *rows0_tmp++ = src_r_1[2] * 0.875f + src_r_1[5] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[3] * 0.375f;
            *rows0_tmp++ = src_r_1[1] * 0.625f + src_r_1[4] * 0.375f;
            *rows0_tmp++ = src_r_1[2] * 0.625f + src_r_1[5] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[3] * 0.625f;
            *rows0_tmp++ = src_r_1[1] * 0.375f + src_r_1[4] * 0.625f;
            *rows0_tmp++ = src_r_1[2] * 0.375f + src_r_1[5] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[3] * 0.875f;
            *rows0_tmp++ = src_r_1[1] * 0.125f + src_r_1[4] * 0.875f;
            *rows0_tmp++ = src_r_1[2] * 0.125f + src_r_1[5] * 0.875f;

            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_result = neon::vmla(vqf32_c_x7, vqf32_n0, 0.125f);
            float32x4_t vqf32_n1_result = neon::vmla(vqf32_c_x5, vqf32_n0, 0.375f);

            neon::vstore(dst_row0, vqf32_n0_result);
            neon::vstore(dst_row1, vqf32_n1_result);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.875f + rows1_y[0] * 0.125f;
            *dst_row1++ = rows0_y[0] * 0.625f + rows1_y[0] * 0.375f;

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

#if defined(AURA_ENABLE_NEON_FP16)
// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    MI_S32 owidth = dst.GetSizes().m_width;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 owidth_align4 = owidth & (-4);
        MI_S32 x = 0;
        for (; x < owidth_align4; x += 4)
        {
            float16x8x3_t v3qf16_c  = neon::vload3q(src_row0);
            float16x8x3_t v3qf16_n0 = neon::vload3q(src_row1);

            float16x4x3_t v3df16_result;
            float16x8x2_t v2qf16_ch = neon::vuzp(v3qf16_c.val[0], v3qf16_n0.val[0]);
            float32x4_t vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])));
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])), vqf32_ch);
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[1])), vqf32_ch);
            v3df16_result.val[0] = neon::vcvt<Tp>(neon::vmul(vqf32_ch, vqf32_const_1_4));

            v2qf16_ch = neon::vuzp(v3qf16_c.val[1], v3qf16_n0.val[1]);
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])));
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])), vqf32_ch);
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[1])), vqf32_ch);
            v3df16_result.val[1] = neon::vcvt<Tp>(neon::vmul(vqf32_ch, vqf32_const_1_4));

            v2qf16_ch = neon::vuzp(v3qf16_c.val[2], v3qf16_n0.val[2]);
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])));
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])), vqf32_ch);
            vqf32_ch = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[1])), vqf32_ch);
            v3df16_result.val[2] = neon::vcvt<Tp>(neon::vmul(vqf32_ch, vqf32_const_1_4));

            neon::vstore(dst_row, v3df16_result);

            src_row0 += 24;
            src_row1 += 24;
            dst_row  += 12;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[3];
            MovlType r1 = src_row1[0] + src_row1[3];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[1] + src_row0[4];
            r1 = src_row1[1] + src_row1[4];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[2] + src_row0[5];
            r1 = src_row1[2] + src_row1[5];
            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 6;
            src_row1 += 6;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC3DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    MI_S32 owidth = dst.GetSizes().m_width;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        MI_S32 width_align4 = owidth & (-4);
        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float16x8x3_t v3qf16_cx0  = neon::vload3q(src_row0);
            float16x8x3_t v3qf16_cx1  = neon::vload3q(src_row0 + 24);
            float16x8x3_t v3qf16_n0x0 = neon::vload3q(src_row1);
            float16x8x3_t v3qf16_n0x1 = neon::vload3q(src_row1 + 24);

            float16x4x3_t v3df16_result;

            float16x8x2_t v2qf16_ch = neon::vuzp(v3qf16_cx0.val[0], v3qf16_n0x0.val[0]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            float32x4_t vqf32_low = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            v2qf16_ch = neon::vuzp(v3qf16_cx1.val[0], v3qf16_n0x1.val[0]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            float32x4_t vqf32_high = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            float32x4_t vqf32_result = neon::vcombine(neon::vadd(neon::vgetlow(vqf32_low), neon::vgethigh(vqf32_low)), neon::vadd(neon::vgetlow(vqf32_high), neon::vgethigh(vqf32_high)));
            v3df16_result.val[0] = neon::vcvt<Tp>(neon::vmul(vqf32_result, vqf32_const_1_4));

            v2qf16_ch = neon::vuzp(v3qf16_cx0.val[1], v3qf16_n0x0.val[1]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            vqf32_low = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            v2qf16_ch = neon::vuzp(v3qf16_cx1.val[1], v3qf16_n0x1.val[1]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            vqf32_high = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            vqf32_result = neon::vcombine(neon::vadd(neon::vgetlow(vqf32_low), neon::vgethigh(vqf32_low)), neon::vadd(neon::vgetlow(vqf32_high), neon::vgethigh(vqf32_high)));
            v3df16_result.val[1] = neon::vcvt<Tp>(neon::vmul(vqf32_result, vqf32_const_1_4));

            v2qf16_ch = neon::vuzp(v3qf16_cx0.val[2], v3qf16_n0x0.val[2]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            vqf32_low = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            v2qf16_ch = neon::vuzp(v3qf16_cx1.val[2], v3qf16_n0x1.val[2]);
            v2qf16_ch = neon::vuzp(v2qf16_ch.val[0], v2qf16_ch.val[1]);
            vqf32_high = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_ch.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_ch.val[1])));
            vqf32_result = neon::vcombine(neon::vadd(neon::vgetlow(vqf32_low), neon::vgethigh(vqf32_low)), neon::vadd(neon::vgetlow(vqf32_high), neon::vgethigh(vqf32_high)));
            v3df16_result.val[2] = neon::vcvt<Tp>(neon::vmul(vqf32_result, vqf32_const_1_4));

            neon::vstore(dst_row, v3df16_result);

            src_row0 += 48;
            src_row1 += 48;
            dst_row  += 12;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[3] + src_row0[6];
            MovlType r1 = src_row1[3] + src_row1[6];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[4] + src_row0[7];
            r1 = src_row1[4] + src_row1[7];
            *dst_row++ = (r0 + r1) * 0.25f;
            r0 = src_row0[5] + src_row0[8];
            r1 = src_row1[5] + src_row1[8];
            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 12;
            src_row1 += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC3UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float16x4x3_t v3df16_x0 = neon::vload3(src_row0);
        float16x4x3_t v3df16_x1 = neon::vload3(src_row0 + 3);

        float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.75f);
        float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.75f);
        float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.75f);
        vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.75f);
        float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.75f);
        vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.75f);
        float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

        float32x4x3_t v3qf32_result;
        v3qf32_result.val[0] = v2qf32_ch0.val[0];
        v3qf32_result.val[1] = v2qf32_ch1.val[0];
        v3qf32_result.val[2] = v2qf32_ch2.val[0];
        neon::vstore(rows1_tmp, v3qf32_result);

        v3qf32_result.val[0] = v2qf32_ch0.val[1];
        v3qf32_result.val[1] = v2qf32_ch1.val[1];
        v3qf32_result.val[2] = v2qf32_ch2.val[1];
        neon::vstore(rows1_tmp + 12, v3qf32_result);

        src_row0 += 12;
        rows1_tmp += 24;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.75f + src_row0[3] * 0.25f;
        *rows1_tmp++ = src_row0[1] * 0.75f + src_row0[4] * 0.25f;
        *rows1_tmp++ = src_row0[2] * 0.75f + src_row0[5] * 0.25f;
        *rows1_tmp++ = src_row0[0] * 0.25f + src_row0[3] * 0.75f;
        *rows1_tmp++ = src_row0[1] * 0.25f + src_row0[4] * 0.75f;
        *rows1_tmp++ = src_row0[2] * 0.25f + src_row0[5] * 0.75f;
        src_row0 += 3;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp   = *(src_row0 + 2);

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_r_1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_r_1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75f + src_r_1[3] * 0.25f;
            *rows0_tmp++ = src_r_1[1] * 0.75f + src_r_1[4] * 0.25f;
            *rows0_tmp++ = src_r_1[2] * 0.75f + src_r_1[5] * 0.25f;
            *rows0_tmp++ = src_r_1[0] * 0.25f + src_r_1[3] * 0.75f;
            *rows0_tmp++ = src_r_1[1] * 0.25f + src_r_1[4] * 0.75f;
            *rows0_tmp++ = src_r_1[2] * 0.25f + src_r_1[5] * 0.75f;
            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_n0_result = neon::vmla(neon::vmul(vqf32_c, 0.25f), vqf32_n0, 0.75f);
            neon::vstore(dst_row, neon::vcvt<Tp>(vqf32_n0_result));

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (MI_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        MI_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_row1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_row1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows1_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows1_tmp + 12, v3qf32_result);

            src_row1 += 12;
            rows1_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.75f + src_row1[3] * 0.25f;
            *rows1_tmp++ = src_row1[1] * 0.75f + src_row1[4] * 0.25f;
            *rows1_tmp++ = src_row1[2] * 0.75f + src_row1[5] * 0.25f;
            *rows1_tmp++ = src_row1[0] * 0.25f + src_row1[3] * 0.75f;
            *rows1_tmp++ = src_row1[1] * 0.25f + src_row1[4] * 0.75f;
            *rows1_tmp++ = src_row1[2] * 0.25f + src_row1[5] * 0.75f;
            src_row1 += 3;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp   = *(src_row1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmla(neon::vmul(vqf32_n0, 0.25f), vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmla(neon::vmul(vqf32_c, 0.25f), vqf32_n0, 0.75f);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_c_result));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_n0_result));

            dst_row0 += 4;
            dst_row1 += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.75f + rows1_y[0] * 0.25f;
            *dst_row1++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);

        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_r_1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_r_1 + 3);

            float32x4_t vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.75f);
            float32x4_t vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[0]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[0]), 0.75f);
            float32x4x2_t v2qf32_ch0 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[1]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[1]), 0.75f);
            float32x4x2_t v2qf32_ch1 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            vqf32_x0_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.75f);
            vqf32_x1_ch = neon::vmla(neon::vmul(neon::vcvt<MovlType>(v3df16_x0.val[2]), 0.25f), neon::vcvt<MovlType>(v3df16_x1.val[2]), 0.75f);
            float32x4x2_t v2qf32_ch2 = neon::vzip(vqf32_x0_ch, vqf32_x1_ch);

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = v2qf32_ch0.val[0];
            v3qf32_result.val[1] = v2qf32_ch1.val[0];
            v3qf32_result.val[2] = v2qf32_ch2.val[0];
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = v2qf32_ch0.val[1];
            v3qf32_result.val[1] = v2qf32_ch1.val[1];
            v3qf32_result.val[2] = v2qf32_ch2.val[1];
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 24;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75f + src_r_1[3] * 0.25f;
            *rows0_tmp++ = src_r_1[1] * 0.75f + src_r_1[4] * 0.25f;
            *rows0_tmp++ = src_r_1[2] * 0.75f + src_r_1[5] * 0.25f;
            *rows0_tmp++ = src_r_1[0] * 0.25f + src_r_1[3] * 0.75f;
            *rows0_tmp++ = src_r_1[1] * 0.25f + src_r_1[4] * 0.75f;
            *rows0_tmp++ = src_r_1[2] * 0.25f + src_r_1[5] * 0.75f;
            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmla(neon::vmul(vqf32_n0, 0.25f), vqf32_c, 0.75f);
            neon::vstore(dst_row, neon::vcvt<Tp>(vqf32_c_result));

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row++ = rows0_y[0] * 0.75f + rows1_y[0] * 0.25f;

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC3UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + 3 * owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    MI_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    MI_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float16x4x3_t v3df16_x0 = neon::vload3(src_row0);
        float16x4x3_t v3df16_x1 = neon::vload3(src_row0 + 3);
        float32x4x3_t v3qf32_x0, v3qf32_x1;
        v3qf32_x0.val[0] = neon::vcvt<MovlType>(v3df16_x0.val[0]);
        v3qf32_x0.val[1] = neon::vcvt<MovlType>(v3df16_x0.val[1]);
        v3qf32_x0.val[2] = neon::vcvt<MovlType>(v3df16_x0.val[2]);
        v3qf32_x1.val[0] = neon::vcvt<MovlType>(v3df16_x1.val[0]);
        v3qf32_x1.val[1] = neon::vcvt<MovlType>(v3df16_x1.val[1]);
        v3qf32_x1.val[2] = neon::vcvt<MovlType>(v3df16_x1.val[2]);

        float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
        float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
        float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
        float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
        float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
        float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
        float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
        float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
        float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

        vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
        vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
        vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
        vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
        vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
        vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
        vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
        vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
        v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

        vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
        vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
        vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
        vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
        vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
        vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
        vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
        vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
        v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
        v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
        float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
        v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
        float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
        float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

        float32x4x3_t v3qf32_result;
        v3qf32_result.val[0] = vqf32_n0_ch0;
        v3qf32_result.val[1] = vqf32_n0_ch1;
        v3qf32_result.val[2] = vqf32_n0_ch2;
        neon::vstore(rows1_tmp, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n1_ch0;
        v3qf32_result.val[1] = vqf32_n1_ch1;
        v3qf32_result.val[2] = vqf32_n1_ch2;
        neon::vstore(rows1_tmp + 12, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n2_ch0;
        v3qf32_result.val[1] = vqf32_n2_ch1;
        v3qf32_result.val[2] = vqf32_n2_ch2;
        neon::vstore(rows1_tmp + 24, v3qf32_result);

        v3qf32_result.val[0] = vqf32_n3_ch0;
        v3qf32_result.val[1] = vqf32_n3_ch1;
        v3qf32_result.val[2] = vqf32_n3_ch2;
        neon::vstore(rows1_tmp + 36, v3qf32_result);

        src_row0 += 12;
        rows1_tmp += 48;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.875f + src_row0[3] * 0.125f;
        *rows1_tmp++ = src_row0[1] * 0.875f + src_row0[4] * 0.125f;
        *rows1_tmp++ = src_row0[2] * 0.875f + src_row0[5] * 0.125f;
        *rows1_tmp++ = src_row0[0] * 0.625f + src_row0[3] * 0.375f;
        *rows1_tmp++ = src_row0[1] * 0.625f + src_row0[4] * 0.375f;
        *rows1_tmp++ = src_row0[2] * 0.625f + src_row0[5] * 0.375f;
        *rows1_tmp++ = src_row0[0] * 0.375f + src_row0[3] * 0.625f;
        *rows1_tmp++ = src_row0[1] * 0.375f + src_row0[4] * 0.625f;
        *rows1_tmp++ = src_row0[2] * 0.375f + src_row0[5] * 0.625f;
        *rows1_tmp++ = src_row0[0] * 0.125f + src_row0[3] * 0.875f;
        *rows1_tmp++ = src_row0[1] * 0.125f + src_row0[4] * 0.875f;
        *rows1_tmp++ = src_row0[2] * 0.125f + src_row0[5] * 0.875f;

        src_row0 += 3;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp++ = *(src_row0 + 2);
    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *(src_row0 + 1);
    *rows1_tmp   = *(src_row0 + 2);

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = rows1_tmp[0];
            *dst_row0++ = rows1_tmp[1];
            *dst_row0++ = rows1_tmp[2];
            *dst_row1++ = rows1_tmp[0];
            *dst_row1++ = rows1_tmp[1];
            *dst_row1++ = rows1_tmp[2];

            rows1_tmp += 3;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_r_1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_r_1 + 3);
            float32x4x3_t v3qf32_x0, v3qf32_x1;
            v3qf32_x0.val[0] = neon::vcvt<MovlType>(v3df16_x0.val[0]);
            v3qf32_x0.val[1] = neon::vcvt<MovlType>(v3df16_x0.val[1]);
            v3qf32_x0.val[2] = neon::vcvt<MovlType>(v3df16_x0.val[2]);
            v3qf32_x1.val[0] = neon::vcvt<MovlType>(v3df16_x1.val[0]);
            v3qf32_x1.val[1] = neon::vcvt<MovlType>(v3df16_x1.val[1]);
            v3qf32_x1.val[2] = neon::vcvt<MovlType>(v3df16_x1.val[2]);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[3] * 0.125f;
            *rows0_tmp++ = src_r_1[1] * 0.875f + src_r_1[4] * 0.125f;
            *rows0_tmp++ = src_r_1[2] * 0.875f + src_r_1[5] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[3] * 0.375f;
            *rows0_tmp++ = src_r_1[1] * 0.625f + src_r_1[4] * 0.375f;
            *rows0_tmp++ = src_r_1[2] * 0.625f + src_r_1[5] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[3] * 0.625f;
            *rows0_tmp++ = src_r_1[1] * 0.375f + src_r_1[4] * 0.625f;
            *rows0_tmp++ = src_r_1[2] * 0.375f + src_r_1[5] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[3] * 0.875f;
            *rows0_tmp++ = src_r_1[1] * 0.125f + src_r_1[4] * 0.875f;
            *rows0_tmp++ = src_r_1[2] * 0.125f + src_r_1[5] * 0.875f;

            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_n2_result = neon::vmla(vqf32_c_x3, vqf32_n0, 0.625f);
            float32x4_t vqf32_n3_result = neon::vmla(vqf32_c_x1, vqf32_n0, 0.875f);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_n2_result));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_n3_result));

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row1++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (MI_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        MI_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_row1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_row1 + 3);
            float32x4x3_t v3qf32_x0, v3qf32_x1;
            v3qf32_x0.val[0] = neon::vcvt<MovlType>(v3df16_x0.val[0]);
            v3qf32_x0.val[1] = neon::vcvt<MovlType>(v3df16_x0.val[1]);
            v3qf32_x0.val[2] = neon::vcvt<MovlType>(v3df16_x0.val[2]);
            v3qf32_x1.val[0] = neon::vcvt<MovlType>(v3df16_x1.val[0]);
            v3qf32_x1.val[1] = neon::vcvt<MovlType>(v3df16_x1.val[1]);
            v3qf32_x1.val[2] = neon::vcvt<MovlType>(v3df16_x1.val[2]);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows1_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows1_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows1_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows1_tmp + 36, v3qf32_result);

            src_row1 += 12;
            rows1_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.875f + src_row1[3] * 0.125f;
            *rows1_tmp++ = src_row1[1] * 0.875f + src_row1[4] * 0.125f;
            *rows1_tmp++ = src_row1[2] * 0.875f + src_row1[5] * 0.125f;
            *rows1_tmp++ = src_row1[0] * 0.625f + src_row1[3] * 0.375f;
            *rows1_tmp++ = src_row1[1] * 0.625f + src_row1[4] * 0.375f;
            *rows1_tmp++ = src_row1[2] * 0.625f + src_row1[5] * 0.375f;
            *rows1_tmp++ = src_row1[0] * 0.375f + src_row1[3] * 0.625f;
            *rows1_tmp++ = src_row1[1] * 0.375f + src_row1[4] * 0.625f;
            *rows1_tmp++ = src_row1[2] * 0.375f + src_row1[5] * 0.625f;
            *rows1_tmp++ = src_row1[0] * 0.125f + src_row1[3] * 0.875f;
            *rows1_tmp++ = src_row1[1] * 0.125f + src_row1[4] * 0.875f;
            *rows1_tmp++ = src_row1[2] * 0.125f + src_row1[5] * 0.875f;

            src_row1 += 3;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp++ = *(src_row1 + 2);
        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *(src_row1 + 1);
        *rows1_tmp   = *(src_row1 + 2);

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_result = neon::vmla(vqf32_c_x7, vqf32_n0, 0.125f);
            float32x4_t vqf32_n1_result = neon::vmla(vqf32_c_x5, vqf32_n0, 0.375f);
            float32x4_t vqf32_n2_result = neon::vmla(vqf32_c_x3, vqf32_n0, 0.625f);
            float32x4_t vqf32_n3_result = neon::vmla(vqf32_c_x1, vqf32_n0, 0.875f);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_n0_result));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_n1_result));
            neon::vstore(dst_row2, neon::vcvt<Tp>(vqf32_n2_result));
            neon::vstore(dst_row3, neon::vcvt<Tp>(vqf32_n3_result));

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
            dst_row2 += 4;
            dst_row3 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.875f + rows1_y[0] * 0.125f;
            *dst_row1++ = rows0_y[0] * 0.625f + rows1_y[0] * 0.375f;
            *dst_row2++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row3++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    dst_row0 = dst.Ptr<Tp>(end_row - 2);
    dst_row1 = dst.Ptr<Tp>(end_row - 1);

    if (oheight == end_row)
    {
        rows1_tmp = rows1;
        for (MI_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = rows1_tmp[0];
            *dst_row0++ = rows1_tmp[1];
            *dst_row0++ = rows1_tmp[2];
            *dst_row1++ = rows1_tmp[0];
            *dst_row1++ = rows1_tmp[1];
            *dst_row1++ = rows1_tmp[2];
            rows1_tmp += 3;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        MI_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float16x4x3_t v3df16_x0 = neon::vload3(src_r_1);
            float16x4x3_t v3df16_x1 = neon::vload3(src_r_1 + 3);
            float32x4x3_t v3qf32_x0, v3qf32_x1;
            v3qf32_x0.val[0] = neon::vcvt<MovlType>(v3df16_x0.val[0]);
            v3qf32_x0.val[1] = neon::vcvt<MovlType>(v3df16_x0.val[1]);
            v3qf32_x0.val[2] = neon::vcvt<MovlType>(v3df16_x0.val[2]);
            v3qf32_x1.val[0] = neon::vcvt<MovlType>(v3df16_x1.val[0]);
            v3qf32_x1.val[1] = neon::vcvt<MovlType>(v3df16_x1.val[1]);
            v3qf32_x1.val[2] = neon::vcvt<MovlType>(v3df16_x1.val[2]);

            float32x4_t vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[0], 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[0], 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[0], 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[0], 0.875f);
            float32x4_t vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[0], 0.125f);
            float32x4_t vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[0], 0.375f);
            float32x4_t vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[0], 0.625f);
            float32x4_t vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[0], 0.875f);
            float32x4x2_t v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            float32x4x2_t v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            float32x4x2_t v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch0 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch0 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch0 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[1], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[1], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[1], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[1], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[1], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[1], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[1], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[1], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch1 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch1 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch1 = v2qf32_tmp.val[1];

            vqf32_x0_x1 = neon::vmul(v3qf32_x0.val[2], 0.125f);
            vqf32_x0_x3 = neon::vmul(v3qf32_x0.val[2], 0.375f);
            vqf32_x0_x5 = neon::vmul(v3qf32_x0.val[2], 0.625f);
            vqf32_x0_x7 = neon::vmul(v3qf32_x0.val[2], 0.875f);
            vqf32_n0 = neon::vmla(vqf32_x0_x7, v3qf32_x1.val[2], 0.125f);
            vqf32_n1 = neon::vmla(vqf32_x0_x5, v3qf32_x1.val[2], 0.375f);
            vqf32_n2 = neon::vmla(vqf32_x0_x3, v3qf32_x1.val[2], 0.625f);
            vqf32_n3 = neon::vmla(vqf32_x0_x1, v3qf32_x1.val[2], 0.875f);
            v2qf32_n0n2 = neon::vzip(vqf32_n0, vqf32_n2);
            v2qf32_n1n3 = neon::vzip(vqf32_n1, vqf32_n3);
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[0], v2qf32_n1n3.val[0]);
            float32x4_t vqf32_n0_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n1_ch2 = v2qf32_tmp.val[1];
            v2qf32_tmp = neon::vzip(v2qf32_n0n2.val[1], v2qf32_n1n3.val[1]);
            float32x4_t vqf32_n2_ch2 = v2qf32_tmp.val[0];
            float32x4_t vqf32_n3_ch2 = v2qf32_tmp.val[1];

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = vqf32_n0_ch0;
            v3qf32_result.val[1] = vqf32_n0_ch1;
            v3qf32_result.val[2] = vqf32_n0_ch2;
            neon::vstore(rows0_tmp, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n1_ch0;
            v3qf32_result.val[1] = vqf32_n1_ch1;
            v3qf32_result.val[2] = vqf32_n1_ch2;
            neon::vstore(rows0_tmp + 12, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n2_ch0;
            v3qf32_result.val[1] = vqf32_n2_ch1;
            v3qf32_result.val[2] = vqf32_n2_ch2;
            neon::vstore(rows0_tmp + 24, v3qf32_result);

            v3qf32_result.val[0] = vqf32_n3_ch0;
            v3qf32_result.val[1] = vqf32_n3_ch1;
            v3qf32_result.val[2] = vqf32_n3_ch2;
            neon::vstore(rows0_tmp + 36, v3qf32_result);

            src_r_1 += 12;
            rows0_tmp += 48;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[3] * 0.125f;
            *rows0_tmp++ = src_r_1[1] * 0.875f + src_r_1[4] * 0.125f;
            *rows0_tmp++ = src_r_1[2] * 0.875f + src_r_1[5] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[3] * 0.375f;
            *rows0_tmp++ = src_r_1[1] * 0.625f + src_r_1[4] * 0.375f;
            *rows0_tmp++ = src_r_1[2] * 0.625f + src_r_1[5] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[3] * 0.625f;
            *rows0_tmp++ = src_r_1[1] * 0.375f + src_r_1[4] * 0.625f;
            *rows0_tmp++ = src_r_1[2] * 0.375f + src_r_1[5] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[3] * 0.875f;
            *rows0_tmp++ = src_r_1[1] * 0.125f + src_r_1[4] * 0.875f;
            *rows0_tmp++ = src_r_1[2] * 0.125f + src_r_1[5] * 0.875f;

            src_r_1 += 3;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp++ = *(src_r_1 + 2);
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *(src_r_1 + 1);
        *rows0_tmp   = *(src_r_1 + 2);

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        MI_S32 owidth_x3 = owidth * 3;
        MI_S32 owidth_x3_align4 = owidth_x3 & (-4);
        x = 0;
        for(; x < owidth_x3_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_result = neon::vmla(vqf32_c_x7, vqf32_n0, 0.125f);
            float32x4_t vqf32_n1_result = neon::vmla(vqf32_c_x5, vqf32_n0, 0.375f);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_n0_result));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_n1_result));

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth_x3; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.875f + rows1_y[0] * 0.125f;
            *dst_row1++ = rows0_y[0] * 0.625f + rows1_y[0] * 0.375f;

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}
#endif

template <typename Tp>
static Status ResizeBnFastC3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    MI_S32 owidth  = dst.GetSizes().m_width;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        return Status::ERROR;
    }

    MI_F32 scale_x = static_cast<MI_F64>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    MI_F32 scale_y = static_cast<MI_F64>(src.GetSizes().m_height) / dst.GetSizes().m_height;

    if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 2.f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnC3DownX2NeonImpl<Tp>, ctx, src, dst);
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnC3DownX4NeonImpl<Tp>, ctx, src, dst);
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.5f))
    {
        ThreadBuffer thread_buffer(ctx, owidth * 6 * sizeof(MovlType));
        ret = wp->ParallelFor(0, AURA_ALIGN(dst.GetSizes().m_height, 2) / 2, ResizeBnC3UpX2NeonImpl<Tp>, ctx, src, dst, std::ref(thread_buffer));
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.25f))
    {
        ThreadBuffer thread_buffer(ctx, owidth * 6 * sizeof(MovlType));
        ret = wp->ParallelFor(0, AURA_ALIGN(dst.GetSizes().m_height, 4) / 4, ResizeBnC3UpX4NeonImpl<Tp>, ctx, src, dst, std::ref(thread_buffer));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "spacial scale param error");
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeBnFastC3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnFastC3NeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnFastC3NeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnFastC3NeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnFastC3NeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeBnFastC3NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeBnFastC3NeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC3NeonHelper run failed, type: MI_F32");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport elem type F64 or F16");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura