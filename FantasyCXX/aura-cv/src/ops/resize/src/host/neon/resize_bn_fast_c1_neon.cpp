#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = DT_U8, DT_S8
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            auto v2d8_c  = neon::vload2(src_row0);
            auto v2d8_n0 = neon::vload2(src_row1);

            auto vq16_c  = neon::vaddl(v2d8_c.val[0], v2d8_c.val[1]);
            auto vq16_n0 = neon::vaddl(v2d8_n0.val[0], v2d8_n0.val[1]);

            auto vd8_result = neon::vrshrn_n<2>(neon::vadd(vq16_c, vq16_n0));

            neon::vstore(dst_row, vd8_result);

            src_row0 += 16;
            src_row1 += 16;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[1];
            MovlType r1 = src_row1[0] + src_row1[1];

            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 2;
            src_row1 += 2;
        }
    }

    return Status::OK;
}

// Tp = DT_U8, DT_S8
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnC1DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            auto v4d8_c  = neon::vload4(src_row0);
            auto v4d8_n0 = neon::vload4(src_row1);

            auto vq16_x0 = neon::vaddl(v4d8_c.val[1], v4d8_n0.val[1]);
            auto vq16_x1 = neon::vaddl(v4d8_c.val[2], v4d8_n0.val[2]);

            auto vd8_result = neon::vrshrn_n<2>(neon::vadd(vq16_x0, vq16_x1));

            neon::vstore(dst_row, vd8_result);

            src_row0 += 32;
            src_row1 += 32;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType res0 = src_row0[1] + src_row0[2];
            MovlType res1 = src_row1[1] + src_row1[2];

            *dst_row++ = SaturateCast<Tp>((res0 + res1 + 2) >> 2);

            src_row0 += 4;
            src_row1 += 4;
        }
    }

    return Status::OK;
}

// Tp = DT_U8, DT_S8
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnC1UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 2>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1   = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) * 4;
    DT_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto vq16_x0 = neon::vmovl(neon::vload1(src_row0));
        auto vq16_x1 = neon::vmovl(neon::vload1(src_row0 + 1));

        MVType v2q16_result;
        v2q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(3));
        v2q16_result.val[1] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(3));

        neon::vstore(rows1_tmp, v2q16_result);

        src_row0 += 8;
        rows1_tmp += 16;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[1];
        *rows1_tmp++ = src_row0[0]     + src_row0[1] * 3;
        src_row0++;
    }
    *rows1_tmp = (*src_row0) * 4;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) * 4;
        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vmovl(neon::vload1(src_r_1));
            auto vq16_x1 = neon::vmovl(neon::vload1(src_r_1 + 1));

            MVType v2q16_result;
            v2q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(3));
            v2q16_result.val[1] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(3));

            neon::vstore(rows0_tmp, v2q16_result);

            src_r_1 += 8;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 3;
            src_r_1++;
        }
        *rows0_tmp = (*src_r_1) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
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

        for (; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (DT_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        DT_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = (*src_row1) * 4;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_c  = neon::vmovl(neon::vload1(src_row1));
            auto vq16_n0 = neon::vmovl(neon::vload1(src_row1 + 1));

            MVType v2q16_result;
            v2q16_result.val[0] = neon::vmla(vq16_n0, vq16_c, static_cast<MovlType>(3));
            v2q16_result.val[1] = neon::vmla(vq16_c, vq16_n0, static_cast<MovlType>(3));

            neon::vstore(rows1_tmp, v2q16_result);

            src_row1 += 8;
            rows1_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[1];
            *rows1_tmp++ = src_row1[0]     + src_row1[1] * 3;
            src_row1++;
        }
        *rows1_tmp = (*src_row1) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
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

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) * 4;
        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vmovl(neon::vload1(src_r_1));
            auto vq16_x1 = neon::vmovl(neon::vload1(src_r_1 + 1));

            MVType v2q16_result;
            v2q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(3));
            v2q16_result.val[1] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(3));

            neon::vstore(rows0_tmp, v2q16_result);

            src_r_1 += 8;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 3;
            src_r_1++;
        }
        *rows0_tmp = (*src_r_1) * 4;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_result = neon::vmla(vq16_n0, vq16_c, static_cast<MovlType>(3));
            auto vd8_result = neon::vrshrn_n<4>(vq16_n0_result);

            neon::vstore(dst_row, vd8_result);

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_U8, DT_S8
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnC1UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 4>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    DT_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto vq16_x0 = neon::vmovl(neon::vload1(src_row0));
        auto vq16_x1 = neon::vmovl(neon::vload1(src_row0 + 1));

        auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
        auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));

        MVType v4q16_result;
        v4q16_result.val[3] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
        v4q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
        v4q16_result.val[1] = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
        v4q16_result.val[2] = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));

        neon::vstore(rows1_tmp, v4q16_result);

        src_row0 += 8;
        rows1_tmp += 32;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 7 + src_row0[1];
        *rows1_tmp++ = src_row0[0] * 5 + src_row0[1] * 3;
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[1] * 5;
        *rows1_tmp++ = src_row0[0]     + src_row0[1] * 7;

        src_row0++;
    }
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp   = (*src_row0) << 3;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);

            rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vmovl(neon::vload1(src_r_1));
            auto vq16_x1 = neon::vmovl(neon::vload1(src_r_1 + 1));

            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));

            MVType v4q16_result;
            v4q16_result.val[3] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v4q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            v4q16_result.val[1] = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            v4q16_result.val[2] = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));

            neon::vstore(rows0_tmp, v4q16_result);

            src_r_1 += 8;
            rows0_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[1] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 7;

            src_r_1++;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp   = (*src_r_1) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_x5 = neon::vmul(vq16_n0, static_cast<MovlType>(5));

            auto vq16_result0 = neon::vmla(vq16_n0_x5, vq16_c, static_cast<MovlType>(3));
            auto vq16_result1 = neon::vmla(vq16_c,    vq16_n0, static_cast<MovlType>(7));

            auto vd8_result0 = neon::vrshrn_n<6>(vq16_result0);
            auto vd8_result1 = neon::vrshrn_n<6>(vq16_result1);

            neon::vstore(dst_row0, vd8_result0);
            neon::vstore(dst_row1, vd8_result1);

            rows0_y += 8;
            rows1_y += 8;
            dst_row0 += 8;
            dst_row1 += 8;
        }

        for (; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (DT_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        DT_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*src_row1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vmovl(neon::vload1(src_row1));
            auto vq16_x1 = neon::vmovl(neon::vload1(src_row1 + 1));

            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));

            MVType v4q16_result;
            v4q16_result.val[3] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v4q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            v4q16_result.val[1] = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            v4q16_result.val[2] = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));

            neon::vstore(rows1_tmp, v4q16_result);

            src_row1 += 8;
            rows1_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 7 + src_row1[1];
            *rows1_tmp++ = src_row1[0] * 5 + src_row1[1] * 3;
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[1] * 5;
            *rows1_tmp++ = src_row1[0]     + src_row1[1] * 7;

            src_row1++;
        }
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp   = (*src_row1) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
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

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vmovl(neon::vload1(src_r_1));
            auto vq16_x1 = neon::vmovl(neon::vload1(src_r_1 + 1));

            auto vq16_x1_x3 = neon::vmul(vq16_x1, static_cast<MovlType>(3));
            auto vq16_x1_x5 = neon::vmul(vq16_x1, static_cast<MovlType>(5));

            MVType v4q16_result;
            v4q16_result.val[3] = neon::vmla(vq16_x0, vq16_x1, static_cast<MovlType>(7));
            v4q16_result.val[0] = neon::vmla(vq16_x1, vq16_x0, static_cast<MovlType>(7));
            v4q16_result.val[1] = neon::vmla(vq16_x1_x3, vq16_x0, static_cast<MovlType>(5));
            v4q16_result.val[2] = neon::vmla(vq16_x1_x5, vq16_x0, static_cast<MovlType>(3));

            neon::vstore(rows0_tmp, v4q16_result);

            src_r_1 += 8;
            rows0_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[1] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 7;

            src_r_1++;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp   = (*src_r_1) << 3;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq16_c  = neon::vload1q(rows0_y);
            auto vq16_n0 = neon::vload1q(rows1_y);

            auto vq16_n0_x3 = neon::vmul(vq16_n0, static_cast<MovlType>(3));

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

        for (; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_U16, DT_S16
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            auto v2q16_c  = neon::vload2q(src_row0);
            auto v2q16_n0 = neon::vload2q(src_row1);

            auto vq32_c = neon::vaddl(neon::vgetlow(v2q16_c.val[0]), neon::vgetlow(v2q16_n0.val[0]));
            vq32_c = neon::vadd(vq32_c, neon::vmovl(neon::vgetlow(v2q16_c.val[1])));
            vq32_c = neon::vadd(vq32_c, neon::vmovl(neon::vgetlow(v2q16_n0.val[1])));
            auto vq32_n0 = neon::vaddl(neon::vgethigh(v2q16_c.val[0]), neon::vgethigh(v2q16_n0.val[0]));
            vq32_n0 = neon::vadd(vq32_n0, neon::vmovl(neon::vgethigh(v2q16_c.val[1])));
            vq32_n0 = neon::vadd(vq32_n0, neon::vmovl(neon::vgethigh(v2q16_n0.val[1])));

            auto vq16_result = neon::vcombine(neon::vrshrn_n<2>(vq32_c), neon::vrshrn_n<2>(vq32_n0));

            neon::vstore(dst_row, vq16_result);

            src_row0 += 16;
            src_row1 += 16;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[1];
            MovlType r1 = src_row1[0] + src_row1[1];

            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 2;
            src_row1 += 2;
        }
    }

    return Status::OK;
}

// Tp = DT_U16, DT_S16
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnC1DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            auto v4q16_c  = neon::vload4q(src_row0);
            auto v4q16_n0 = neon::vload4q(src_row1);

            auto vq32_x0 = neon::vaddl(neon::vgetlow(v4q16_c.val[1]), neon::vgetlow(v4q16_n0.val[1]));
            vq32_x0 = neon::vadd(vq32_x0, neon::vmovl(neon::vgetlow(v4q16_c.val[2])));
            vq32_x0 = neon::vadd(vq32_x0, neon::vmovl(neon::vgetlow(v4q16_n0.val[2])));
            auto vq32_x1 = neon::vaddl(neon::vgethigh(v4q16_c.val[1]), neon::vgethigh(v4q16_n0.val[1]));
            vq32_x1 = neon::vadd(vq32_x1, neon::vmovl(neon::vgethigh(v4q16_c.val[2])));
            vq32_x1 = neon::vadd(vq32_x1, neon::vmovl(neon::vgethigh(v4q16_n0.val[2])));

            auto vq16_result = neon::vcombine(neon::vrshrn_n<2>(vq32_x0), neon::vrshrn_n<2>(vq32_x1));

            neon::vstore(dst_row, vq16_result);

            src_row0 += 32;
            src_row1 += 32;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[1] + src_row0[2];
            MovlType r1 = src_row1[1] + src_row1[2];

            *dst_row++ = SaturateCast<Tp>((r0 + r1 + 2) >> 2);

            src_row0 += 4;
            src_row1 += 4;
        }
    }

    return Status::OK;
}

// Tp = DT_U16, DT_S16
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnC1UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 2>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) * 4;
    DT_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto vq16_x0 = neon::vload1q(src_row0);
        auto vq16_x1 = neon::vload1q(src_row0 + 1);

        MVType v2q32_x0, v2q32_x1;
        v2q32_x0.val[0] = neon::vmovl(neon::vgetlow(vq16_x1));
        v2q32_x0.val[1] = neon::vmovl(neon::vgetlow(vq16_x0));
        v2q32_x0.val[0] = neon::vmlal(v2q32_x0.val[0], neon::vgetlow(vq16_x0), static_cast<Tp>(3));
        v2q32_x0.val[1] = neon::vmlal(v2q32_x0.val[1], neon::vgetlow(vq16_x1), static_cast<Tp>(3));

        v2q32_x1.val[0] = neon::vmovl(neon::vgethigh(vq16_x1));
        v2q32_x1.val[1] = neon::vmovl(neon::vgethigh(vq16_x0));
        v2q32_x1.val[0] = neon::vmlal(v2q32_x1.val[0], neon::vgethigh(vq16_x0), static_cast<Tp>(3));
        v2q32_x1.val[1] = neon::vmlal(v2q32_x1.val[1], neon::vgethigh(vq16_x1), static_cast<Tp>(3));

        neon::vstore(rows1_tmp, v2q32_x0);
        neon::vstore(rows1_tmp + 8, v2q32_x1);

        src_row0 += 8;
        rows1_tmp += 16;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[1];
        *rows1_tmp++ = src_row0[0]     + src_row0[1] * 3;

        src_row0++;
    }
    *rows1_tmp = (*src_row0) * 4;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) * 4;
        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0  = neon::vload1q(src_r_1);
            auto vq16_x1 = neon::vload1q(src_r_1 + 1);

            MVType v2q32_x0, v2q32_x1;
            v2q32_x0.val[0] = neon::vmovl(neon::vgetlow(vq16_x1));
            v2q32_x0.val[1] = neon::vmovl(neon::vgetlow(vq16_x0));
            v2q32_x0.val[0] = neon::vmlal(v2q32_x0.val[0], neon::vgetlow(vq16_x0), static_cast<Tp>(3));
            v2q32_x0.val[1] = neon::vmlal(v2q32_x0.val[1], neon::vgetlow(vq16_x1), static_cast<Tp>(3));

            v2q32_x1.val[0] = neon::vmovl(neon::vgethigh(vq16_x1));
            v2q32_x1.val[1] = neon::vmovl(neon::vgethigh(vq16_x0));
            v2q32_x1.val[0] = neon::vmlal(v2q32_x1.val[0], neon::vgethigh(vq16_x0), static_cast<Tp>(3));
            v2q32_x1.val[1] = neon::vmlal(v2q32_x1.val[1], neon::vgethigh(vq16_x1), static_cast<Tp>(3));

            neon::vstore(rows0_tmp, v2q32_x0);
            neon::vstore(rows0_tmp + 8, v2q32_x1);

            src_r_1 += 8;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 3;

            src_r_1++;
        }
        *rows0_tmp = (*src_r_1) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq32_cx0  = neon::vload1q(rows0_y);
            auto vq32_n0x0 = neon::vload1q(rows1_y);
            auto vq32_cx1  = neon::vload1q(rows0_y + 4);
            auto vq32_n0x1 = neon::vload1q(rows1_y + 4);

            auto vq32_n0x0_result = neon::vmla(vq32_cx0, vq32_n0x0, static_cast<MovlType>(3));
            auto vq32_n0x1_result = neon::vmla(vq32_cx1, vq32_n0x1, static_cast<MovlType>(3));

            auto vd16_n0x0 = neon::vrshrn_n<4>(vq32_n0x0_result);
            auto vd16_n0x1 = neon::vrshrn_n<4>(vq32_n0x1_result);

            auto vq16_n0_result = neon::vcombine(vd16_n0x0, vd16_n0x1);

            neon::vstore(dst_row, vq16_n0_result);

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] + rows1_y[0] * 3 + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (DT_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        DT_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = (*src_row1) * 4;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vload1q(src_row1);
            auto vq16_x1 = neon::vload1q(src_row1 + 1);

            MVType v2q32_x0, v2q32_x1;
            v2q32_x0.val[0] = neon::vmovl(neon::vgetlow(vq16_x1));
            v2q32_x0.val[1] = neon::vmovl(neon::vgetlow(vq16_x0));
            v2q32_x0.val[0] = neon::vmlal(v2q32_x0.val[0], neon::vgetlow(vq16_x0), static_cast<Tp>(3));
            v2q32_x0.val[1] = neon::vmlal(v2q32_x0.val[1], neon::vgetlow(vq16_x1), static_cast<Tp>(3));

            v2q32_x1.val[0] = neon::vmovl(neon::vgethigh(vq16_x1));
            v2q32_x1.val[1] = neon::vmovl(neon::vgethigh(vq16_x0));
            v2q32_x1.val[0] = neon::vmlal(v2q32_x1.val[0], neon::vgethigh(vq16_x0), static_cast<Tp>(3));
            v2q32_x1.val[1] = neon::vmlal(v2q32_x1.val[1], neon::vgethigh(vq16_x1), static_cast<Tp>(3));

            neon::vstore(rows1_tmp, v2q32_x0);
            neon::vstore(rows1_tmp + 8, v2q32_x1);

            src_row1 += 8;
            rows1_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[1];
            *rows1_tmp++ = src_row1[0]     + src_row1[1] * 3;

            src_row1++;
        }
        *rows1_tmp = (*src_row1) * 4;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq32_cx0  = neon::vload1q(rows0_y);
            auto vq32_n0x0 = neon::vload1q(rows1_y);
            auto vq32_cx1  = neon::vload1q(rows0_y + 4);
            auto vq32_n0x1 = neon::vload1q(rows1_y + 4);

            auto vq32_cx0_result  = neon::vmla(vq32_n0x0, vq32_cx0, static_cast<MovlType>(3));
            auto vq32_n0x0_result = neon::vmla(vq32_cx0, vq32_n0x0, static_cast<MovlType>(3));
            auto vq32_cx1_result  = neon::vmla(vq32_n0x1, vq32_cx1, static_cast<MovlType>(3));
            auto vq32_n0x1_result = neon::vmla(vq32_cx1, vq32_n0x1, static_cast<MovlType>(3));

            auto vd16_cx0  = neon::vrshrn_n<4>(vq32_cx0_result);
            auto vd16_n0x0 = neon::vrshrn_n<4>(vq32_n0x0_result);
            auto vd16_cx1  = neon::vrshrn_n<4>(vq32_cx1_result);
            auto vd16_n0x1 = neon::vrshrn_n<4>(vq32_n0x1_result);

            auto vq16_c_result  = neon::vcombine(vd16_cx0, vd16_cx1);
            auto vq16_n0_result = neon::vcombine(vd16_n0x0, vd16_n0x1);

            neon::vstore(dst_row0, vq16_c_result);
            neon::vstore(dst_row1, vq16_n0_result);

            dst_row0 += 8;
            dst_row1 += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>(((*rows1_tmp++) + 2) >> 2);
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) * 4;
        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vload1q(src_r_1);
            auto vq16_x1 = neon::vload1q(src_r_1 + 1);

            MVType v2q32_x0, v2q32_x1;
            v2q32_x0.val[0] = neon::vmovl(neon::vgetlow(vq16_x1));
            v2q32_x0.val[1] = neon::vmovl(neon::vgetlow(vq16_x0));
            v2q32_x0.val[0] = neon::vmlal(v2q32_x0.val[0], neon::vgetlow(vq16_x0), static_cast<Tp>(3));
            v2q32_x0.val[1] = neon::vmlal(v2q32_x0.val[1], neon::vgetlow(vq16_x1), static_cast<Tp>(3));

            v2q32_x1.val[0] = neon::vmovl(neon::vgethigh(vq16_x1));
            v2q32_x1.val[1] = neon::vmovl(neon::vgethigh(vq16_x0));
            v2q32_x1.val[0] = neon::vmlal(v2q32_x1.val[0], neon::vgethigh(vq16_x0), static_cast<Tp>(3));
            v2q32_x1.val[1] = neon::vmlal(v2q32_x1.val[1], neon::vgethigh(vq16_x1), static_cast<Tp>(3));

            neon::vstore(rows0_tmp, v2q32_x0);
            neon::vstore(rows0_tmp + 8, v2q32_x1);

            src_r_1 += 8;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 3;

            src_r_1++;
        }
        *rows0_tmp = (*src_r_1) * 4;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align8 = owidth & (-8);
        x = 0;
        for(; x < owidth_align8; x += 8)
        {
            auto vq32_cx0  = neon::vload1q(rows0_y);
            auto vq32_n0x0 = neon::vload1q(rows1_y);
            auto vq32_cx1  = neon::vload1q(rows0_y + 4);
            auto vq32_n0x1 = neon::vload1q(rows1_y + 4);

            auto vq32_n0x0_result = neon::vmla(vq32_n0x0, vq32_cx0, static_cast<MovlType>(3));
            auto vq32_n0x1_result = neon::vmla(vq32_n0x1, vq32_cx1, static_cast<MovlType>(3));

            auto vd16_n0x0 = neon::vrshrn_n<4>(vq32_n0x0_result);
            auto vd16_n0x1 = neon::vrshrn_n<4>(vq32_n0x1_result);

            auto vq16_n0_result = neon::vcombine(vd16_n0x0, vd16_n0x1);

            neon::vstore(dst_row, vq16_n0_result);

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] + 8) >> 4);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_U16, DT_S16
template <typename Tp>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnC1UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 4>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align8 = (iwidth - 1) & (-8);

    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp++ = (*src_row0) << 3;
    DT_S32 x = 0;
    for (; x < iwidth_1_align8; x += 8)
    {
        auto vq16_x0 = neon::vload1q(src_row0);
        auto vq16_x1 = neon::vload1q(src_row0 + 1);

        auto vq32_x0_l_x1 = neon::vmovl(neon::vgetlow(vq16_x0));
        auto vq32_x1_l_x1 = neon::vmovl(neon::vgetlow(vq16_x1));

        auto vq32_x0_h_x1 = neon::vmovl(neon::vgethigh(vq16_x0));
        auto vq32_x1_h_x1 = neon::vmovl(neon::vgethigh(vq16_x1));

        auto vq32_x0_l_x3 = neon::vmul(vq32_x0_l_x1, static_cast<MovlType>(3));
        auto vq32_x1_l_x3 = neon::vmul(vq32_x1_l_x1, static_cast<MovlType>(3));

        auto vq32_x0_h_x3 = neon::vmul(vq32_x0_h_x1, static_cast<MovlType>(3));
        auto vq32_x1_h_x3 = neon::vmul(vq32_x1_h_x1, static_cast<MovlType>(3));

        MVType v4q32_x0, v4q32_x1;
        v4q32_x0.val[0] = neon::vmla(vq32_x1_l_x1, vq32_x0_l_x1, static_cast<MovlType>(7));
        v4q32_x0.val[1] = neon::vmla(vq32_x1_l_x3, vq32_x0_l_x1, static_cast<MovlType>(5));
        v4q32_x0.val[2] = neon::vmla(vq32_x0_l_x3, vq32_x1_l_x1, static_cast<MovlType>(5));
        v4q32_x0.val[3] = neon::vmla(vq32_x0_l_x1, vq32_x1_l_x1, static_cast<MovlType>(7));

        v4q32_x1.val[0] = neon::vmla(vq32_x1_h_x1, vq32_x0_h_x1, static_cast<MovlType>(7));
        v4q32_x1.val[1] = neon::vmla(vq32_x1_h_x3, vq32_x0_h_x1, static_cast<MovlType>(5));
        v4q32_x1.val[2] = neon::vmla(vq32_x0_h_x3, vq32_x1_h_x1, static_cast<MovlType>(5));
        v4q32_x1.val[3] = neon::vmla(vq32_x0_h_x1, vq32_x1_h_x1, static_cast<MovlType>(7));

        neon::vstore(rows1_tmp, v4q32_x0);
        neon::vstore(rows1_tmp + 16, v4q32_x1);

        src_row0 += 8;
        rows1_tmp += 32;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 7 + src_row0[1];
        *rows1_tmp++ = src_row0[0] * 5 + src_row0[1] * 3;
        *rows1_tmp++ = src_row0[0] * 3 + src_row0[1] * 5;
        *rows1_tmp++ = src_row0[0]     + src_row0[1] * 7;

        src_row0++;
    }
    *rows1_tmp++ = (*src_row0) << 3;
    *rows1_tmp   = (*src_row0) << 3;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);

            rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vload1q(src_r_1);
            auto vq16_x1 = neon::vload1q(src_r_1 + 1);

            auto vq32_x0_l_x1 = neon::vmovl(neon::vgetlow(vq16_x0));
            auto vq32_x1_l_x1 = neon::vmovl(neon::vgetlow(vq16_x1));

            auto vq32_x0_h_x1 = neon::vmovl(neon::vgethigh(vq16_x0));
            auto vq32_x1_h_x1 = neon::vmovl(neon::vgethigh(vq16_x1));

            auto vq32_x0_l_x3 = neon::vmul(vq32_x0_l_x1, static_cast<MovlType>(3));
            auto vq32_x1_l_x3 = neon::vmul(vq32_x1_l_x1, static_cast<MovlType>(3));

            auto vq32_x0_h_x3 = neon::vmul(vq32_x0_h_x1, static_cast<MovlType>(3));
            auto vq32_x1_h_x3 = neon::vmul(vq32_x1_h_x1, static_cast<MovlType>(3));

            MVType v4q32_x0, v4q32_x1;
            v4q32_x0.val[0] = neon::vmla(vq32_x1_l_x1, vq32_x0_l_x1, static_cast<MovlType>(7));
            v4q32_x0.val[1] = neon::vmla(vq32_x1_l_x3, vq32_x0_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[2] = neon::vmla(vq32_x0_l_x3, vq32_x1_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[3] = neon::vmla(vq32_x0_l_x1, vq32_x1_l_x1, static_cast<MovlType>(7));

            v4q32_x1.val[0] = neon::vmla(vq32_x1_h_x1, vq32_x0_h_x1, static_cast<MovlType>(7));
            v4q32_x1.val[1] = neon::vmla(vq32_x1_h_x3, vq32_x0_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[2] = neon::vmla(vq32_x0_h_x3, vq32_x1_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[3] = neon::vmla(vq32_x0_h_x1, vq32_x1_h_x1, static_cast<MovlType>(7));

            neon::vstore(rows0_tmp, v4q32_x0);
            neon::vstore(rows0_tmp + 16, v4q32_x1);

            src_r_1 += 8;
            rows0_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[1] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 7;

            src_r_1++;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp   = (*src_r_1) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_c_x3    = neon::vmul(vq32_c, static_cast<MovlType>(3));
            auto vq32_result0 = neon::vmla(vq32_c_x3, vq32_n0, static_cast<MovlType>(5));
            auto vq32_result1 = neon::vmla(vq32_c, vq32_n0, static_cast<MovlType>(7));

            auto vd16_result0 = neon::vrshrn_n<6>(vq32_result0);
            auto vd16_result1 = neon::vrshrn_n<6>(vq32_result1);

            neon::vstore(dst_row0, vd16_result0);
            neon::vstore(dst_row1, vd16_result1);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 3 + rows1_y[0] * 5 + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0]     + rows1_y[0] * 7 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (DT_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        DT_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp++ = (*src_row1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vload1q(src_row1);
            auto vq16_x1 = neon::vload1q(src_row1 + 1);

            auto vq32_x0_l_x1 = neon::vmovl(neon::vgetlow(vq16_x0));
            auto vq32_x1_l_x1 = neon::vmovl(neon::vgetlow(vq16_x1));

            auto vq32_x0_h_x1 = neon::vmovl(neon::vgethigh(vq16_x0));
            auto vq32_x1_h_x1 = neon::vmovl(neon::vgethigh(vq16_x1));

            auto vq32_x0_l_x3 = neon::vmul(vq32_x0_l_x1, static_cast<MovlType>(3));
            auto vq32_x1_l_x3 = neon::vmul(vq32_x1_l_x1, static_cast<MovlType>(3));

            auto vq32_x0_h_x3 = neon::vmul(vq32_x0_h_x1, static_cast<MovlType>(3));
            auto vq32_x1_h_x3 = neon::vmul(vq32_x1_h_x1, static_cast<MovlType>(3));

            MVType v4q32_x0, v4q32_x1;
            v4q32_x0.val[0] = neon::vmla(vq32_x1_l_x1, vq32_x0_l_x1, static_cast<MovlType>(7));
            v4q32_x0.val[1] = neon::vmla(vq32_x1_l_x3, vq32_x0_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[2] = neon::vmla(vq32_x0_l_x3, vq32_x1_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[3] = neon::vmla(vq32_x0_l_x1, vq32_x1_l_x1, static_cast<MovlType>(7));

            v4q32_x1.val[0] = neon::vmla(vq32_x1_h_x1, vq32_x0_h_x1, static_cast<MovlType>(7));
            v4q32_x1.val[1] = neon::vmla(vq32_x1_h_x3, vq32_x0_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[2] = neon::vmla(vq32_x0_h_x3, vq32_x1_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[3] = neon::vmla(vq32_x0_h_x1, vq32_x1_h_x1, static_cast<MovlType>(7));

            neon::vstore(rows1_tmp, v4q32_x0);
            neon::vstore(rows1_tmp + 16, v4q32_x1);

            src_row1 += 8;
            rows1_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 7 + src_row1[1];
            *rows1_tmp++ = src_row1[0] * 5 + src_row1[1] * 3;
            *rows1_tmp++ = src_row1[0] * 3 + src_row1[1] * 5;
            *rows1_tmp++ = src_row1[0]     + src_row1[1] * 7;

            src_row1++;
        }
        *rows1_tmp++ = (*src_row1) << 3;
        *rows1_tmp   = (*src_row1) << 3;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_c_x3  = neon::vmul(vq32_c, static_cast<MovlType>(3));
            auto vq32_n0_x3 = neon::vmul(vq32_n0, static_cast<MovlType>(3));

            auto vq32_result0 = neon::vmla(vq32_n0, vq32_c, static_cast<MovlType>(7));
            auto vq32_result1 = neon::vmla(vq32_n0_x3, vq32_c, static_cast<MovlType>(5));
            auto vq32_result2 = neon::vmla(vq32_c_x3, vq32_n0, static_cast<MovlType>(5));
            auto vq32_result3 = neon::vmla(vq32_c, vq32_n0, static_cast<MovlType>(7));

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

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            *dst_row1++ = SaturateCast<Tp>((*rows1_tmp + 4) >> 3);
            rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp++ = (*src_r_1) << 3;

        DT_S32 x = 0;
        for (; x < iwidth_1_align8; x += 8)
        {
            auto vq16_x0 = neon::vload1q(src_r_1);
            auto vq16_x1 = neon::vload1q(src_r_1 + 1);

            auto vq32_x0_l_x1 = neon::vmovl(neon::vgetlow(vq16_x0));
            auto vq32_x1_l_x1 = neon::vmovl(neon::vgetlow(vq16_x1));

            auto vq32_x0_h_x1 = neon::vmovl(neon::vgethigh(vq16_x0));
            auto vq32_x1_h_x1 = neon::vmovl(neon::vgethigh(vq16_x1));

            auto vq32_x0_l_x3 = neon::vmul(vq32_x0_l_x1, static_cast<MovlType>(3));
            auto vq32_x1_l_x3 = neon::vmul(vq32_x1_l_x1, static_cast<MovlType>(3));

            auto vq32_x0_h_x3 = neon::vmul(vq32_x0_h_x1, static_cast<MovlType>(3));
            auto vq32_x1_h_x3 = neon::vmul(vq32_x1_h_x1, static_cast<MovlType>(3));

            MVType v4q32_x0, v4q32_x1;
            v4q32_x0.val[0] = neon::vmla(vq32_x1_l_x1, vq32_x0_l_x1, static_cast<MovlType>(7));
            v4q32_x0.val[1] = neon::vmla(vq32_x1_l_x3, vq32_x0_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[2] = neon::vmla(vq32_x0_l_x3, vq32_x1_l_x1, static_cast<MovlType>(5));
            v4q32_x0.val[3] = neon::vmla(vq32_x0_l_x1, vq32_x1_l_x1, static_cast<MovlType>(7));

            v4q32_x1.val[0] = neon::vmla(vq32_x1_h_x1, vq32_x0_h_x1, static_cast<MovlType>(7));
            v4q32_x1.val[1] = neon::vmla(vq32_x1_h_x3, vq32_x0_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[2] = neon::vmla(vq32_x0_h_x3, vq32_x1_h_x1, static_cast<MovlType>(5));
            v4q32_x1.val[3] = neon::vmla(vq32_x0_h_x1, vq32_x1_h_x1, static_cast<MovlType>(7));

            neon::vstore(rows0_tmp, v4q32_x0);
            neon::vstore(rows0_tmp + 16, v4q32_x1);

            src_r_1 += 8;
            rows0_tmp += 32;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 7 + src_r_1[1];
            *rows0_tmp++ = src_r_1[0] * 5 + src_r_1[1] * 3;
            *rows0_tmp++ = src_r_1[0] * 3 + src_r_1[1] * 5;
            *rows0_tmp++ = src_r_1[0]     + src_r_1[1] * 7;

            src_r_1++;
        }
        *rows0_tmp++ = (*src_r_1) << 3;
        *rows0_tmp   = (*src_r_1) << 3;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            auto vq32_c  = neon::vload1q(rows0_y);
            auto vq32_n0 = neon::vload1q(rows1_y);

            auto vq32_n0_x3   = neon::vmul(vq32_n0, static_cast<MovlType>(3));
            auto vq32_result0 = neon::vmla(vq32_n0, vq32_c, static_cast<MovlType>(7));
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

        for (; x < owidth; x++)
        {
            *dst_row0++ = SaturateCast<Tp>((rows0_y[0] * 7 + rows1_y[0]     + 32) >> 6);
            *dst_row1++ = SaturateCast<Tp>((rows0_y[0] * 5 + rows1_y[0] * 3 + 32) >> 6);

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_F32
template <typename Tp>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeBnC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align4 = owidth & (-4);
        DT_S32 x = 0;
        for (; x < owidth_align4; x += 4)
        {
            float32x4x2_t v2qf32_x0 = neon::vload2q(src_row0);
            float32x4x2_t v2qf32_x1 = neon::vload2q(src_row1);

            float32x4_t vqf32_x0 = neon::vadd(v2qf32_x0.val[0], v2qf32_x0.val[1]);
            float32x4_t vqf32_x1 = neon::vadd(v2qf32_x1.val[0], v2qf32_x1.val[1]);

            float32x4_t vqf32_result = neon::vadd(vqf32_x0, vqf32_x1);
            vqf32_result = neon::vmul(vqf32_result, vqf32_const_1_4);

            neon::vstore(dst_row, vqf32_result);

            src_row0 += 8;
            src_row1 += 8;
            dst_row  += 4;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[1];
            MovlType r1 = src_row1[0] + src_row1[1];

            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 2;
            src_row1 += 2;
        }
    }

    return Status::OK;
}

// Tp = DT_F32
template <typename Tp>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeBnC1DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align4 = owidth & (-4);
        DT_S32 x = 0;
        for (; x < owidth_align4; x += 4)
        {
            auto v4qf32_x0 = neon::vload4q(src_row0);
            auto v4qf32_x1 = neon::vload4q(src_row1);

            float32x4_t vqf32_x0 = neon::vadd(v4qf32_x0.val[1], v4qf32_x1.val[1]);
            float32x4_t vqf32_x1 = neon::vadd(v4qf32_x0.val[2], v4qf32_x1.val[2]);

            float32x4_t vqf32_result = neon::vadd(vqf32_x0, vqf32_x1);
            vqf32_result = neon::vmul(vqf32_result, vqf32_const_1_4);

            neon::vstore(dst_row, vqf32_result);

            src_row0 += 16;
            src_row1 += 16;
            dst_row  += 4;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[1] + src_row0[2];
            MovlType r1 = src_row1[1] + src_row1[2];

            *dst_row++ = (r0 + r1) * 0.25f;

            src_row0 += 4;
            src_row1 += 4;
        }
    }

    return Status::OK;
}

// Tp = DT_F32
template <typename Tp>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeBnC1UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 2>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    DT_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4_t vqf32_x0 = neon::vload1q(src_row0);
        float32x4_t vqf32_x1 = neon::vload1q(src_row0 + 1);

        float32x4_t vqf32_x0_result = neon::vmul(vqf32_x0, 0.75f);
        float32x4_t vqf32_x1_result = neon::vmul(vqf32_x1, 0.75f);

        MVType v2qf32_result;
        v2qf32_result.val[0] = neon::vmla(vqf32_x0_result, vqf32_x1, 0.25f);
        v2qf32_result.val[1] = neon::vmla(vqf32_x1_result, vqf32_x0, 0.25f);

        neon::vstore(rows1_tmp, v2qf32_result);

        src_row0 += 4;
        rows1_tmp += 8;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.75f + src_row0[1] * 0.25f;
        *rows1_tmp++ = src_row0[0] * 0.25f + src_row0[1] * 0.75f;

        src_row0++;
    }
    *rows1_tmp = *src_row0;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(src_r_1);
            float32x4_t vqf32_n0 = neon::vload1q(src_r_1 + 1);

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            MVType v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows0_tmp, v2qf32_result);

            src_r_1 += 4;
            rows0_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75 + src_r_1[1] * 0.25;
            *rows0_tmp++ = src_r_1[0] * 0.25 + src_r_1[1] * 0.75;

            src_r_1++;
        }
        *rows0_tmp = *src_r_1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);
            vqf32_n0_result = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(dst_row, vqf32_n0_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (DT_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        DT_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = *src_row1;

        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(src_row1);
            float32x4_t vqf32_n0 = neon::vload1q(src_row1 + 1);

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            MVType v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows1_tmp, v2qf32_result);

            src_row1 += 4;
            rows1_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.75 + src_row1[1] * 0.25;
            *rows1_tmp++ = src_row1[0] * 0.25 + src_row1[1] * 0.75;

            src_row1++;
        }
        *rows1_tmp = *src_row1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            vqf32_c_result  = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            vqf32_n0_result = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(dst_row0, vqf32_c_result);
            neon::vstore(dst_row1, vqf32_n0_result);

            dst_row0 += 4;
            dst_row1 += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(src_r_1);
            float32x4_t vqf32_n0 = neon::vload1q(src_r_1 + 1);

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            MVType v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows0_tmp, v2qf32_result);

            src_r_1 += 4;
            rows0_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75 + src_r_1[1] * 0.25;
            *rows0_tmp++ = src_r_1[0] * 0.25 + src_r_1[1] * 0.75;

            src_r_1++;
        }
        *rows0_tmp = *src_r_1;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result = neon::vmul(vqf32_c, 0.75f);
            vqf32_c_result = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);

            neon::vstore(dst_row, vqf32_c_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = rows0_y[0] * 0.75f + rows1_y[0] * 0.25f;

            rows0_y++;
            rows1_y++;
        }
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_F32
template <typename Tp>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeBnC1UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType   = typename neon::MQVector<MovlType, 4>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *src_row0;
    DT_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4_t vqf32_x0 = neon::vload1q(src_row0);
        float32x4_t vqf32_x1 = neon::vload1q(src_row0 + 1);

        float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
        float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
        float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
        float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

        float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
        float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
        float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
        float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

        MVType v4qf32_result;
        v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
        v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
        v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
        v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

        neon::vstore(rows1_tmp, v4qf32_result);

        src_row0 += 4;
        rows1_tmp += 16;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.875f + src_row0[1] * 0.125f;
        *rows1_tmp++ = src_row0[0] * 0.625f + src_row0[1] * 0.375f;
        *rows1_tmp++ = src_row0[0] * 0.375f + src_row0[1] * 0.625f;
        *rows1_tmp++ = src_row0[0] * 0.125f + src_row0[1] * 0.875f;

        src_row0++;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp   = *src_row0;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = *rows1_tmp;
            *dst_row1++ = *rows1_tmp;

            rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *src_r_1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vload1q(src_r_1);
            float32x4_t vqf32_x1 = neon::vload1q(src_r_1 + 1);

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            MVType v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows0_tmp, v4qf32_result);

            src_r_1 += 4;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[1] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[1] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[1] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[1] * 0.875f;

            src_r_1++;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp   = *src_r_1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1  = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3  = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_n0_x5 = neon::vmul(vqf32_n0, 0.625f);
            float32x4_t vqf32_n0_x7 = neon::vmul(vqf32_n0, 0.875f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x3, vqf32_n0_x5);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x1, vqf32_n0_x7);

            neon::vstore(dst_row0, vqf32_result0);
            neon::vstore(dst_row1, vqf32_result1);

            rows0_y += 4;
            rows1_y += 4;
            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row1++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (DT_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        DT_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *src_row1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vload1q(src_row1);
            float32x4_t vqf32_x1 = neon::vload1q(src_row1 + 1);

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            MVType v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows1_tmp, v4qf32_result);

            src_row1 += 4;
            rows1_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.875f + src_row1[1] * 0.125f;
            *rows1_tmp++ = src_row1[0] * 0.625f + src_row1[1] * 0.375f;
            *rows1_tmp++ = src_row1[0] * 0.375f + src_row1[1] * 0.625f;
            *rows1_tmp++ = src_row1[0] * 0.125f + src_row1[1] * 0.875f;

            src_row1++;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp   = *src_row1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);

            float32x4_t vqf32_n0x1  = neon::vmul(vqf32_n0, 0.125f);
            float32x4_t vqf32_n0_x3 = neon::vmul(vqf32_n0, 0.375f);
            float32x4_t vqf32_n0_x5 = neon::vmul(vqf32_n0, 0.625f);
            float32x4_t vqf32_n0_x7 = neon::vmul(vqf32_n0, 0.875f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x7, vqf32_n0x1);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x5, vqf32_n0_x3);
            float32x4_t vqf32_result2 = neon::vadd(vqf32_c_x3, vqf32_n0_x5);
            float32x4_t vqf32_result3 = neon::vadd(vqf32_c_x1, vqf32_n0_x7);

            neon::vstore(dst_row0, vqf32_result0);
            neon::vstore(dst_row1, vqf32_result1);
            neon::vstore(dst_row2, vqf32_result2);
            neon::vstore(dst_row3, vqf32_result3);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
            dst_row2 += 4;
            dst_row3 += 4;
        }

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = *rows1_tmp;
            *dst_row1++ = *rows1_tmp;
            rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *src_r_1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vload1q(src_r_1);
            float32x4_t vqf32_x1 = neon::vload1q(src_r_1 + 1);

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            MVType v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows0_tmp, v4qf32_result);

            src_r_1 += 4;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[1] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[1] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[1] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[1] * 0.875f;

            src_r_1++;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp   = *src_r_1;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_x1 = neon::vmul(vqf32_n0, 0.125f);
            float32x4_t vqf32_n0_x3 = neon::vmul(vqf32_n0, 0.375f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x7, vqf32_n0_x1);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x5, vqf32_n0_x3);

            neon::vstore(dst_row0, vqf32_result0);
            neon::vstore(dst_row1, vqf32_result1);

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth; x++)
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
ResizeBnC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 2;

        const Tp *src_row0 = src.Ptr<Tp>(sy);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            float16x8x2_t v2qf16_c  = neon::vload2q(src_row0);
            float16x8x2_t v2qf16_n0 = neon::vload2q(src_row1);

            float32x4_t vqf32_x0 = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v2qf16_c.val[0])), neon::vcvt<MovlType>(neon::vgetlow(v2qf16_n0.val[0])));
            vqf32_x0 = neon::vadd(vqf32_x0, neon::vcvt<MovlType>(neon::vgetlow(v2qf16_c.val[1])));
            vqf32_x0 = neon::vadd(vqf32_x0, neon::vcvt<MovlType>(neon::vgetlow(v2qf16_n0.val[1])));
            float32x4_t vqf32_x1 = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v2qf16_c.val[0])), neon::vcvt<MovlType>(neon::vgethigh(v2qf16_n0.val[0])));
            vqf32_x1 = neon::vadd(vqf32_x1, neon::vcvt<MovlType>(neon::vgethigh(v2qf16_c.val[1])));
            vqf32_x1 = neon::vadd(vqf32_x1, neon::vcvt<MovlType>(neon::vgethigh(v2qf16_n0.val[1])));

            vqf32_x0 = neon::vmul(vqf32_x0, vqf32_const_1_4);
            vqf32_x1 = neon::vmul(vqf32_x1, vqf32_const_1_4);
            float16x8_t vqf16_result = neon::vcombine(neon::vcvt<Tp>(vqf32_x0), neon::vcvt<Tp>(vqf32_x1));
            neon::vstore(dst_row, vqf16_result);

            src_row0 += 16;
            src_row1 += 16;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[0] + src_row0[1];
            MovlType r1 = src_row1[0] + src_row1[1];

            *dst_row++ = SaturateCast<Tp>((r0 + r1) * 0.25f);

            src_row0 += 2;
            src_row1 += 2;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC1DownX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    float32x4_t vqf32_const_1_4;
    neon::vdup(vqf32_const_1_4, 0.25f);

    DT_S32 owidth = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy = y * 4;

        const Tp *src_row0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_row1 = src.Ptr<Tp>(sy + 2);

        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 x = 0;
        for (; x < owidth_align8; x += 8)
        {
            float16x8x4_t v4qf16_c  = neon::vload4q(src_row0);
            float16x8x4_t v4qf16_n0 = neon::vload4q(src_row1);

            float32x4_t vqf32_x0 = neon::vadd(neon::vcvt<MovlType>(neon::vgetlow(v4qf16_c.val[1])), neon::vcvt<MovlType>(neon::vgetlow(v4qf16_n0.val[1])));
            vqf32_x0 = neon::vadd(vqf32_x0, neon::vcvt<MovlType>(neon::vgetlow(v4qf16_c.val[2])));
            vqf32_x0 = neon::vadd(vqf32_x0, neon::vcvt<MovlType>(neon::vgetlow(v4qf16_n0.val[2])));
            float32x4_t vqf32_x1 = neon::vadd(neon::vcvt<MovlType>(neon::vgethigh(v4qf16_c.val[1])), neon::vcvt<MovlType>(neon::vgethigh(v4qf16_n0.val[1])));
            vqf32_x1 = neon::vadd(vqf32_x1, neon::vcvt<MovlType>(neon::vgethigh(v4qf16_c.val[2])));
            vqf32_x1 = neon::vadd(vqf32_x1, neon::vcvt<MovlType>(neon::vgethigh(v4qf16_n0.val[2])));

            vqf32_x0 = neon::vmul(vqf32_x0, vqf32_const_1_4);
            vqf32_x1 = neon::vmul(vqf32_x1, vqf32_const_1_4);
            float16x8_t vqf16_result = neon::vcombine(neon::vcvt<Tp>(vqf32_x0), neon::vcvt<Tp>(vqf32_x1));
            neon::vstore(dst_row, vqf16_result);

            src_row0 += 32;
            src_row1 += 32;
            dst_row += 8;
        }

        for (; x < owidth; x++)
        {
            MovlType r0 = src_row0[1] + src_row0[2];
            MovlType r1 = src_row1[1] + src_row1[2];

            *dst_row++ = SaturateCast<Tp>((r0 + r1) * 0.25f);

            src_row0 += 4;
            src_row1 += 4;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnC1UpX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 1;
    end_row = Min(end_row << 1, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 1) / 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    DT_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4_t vqf32_x0 = neon::vcvt<MovlType>(neon::vload1(src_row0));
        float32x4_t vqf32_x1 = neon::vcvt<MovlType>(neon::vload1(src_row0 + 1));

        float32x4_t vqf32_x0_result = neon::vmul(vqf32_x0, 0.75f);
        float32x4_t vqf32_x1_result = neon::vmul(vqf32_x1, 0.75f);

        float32x4x2_t v2qf32_result;
        v2qf32_result.val[0] = neon::vmla(vqf32_x0_result, vqf32_x1, 0.25f);
        v2qf32_result.val[1] = neon::vmla(vqf32_x1_result, vqf32_x0, 0.25f);

        neon::vstore(rows1_tmp, v2qf32_result);

        src_row0 += 4;
        rows1_tmp += 8;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.75f + src_row0[1] * 0.25f;
        *rows1_tmp++ = src_row0[0] * 0.25f + src_row0[1] * 0.75f;

        src_row0++;
    }
    *rows1_tmp = *src_row0;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vcvt<MovlType>(neon::vload1(src_r_1));
            float32x4_t vqf32_n0 = neon::vcvt<MovlType>(neon::vload1(src_r_1 + 1));

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            float32x4x2_t v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows0_tmp, v2qf32_result);

            src_r_1 += 4;
            rows0_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75 + src_r_1[1] * 0.25;
            *rows0_tmp++ = src_r_1[0] * 0.25 + src_r_1[1] * 0.75;

            src_r_1++;
        }
        *rows0_tmp = *src_r_1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);
            vqf32_n0_result = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);


            neon::vstore(dst_row, neon::vcvt<Tp>(vqf32_n0_result));

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row++ = rows0_y[0] * 0.25f + rows1_y[0] * 0.75f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 1);

    for (DT_S32 y = start_row + 1; y < end_row - 1; y += 2)
    {
        DT_S32 sy = (y - 1) >> 1;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;
        *rows1_tmp++ = *src_row1;

        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vcvt<MovlType>(neon::vload1(src_row1));
            float32x4_t vqf32_n0 = neon::vcvt<MovlType>(neon::vload1(src_row1 + 1));

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            float32x4x2_t v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows1_tmp, v2qf32_result);

            src_row1 += 4;
            rows1_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.75 + src_row1[1] * 0.25;
            *rows1_tmp++ = src_row1[0] * 0.25 + src_row1[1] * 0.75;

            src_row1++;
        }
        *rows1_tmp = *src_row1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            vqf32_c_result  = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            vqf32_n0_result = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_c_result));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_n0_result));

            dst_row0 += 4;
            dst_row1 += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row++ = *rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        DT_S32 x = 0;
        for (; x< iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vcvt<MovlType>(neon::vload1(src_r_1));
            float32x4_t vqf32_n0 = neon::vcvt<MovlType>(neon::vload1(src_r_1 + 1));

            float32x4_t vqf32_c_result  = neon::vmul(vqf32_c, 0.75f);
            float32x4_t vqf32_n0_result = neon::vmul(vqf32_n0, 0.75f);

            float32x4x2_t v2qf32_result;
            v2qf32_result.val[0] = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);
            v2qf32_result.val[1] = neon::vmla(vqf32_n0_result, vqf32_c, 0.25f);

            neon::vstore(rows0_tmp, v2qf32_result);

            src_r_1 += 4;
            rows0_tmp += 8;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.75 + src_r_1[1] * 0.25;
            *rows0_tmp++ = src_r_1[0] * 0.25 + src_r_1[1] * 0.75;

            src_r_1++;
        }
        *rows0_tmp = *src_r_1;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_result = neon::vmul(vqf32_c, 0.75f);
            vqf32_c_result = neon::vmla(vqf32_c_result, vqf32_n0, 0.25f);

            neon::vstore(dst_row, neon::vcvt<Tp>(vqf32_c_result));

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
        }

        for (; x < owidth; x++)
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
ResizeBnC1UpX4NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    MovlType *rows = thread_buffer.GetThreadData<MovlType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MovlType *rows0 = rows;
    MovlType *rows1 = rows + owidth;

    start_row = start_row << 2;
    end_row   = Min(end_row << 2, oheight);

    const Tp *src_row0  = src.Ptr<Tp>((start_row + 2) >> 2);
    const Tp *src_r_1 = src_row0 - src.GetRowPitch() / sizeof(Tp);
    MovlType *rows0_tmp = rows0;
    MovlType *rows1_tmp = rows1;

    DT_S32 iwidth_1_align4 = (iwidth - 1) & (-4);

    *rows1_tmp++ = *src_row0;
    *rows1_tmp++ = *src_row0;
    DT_S32 x = 0;
    for (; x < iwidth_1_align4; x += 4)
    {
        float32x4_t vqf32_x0 = neon::vcvt<MovlType>(neon::vload1(src_row0));
        float32x4_t vqf32_x1 = neon::vcvt<MovlType>(neon::vload1(src_row0 + 1));

        float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
        float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
        float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
        float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

        float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
        float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
        float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
        float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

        float32x4x4_t v4qf32_result;
        v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
        v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
        v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
        v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

        neon::vstore(rows1_tmp, v4qf32_result);

        src_row0 += 4;
        rows1_tmp += 16;
    }
    for (; x < iwidth - 1; x++)
    {
        *rows1_tmp++ = src_row0[0] * 0.875f + src_row0[1] * 0.125f;
        *rows1_tmp++ = src_row0[0] * 0.625f + src_row0[1] * 0.375f;
        *rows1_tmp++ = src_row0[0] * 0.375f + src_row0[1] * 0.625f;
        *rows1_tmp++ = src_row0[0] * 0.125f + src_row0[1] * 0.875f;

        src_row0++;
    }
    *rows1_tmp++ = *src_row0;
    *rows1_tmp   = *src_row0;

    Tp *dst_row0 = dst.Ptr<Tp>(start_row);
    Tp *dst_row1 = dst.Ptr<Tp>(start_row + 1);

    if (0 == start_row)
    {
        rows1_tmp = rows1;
        for (DT_S32 x = 0; x < owidth; ++x)
        {
            *dst_row0++ = *rows1_tmp;
            *dst_row1++ = *rows1_tmp;

            rows1_tmp++;
        }
    }
    else
    {
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *src_r_1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vcvt<MovlType>(neon::vload1(src_r_1));
            float32x4_t vqf32_x1 = neon::vcvt<MovlType>(neon::vload1(src_r_1 + 1));

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            float32x4x4_t v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows0_tmp, v4qf32_result);

            src_r_1 += 4;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[1] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[1] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[1] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[1] * 0.875f;

            src_r_1++;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp   = *src_r_1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1  = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3  = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_n0_x5 = neon::vmul(vqf32_n0, 0.625f);
            float32x4_t vqf32_n0_x7 = neon::vmul(vqf32_n0, 0.875f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x3, vqf32_n0_x5);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x1, vqf32_n0_x7);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_result0));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_result1));

            rows0_y += 4;
            rows1_y += 4;
            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth; x++)
        {
            *dst_row0++ = rows0_y[0] * 0.375f + rows1_y[0] * 0.625f;
            *dst_row1++ = rows0_y[0] * 0.125f + rows1_y[0] * 0.875f;

            rows0_y++;
            rows1_y++;
        }
    }

    src_r_1 = src.Ptr<Tp>(end_row >> 2);

    for (DT_S32 y = start_row + 2; y < end_row - 2; y += 4)
    {
        DT_S32 sy = (y - 2) >> 2;

        MovlType *rows0_old = rows0;
        rows0 = rows1;
        rows1 = rows0_old;

        const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

        rows1_tmp = rows1;

        *rows1_tmp++ = *src_row1;
        *rows1_tmp++ = *src_row1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vcvt<MovlType>(neon::vload1(src_row1));
            float32x4_t vqf32_x1 = neon::vcvt<MovlType>(neon::vload1(src_row1 + 1));

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            float32x4x4_t v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows1_tmp, v4qf32_result);

            src_row1 += 4;
            rows1_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows1_tmp++ = src_row1[0] * 0.875f + src_row1[1] * 0.125f;
            *rows1_tmp++ = src_row1[0] * 0.625f + src_row1[1] * 0.375f;
            *rows1_tmp++ = src_row1[0] * 0.375f + src_row1[1] * 0.625f;
            *rows1_tmp++ = src_row1[0] * 0.125f + src_row1[1] * 0.875f;

            src_row1++;
        }
        *rows1_tmp++ = *src_row1;
        *rows1_tmp   = *src_row1;

        MovlType *rows0_y = rows0;
        MovlType *rows1_y = rows1;

        Tp *dst_row0 = dst.Ptr<Tp>(y);
        Tp *dst_row1 = dst.Ptr<Tp>(y + 1);
        Tp *dst_row2 = dst.Ptr<Tp>(y + 2);
        Tp *dst_row3 = dst.Ptr<Tp>(y + 3);

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x1 = neon::vmul(vqf32_c, 0.125f);
            float32x4_t vqf32_c_x3 = neon::vmul(vqf32_c, 0.375f);
            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);

            float32x4_t vqf32_n0x1  = neon::vmul(vqf32_n0, 0.125f);
            float32x4_t vqf32_n0_x3 = neon::vmul(vqf32_n0, 0.375f);
            float32x4_t vqf32_n0_x5 = neon::vmul(vqf32_n0, 0.625f);
            float32x4_t vqf32_n0_x7 = neon::vmul(vqf32_n0, 0.875f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x7, vqf32_n0x1);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x5, vqf32_n0_x3);
            float32x4_t vqf32_result2 = neon::vadd(vqf32_c_x3, vqf32_n0_x5);
            float32x4_t vqf32_result3 = neon::vadd(vqf32_c_x1, vqf32_n0_x7);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_result0));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_result1));
            neon::vstore(dst_row2, neon::vcvt<Tp>(vqf32_result2));
            neon::vstore(dst_row3, neon::vcvt<Tp>(vqf32_result3));

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
            dst_row2 += 4;
            dst_row3 += 4;
        }

        for (; x < owidth; x++)
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
        for (DT_S32 x = 0; x < owidth; x++)
        {
            *dst_row0++ = *rows1_tmp;
            *dst_row1++ = *rows1_tmp;
            rows1_tmp++;
        }
    }
    else
    {
        rows0_tmp = rows0;
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp++ = *src_r_1;

        DT_S32 x = 0;
        for (; x < iwidth_1_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vcvt<MovlType>(neon::vload1(src_r_1));
            float32x4_t vqf32_x1 = neon::vcvt<MovlType>(neon::vload1(src_r_1 + 1));

            float32x4_t vqf32_x0_x1 = neon::vmul(vqf32_x0, 0.125f);
            float32x4_t vqf32_x0_x3 = neon::vmul(vqf32_x0, 0.375f);
            float32x4_t vqf32_x0_x5 = neon::vmul(vqf32_x0, 0.625f);
            float32x4_t vqf32_x0_x7 = neon::vmul(vqf32_x0, 0.875f);

            float32x4_t vqf32_x1_x1 = neon::vmul(vqf32_x1, 0.125f);
            float32x4_t vqf32_x1_x3 = neon::vmul(vqf32_x1, 0.375f);
            float32x4_t vqf32_x1_x5 = neon::vmul(vqf32_x1, 0.625f);
            float32x4_t vqf32_x1_x7 = neon::vmul(vqf32_x1, 0.875f);

            float32x4x4_t v4qf32_result;
            v4qf32_result.val[0] = neon::vadd(vqf32_x0_x7, vqf32_x1_x1);
            v4qf32_result.val[1] = neon::vadd(vqf32_x0_x5, vqf32_x1_x3);
            v4qf32_result.val[2] = neon::vadd(vqf32_x0_x3, vqf32_x1_x5);
            v4qf32_result.val[3] = neon::vadd(vqf32_x0_x1, vqf32_x1_x7);

            neon::vstore(rows0_tmp, v4qf32_result);

            src_r_1 += 4;
            rows0_tmp += 16;
        }
        for (; x < iwidth - 1; x++)
        {
            *rows0_tmp++ = src_r_1[0] * 0.875f + src_r_1[1] * 0.125f;
            *rows0_tmp++ = src_r_1[0] * 0.625f + src_r_1[1] * 0.375f;
            *rows0_tmp++ = src_r_1[0] * 0.375f + src_r_1[1] * 0.625f;
            *rows0_tmp++ = src_r_1[0] * 0.125f + src_r_1[1] * 0.875f;

            src_r_1++;
        }
        *rows0_tmp++ = *src_r_1;
        *rows0_tmp   = *src_r_1;

        MovlType *rows0_y = rows1;
        MovlType *rows1_y = rows0;

        DT_S32 owidth_align4 = owidth & (-4);
        x = 0;
        for(; x < owidth_align4; x += 4)
        {
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);

            float32x4_t vqf32_c_x5 = neon::vmul(vqf32_c, 0.625f);
            float32x4_t vqf32_c_x7 = neon::vmul(vqf32_c, 0.875f);
            float32x4_t vqf32_n0_x1 = neon::vmul(vqf32_n0, 0.125f);
            float32x4_t vqf32_n0_x3 = neon::vmul(vqf32_n0, 0.375f);

            float32x4_t vqf32_result0 = neon::vadd(vqf32_c_x7, vqf32_n0_x1);
            float32x4_t vqf32_result1 = neon::vadd(vqf32_c_x5, vqf32_n0_x3);

            neon::vstore(dst_row0, neon::vcvt<Tp>(vqf32_result0));
            neon::vstore(dst_row1, neon::vcvt<Tp>(vqf32_result1));

            rows0_y += 4;
            rows1_y += 4;

            dst_row0 += 4;
            dst_row1 += 4;
        }

        for (; x < owidth; x++)
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
static Status ResizeBnFastC1NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    using MovlType = typename ResizeBnCuTraits<Tp>::MovlType;
    DT_S32 owidth  = dst.GetSizes().m_width;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        return Status::ERROR;
    }

    DT_F32 scale_x = static_cast<DT_F64>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    DT_F32 scale_y = static_cast<DT_F64>(src.GetSizes().m_height) / dst.GetSizes().m_height;

    if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 2.f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnC1DownX2NeonImpl<Tp>, ctx, src, dst);
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnC1DownX4NeonImpl<Tp>, ctx, src, dst);
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.5f))
    {
        ThreadBuffer thread_buffer(ctx, owidth * 2 * sizeof(MovlType));
        ret = wp->ParallelFor(0, AURA_ALIGN(dst.GetSizes().m_height, 2) / 2, ResizeBnC1UpX2NeonImpl<Tp>, ctx, src, dst, std::ref(thread_buffer));
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.25f))
    {
        ThreadBuffer thread_buffer(ctx, owidth * 2 * sizeof(MovlType));
        ret = wp->ParallelFor(0, AURA_ALIGN(dst.GetSizes().m_height, 4) / 4, ResizeBnC1UpX4NeonImpl<Tp>, ctx, src, dst, std::ref(thread_buffer));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "spacial scale param error");
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeBnFastC1Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnFastC1NeonHelper<DT_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnFastC1NeonHelper<DT_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnFastC1NeonHelper<DT_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnFastC1NeonHelper<DT_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: DT_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeBnFastC1NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeBnFastC1NeonHelper<DT_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastC1NeonHelper run failed, type: DT_F32");
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