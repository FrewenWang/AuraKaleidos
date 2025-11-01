#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeCuC3DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    using MVType        = typename neon::MDVector<Tp, 3>::MVType;
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align8 = owidth & (-8);

    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        // hresize two row
        MI_S32 sy        = dy << 2;
        const Tp *src_c  = src.Ptr<Tp>(sy);
        const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        Tp *dst_row      = dst.Ptr<Tp>(dy);

        MI_S32 x = 0;
        for (; x < width_align8; x += 8)
        {
            auto v3q8_cx0  = neon::vload3q(src_c);
            auto v3q8_cx1  = neon::vload3q(src_c + 48);
            auto v3q8_n0x0 = neon::vload3q(src_n0);
            auto v3q8_n0x1 = neon::vload3q(src_n0 + 48);
            auto v3q8_n1x0 = neon::vload3q(src_n1);
            auto v3q8_n1x1 = neon::vload3q(src_n1 + 48);
            auto v3q8_n2x0 = neon::vload3q(src_n2);
            auto v3q8_n2x1 = neon::vload3q(src_n2 + 48);

            auto v2q8_c_0  = neon::vuzp(v3q8_cx0.val[0], v3q8_cx1.val[0]);
            auto v2q8_c_1  = neon::vuzp(v2q8_c_0.val[0], v2q8_c_0.val[1]);
            auto v2q8_c_2  = neon::vuzp(v3q8_cx0.val[1], v3q8_cx1.val[1]);
            auto v2q8_c_3  = neon::vuzp(v2q8_c_2.val[0], v2q8_c_2.val[1]);
            auto v2q8_c_4  = neon::vuzp(v3q8_cx0.val[2], v3q8_cx1.val[2]);
            auto v2q8_c_5  = neon::vuzp(v2q8_c_4.val[0], v2q8_c_4.val[1]);
            auto v2q8_n0_0 = neon::vuzp(v3q8_n0x0.val[0], v3q8_n0x1.val[0]);
            auto v2q8_n0_1 = neon::vuzp(v2q8_n0_0.val[0], v2q8_n0_0.val[1]);
            auto v2q8_n0_2 = neon::vuzp(v3q8_n0x0.val[1], v3q8_n0x1.val[1]);
            auto v2q8_n0_3 = neon::vuzp(v2q8_n0_2.val[0], v2q8_n0_2.val[1]);
            auto v2q8_n0_4 = neon::vuzp(v3q8_n0x0.val[2], v3q8_n0x1.val[2]);
            auto v2q8_n0_5 = neon::vuzp(v2q8_n0_4.val[0], v2q8_n0_4.val[1]);
            auto v2q8_n1_0 = neon::vuzp(v3q8_n1x0.val[0], v3q8_n1x1.val[0]);
            auto v2q8_n1_1 = neon::vuzp(v2q8_n1_0.val[0], v2q8_n1_0.val[1]);
            auto v2q8_n1_2 = neon::vuzp(v3q8_n1x0.val[1], v3q8_n1x1.val[1]);
            auto v2q8_n1_3 = neon::vuzp(v2q8_n1_2.val[0], v2q8_n1_2.val[1]);
            auto v2q8_n1_4 = neon::vuzp(v3q8_n1x0.val[2], v3q8_n1x1.val[2]);
            auto v2q8_n1_5 = neon::vuzp(v2q8_n1_4.val[0], v2q8_n1_4.val[1]);
            auto v2q8_n2_0 = neon::vuzp(v3q8_n2x0.val[0], v3q8_n2x1.val[0]);
            auto v2q8_n2_1 = neon::vuzp(v2q8_n2_0.val[0], v2q8_n2_0.val[1]);
            auto v2q8_n2_2 = neon::vuzp(v3q8_n2x0.val[1], v3q8_n2x1.val[1]);
            auto v2q8_n2_3 = neon::vuzp(v2q8_n2_2.val[0], v2q8_n2_2.val[1]);
            auto v2q8_n2_4 = neon::vuzp(v3q8_n2x0.val[2], v3q8_n2x1.val[2]);
            auto v2q8_n2_5 = neon::vuzp(v2q8_n2_4.val[0], v2q8_n2_4.val[1]);

            int16x8_t vqs16_c_c0_12  = neon::vaddl(neon::vgethigh(v2q8_c_1.val[0]), neon::vgetlow(v2q8_c_1.val[1]));
            int16x8_t vqs16_c_c0_03  = neon::vaddl(neon::vgetlow(v2q8_c_1.val[0]), neon::vgethigh(v2q8_c_1.val[1]));
            int16x8_t vqs16_c_c1_12  = neon::vaddl(neon::vgethigh(v2q8_c_3.val[0]), neon::vgetlow(v2q8_c_3.val[1]));
            int16x8_t vqs16_c_c1_03  = neon::vaddl(neon::vgetlow(v2q8_c_3.val[0]), neon::vgethigh(v2q8_c_3.val[1]));
            int16x8_t vqs16_c_c2_12  = neon::vaddl(neon::vgethigh(v2q8_c_5.val[0]), neon::vgetlow(v2q8_c_5.val[1]));
            int16x8_t vqs16_c_c2_03  = neon::vaddl(neon::vgetlow(v2q8_c_5.val[0]), neon::vgethigh(v2q8_c_5.val[1]));
            int16x8_t vqs16_n0_c0_12 = neon::vaddl(neon::vgethigh(v2q8_n0_1.val[0]), neon::vgetlow(v2q8_n0_1.val[1]));
            int16x8_t vqs16_n0_c0_03 = neon::vaddl(neon::vgetlow(v2q8_n0_1.val[0]), neon::vgethigh(v2q8_n0_1.val[1]));
            int16x8_t vqs16_n0_c1_12 = neon::vaddl(neon::vgethigh(v2q8_n0_3.val[0]), neon::vgetlow(v2q8_n0_3.val[1]));
            int16x8_t vqs16_n0_c1_03 = neon::vaddl(neon::vgetlow(v2q8_n0_3.val[0]), neon::vgethigh(v2q8_n0_3.val[1]));
            int16x8_t vqs16_n0_c2_12 = neon::vaddl(neon::vgethigh(v2q8_n0_5.val[0]), neon::vgetlow(v2q8_n0_5.val[1]));
            int16x8_t vqs16_n0_c2_03 = neon::vaddl(neon::vgetlow(v2q8_n0_5.val[0]), neon::vgethigh(v2q8_n0_5.val[1]));
            int16x8_t vqs16_n1_c0_12 = neon::vaddl(neon::vgethigh(v2q8_n1_1.val[0]), neon::vgetlow(v2q8_n1_1.val[1]));
            int16x8_t vqs16_n1_c0_03 = neon::vaddl(neon::vgetlow(v2q8_n1_1.val[0]), neon::vgethigh(v2q8_n1_1.val[1]));
            int16x8_t vqs16_n1_c1_12 = neon::vaddl(neon::vgethigh(v2q8_n1_3.val[0]), neon::vgetlow(v2q8_n1_3.val[1]));
            int16x8_t vqs16_n1_c1_03 = neon::vaddl(neon::vgetlow(v2q8_n1_3.val[0]), neon::vgethigh(v2q8_n1_3.val[1]));
            int16x8_t vqs16_n1_c2_12 = neon::vaddl(neon::vgethigh(v2q8_n1_5.val[0]), neon::vgetlow(v2q8_n1_5.val[1]));
            int16x8_t vqs16_n1_c2_03 = neon::vaddl(neon::vgetlow(v2q8_n1_5.val[0]), neon::vgethigh(v2q8_n1_5.val[1]));
            int16x8_t vqs16_n2_c0_12 = neon::vaddl(neon::vgethigh(v2q8_n2_1.val[0]), neon::vgetlow(v2q8_n2_1.val[1]));
            int16x8_t vqs16_n2_c0_03 = neon::vaddl(neon::vgetlow(v2q8_n2_1.val[0]), neon::vgethigh(v2q8_n2_1.val[1]));
            int16x8_t vqs16_n2_c1_12 = neon::vaddl(neon::vgethigh(v2q8_n2_3.val[0]), neon::vgetlow(v2q8_n2_3.val[1]));
            int16x8_t vqs16_n2_c1_03 = neon::vaddl(neon::vgetlow(v2q8_n2_3.val[0]), neon::vgethigh(v2q8_n2_3.val[1]));
            int16x8_t vqs16_n2_c2_12 = neon::vaddl(neon::vgethigh(v2q8_n2_5.val[0]), neon::vgetlow(v2q8_n2_5.val[1]));
            int16x8_t vqs16_n2_c2_03 = neon::vaddl(neon::vgetlow(v2q8_n2_5.val[0]), neon::vgethigh(v2q8_n2_5.val[1]));

            int16x8_t vqs16_c_c0_x19     = neon::vmul(vqs16_c_c0_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c_c0_result  = neon::vmls(vqs16_c_c0_x19, vqs16_c_c0_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_c_c1_x19     = neon::vmul(vqs16_c_c1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c_c1_result  = neon::vmls(vqs16_c_c1_x19, vqs16_c_c1_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_c_c2_x19     = neon::vmul(vqs16_c_c2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c_c2_result  = neon::vmls(vqs16_c_c2_x19, vqs16_c_c2_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n0_c0_x19    = neon::vmul(vqs16_n0_c0_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n0_c0_result = neon::vmls(vqs16_n0_c0_x19, vqs16_n0_c0_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n0_c1_x19    = neon::vmul(vqs16_n0_c1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n0_c1_result = neon::vmls(vqs16_n0_c1_x19, vqs16_n0_c1_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n0_c2_x19    = neon::vmul(vqs16_n0_c2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n0_c2_result = neon::vmls(vqs16_n0_c2_x19, vqs16_n0_c2_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n1_c0_x19    = neon::vmul(vqs16_n1_c0_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n1_c0_result = neon::vmls(vqs16_n1_c0_x19, vqs16_n1_c0_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n1_c1_x19    = neon::vmul(vqs16_n1_c1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n1_c1_result = neon::vmls(vqs16_n1_c1_x19, vqs16_n1_c1_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n1_c2_x19    = neon::vmul(vqs16_n1_c2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n1_c2_result = neon::vmls(vqs16_n1_c2_x19, vqs16_n1_c2_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n2_c0_x19    = neon::vmul(vqs16_n2_c0_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n2_c0_result = neon::vmls(vqs16_n2_c0_x19, vqs16_n2_c0_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n2_c1_x19    = neon::vmul(vqs16_n2_c1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n2_c1_result = neon::vmls(vqs16_n2_c1_x19, vqs16_n2_c1_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n2_c2_x19    = neon::vmul(vqs16_n2_c2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n2_c2_result = neon::vmls(vqs16_n2_c2_x19, vqs16_n2_c2_03, static_cast<MI_S16>(3));

            int32x4_t vqs32_c0_result_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_c0_result), neon::vgetlow(vqs16_n1_c0_result));
            int32x4_t vqs32_c0_result_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_c0_result), neon::vgethigh(vqs16_n1_c0_result));
            int32x4_t vqs32_c0_result_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_c0_result), neon::vgetlow(vqs16_n2_c0_result));
            int32x4_t vqs32_c0_result_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_c0_result), neon::vgethigh(vqs16_n2_c0_result));
            int32x4_t vqs32_c1_result_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_c1_result), neon::vgetlow(vqs16_n1_c1_result));
            int32x4_t vqs32_c1_result_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_c1_result), neon::vgethigh(vqs16_n1_c1_result));
            int32x4_t vqs32_c1_result_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_c1_result), neon::vgetlow(vqs16_n2_c1_result));
            int32x4_t vqs32_c1_result_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_c1_result), neon::vgethigh(vqs16_n2_c1_result));
            int32x4_t vqs32_c2_result_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_c2_result), neon::vgetlow(vqs16_n1_c2_result));
            int32x4_t vqs32_c2_result_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_c2_result), neon::vgethigh(vqs16_n1_c2_result));
            int32x4_t vqs32_c2_result_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_c2_result), neon::vgetlow(vqs16_n2_c2_result));
            int32x4_t vqs32_c2_result_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_c2_result), neon::vgethigh(vqs16_n2_c2_result));

            int32x4_t vqs32_c0_x19_lo    = neon::vmul(vqs32_c0_result_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c0_result_lo = neon::vmls(vqs32_c0_x19_lo, vqs32_c0_result_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c0_x19_hi    = neon::vmul(vqs32_c0_result_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c0_result_hi = neon::vmls(vqs32_c0_x19_hi, vqs32_c0_result_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c1_x19_lo    = neon::vmul(vqs32_c1_result_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c1_result_lo = neon::vmls(vqs32_c1_x19_lo, vqs32_c1_result_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c1_x19_hi    = neon::vmul(vqs32_c1_result_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c1_result_hi = neon::vmls(vqs32_c1_x19_hi, vqs32_c1_result_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c2_x19_lo    = neon::vmul(vqs32_c2_result_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c2_result_lo = neon::vmls(vqs32_c2_x19_lo, vqs32_c2_result_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c2_x19_hi    = neon::vmul(vqs32_c2_result_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c2_result_hi = neon::vmls(vqs32_c2_x19_hi, vqs32_c2_result_hi03, static_cast<MI_S32>(3));

            int16x4_t vds16_c0_des_lo = neon::vrshrn_n<10>(vqs32_c0_result_lo);
            int16x4_t vds16_c0_des_hi = neon::vrshrn_n<10>(vqs32_c0_result_hi);
            int16x4_t vds16_c1_des_lo = neon::vrshrn_n<10>(vqs32_c1_result_lo);
            int16x4_t vds16_c1_des_hi = neon::vrshrn_n<10>(vqs32_c1_result_hi);
            int16x4_t vds16_c2_des_lo = neon::vrshrn_n<10>(vqs32_c2_result_lo);
            int16x4_t vds16_c2_des_hi = neon::vrshrn_n<10>(vqs32_c2_result_hi);

            MVType mvd8_result;
            if (std::is_same<Tp, MI_U8>::value)
            {
                int16x8_t vqs16_zero;
                neon::vdup(vqs16_zero, static_cast<MI_S16>(0));
                int16x8_t vqs16_255;
                neon::vdup(vqs16_255, static_cast<MI_S16>(255));

                int16x8_t vqs16_c0_des  = neon::vcombine(vds16_c0_des_lo, vds16_c0_des_hi);
                int16x8_t vqs16_c1_des  = neon::vcombine(vds16_c1_des_lo, vds16_c1_des_hi);
                int16x8_t vqs16_c2_des  = neon::vcombine(vds16_c2_des_lo, vds16_c2_des_hi);
                vqs16_c0_des            = neon::vmax(vqs16_c0_des, vqs16_zero);
                vqs16_c0_des            = neon::vmin(vqs16_c0_des, vqs16_255);
                vqs16_c1_des            = neon::vmax(vqs16_c1_des, vqs16_zero);
                vqs16_c1_des            = neon::vmin(vqs16_c1_des, vqs16_255);
                vqs16_c2_des            = neon::vmax(vqs16_c2_des, vqs16_zero);
                vqs16_c2_des            = neon::vmin(vqs16_c2_des, vqs16_255);
                uint16x8_t vdu16_c0_des = neon::vreinterpret(vqs16_c0_des);
                uint16x8_t vdu16_c1_des = neon::vreinterpret(vqs16_c1_des);
                uint16x8_t vdu16_c2_des = neon::vreinterpret(vqs16_c2_des);
                mvd8_result.val[0]      = neon::vmovn(vdu16_c0_des);
                mvd8_result.val[1]      = neon::vmovn(vdu16_c1_des);
                mvd8_result.val[2]      = neon::vmovn(vdu16_c2_des);
                neon::vstore(dst_row, mvd8_result);
            }
            else
            {
                int16x8_t vqs16_n128;
                neon::vdup(vqs16_n128, static_cast<MI_S16>(-128));
                int16x8_t vqs16_p127;
                neon::vdup(vqs16_p127, static_cast<MI_S16>(127));

                int16x8_t vqs16_c0_des = neon::vcombine(vds16_c0_des_lo, vds16_c0_des_hi);
                int16x8_t vqs16_c1_des = neon::vcombine(vds16_c1_des_lo, vds16_c1_des_hi);
                int16x8_t vqs16_c2_des = neon::vcombine(vds16_c2_des_lo, vds16_c2_des_hi);
                vqs16_c0_des           = neon::vmax(vqs16_c0_des, vqs16_n128);
                vqs16_c0_des           = neon::vmin(vqs16_c0_des, vqs16_p127);
                vqs16_c1_des           = neon::vmax(vqs16_c1_des, vqs16_n128);
                vqs16_c1_des           = neon::vmin(vqs16_c1_des, vqs16_p127);
                vqs16_c2_des           = neon::vmax(vqs16_c2_des, vqs16_n128);
                vqs16_c2_des           = neon::vmin(vqs16_c2_des, vqs16_p127);
                mvd8_result.val[0]     = neon::vmovn(vqs16_c0_des);
                mvd8_result.val[1]     = neon::vmovn(vqs16_c1_des);
                mvd8_result.val[2]     = neon::vmovn(vqs16_c2_des);
                neon::vstore(dst_row, mvd8_result);
            }

            dst_row += 24;
            src_c   += 96;
            src_n0  += 96;
            src_n1  += 96;
            src_n2  += 96;
        }

        for (; x < owidth; x++)
        {
            MI_S32 y00   = (src_c[3] + src_c[6]) * 19 - (src_c[0] + src_c[9]) * 3;
            MI_S32 y10   = (src_n0[3] + src_n0[6]) * 19 - (src_n0[0] + src_n0[9]) * 3;
            MI_S32 y20   = (src_n1[3] + src_n1[6]) * 19 - (src_n1[0] + src_n1[9]) * 3;
            MI_S32 y30   = (src_n2[3] + src_n2[6]) * 19 - (src_n2[0] + src_n2[9]) * 3;
            MI_S32 temp0 = (y10 * 19 - y00 * 3 + y20 * 19 - y30 * 3 + 512) >> 10;
            *dst_row     = SaturateCast<Tp>(temp0);

            MI_S32 y01     = (src_c[4] + src_c[7]) * 19 - (src_c[1] + src_c[10]) * 3;
            MI_S32 y11     = (src_n0[4] + src_n0[7]) * 19 - (src_n0[1] + src_n0[10]) * 3;
            MI_S32 y21     = (src_n1[4] + src_n1[7]) * 19 - (src_n1[1] + src_n1[10]) * 3;
            MI_S32 y31     = (src_n2[4] + src_n2[7]) * 19 - (src_n2[1] + src_n2[10]) * 3;
            MI_S32 temp1   = (y11 * 19 - y01 * 3 + y21 * 19 - y31 * 3 + 512) >> 10;
            *(dst_row + 1) = SaturateCast<Tp>(temp1);

            MI_S32 y02     = (src_c[5] + src_c[8]) * 19 - (src_c[2] + src_c[11]) * 3;
            MI_S32 y12     = (src_n0[5] + src_n0[8]) * 19 - (src_n0[2] + src_n0[11]) * 3;
            MI_S32 y22     = (src_n1[5] + src_n1[8]) * 19 - (src_n1[2] + src_n1[11]) * 3;
            MI_S32 y32     = (src_n2[5] + src_n2[8]) * 19 - (src_n2[2] + src_n2[11]) * 3;
            MI_S32 temp2   = (y12 * 19 - y02 * 3 + y22 * 19 - y32 * 3 + 512) >> 10;
            *(dst_row + 2) = SaturateCast<Tp>(temp2);

            dst_row += 3;
            src_c   += 12;
            src_n0  += 12;
            src_n1  += 12;
            src_n2  += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeCuC3DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    using MVType        = typename neon::MDVector<Tp, 3>::MVType;
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align4 = owidth & (-4);

    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        // hresize two row
        MI_S32 sy        = dy << 2;
        const Tp *src_c  = src.Ptr<Tp>(sy);
        const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        Tp *dst_row      = dst.Ptr<Tp>(dy);

        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            auto v3q16_cx0  = neon::vload3q(src_c);
            auto v3q16_cx1  = neon::vload3q(src_c + 24);
            auto v3q16_n0x0 = neon::vload3q(src_n0);
            auto v3q16_n0x1 = neon::vload3q(src_n0 + 24);
            auto v3q16_n1x0 = neon::vload3q(src_n1);
            auto v3q16_n1x1 = neon::vload3q(src_n1 + 24);
            auto v3q16_n2x0 = neon::vload3q(src_n2);
            auto v3q16_n2x1 = neon::vload3q(src_n2 + 24);

            auto v2q16_c_0  = neon::vuzp(v3q16_cx0.val[0], v3q16_cx1.val[0]);
            auto v2q16_c_1  = neon::vuzp(v2q16_c_0.val[0], v2q16_c_0.val[1]);
            auto v2q16_c_2  = neon::vuzp(v3q16_cx0.val[1], v3q16_cx1.val[1]);
            auto v2q16_c_3  = neon::vuzp(v2q16_c_2.val[0], v2q16_c_2.val[1]);
            auto v2q16_c_4  = neon::vuzp(v3q16_cx0.val[2], v3q16_cx1.val[2]);
            auto v2q16_c_5  = neon::vuzp(v2q16_c_4.val[0], v2q16_c_4.val[1]);
            auto v2q16_n0_0 = neon::vuzp(v3q16_n0x0.val[0], v3q16_n0x1.val[0]);
            auto v2q16_n0_1 = neon::vuzp(v2q16_n0_0.val[0], v2q16_n0_0.val[1]);
            auto v2q16_n0_2 = neon::vuzp(v3q16_n0x0.val[1], v3q16_n0x1.val[1]);
            auto v2q16_n0_3 = neon::vuzp(v2q16_n0_2.val[0], v2q16_n0_2.val[1]);
            auto v2q16_n0_4 = neon::vuzp(v3q16_n0x0.val[2], v3q16_n0x1.val[2]);
            auto v2q16_n0_5 = neon::vuzp(v2q16_n0_4.val[0], v2q16_n0_4.val[1]);
            auto v2q16_n1_0 = neon::vuzp(v3q16_n1x0.val[0], v3q16_n1x1.val[0]);
            auto v2q16_n1_1 = neon::vuzp(v2q16_n1_0.val[0], v2q16_n1_0.val[1]);
            auto v2q16_n1_2 = neon::vuzp(v3q16_n1x0.val[1], v3q16_n1x1.val[1]);
            auto v2q16_n1_3 = neon::vuzp(v2q16_n1_2.val[0], v2q16_n1_2.val[1]);
            auto v2q16_n1_4 = neon::vuzp(v3q16_n1x0.val[2], v3q16_n1x1.val[2]);
            auto v2q16_n1_5 = neon::vuzp(v2q16_n1_4.val[0], v2q16_n1_4.val[1]);
            auto v2q16_n2_0 = neon::vuzp(v3q16_n2x0.val[0], v3q16_n2x1.val[0]);
            auto v2q16_n2_1 = neon::vuzp(v2q16_n2_0.val[0], v2q16_n2_0.val[1]);
            auto v2q16_n2_2 = neon::vuzp(v3q16_n2x0.val[1], v3q16_n2x1.val[1]);
            auto v2q16_n2_3 = neon::vuzp(v2q16_n2_2.val[0], v2q16_n2_2.val[1]);
            auto v2q16_n2_4 = neon::vuzp(v3q16_n2x0.val[2], v3q16_n2x1.val[2]);
            auto v2q16_n2_5 = neon::vuzp(v2q16_n2_4.val[0], v2q16_n2_4.val[1]);

            int32x4_t vqs32_c_c0_12  = neon::vaddl(neon::vgethigh(v2q16_c_1.val[0]), neon::vgetlow(v2q16_c_1.val[1]));
            int32x4_t vqs32_c_c0_03  = neon::vaddl(neon::vgetlow(v2q16_c_1.val[0]), neon::vgethigh(v2q16_c_1.val[1]));
            int32x4_t vqs32_c_c1_12  = neon::vaddl(neon::vgethigh(v2q16_c_3.val[0]), neon::vgetlow(v2q16_c_3.val[1]));
            int32x4_t vqs32_c_c1_03  = neon::vaddl(neon::vgetlow(v2q16_c_3.val[0]), neon::vgethigh(v2q16_c_3.val[1]));
            int32x4_t vqs32_c_c2_12  = neon::vaddl(neon::vgethigh(v2q16_c_5.val[0]), neon::vgetlow(v2q16_c_5.val[1]));
            int32x4_t vqs32_c_c2_03  = neon::vaddl(neon::vgetlow(v2q16_c_5.val[0]), neon::vgethigh(v2q16_c_5.val[1]));
            int32x4_t vqs32_n0_c0_12 = neon::vaddl(neon::vgethigh(v2q16_n0_1.val[0]), neon::vgetlow(v2q16_n0_1.val[1]));
            int32x4_t vqs32_n0_c0_03 = neon::vaddl(neon::vgetlow(v2q16_n0_1.val[0]), neon::vgethigh(v2q16_n0_1.val[1]));
            int32x4_t vqs32_n0_c1_12 = neon::vaddl(neon::vgethigh(v2q16_n0_3.val[0]), neon::vgetlow(v2q16_n0_3.val[1]));
            int32x4_t vqs32_n0_c1_03 = neon::vaddl(neon::vgetlow(v2q16_n0_3.val[0]), neon::vgethigh(v2q16_n0_3.val[1]));
            int32x4_t vqs32_n0_c2_12 = neon::vaddl(neon::vgethigh(v2q16_n0_5.val[0]), neon::vgetlow(v2q16_n0_5.val[1]));
            int32x4_t vqs32_n0_c2_03 = neon::vaddl(neon::vgetlow(v2q16_n0_5.val[0]), neon::vgethigh(v2q16_n0_5.val[1]));
            int32x4_t vqs32_n1_c0_12 = neon::vaddl(neon::vgethigh(v2q16_n1_1.val[0]), neon::vgetlow(v2q16_n1_1.val[1]));
            int32x4_t vqs32_n1_c0_03 = neon::vaddl(neon::vgetlow(v2q16_n1_1.val[0]), neon::vgethigh(v2q16_n1_1.val[1]));
            int32x4_t vqs32_n1_c1_12 = neon::vaddl(neon::vgethigh(v2q16_n1_3.val[0]), neon::vgetlow(v2q16_n1_3.val[1]));
            int32x4_t vqs32_n1_c1_03 = neon::vaddl(neon::vgetlow(v2q16_n1_3.val[0]), neon::vgethigh(v2q16_n1_3.val[1]));
            int32x4_t vqs32_n1_c2_12 = neon::vaddl(neon::vgethigh(v2q16_n1_5.val[0]), neon::vgetlow(v2q16_n1_5.val[1]));
            int32x4_t vqs32_n1_c2_03 = neon::vaddl(neon::vgetlow(v2q16_n1_5.val[0]), neon::vgethigh(v2q16_n1_5.val[1]));
            int32x4_t vqs32_n2_c0_12 = neon::vaddl(neon::vgethigh(v2q16_n2_1.val[0]), neon::vgetlow(v2q16_n2_1.val[1]));
            int32x4_t vqs32_n2_c0_03 = neon::vaddl(neon::vgetlow(v2q16_n2_1.val[0]), neon::vgethigh(v2q16_n2_1.val[1]));
            int32x4_t vqs32_n2_c1_12 = neon::vaddl(neon::vgethigh(v2q16_n2_3.val[0]), neon::vgetlow(v2q16_n2_3.val[1]));
            int32x4_t vqs32_n2_c1_03 = neon::vaddl(neon::vgetlow(v2q16_n2_3.val[0]), neon::vgethigh(v2q16_n2_3.val[1]));
            int32x4_t vqs32_n2_c2_12 = neon::vaddl(neon::vgethigh(v2q16_n2_5.val[0]), neon::vgetlow(v2q16_n2_5.val[1]));
            int32x4_t vqs32_n2_c2_03 = neon::vaddl(neon::vgetlow(v2q16_n2_5.val[0]), neon::vgethigh(v2q16_n2_5.val[1]));

            int32x4_t vqs32_c_c0_x19     = neon::vmul(vqs32_c_c0_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c_c0_result  = neon::vmls(vqs32_c_c0_x19, vqs32_c_c0_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c_c1_x19     = neon::vmul(vqs32_c_c1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c_c1_result  = neon::vmls(vqs32_c_c1_x19, vqs32_c_c1_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c_c2_x19     = neon::vmul(vqs32_c_c2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c_c2_result  = neon::vmls(vqs32_c_c2_x19, vqs32_c_c2_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n0_c0_x19    = neon::vmul(vqs32_n0_c0_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n0_c0_result = neon::vmls(vqs32_n0_c0_x19, vqs32_n0_c0_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n0_c1_x19    = neon::vmul(vqs32_n0_c1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n0_c1_result = neon::vmls(vqs32_n0_c1_x19, vqs32_n0_c1_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n0_c2_x19    = neon::vmul(vqs32_n0_c2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n0_c2_result = neon::vmls(vqs32_n0_c2_x19, vqs32_n0_c2_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n1_c0_x19    = neon::vmul(vqs32_n1_c0_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n1_c0_result = neon::vmls(vqs32_n1_c0_x19, vqs32_n1_c0_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n1_c1_x19    = neon::vmul(vqs32_n1_c1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n1_c1_result = neon::vmls(vqs32_n1_c1_x19, vqs32_n1_c1_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n1_c2_x19    = neon::vmul(vqs32_n1_c2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n1_c2_result = neon::vmls(vqs32_n1_c2_x19, vqs32_n1_c2_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n2_c0_x19    = neon::vmul(vqs32_n2_c0_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n2_c0_result = neon::vmls(vqs32_n2_c0_x19, vqs32_n2_c0_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n2_c1_x19    = neon::vmul(vqs32_n2_c1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n2_c1_result = neon::vmls(vqs32_n2_c1_x19, vqs32_n2_c1_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n2_c2_x19    = neon::vmul(vqs32_n2_c2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n2_c2_result = neon::vmls(vqs32_n2_c2_x19, vqs32_n2_c2_03, static_cast<MI_S32>(3));

            int32x4_t vqs32_c0_result_12 = neon::vadd(vqs32_n0_c0_result, vqs32_n1_c0_result);
            int32x4_t vqs32_c0_result_03 = neon::vadd(vqs32_c_c0_result, vqs32_n2_c0_result);
            int32x4_t vqs32_c1_result_12 = neon::vadd(vqs32_n0_c1_result, vqs32_n1_c1_result);
            int32x4_t vqs32_c1_result_03 = neon::vadd(vqs32_c_c1_result, vqs32_n2_c1_result);
            int32x4_t vqs32_c2_result_12 = neon::vadd(vqs32_n0_c2_result, vqs32_n1_c2_result);
            int32x4_t vqs32_c2_result_03 = neon::vadd(vqs32_c_c2_result, vqs32_n2_c2_result);

            int32x4_t vqs32_c0_x19    = neon::vmul(vqs32_c0_result_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c0_result = neon::vmls(vqs32_c0_x19, vqs32_c0_result_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c1_x19    = neon::vmul(vqs32_c1_result_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c1_result = neon::vmls(vqs32_c1_x19, vqs32_c1_result_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c2_x19    = neon::vmul(vqs32_c2_result_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c2_result = neon::vmls(vqs32_c2_x19, vqs32_c2_result_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_c0_des    = neon::vrshr_n<10>(vqs32_c0_result);
            int32x4_t vqs32_c1_des    = neon::vrshr_n<10>(vqs32_c1_result);
            int32x4_t vqs32_c2_des    = neon::vrshr_n<10>(vqs32_c2_result);

            MVType mvd16_result;
            if (std::is_same<Tp, MI_U16>::value)
            {
                int32x4_t vqs32_zero;
                neon::vdup(vqs32_zero, static_cast<MI_S32>(0));
                int32x4_t vqs32_65535;
                neon::vdup(vqs32_65535, static_cast<MI_S32>(65535));

                vqs32_c0_des = neon::vmax(vqs32_c0_des, vqs32_zero);
                vqs32_c0_des = neon::vmin(vqs32_c0_des, vqs32_65535);
                vqs32_c1_des = neon::vmax(vqs32_c1_des, vqs32_zero);
                vqs32_c1_des = neon::vmin(vqs32_c1_des, vqs32_65535);
                vqs32_c2_des = neon::vmax(vqs32_c2_des, vqs32_zero);
                vqs32_c2_des = neon::vmin(vqs32_c2_des, vqs32_65535);

                int16x4_t vds16_c0_des = neon::vmovn(vqs32_c0_des);
                int16x4_t vds16_c1_des = neon::vmovn(vqs32_c1_des);
                int16x4_t vds16_c2_des = neon::vmovn(vqs32_c2_des);
                mvd16_result.val[0]    = neon::vreinterpret(vds16_c0_des);
                mvd16_result.val[1]    = neon::vreinterpret(vds16_c1_des);
                mvd16_result.val[2]    = neon::vreinterpret(vds16_c2_des);
                neon::vstore(dst_row, mvd16_result);
            }
            else
            {
                int32x4_t vqs32_n32768;
                neon::vdup(vqs32_n32768, static_cast<MI_S32>(-32768));
                int32x4_t vqs32_p32767;
                neon::vdup(vqs32_p32767, static_cast<MI_S32>(32767));

                vqs32_c0_des = neon::vmax(vqs32_c0_des, vqs32_n32768);
                vqs32_c0_des = neon::vmin(vqs32_c0_des, vqs32_p32767);
                vqs32_c1_des = neon::vmax(vqs32_c1_des, vqs32_n32768);
                vqs32_c1_des = neon::vmin(vqs32_c1_des, vqs32_p32767);
                vqs32_c2_des = neon::vmax(vqs32_c2_des, vqs32_n32768);
                vqs32_c2_des = neon::vmin(vqs32_c2_des, vqs32_p32767);

                mvd16_result.val[0] = neon::vmovn(vqs32_c0_des);
                mvd16_result.val[1] = neon::vmovn(vqs32_c1_des);
                mvd16_result.val[2] = neon::vmovn(vqs32_c2_des);
                neon::vstore(dst_row, mvd16_result);
            }

            dst_row += 12;
            src_c   += 48;
            src_n0  += 48;
            src_n1  += 48;
            src_n2  += 48;
        }

        for (; x < owidth; x++)
        {
            MI_S32 y00   = (src_c[3] + src_c[6]) * 19 - (src_c[0] + src_c[9]) * 3;
            MI_S32 y10   = (src_n0[3] + src_n0[6]) * 19 - (src_n0[0] + src_n0[9]) * 3;
            MI_S32 y20   = (src_n1[3] + src_n1[6]) * 19 - (src_n1[0] + src_n1[9]) * 3;
            MI_S32 y30   = (src_n2[3] + src_n2[6]) * 19 - (src_n2[0] + src_n2[9]) * 3;
            MI_S32 temp0 = (y10 * 19 - y00 * 3 + y20 * 19 - y30 * 3 + 512) >> 10;
            *dst_row     = SaturateCast<Tp>(temp0);

            MI_S32 y01     = (src_c[4] + src_c[7]) * 19 - (src_c[1] + src_c[10]) * 3;
            MI_S32 y11     = (src_n0[4] + src_n0[7]) * 19 - (src_n0[1] + src_n0[10]) * 3;
            MI_S32 y21     = (src_n1[4] + src_n1[7]) * 19 - (src_n1[1] + src_n1[10]) * 3;
            MI_S32 y31     = (src_n2[4] + src_n2[7]) * 19 - (src_n2[1] + src_n2[10]) * 3;
            MI_S32 temp1   = (y11 * 19 - y01 * 3 + y21 * 19 - y31 * 3 + 512) >> 10;
            *(dst_row + 1) = SaturateCast<Tp>(temp1);

            MI_S32 y02     = (src_c[5] + src_c[8]) * 19 - (src_c[2] + src_c[11]) * 3;
            MI_S32 y12     = (src_n0[5] + src_n0[8]) * 19 - (src_n0[2] + src_n0[11]) * 3;
            MI_S32 y22     = (src_n1[5] + src_n1[8]) * 19 - (src_n1[2] + src_n1[11]) * 3;
            MI_S32 y32     = (src_n2[5] + src_n2[8]) * 19 - (src_n2[2] + src_n2[11]) * 3;
            MI_S32 temp2   = (y12 * 19 - y02 * 3 + y22 * 19 - y32 * 3 + 512) >> 10;
            *(dst_row + 2) = SaturateCast<Tp>(temp2);

            dst_row += 3;
            src_c   += 12;
            src_n0  += 12;
            src_n1  += 12;
            src_n2  += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeCuC3DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align4 = owidth & (-4);

    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        // hresize two row
        MI_S32 sy        = dy << 2;
        const Tp *src_c  = src.Ptr<Tp>(sy);
        const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        Tp *dst_row      = dst.Ptr<Tp>(dy);

        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float16x8x3_t v3qf16_cx0  = neon::vload3q(src_c);
            float16x8x3_t v3qf16_cx1  = neon::vload3q(src_c + 24);
            float16x8x3_t v3qf16_n0x0 = neon::vload3q(src_n0);
            float16x8x3_t v3qf16_n0x1 = neon::vload3q(src_n0 + 24);
            float16x8x3_t v3qf16_n1x0 = neon::vload3q(src_n1);
            float16x8x3_t v3qf16_n1x1 = neon::vload3q(src_n1 + 24);
            float16x8x3_t v3qf16_n2x0 = neon::vload3q(src_n2);
            float16x8x3_t v3qf16_n2x1 = neon::vload3q(src_n2 + 24);

            float16x8x2_t v2qf16_c_0  = neon::vuzp(v3qf16_cx0.val[0],  v3qf16_cx1.val[0]);
            float16x8x2_t v2qf16_c_1  = neon::vuzp(v2qf16_c_0.val[0],  v2qf16_c_0.val[1]);
            float16x8x2_t v2qf16_c_2  = neon::vuzp(v3qf16_cx0.val[1],  v3qf16_cx1.val[1]);
            float16x8x2_t v2qf16_c_3  = neon::vuzp(v2qf16_c_2.val[0],  v2qf16_c_2.val[1]);
            float16x8x2_t v2qf16_c_4  = neon::vuzp(v3qf16_cx0.val[2],  v3qf16_cx1.val[2]);
            float16x8x2_t v2qf16_c_5  = neon::vuzp(v2qf16_c_4.val[0],  v2qf16_c_4.val[1]);
            float16x8x2_t v2qf16_n0_0 = neon::vuzp(v3qf16_n0x0.val[0], v3qf16_n0x1.val[0]);
            float16x8x2_t v2qf16_n0_1 = neon::vuzp(v2qf16_n0_0.val[0], v2qf16_n0_0.val[1]);
            float16x8x2_t v2qf16_n0_2 = neon::vuzp(v3qf16_n0x0.val[1], v3qf16_n0x1.val[1]);
            float16x8x2_t v2qf16_n0_3 = neon::vuzp(v2qf16_n0_2.val[0], v2qf16_n0_2.val[1]);
            float16x8x2_t v2qf16_n0_4 = neon::vuzp(v3qf16_n0x0.val[2], v3qf16_n0x1.val[2]);
            float16x8x2_t v2qf16_n0_5 = neon::vuzp(v2qf16_n0_4.val[0], v2qf16_n0_4.val[1]);
            float16x8x2_t v2qf16_n1_0 = neon::vuzp(v3qf16_n1x0.val[0], v3qf16_n1x1.val[0]);
            float16x8x2_t v2qf16_n1_1 = neon::vuzp(v2qf16_n1_0.val[0], v2qf16_n1_0.val[1]);
            float16x8x2_t v2qf16_n1_2 = neon::vuzp(v3qf16_n1x0.val[1], v3qf16_n1x1.val[1]);
            float16x8x2_t v2qf16_n1_3 = neon::vuzp(v2qf16_n1_2.val[0], v2qf16_n1_2.val[1]);
            float16x8x2_t v2qf16_n1_4 = neon::vuzp(v3qf16_n1x0.val[2], v3qf16_n1x1.val[2]);
            float16x8x2_t v2qf16_n1_5 = neon::vuzp(v2qf16_n1_4.val[0], v2qf16_n1_4.val[1]);
            float16x8x2_t v2qf16_n2_0 = neon::vuzp(v3qf16_n2x0.val[0], v3qf16_n2x1.val[0]);
            float16x8x2_t v2qf16_n2_1 = neon::vuzp(v2qf16_n2_0.val[0], v2qf16_n2_0.val[1]);
            float16x8x2_t v2qf16_n2_2 = neon::vuzp(v3qf16_n2x0.val[1], v3qf16_n2x1.val[1]);
            float16x8x2_t v2qf16_n2_3 = neon::vuzp(v2qf16_n2_2.val[0], v2qf16_n2_2.val[1]);
            float16x8x2_t v2qf16_n2_4 = neon::vuzp(v3qf16_n2x0.val[2], v3qf16_n2x1.val[2]);
            float16x8x2_t v2qf16_n2_5 = neon::vuzp(v2qf16_n2_4.val[0], v2qf16_n2_4.val[1]);

            float32x4_t vqf32_c_c0_12  = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_1.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_1.val[1])));
            float32x4_t vqf32_c_c0_03  = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_1.val[0])),   neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_1.val[1])));
            float32x4_t vqf32_c_c1_12  = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_3.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_3.val[1])));
            float32x4_t vqf32_c_c1_03  = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_3.val[0])),   neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_3.val[1])));
            float32x4_t vqf32_c_c2_12  = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_5.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_5.val[1])));
            float32x4_t vqf32_c_c2_03  = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_c_5.val[0])),   neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_c_5.val[1])));
            float32x4_t vqf32_n0_c0_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_1.val[1])));
            float32x4_t vqf32_n0_c0_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_1.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_1.val[1])));
            float32x4_t vqf32_n0_c1_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_3.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_3.val[1])));
            float32x4_t vqf32_n0_c1_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_3.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_3.val[1])));
            float32x4_t vqf32_n0_c2_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_5.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_5.val[1])));
            float32x4_t vqf32_n0_c2_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0_5.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n0_5.val[1])));
            float32x4_t vqf32_n1_c0_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_1.val[1])));
            float32x4_t vqf32_n1_c0_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_1.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_1.val[1])));
            float32x4_t vqf32_n1_c1_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_3.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_3.val[1])));
            float32x4_t vqf32_n1_c1_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_3.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_3.val[1])));
            float32x4_t vqf32_n1_c2_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_5.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_5.val[1])));
            float32x4_t vqf32_n1_c2_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1_5.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n1_5.val[1])));
            float32x4_t vqf32_n2_c0_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_1.val[1])));
            float32x4_t vqf32_n2_c0_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_1.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_1.val[1])));
            float32x4_t vqf32_n2_c1_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_3.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_3.val[1])));
            float32x4_t vqf32_n2_c1_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_3.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_3.val[1])));
            float32x4_t vqf32_n2_c2_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_5.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_5.val[1])));
            float32x4_t vqf32_n2_c2_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2_5.val[0])),  neon::vcvt<MI_F32>(neon::vgethigh(v2qf16_n2_5.val[1])));

            float32x4_t vqf32_c_c0_x19     = neon::vmul(vqf32_c_c0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_c_c0_x3      = neon::vmul(vqf32_c_c0_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_c_c0_result  = neon::vadd(vqf32_c_c0_x19, vqf32_c_c0_x3);
            float32x4_t vqf32_c_c1_x19     = neon::vmul(vqf32_c_c1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_c_c1_x3      = neon::vmul(vqf32_c_c1_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_c_c1_result  = neon::vadd(vqf32_c_c1_x19, vqf32_c_c1_x3);
            float32x4_t vqf32_c_c2_x19     = neon::vmul(vqf32_c_c2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_c_c2_x3      = neon::vmul(vqf32_c_c2_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_c_c2_result  = neon::vadd(vqf32_c_c2_x19, vqf32_c_c2_x3);
            float32x4_t vqf32_n0_c0_x19    = neon::vmul(vqf32_n0_c0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n0_c0_x3     = neon::vmul(vqf32_n0_c0_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n0_c0_result = neon::vadd(vqf32_n0_c0_x19, vqf32_n0_c0_x3);
            float32x4_t vqf32_n0_c1_x19    = neon::vmul(vqf32_n0_c1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n0_c1_x3     = neon::vmul(vqf32_n0_c1_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n0_c1_result = neon::vadd(vqf32_n0_c1_x19, vqf32_n0_c1_x3);
            float32x4_t vqf32_n0_c2_x19    = neon::vmul(vqf32_n0_c2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n0_c2_x3     = neon::vmul(vqf32_n0_c2_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n0_c2_result = neon::vadd(vqf32_n0_c2_x19, vqf32_n0_c2_x3);
            float32x4_t vqf32_n1_c0_x19    = neon::vmul(vqf32_n1_c0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n1_c0_x3     = neon::vmul(vqf32_n1_c0_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n1_c0_result = neon::vadd(vqf32_n1_c0_x19, vqf32_n1_c0_x3);
            float32x4_t vqf32_n1_c1_x19    = neon::vmul(vqf32_n1_c1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n1_c1_x3     = neon::vmul(vqf32_n1_c1_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n1_c1_result = neon::vadd(vqf32_n1_c1_x19, vqf32_n1_c1_x3);
            float32x4_t vqf32_n1_c2_x19    = neon::vmul(vqf32_n1_c2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n1_c2_x3     = neon::vmul(vqf32_n1_c2_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n1_c2_result = neon::vadd(vqf32_n1_c2_x19, vqf32_n1_c2_x3);
            float32x4_t vqf32_n2_c0_x19    = neon::vmul(vqf32_n2_c0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n2_c0_x3     = neon::vmul(vqf32_n2_c0_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n2_c0_result = neon::vadd(vqf32_n2_c0_x19, vqf32_n2_c0_x3);
            float32x4_t vqf32_n2_c1_x19    = neon::vmul(vqf32_n2_c1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n2_c1_x3     = neon::vmul(vqf32_n2_c1_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n2_c1_result = neon::vadd(vqf32_n2_c1_x19, vqf32_n2_c1_x3);
            float32x4_t vqf32_n2_c2_x19    = neon::vmul(vqf32_n2_c2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n2_c2_x3     = neon::vmul(vqf32_n2_c2_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n2_c2_result = neon::vadd(vqf32_n2_c2_x19, vqf32_n2_c2_x3);

            float32x4_t vqf32_c0_result_12 = neon::vadd(vqf32_n0_c0_result, vqf32_n1_c0_result);
            float32x4_t vqf32_c0_result_03 = neon::vadd(vqf32_c_c0_result, vqf32_n2_c0_result);
            float32x4_t vqf32_c1_result_12 = neon::vadd(vqf32_n0_c1_result, vqf32_n1_c1_result);
            float32x4_t vqf32_c1_result_03 = neon::vadd(vqf32_c_c1_result, vqf32_n2_c1_result);
            float32x4_t vqf32_c2_result_12 = neon::vadd(vqf32_n0_c2_result, vqf32_n1_c2_result);
            float32x4_t vqf32_c2_result_03 = neon::vadd(vqf32_c_c2_result, vqf32_n2_c2_result);

            float16x4x3_t v3df16_result;
            float32x4_t vqf32_c0_x19 = neon::vmul(vqf32_c0_result_12, static_cast<MI_F32>(0.59375));
            v3df16_result.val[0]     = neon::vcvt<MI_F16>(neon::vadd(vqf32_c0_x19, neon::vmul(vqf32_c0_result_03, static_cast<MI_F32>(-0.09375))));
            float32x4_t vqf32_c1_x19 = neon::vmul(vqf32_c1_result_12, static_cast<MI_F32>(0.59375));
            v3df16_result.val[1]     = neon::vcvt<MI_F16>(neon::vadd(vqf32_c1_x19, neon::vmul(vqf32_c1_result_03, static_cast<MI_F32>(-0.09375))));
            float32x4_t vqf32_c2_x19 = neon::vmul(vqf32_c2_result_12, static_cast<MI_F32>(0.59375));
            v3df16_result.val[2]     = neon::vcvt<MI_F16>(neon::vadd(vqf32_c2_x19, neon::vmul(vqf32_c2_result_03, static_cast<MI_F32>(-0.09375))));
            neon::vstore(dst_row, v3df16_result);

            dst_row += 12;
            src_c   += 48;
            src_n0  += 48;
            src_n1  += 48;
            src_n2  += 48;
        }

        for (; x < owidth; x++)
        {
            MI_F32 y00 = (src_c[3] + src_c[6]) * 0.59375f - (src_c[0] + src_c[9]) * 0.09375f;
            MI_F32 y10 = (src_n0[3] + src_n0[6]) * 0.59375f - (src_n0[0] + src_n0[9]) * 0.09375f;
            MI_F32 y20 = (src_n1[3] + src_n1[6]) * 0.59375f - (src_n1[0] + src_n1[9]) * 0.09375f;
            MI_F32 y30 = (src_n2[3] + src_n2[6]) * 0.59375f - (src_n2[0] + src_n2[9]) * 0.09375f;
            *dst_row   = SaturateCast<Tp>((y10 + y20) * 0.59375f - (y00 + y30) * 0.09375f);

            MI_F32 y01     = (src_c[4] + src_c[7]) * 0.59375f - (src_c[1] + src_c[10]) * 0.09375f;
            MI_F32 y11     = (src_n0[4] + src_n0[7]) * 0.59375f - (src_n0[1] + src_n0[10]) * 0.09375f;
            MI_F32 y21     = (src_n1[4] + src_n1[7]) * 0.59375f - (src_n1[1] + src_n1[10]) * 0.09375f;
            MI_F32 y31     = (src_n2[4] + src_n2[7]) * 0.59375f - (src_n2[1] + src_n2[10]) * 0.09375f;
            *(dst_row + 1) = SaturateCast<Tp>((y11 + y21) * 0.59375f - (y01 + y31) * 0.09375f);

            MI_F32 y02     = (src_c[5] + src_c[8]) * 0.59375f - (src_c[2] + src_c[11]) * 0.09375f;
            MI_F32 y12     = (src_n0[5] + src_n0[8]) * 0.59375f - (src_n0[2] + src_n0[11]) * 0.09375f;
            MI_F32 y22     = (src_n1[5] + src_n1[8]) * 0.59375f - (src_n1[2] + src_n1[11]) * 0.09375f;
            MI_F32 y32     = (src_n2[5] + src_n2[8]) * 0.59375f - (src_n2[2] + src_n2[11]) * 0.09375f;
            *(dst_row + 2) = SaturateCast<Tp>((y12 + y22) * 0.59375f - (y02 + y32) * 0.09375f);

            dst_row += 3;
            src_c   += 12;
            src_n0  += 12;
            src_n1  += 12;
            src_n2  += 12;
        }
    }

    return Status::OK;
}
#endif

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeCuC3DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align2 = owidth & (-2);

    for (MI_S32 dy = start_row; dy < end_row; dy++)
    {
        // hresize two row
        MI_S32 sy        = dy << 2;
        const Tp *src_c  = src.Ptr<Tp>(sy);
        const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        Tp *dst_row      = dst.Ptr<Tp>(dy);

        MI_S32 x = 0;
        for (; x < width_align2; x += 2)
        {
            float32x4x3_t v3qf32_cx0  = neon::vload3q(src_c);
            float32x4x3_t v3qf32_cx1  = neon::vload3q(src_c + 12);
            float32x4x3_t v3qf32_n0x0 = neon::vload3q(src_n0);
            float32x4x3_t v3qf32_n0x1 = neon::vload3q(src_n0 + 12);
            float32x4x3_t v3qf32_n1x0 = neon::vload3q(src_n1);
            float32x4x3_t v3qf32_n1x1 = neon::vload3q(src_n1 + 12);
            float32x4x3_t v3qf32_n2x0 = neon::vload3q(src_n2);
            float32x4x3_t v3qf32_n2x1 = neon::vload3q(src_n2 + 12);

            float32x4x2_t v2qf32_c_0  = neon::vuzp(v3qf32_cx0.val[0], v3qf32_cx1.val[0]);
            float32x4x2_t v2qf32_c_1  = neon::vuzp(v2qf32_c_0.val[0], v2qf32_c_0.val[1]);
            float32x4x2_t v2qf32_c_2  = neon::vuzp(v3qf32_cx0.val[1], v3qf32_cx1.val[1]);
            float32x4x2_t v2qf32_c_3  = neon::vuzp(v2qf32_c_2.val[0], v2qf32_c_2.val[1]);
            float32x4x2_t v2qf32_c_4  = neon::vuzp(v3qf32_cx0.val[2], v3qf32_cx1.val[2]);
            float32x4x2_t v2qf32_c_5  = neon::vuzp(v2qf32_c_4.val[0], v2qf32_c_4.val[1]);
            float32x4x2_t v2qf32_n0_0 = neon::vuzp(v3qf32_n0x0.val[0], v3qf32_n0x1.val[0]);
            float32x4x2_t v2qf32_n0_1 = neon::vuzp(v2qf32_n0_0.val[0], v2qf32_n0_0.val[1]);
            float32x4x2_t v2qf32_n0_2 = neon::vuzp(v3qf32_n0x0.val[1], v3qf32_n0x1.val[1]);
            float32x4x2_t v2qf32_n0_3 = neon::vuzp(v2qf32_n0_2.val[0], v2qf32_n0_2.val[1]);
            float32x4x2_t v2qf32_n0_4 = neon::vuzp(v3qf32_n0x0.val[2], v3qf32_n0x1.val[2]);
            float32x4x2_t v2qf32_n0_5 = neon::vuzp(v2qf32_n0_4.val[0], v2qf32_n0_4.val[1]);
            float32x4x2_t v2qf32_n1_0 = neon::vuzp(v3qf32_n1x0.val[0], v3qf32_n1x1.val[0]);
            float32x4x2_t v2qf32_n1_1 = neon::vuzp(v2qf32_n1_0.val[0], v2qf32_n1_0.val[1]);
            float32x4x2_t v2qf32_n1_2 = neon::vuzp(v3qf32_n1x0.val[1], v3qf32_n1x1.val[1]);
            float32x4x2_t v2qf32_n1_3 = neon::vuzp(v2qf32_n1_2.val[0], v2qf32_n1_2.val[1]);
            float32x4x2_t v2qf32_n1_4 = neon::vuzp(v3qf32_n1x0.val[2], v3qf32_n1x1.val[2]);
            float32x4x2_t v2qf32_n1_5 = neon::vuzp(v2qf32_n1_4.val[0], v2qf32_n1_4.val[1]);
            float32x4x2_t v2qf32_n2_0 = neon::vuzp(v3qf32_n2x0.val[0], v3qf32_n2x1.val[0]);
            float32x4x2_t v2qf32_n2_1 = neon::vuzp(v2qf32_n2_0.val[0], v2qf32_n2_0.val[1]);
            float32x4x2_t v2qf32_n2_2 = neon::vuzp(v3qf32_n2x0.val[1], v3qf32_n2x1.val[1]);
            float32x4x2_t v2qf32_n2_3 = neon::vuzp(v2qf32_n2_2.val[0], v2qf32_n2_2.val[1]);
            float32x4x2_t v2qf32_n2_4 = neon::vuzp(v3qf32_n2x0.val[2], v3qf32_n2x1.val[2]);
            float32x4x2_t v2qf32_n2_5 = neon::vuzp(v2qf32_n2_4.val[0], v2qf32_n2_4.val[1]);

            float32x2_t vdf32_c_c0_12  = neon::vadd(neon::vgethigh(v2qf32_c_1.val[0]), neon::vgetlow(v2qf32_c_1.val[1]));
            float32x2_t vdf32_c_c0_03  = neon::vadd(neon::vgetlow(v2qf32_c_1.val[0]), neon::vgethigh(v2qf32_c_1.val[1]));
            float32x2_t vdf32_c_c1_12  = neon::vadd(neon::vgethigh(v2qf32_c_3.val[0]), neon::vgetlow(v2qf32_c_3.val[1]));
            float32x2_t vdf32_c_c1_03  = neon::vadd(neon::vgetlow(v2qf32_c_3.val[0]), neon::vgethigh(v2qf32_c_3.val[1]));
            float32x2_t vdf32_c_c2_12  = neon::vadd(neon::vgethigh(v2qf32_c_5.val[0]), neon::vgetlow(v2qf32_c_5.val[1]));
            float32x2_t vdf32_c_c2_03  = neon::vadd(neon::vgetlow(v2qf32_c_5.val[0]), neon::vgethigh(v2qf32_c_5.val[1]));
            float32x2_t vdf32_n0_c0_12 = neon::vadd(neon::vgethigh(v2qf32_n0_1.val[0]), neon::vgetlow(v2qf32_n0_1.val[1]));
            float32x2_t vdf32_n0_c0_03 = neon::vadd(neon::vgetlow(v2qf32_n0_1.val[0]), neon::vgethigh(v2qf32_n0_1.val[1]));
            float32x2_t vdf32_n0_c1_12 = neon::vadd(neon::vgethigh(v2qf32_n0_3.val[0]), neon::vgetlow(v2qf32_n0_3.val[1]));
            float32x2_t vdf32_n0_c1_03 = neon::vadd(neon::vgetlow(v2qf32_n0_3.val[0]), neon::vgethigh(v2qf32_n0_3.val[1]));
            float32x2_t vdf32_n0_c2_12 = neon::vadd(neon::vgethigh(v2qf32_n0_5.val[0]), neon::vgetlow(v2qf32_n0_5.val[1]));
            float32x2_t vdf32_n0_c2_03 = neon::vadd(neon::vgetlow(v2qf32_n0_5.val[0]), neon::vgethigh(v2qf32_n0_5.val[1]));
            float32x2_t vdf32_n1_c0_12 = neon::vadd(neon::vgethigh(v2qf32_n1_1.val[0]), neon::vgetlow(v2qf32_n1_1.val[1]));
            float32x2_t vdf32_n1_c0_03 = neon::vadd(neon::vgetlow(v2qf32_n1_1.val[0]), neon::vgethigh(v2qf32_n1_1.val[1]));
            float32x2_t vdf32_n1_c1_12 = neon::vadd(neon::vgethigh(v2qf32_n1_3.val[0]), neon::vgetlow(v2qf32_n1_3.val[1]));
            float32x2_t vdf32_n1_c1_03 = neon::vadd(neon::vgetlow(v2qf32_n1_3.val[0]), neon::vgethigh(v2qf32_n1_3.val[1]));
            float32x2_t vdf32_n1_c2_12 = neon::vadd(neon::vgethigh(v2qf32_n1_5.val[0]), neon::vgetlow(v2qf32_n1_5.val[1]));
            float32x2_t vdf32_n1_c2_03 = neon::vadd(neon::vgetlow(v2qf32_n1_5.val[0]), neon::vgethigh(v2qf32_n1_5.val[1]));
            float32x2_t vdf32_n2_c0_12 = neon::vadd(neon::vgethigh(v2qf32_n2_1.val[0]), neon::vgetlow(v2qf32_n2_1.val[1]));
            float32x2_t vdf32_n2_c0_03 = neon::vadd(neon::vgetlow(v2qf32_n2_1.val[0]), neon::vgethigh(v2qf32_n2_1.val[1]));
            float32x2_t vdf32_n2_c1_12 = neon::vadd(neon::vgethigh(v2qf32_n2_3.val[0]), neon::vgetlow(v2qf32_n2_3.val[1]));
            float32x2_t vdf32_n2_c1_03 = neon::vadd(neon::vgetlow(v2qf32_n2_3.val[0]), neon::vgethigh(v2qf32_n2_3.val[1]));
            float32x2_t vdf32_n2_c2_12 = neon::vadd(neon::vgethigh(v2qf32_n2_5.val[0]), neon::vgetlow(v2qf32_n2_5.val[1]));
            float32x2_t vdf32_n2_c2_03 = neon::vadd(neon::vgetlow(v2qf32_n2_5.val[0]), neon::vgethigh(v2qf32_n2_5.val[1]));

            float32x2_t vdf32_c_c0_x19     = neon::vmul(vdf32_c_c0_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_c_c0_result  = neon::vmls(vdf32_c_c0_x19, vdf32_c_c0_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_c_c1_x19     = neon::vmul(vdf32_c_c1_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_c_c1_result  = neon::vmls(vdf32_c_c1_x19, vdf32_c_c1_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_c_c2_x19     = neon::vmul(vdf32_c_c2_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_c_c2_result  = neon::vmls(vdf32_c_c2_x19, vdf32_c_c2_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n0_c0_x19    = neon::vmul(vdf32_n0_c0_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n0_c0_result = neon::vmls(vdf32_n0_c0_x19, vdf32_n0_c0_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n0_c1_x19    = neon::vmul(vdf32_n0_c1_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n0_c1_result = neon::vmls(vdf32_n0_c1_x19, vdf32_n0_c1_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n0_c2_x19    = neon::vmul(vdf32_n0_c2_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n0_c2_result = neon::vmls(vdf32_n0_c2_x19, vdf32_n0_c2_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n1_c0_x19    = neon::vmul(vdf32_n1_c0_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n1_c0_result = neon::vmls(vdf32_n1_c0_x19, vdf32_n1_c0_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n1_c1_x19    = neon::vmul(vdf32_n1_c1_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n1_c1_result = neon::vmls(vdf32_n1_c1_x19, vdf32_n1_c1_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n1_c2_x19    = neon::vmul(vdf32_n1_c2_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n1_c2_result = neon::vmls(vdf32_n1_c2_x19, vdf32_n1_c2_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n2_c0_x19    = neon::vmul(vdf32_n2_c0_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n2_c0_result = neon::vmls(vdf32_n2_c0_x19, vdf32_n2_c0_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n2_c1_x19    = neon::vmul(vdf32_n2_c1_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n2_c1_result = neon::vmls(vdf32_n2_c1_x19, vdf32_n2_c1_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_n2_c2_x19    = neon::vmul(vdf32_n2_c2_12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_n2_c2_result = neon::vmls(vdf32_n2_c2_x19, vdf32_n2_c2_03, static_cast<MI_F32>(0.09375));

            float32x2_t vdf32_c0_result_12 = neon::vadd(vdf32_n0_c0_result, vdf32_n1_c0_result);
            float32x2_t vdf32_c0_result_03 = neon::vadd(vdf32_c_c0_result, vdf32_n2_c0_result);
            float32x2_t vdf32_c1_result_12 = neon::vadd(vdf32_n0_c1_result, vdf32_n1_c1_result);
            float32x2_t vdf32_c1_result_03 = neon::vadd(vdf32_c_c1_result, vdf32_n2_c1_result);
            float32x2_t vdf32_c2_result_12 = neon::vadd(vdf32_n0_c2_result, vdf32_n1_c2_result);
            float32x2_t vdf32_c2_result_03 = neon::vadd(vdf32_c_c2_result, vdf32_n2_c2_result);

            float32x2x3_t v3df32_result;
            float32x2_t vdf32_c0_x19 = neon::vmul(vdf32_c0_result_12, static_cast<MI_F32>(0.59375));
            v3df32_result.val[0]     = neon::vmls(vdf32_c0_x19, vdf32_c0_result_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_c1_x19 = neon::vmul(vdf32_c1_result_12, static_cast<MI_F32>(0.59375));
            v3df32_result.val[1]     = neon::vmls(vdf32_c1_x19, vdf32_c1_result_03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_c2_x19 = neon::vmul(vdf32_c2_result_12, static_cast<MI_F32>(0.59375));
            v3df32_result.val[2]     = neon::vmls(vdf32_c2_x19, vdf32_c2_result_03, static_cast<MI_F32>(0.09375));
            neon::vstore(dst_row, v3df32_result);

            dst_row += 6;
            src_c   += 24;
            src_n0  += 24;
            src_n1  += 24;
            src_n2  += 24;
        }

        for (; x < owidth; x++)
        {
            MI_F32 y00 = (src_c[3] + src_c[6]) * 0.59375f - (src_c[0] + src_c[9]) * 0.09375f;
            MI_F32 y10 = (src_n0[3] + src_n0[6]) * 0.59375f - (src_n0[0] + src_n0[9]) * 0.09375f;
            MI_F32 y20 = (src_n1[3] + src_n1[6]) * 0.59375f - (src_n1[0] + src_n1[9]) * 0.09375f;
            MI_F32 y30 = (src_n2[3] + src_n2[6]) * 0.59375f - (src_n2[0] + src_n2[9]) * 0.09375f;
            *dst_row   = (y10 + y20) * 0.59375f - (y00 + y30) * 0.09375f;

            MI_F32 y01     = (src_c[4] + src_c[7]) * 0.59375f - (src_c[1] + src_c[10]) * 0.09375f;
            MI_F32 y11     = (src_n0[4] + src_n0[7]) * 0.59375f - (src_n0[1] + src_n0[10]) * 0.09375f;
            MI_F32 y21     = (src_n1[4] + src_n1[7]) * 0.59375f - (src_n1[1] + src_n1[10]) * 0.09375f;
            MI_F32 y31     = (src_n2[4] + src_n2[7]) * 0.59375f - (src_n2[1] + src_n2[10]) * 0.09375f;
            *(dst_row + 1) = (y11 + y21) * 0.59375f - (y01 + y31) * 0.09375f;

            MI_F32 y02     = (src_c[5] + src_c[8]) * 0.59375f - (src_c[2] + src_c[11]) * 0.09375f;
            MI_F32 y12     = (src_n0[5] + src_n0[8]) * 0.59375f - (src_n0[2] + src_n0[11]) * 0.09375f;
            MI_F32 y22     = (src_n1[5] + src_n1[8]) * 0.59375f - (src_n1[2] + src_n1[11]) * 0.09375f;
            MI_F32 y32     = (src_n2[5] + src_n2[8]) * 0.59375f - (src_n2[2] + src_n2[11]) * 0.09375f;
            *(dst_row + 2) = (y12 + y22) * 0.59375f - (y02 + y32) * 0.09375f;

            dst_row += 3;
            src_c   += 12;
            src_n0  += 12;
            src_n1  += 12;
            src_n2  += 12;
        }
    }

    return Status::OK;
}

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeCuC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using WMVType  = typename neon::MQVector<Tp, 3>::MVType;
    using MVType   = typename neon::MDVector<Tp, 3>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_S16 *rows = thread_buffer.GetThreadData<MI_S16>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_S16 *rows0 = rows;
    MI_S16 *rows1 = rows0 + owidth * channel;
    MI_S16 *rows2 = rows1 + owidth * channel;
    MI_S16 *rows3 = rows2 + owidth * channel;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_S16 *rows0_x = rows0;
    MI_S16 *rows1_x = rows1;
    MI_S16 *rows2_x = rows2;
    MI_S16 *rows3_x = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 16 + src_c[3] * 19 - src_c[6] * 3;
    rows1_x[0] = src_n0[0] * 16 + src_n0[3] * 19 - src_n0[6] * 3;
    rows2_x[0] = src_n1[0] * 16 + src_n1[3] * 19 - src_n1[6] * 3;
    rows3_x[0] = src_n2[0] * 16 + src_n2[3] * 19 - src_n2[6] * 3;
    rows0_x[1] = src_c[1] * 16 + src_c[4] * 19 - src_c[7] * 3;
    rows1_x[1] = src_n0[1] * 16 + src_n0[4] * 19 - src_n0[7] * 3;
    rows2_x[1] = src_n1[1] * 16 + src_n1[4] * 19 - src_n1[7] * 3;
    rows3_x[1] = src_n2[1] * 16 + src_n2[4] * 19 - src_n2[7] * 3;
    rows0_x[2] = src_c[2] * 16 + src_c[5] * 19 - src_c[8] * 3;
    rows1_x[2] = src_n0[2] * 16 + src_n0[5] * 19 - src_n0[8] * 3;
    rows2_x[2] = src_n1[2] * 16 + src_n1[5] * 19 - src_n1[8] * 3;
    rows3_x[2] = src_n2[2] * 16 + src_n2[5] * 19 - src_n2[8] * 3;

    src_c  += channel;
    src_n0 += channel;
    src_n1 += channel;
    src_n2 += channel;

    rows0_x += channel;
    rows1_x += channel;
    rows2_x += channel;
    rows3_x += channel;

    MI_S32 owidth_align8 = (owidth - 2) & (-8);
    MI_S32 dx = 0;
    for (; dx < owidth_align8; dx += 8)
    {
        WMVType wmvq8_cx0  = neon::vload3q(src_c); //val[0]:channel 0; val[1]:channel 1; val[2]: channel 2
        WMVType wmvq8_n0x0 = neon::vload3q(src_n0);
        WMVType wmvq8_n1x0 = neon::vload3q(src_n1);
        WMVType wmvq8_n2x0 = neon::vload3q(src_n2);
        WMVType wmvq8_cx1  = neon::vload3q(src_c + 2 * channel);
        WMVType wmvq8_n0x1 = neon::vload3q(src_n0 + 2 * channel);
        WMVType wmvq8_n1x1 = neon::vload3q(src_n1 + 2 * channel);
        WMVType wmvq8_n2x1 = neon::vload3q(src_n2 + 2 * channel);

        auto v2q8_cx0_tmp0  = neon::vuzp(wmvq8_cx0.val[0], wmvq8_cx0.val[0]);
        auto v2q8_n0x0_tmp0 = neon::vuzp(wmvq8_n0x0.val[0], wmvq8_n0x0.val[0]);
        auto v2q8_n1x0_tmp0 = neon::vuzp(wmvq8_n1x0.val[0], wmvq8_n1x0.val[0]);
        auto v2q8_n2x0_tmp0 = neon::vuzp(wmvq8_n2x0.val[0], wmvq8_n2x0.val[0]);
        auto v2q8_cx1_tmp0  = neon::vuzp(wmvq8_cx1.val[0], wmvq8_cx1.val[0]);
        auto v2q8_n0x1_tmp0 = neon::vuzp(wmvq8_n0x1.val[0], wmvq8_n0x1.val[0]);
        auto v2q8_n1x1_tmp0 = neon::vuzp(wmvq8_n1x1.val[0], wmvq8_n1x1.val[0]);
        auto v2q8_n2x1_tmp0 = neon::vuzp(wmvq8_n2x1.val[0], wmvq8_n2x1.val[0]);

        auto v2q8_cx0_tmp1  = neon::vuzp(wmvq8_cx0.val[1], wmvq8_cx0.val[1]);
        auto v2q8_n0x0_tmp1 = neon::vuzp(wmvq8_n0x0.val[1], wmvq8_n0x0.val[1]);
        auto v2q8_n1x0_tmp1 = neon::vuzp(wmvq8_n1x0.val[1], wmvq8_n1x0.val[1]);
        auto v2q8_n2x0_tmp1 = neon::vuzp(wmvq8_n2x0.val[1], wmvq8_n2x0.val[1]);
        auto v2q8_cx1_tmp1  = neon::vuzp(wmvq8_cx1.val[1], wmvq8_cx1.val[1]);
        auto v2q8_n0x1_tmp1 = neon::vuzp(wmvq8_n0x1.val[1], wmvq8_n0x1.val[1]);
        auto v2q8_n1x1_tmp1 = neon::vuzp(wmvq8_n1x1.val[1], wmvq8_n1x1.val[1]);
        auto v2q8_n2x1_tmp1 = neon::vuzp(wmvq8_n2x1.val[1], wmvq8_n2x1.val[1]);

        auto v2q8_cx0_tmp2  = neon::vuzp(wmvq8_cx0.val[2], wmvq8_cx0.val[2]);
        auto v2q8_n0x0_tmp2 = neon::vuzp(wmvq8_n0x0.val[2], wmvq8_n0x0.val[2]);
        auto v2q8_n1x0_tmp2 = neon::vuzp(wmvq8_n1x0.val[2], wmvq8_n1x0.val[2]);
        auto v2q8_n2x0_tmp2 = neon::vuzp(wmvq8_n2x0.val[2], wmvq8_n2x0.val[2]);
        auto v2q8_cx1_tmp2  = neon::vuzp(wmvq8_cx1.val[2], wmvq8_cx1.val[2]);
        auto v2q8_n0x1_tmp2 = neon::vuzp(wmvq8_n0x1.val[2], wmvq8_n0x1.val[2]);
        auto v2q8_n1x1_tmp2 = neon::vuzp(wmvq8_n1x1.val[2], wmvq8_n1x1.val[2]);
        auto v2q8_n2x1_tmp2 = neon::vuzp(wmvq8_n2x1.val[2], wmvq8_n2x1.val[2]);

        int16x8_t vqs16_c00_12 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp0.val[1]), neon::vgetlow(v2q8_cx1_tmp0.val[0]));
        int16x8_t vqs16_c00_03 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp0.val[0]), neon::vgetlow(v2q8_cx1_tmp0.val[1]));
        int16x8_t vqs16_c01_12 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp0.val[1]), neon::vgetlow(v2q8_n0x1_tmp0.val[0]));
        int16x8_t vqs16_c01_03 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp0.val[0]), neon::vgetlow(v2q8_n0x1_tmp0.val[1]));
        int16x8_t vqs16_c02_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp0.val[1]), neon::vgetlow(v2q8_n1x1_tmp0.val[0]));
        int16x8_t vqs16_c02_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp0.val[0]), neon::vgetlow(v2q8_n1x1_tmp0.val[1]));
        int16x8_t vqs16_c03_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp0.val[1]), neon::vgetlow(v2q8_n2x1_tmp0.val[0]));
        int16x8_t vqs16_c03_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp0.val[0]), neon::vgetlow(v2q8_n2x1_tmp0.val[1]));

        int16x8_t vqs16_c10_12 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp1.val[1]), neon::vgetlow(v2q8_cx1_tmp1.val[0]));
        int16x8_t vqs16_c10_03 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp1.val[0]), neon::vgetlow(v2q8_cx1_tmp1.val[1]));
        int16x8_t vqs16_c11_12 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp1.val[1]), neon::vgetlow(v2q8_n0x1_tmp1.val[0]));
        int16x8_t vqs16_c11_03 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp1.val[0]), neon::vgetlow(v2q8_n0x1_tmp1.val[1]));
        int16x8_t vqs16_c12_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp1.val[1]), neon::vgetlow(v2q8_n1x1_tmp1.val[0]));
        int16x8_t vqs16_c12_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp1.val[0]), neon::vgetlow(v2q8_n1x1_tmp1.val[1]));
        int16x8_t vqs16_c13_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp1.val[1]), neon::vgetlow(v2q8_n2x1_tmp1.val[0]));
        int16x8_t vqs16_c13_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp1.val[0]), neon::vgetlow(v2q8_n2x1_tmp1.val[1]));

        int16x8_t vqs16_c20_12 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp2.val[1]), neon::vgetlow(v2q8_cx1_tmp2.val[0]));
        int16x8_t vqs16_c20_03 = neon::vaddl(neon::vgetlow(v2q8_cx0_tmp2.val[0]), neon::vgetlow(v2q8_cx1_tmp2.val[1]));
        int16x8_t vqs16_c21_12 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp2.val[1]), neon::vgetlow(v2q8_n0x1_tmp2.val[0]));
        int16x8_t vqs16_c21_03 = neon::vaddl(neon::vgetlow(v2q8_n0x0_tmp2.val[0]), neon::vgetlow(v2q8_n0x1_tmp2.val[1]));
        int16x8_t vqs16_c22_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp2.val[1]), neon::vgetlow(v2q8_n1x1_tmp2.val[0]));
        int16x8_t vqs16_c22_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp2.val[0]), neon::vgetlow(v2q8_n1x1_tmp2.val[1]));
        int16x8_t vqs16_c23_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp2.val[1]), neon::vgetlow(v2q8_n2x1_tmp2.val[0]));
        int16x8_t vqs16_c23_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp2.val[0]), neon::vgetlow(v2q8_n2x1_tmp2.val[1]));

        int16x8x3_t v3qs16_result0, v3qs16_result1, v3qs16_result2, v3qs16_result3;
        int16x8_t vqs16_c00_x19 = neon::vmul(vqs16_c00_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c01_x19 = neon::vmul(vqs16_c01_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c02_x19 = neon::vmul(vqs16_c02_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c03_x19 = neon::vmul(vqs16_c03_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c10_x19 = neon::vmul(vqs16_c10_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c11_x19 = neon::vmul(vqs16_c11_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c12_x19 = neon::vmul(vqs16_c12_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c13_x19 = neon::vmul(vqs16_c13_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c20_x19 = neon::vmul(vqs16_c20_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c21_x19 = neon::vmul(vqs16_c21_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c22_x19 = neon::vmul(vqs16_c22_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_c23_x19 = neon::vmul(vqs16_c23_12, static_cast<MI_S16>(19));

        v3qs16_result0.val[0] = neon::vmls(vqs16_c00_x19, vqs16_c00_03, static_cast<MI_S16>(3));
        v3qs16_result1.val[0] = neon::vmls(vqs16_c01_x19, vqs16_c01_03, static_cast<MI_S16>(3));
        v3qs16_result2.val[0] = neon::vmls(vqs16_c02_x19, vqs16_c02_03, static_cast<MI_S16>(3));
        v3qs16_result3.val[0] = neon::vmls(vqs16_c03_x19, vqs16_c03_03, static_cast<MI_S16>(3));
        v3qs16_result0.val[1] = neon::vmls(vqs16_c10_x19, vqs16_c10_03, static_cast<MI_S16>(3));
        v3qs16_result1.val[1] = neon::vmls(vqs16_c11_x19, vqs16_c11_03, static_cast<MI_S16>(3));
        v3qs16_result2.val[1] = neon::vmls(vqs16_c12_x19, vqs16_c12_03, static_cast<MI_S16>(3));
        v3qs16_result3.val[1] = neon::vmls(vqs16_c13_x19, vqs16_c13_03, static_cast<MI_S16>(3));
        v3qs16_result0.val[2] = neon::vmls(vqs16_c20_x19, vqs16_c20_03, static_cast<MI_S16>(3));
        v3qs16_result1.val[2] = neon::vmls(vqs16_c21_x19, vqs16_c21_03, static_cast<MI_S16>(3));
        v3qs16_result2.val[2] = neon::vmls(vqs16_c22_x19, vqs16_c22_03, static_cast<MI_S16>(3));
        v3qs16_result3.val[2] = neon::vmls(vqs16_c23_x19, vqs16_c23_03, static_cast<MI_S16>(3));

        neon::vstore(rows0_x, v3qs16_result0);
        neon::vstore(rows1_x, v3qs16_result1);
        neon::vstore(rows2_x, v3qs16_result2);
        neon::vstore(rows3_x, v3qs16_result3);

        rows0_x += 24;
        rows1_x += 24;
        rows2_x += 24;
        rows3_x += 24;

        src_c  += 48;
        src_n0 += 48;
        src_n1 += 48;
        src_n2 += 48;
    }

    for (; dx < (owidth - 2); dx++)
    {
        rows0_x[0] = src_c[3] * 19 - src_c[0] * 3 + src_c[6] * 19 - src_c[9] * 3;
        rows1_x[0] = src_n0[3] * 19 - src_n0[0] * 3 + src_n0[6] * 19 - src_n0[9] * 3;
        rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 19 - src_n1[9] * 3;
        rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 19 - src_n2[9] * 3;
        rows0_x[1] = src_c[4] * 19 - src_c[1] * 3 + src_c[7] * 19 - src_c[10] * 3;
        rows1_x[1] = src_n0[4] * 19 - src_n0[1] * 3 + src_n0[7] * 19 - src_n0[10] * 3;
        rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 19 - src_n1[10] * 3;
        rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 19 - src_n2[10] * 3;
        rows0_x[2] = src_c[5] * 19 - src_c[2] * 3 + src_c[8] * 19 - src_c[11] * 3;
        rows1_x[2] = src_n0[5] * 19 - src_n0[2] * 3 + src_n0[8] * 19 - src_n0[11] * 3;
        rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 19 - src_n1[11] * 3;
        rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 19 - src_n2[11] * 3;

        rows0_x += channel;
        rows1_x += channel;
        rows2_x += channel;
        rows3_x += channel;
        src_c  += 2 * channel;
        src_n0 += 2 * channel;
        src_n1 += 2 * channel;
        src_n2 += 2 * channel;
    }

    rows0_x[0] = src_c[3] * 19 - src_c[0] * 3 + src_c[6] * 16;
    rows1_x[0] = src_n0[3] * 19 - src_n0[0] * 3 + src_n0[6] * 16;
    rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 16;
    rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 16;
    rows0_x[1] = src_c[4] * 19 - src_c[1] * 3 + src_c[7] * 16;
    rows1_x[1] = src_n0[4] * 19 - src_n0[1] * 3 + src_n0[7] * 16;
    rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 16;
    rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 16;
    rows0_x[2] = src_c[5] * 19 - src_c[2] * 3 + src_c[8] * 16;
    rows1_x[2] = src_n0[5] * 19 - src_n0[2] * 3 + src_n0[8] * 16;
    rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 16;
    rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 16;

    // vresize
    MI_S16 *rows0_y = rows0;
    MI_S16 *rows1_y = rows1;
    MI_S16 *rows2_y = rows2;
    MI_S16 *rows3_y = rows3;

    Tp *dst_row   = dst.Ptr<Tp>(start_row);
    owidth_align8 = owidth & (-8);

    int16x8_t vqs16_zero;
    neon::vdup(vqs16_zero, static_cast<MI_S16>(0));
    int16x8_t vqs16_255;
    neon::vdup(vqs16_255, static_cast<MI_S16>(255));

    dx = 0;
    for (; dx < owidth_align8; dx += 8)
    {
        int16x8x3_t v3qs16_c  = neon::vload3q(rows0_y);
        int16x8x3_t v3qs16_n0 = neon::vload3q(rows1_y);
        int16x8x3_t v3qs16_n1 = neon::vload3q(rows2_y);
        int16x8x3_t v3qs16_n2 = neon::vload3q(rows3_y);

        int32x4_t vqs32_0_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[0]), neon::vgetlow(v3qs16_n1.val[0]));
        int32x4_t vqs32_0_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[0]), neon::vgetlow(v3qs16_n2.val[0]));
        int32x4_t vqs32_1_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[1]), neon::vgetlow(v3qs16_n1.val[1]));
        int32x4_t vqs32_1_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[1]), neon::vgetlow(v3qs16_n2.val[1]));
        int32x4_t vqs32_2_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[2]), neon::vgetlow(v3qs16_n1.val[2]));
        int32x4_t vqs32_2_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[2]), neon::vgetlow(v3qs16_n2.val[2]));
        int32x4_t vqs32_0_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[0]), neon::vgethigh(v3qs16_n1.val[0]));
        int32x4_t vqs32_0_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[0]), neon::vgethigh(v3qs16_n2.val[0]));
        int32x4_t vqs32_1_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[1]), neon::vgethigh(v3qs16_n1.val[1]));
        int32x4_t vqs32_1_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[1]), neon::vgethigh(v3qs16_n2.val[1]));
        int32x4_t vqs32_2_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[2]), neon::vgethigh(v3qs16_n1.val[2]));
        int32x4_t vqs32_2_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[2]), neon::vgethigh(v3qs16_n2.val[2]));

        int32x4_t vqs32_lo0_x19 = neon::vmul(vqs32_0_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi0_x19 = neon::vmul(vqs32_0_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_lo1_x19 = neon::vmul(vqs32_1_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi1_x19 = neon::vmul(vqs32_1_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_lo2_x19 = neon::vmul(vqs32_2_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi2_x19 = neon::vmul(vqs32_2_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_des0_lo = neon::vmls(vqs32_lo0_x19, vqs32_0_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des0_hi = neon::vmls(vqs32_hi0_x19, vqs32_0_hi03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des1_lo = neon::vmls(vqs32_lo1_x19, vqs32_1_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des1_hi = neon::vmls(vqs32_hi1_x19, vqs32_1_hi03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des2_lo = neon::vmls(vqs32_lo2_x19, vqs32_2_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des2_hi = neon::vmls(vqs32_hi2_x19, vqs32_2_hi03, static_cast<MI_S32>(3));

        int16x4_t vds16_des0_lo = neon::vrshrn_n<10>(vqs32_des0_lo);
        int16x4_t vds16_des0_hi = neon::vrshrn_n<10>(vqs32_des0_hi);
        int16x4_t vds16_des1_lo = neon::vrshrn_n<10>(vqs32_des1_lo);
        int16x4_t vds16_des1_hi = neon::vrshrn_n<10>(vqs32_des1_hi);
        int16x4_t vds16_des2_lo = neon::vrshrn_n<10>(vqs32_des2_lo);
        int16x4_t vds16_des2_hi = neon::vrshrn_n<10>(vqs32_des2_hi);

        MVType mvd8_result;
        if (std::is_same<Tp, MI_U8>::value)
        {
            int16x8_t vqs16_des0 = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
            int16x8_t vqs16_des1 = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
            int16x8_t vqs16_des2 = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

            vqs16_des0 = neon::vmax(vqs16_des0, vqs16_zero);
            vqs16_des0 = neon::vmin(vqs16_des0, vqs16_255);
            vqs16_des1 = neon::vmax(vqs16_des1, vqs16_zero);
            vqs16_des1 = neon::vmin(vqs16_des1, vqs16_255);
            vqs16_des2 = neon::vmax(vqs16_des2, vqs16_zero);
            vqs16_des2 = neon::vmin(vqs16_des2, vqs16_255);

            uint16x8_t vqu16_des0 = neon::vreinterpret(vqs16_des0);
            uint16x8_t vqu16_des1 = neon::vreinterpret(vqs16_des1);
            uint16x8_t vqu16_des2 = neon::vreinterpret(vqs16_des2);

            mvd8_result.val[0] = neon::vqmovn(vqu16_des0);
            mvd8_result.val[1] = neon::vqmovn(vqu16_des1);
            mvd8_result.val[2] = neon::vqmovn(vqu16_des2);

            neon::vstore(dst_row, mvd8_result);
        }
        else
        {
            int16x8_t vqs16_des0 = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
            int16x8_t vqs16_des1 = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
            int16x8_t vqs16_des2 = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

            mvd8_result.val[0] = neon::vqmovn(vqs16_des0);
            mvd8_result.val[1] = neon::vqmovn(vqs16_des1);
            mvd8_result.val[2] = neon::vqmovn(vqs16_des2);

            neon::vstore(dst_row, mvd8_result);
        }

        dst_row += 24;
        rows0_y += 24;
        rows1_y += 24;
        rows2_y += 24;
        rows3_y += 24;
    }

    for (; dx < owidth; dx++)
    {
        MI_S32 result = (rows1_y[0] * 19 - rows0_y[0] * 3 + rows2_y[0] * 19 - rows3_y[0] * 3 + 512) >> 10;
        dst_row[0]    = SaturateCast<Tp>(result);
        result        = (rows1_y[1] * 19 - rows0_y[1] * 3 + rows2_y[1] * 19 - rows3_y[1] * 3 + 512) >> 10;
        dst_row[1]    = SaturateCast<Tp>(result);
        result        = (rows1_y[2] * 19 - rows0_y[2] * 3 + rows2_y[2] * 19 - rows3_y[2] * 3 + 512) >> 10;
        dst_row[2]    = SaturateCast<Tp>(result);

        dst_row += channel;
        rows0_y += channel;
        rows1_y += channel;
        rows2_y += channel;
        rows3_y += channel;
    }

    // Line 1 ~ h-1
    for (MI_S32 dy = (start_row + 1); dy < end_row; dy++)
    {
        MI_S32 sy = (dy << 1) - 1;

        // hresize two row
        MI_S16 *rows0_tmp = rows0;
        MI_S16 *rows1_tmp = rows1;
        rows0             = rows2;
        rows1             = rows3;
        rows2             = rows0_tmp;
        rows3             = rows1_tmp;

        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        MI_S16 *rows2_x  = rows2;
        MI_S16 *rows3_x  = rows3;

        if (end_row == oheight && dy == (end_row - 1))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 16 + src_n1[3] * 19 - src_n1[6] * 3;
        rows3_x[0] = src_n2[0] * 16 + src_n2[3] * 19 - src_n2[6] * 3;
        rows2_x[1] = src_n1[1] * 16 + src_n1[4] * 19 - src_n1[7] * 3;
        rows3_x[1] = src_n2[1] * 16 + src_n2[4] * 19 - src_n2[7] * 3;
        rows2_x[2] = src_n1[2] * 16 + src_n1[5] * 19 - src_n1[8] * 3;
        rows3_x[2] = src_n2[2] * 16 + src_n2[5] * 19 - src_n2[8] * 3;

        src_n1  += channel;
        src_n2  += channel;
        rows2_x += channel;
        rows3_x += channel;

        MI_S32 owidth_align8 = (owidth - 2) & (-8);
        MI_S32 dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            WMVType wmvq8_n1x0 = neon::vload3q(src_n1);
            WMVType wmvq8_n2x0 = neon::vload3q(src_n2);
            WMVType wmvq8_n1x1 = neon::vload3q(src_n1 + 2 * channel);
            WMVType wmvq8_n2x1 = neon::vload3q(src_n2 + 2 * channel);

            auto v2q8_n1x0_tmp0 = neon::vuzp(wmvq8_n1x0.val[0], wmvq8_n1x0.val[0]);
            auto v2q8_n2x0_tmp0 = neon::vuzp(wmvq8_n2x0.val[0], wmvq8_n2x0.val[0]);
            auto v2q8_n1x1_tmp0 = neon::vuzp(wmvq8_n1x1.val[0], wmvq8_n1x1.val[0]);
            auto v2q8_n2x1_tmp0 = neon::vuzp(wmvq8_n2x1.val[0], wmvq8_n2x1.val[0]);
            auto v2q8_n1x0_tmp1 = neon::vuzp(wmvq8_n1x0.val[1], wmvq8_n1x0.val[1]);
            auto v2q8_n2x0_tmp1 = neon::vuzp(wmvq8_n2x0.val[1], wmvq8_n2x0.val[1]);
            auto v2q8_n1x1_tmp1 = neon::vuzp(wmvq8_n1x1.val[1], wmvq8_n1x1.val[1]);
            auto v2q8_n2x1_tmp1 = neon::vuzp(wmvq8_n2x1.val[1], wmvq8_n2x1.val[1]);
            auto v2q8_n1x0_tmp2 = neon::vuzp(wmvq8_n1x0.val[2], wmvq8_n1x0.val[2]);
            auto v2q8_n2x0_tmp2 = neon::vuzp(wmvq8_n2x0.val[2], wmvq8_n2x0.val[2]);
            auto v2q8_n1x1_tmp2 = neon::vuzp(wmvq8_n1x1.val[2], wmvq8_n1x1.val[2]);
            auto v2q8_n2x1_tmp2 = neon::vuzp(wmvq8_n2x1.val[2], wmvq8_n2x1.val[2]);

            int16x8_t vqs16_c02_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp0.val[1]), neon::vgetlow(v2q8_n1x1_tmp0.val[0]));
            int16x8_t vqs16_c02_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp0.val[0]), neon::vgetlow(v2q8_n1x1_tmp0.val[1]));
            int16x8_t vqs16_c03_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp0.val[1]), neon::vgetlow(v2q8_n2x1_tmp0.val[0]));
            int16x8_t vqs16_c03_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp0.val[0]), neon::vgetlow(v2q8_n2x1_tmp0.val[1]));
            int16x8_t vqs16_c12_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp1.val[1]), neon::vgetlow(v2q8_n1x1_tmp1.val[0]));
            int16x8_t vqs16_c12_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp1.val[0]), neon::vgetlow(v2q8_n1x1_tmp1.val[1]));
            int16x8_t vqs16_c13_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp1.val[1]), neon::vgetlow(v2q8_n2x1_tmp1.val[0]));
            int16x8_t vqs16_c13_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp1.val[0]), neon::vgetlow(v2q8_n2x1_tmp1.val[1]));
            int16x8_t vqs16_c22_12 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp2.val[1]), neon::vgetlow(v2q8_n1x1_tmp2.val[0]));
            int16x8_t vqs16_c22_03 = neon::vaddl(neon::vgetlow(v2q8_n1x0_tmp2.val[0]), neon::vgetlow(v2q8_n1x1_tmp2.val[1]));
            int16x8_t vqs16_c23_12 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp2.val[1]), neon::vgetlow(v2q8_n2x1_tmp2.val[0]));
            int16x8_t vqs16_c23_03 = neon::vaddl(neon::vgetlow(v2q8_n2x0_tmp2.val[0]), neon::vgetlow(v2q8_n2x1_tmp2.val[1]));

            int16x8_t vqs16_c02_x19 = neon::vmul(vqs16_c02_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c03_x19 = neon::vmul(vqs16_c03_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c12_x19 = neon::vmul(vqs16_c12_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c13_x19 = neon::vmul(vqs16_c13_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c22_x19 = neon::vmul(vqs16_c22_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c23_x19 = neon::vmul(vqs16_c23_12, static_cast<MI_S16>(19));

            int16x8x3_t v3qs16_result2, v3qs16_result3;
            v3qs16_result2.val[0] = neon::vmls(vqs16_c02_x19, vqs16_c02_03, static_cast<MI_S16>(3));
            v3qs16_result3.val[0] = neon::vmls(vqs16_c03_x19, vqs16_c03_03, static_cast<MI_S16>(3));
            v3qs16_result2.val[1] = neon::vmls(vqs16_c12_x19, vqs16_c12_03, static_cast<MI_S16>(3));
            v3qs16_result3.val[1] = neon::vmls(vqs16_c13_x19, vqs16_c13_03, static_cast<MI_S16>(3));
            v3qs16_result2.val[2] = neon::vmls(vqs16_c22_x19, vqs16_c22_03, static_cast<MI_S16>(3));
            v3qs16_result3.val[2] = neon::vmls(vqs16_c23_x19, vqs16_c23_03, static_cast<MI_S16>(3));

            neon::vstore(rows2_x, v3qs16_result2);
            neon::vstore(rows3_x, v3qs16_result3);

            rows2_x += 24;
            rows3_x += 24;
            src_n1  += 48;
            src_n2  += 48;
        }

        for (; dx < (owidth - 2); dx++)
        {
            rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 19 - src_n1[9] * 3;
            rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 19 - src_n2[9] * 3;
            rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 19 - src_n1[10] * 3;
            rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 19 - src_n2[10] * 3;
            rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 19 - src_n1[11] * 3;
            rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 19 - src_n2[11] * 3;

            rows2_x += channel;
            rows3_x += channel;
            src_n1  += 2 * channel;
            src_n2  += 2 * channel;
        }

        rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 16;
        rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 16;
        rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 16;
        rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 16;
        rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 16;
        rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 16;

        // vresize
        MI_S16 *rows0_y = rows0;
        MI_S16 *rows1_y = rows1;
        MI_S16 *rows2_y = rows2;
        MI_S16 *rows3_y = rows3;

        Tp *dst_row = dst.Ptr<Tp>(dy);
        owidth_align8 = owidth & (-8);
        dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            int16x8x3_t v3qs16_c  = neon::vload3q(rows0_y);
            int16x8x3_t v3qs16_n0 = neon::vload3q(rows1_y);
            int16x8x3_t v3qs16_n1 = neon::vload3q(rows2_y);
            int16x8x3_t v3qs16_n2 = neon::vload3q(rows3_y);

            int32x4_t vqs32_0_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[0]), neon::vgetlow(v3qs16_n1.val[0]));
            int32x4_t vqs32_0_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[0]), neon::vgetlow(v3qs16_n2.val[0]));
            int32x4_t vqs32_1_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[1]), neon::vgetlow(v3qs16_n1.val[1]));
            int32x4_t vqs32_1_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[1]), neon::vgetlow(v3qs16_n2.val[1]));
            int32x4_t vqs32_2_lo12 = neon::vaddl(neon::vgetlow(v3qs16_n0.val[2]), neon::vgetlow(v3qs16_n1.val[2]));
            int32x4_t vqs32_2_lo03 = neon::vaddl(neon::vgetlow(v3qs16_c.val[2]), neon::vgetlow(v3qs16_n2.val[2]));
            int32x4_t vqs32_0_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[0]), neon::vgethigh(v3qs16_n1.val[0]));
            int32x4_t vqs32_0_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[0]), neon::vgethigh(v3qs16_n2.val[0]));
            int32x4_t vqs32_1_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[1]), neon::vgethigh(v3qs16_n1.val[1]));
            int32x4_t vqs32_1_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[1]), neon::vgethigh(v3qs16_n2.val[1]));
            int32x4_t vqs32_2_hi12 = neon::vaddl(neon::vgethigh(v3qs16_n0.val[2]), neon::vgethigh(v3qs16_n1.val[2]));
            int32x4_t vqs32_2_hi03 = neon::vaddl(neon::vgethigh(v3qs16_c.val[2]), neon::vgethigh(v3qs16_n2.val[2]));

            int32x4_t vqs32_lo0_x19 = neon::vmul(vqs32_0_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi0_x19 = neon::vmul(vqs32_0_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_lo1_x19 = neon::vmul(vqs32_1_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi1_x19 = neon::vmul(vqs32_1_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_lo2_x19 = neon::vmul(vqs32_2_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi2_x19 = neon::vmul(vqs32_2_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_des0_lo = neon::vmls(vqs32_lo0_x19, vqs32_0_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des0_hi = neon::vmls(vqs32_hi0_x19, vqs32_0_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des1_lo = neon::vmls(vqs32_lo1_x19, vqs32_1_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des1_hi = neon::vmls(vqs32_hi1_x19, vqs32_1_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des2_lo = neon::vmls(vqs32_lo2_x19, vqs32_2_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des2_hi = neon::vmls(vqs32_hi2_x19, vqs32_2_hi03, static_cast<MI_S32>(3));

            int16x4_t vds16_des0_lo = neon::vrshrn_n<10>(vqs32_des0_lo);
            int16x4_t vds16_des0_hi = neon::vrshrn_n<10>(vqs32_des0_hi);
            int16x4_t vds16_des1_lo = neon::vrshrn_n<10>(vqs32_des1_lo);
            int16x4_t vds16_des1_hi = neon::vrshrn_n<10>(vqs32_des1_hi);
            int16x4_t vds16_des2_lo = neon::vrshrn_n<10>(vqs32_des2_lo);
            int16x4_t vds16_des2_hi = neon::vrshrn_n<10>(vqs32_des2_hi);

            MVType mvd8_result;
            if (std::is_same<Tp, MI_U8>::value)
            {
                int16x8_t vqs16_des0 = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
                int16x8_t vqs16_des1 = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
                int16x8_t vqs16_des2 = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

                vqs16_des0 = neon::vmax(vqs16_des0, vqs16_zero);
                vqs16_des0 = neon::vmin(vqs16_des0, vqs16_255);
                vqs16_des1 = neon::vmax(vqs16_des1, vqs16_zero);
                vqs16_des1 = neon::vmin(vqs16_des1, vqs16_255);
                vqs16_des2 = neon::vmax(vqs16_des2, vqs16_zero);
                vqs16_des2 = neon::vmin(vqs16_des2, vqs16_255);

                uint16x8_t vqu16_des0 = neon::vreinterpret(vqs16_des0);
                uint16x8_t vqu16_des1 = neon::vreinterpret(vqs16_des1);
                uint16x8_t vqu16_des2 = neon::vreinterpret(vqs16_des2);

                mvd8_result.val[0] = neon::vqmovn(vqu16_des0);
                mvd8_result.val[1] = neon::vqmovn(vqu16_des1);
                mvd8_result.val[2] = neon::vqmovn(vqu16_des2);

                neon::vstore(dst_row, mvd8_result);
            }
            else
            {
                int16x8_t vqs16_des0 = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
                int16x8_t vqs16_des1 = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
                int16x8_t vqs16_des2 = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

                mvd8_result.val[0] = neon::vqmovn(vqs16_des0);
                mvd8_result.val[1] = neon::vqmovn(vqs16_des1);
                mvd8_result.val[2] = neon::vqmovn(vqs16_des2);

                neon::vstore(dst_row, mvd8_result);
            }

            dst_row += 24;
            rows0_y += 24;
            rows1_y += 24;
            rows2_y += 24;
            rows3_y += 24;
        }

        for (; dx < owidth; dx++)
        {
            MI_S32 result = (rows1_y[0] * 19 - rows0_y[0] * 3 + rows2_y[0] * 19 - rows3_y[0] * 3 + 512) >> 10;
            dst_row[0]    = SaturateCast<Tp>(result);
            result        = (rows1_y[1] * 19 - rows0_y[1] * 3 + rows2_y[1] * 19 - rows3_y[1] * 3 + 512) >> 10;
            dst_row[1]    = SaturateCast<Tp>(result);
            result        = (rows1_y[2] * 19 - rows0_y[2] * 3 + rows2_y[2] * 19 - rows3_y[2] * 3 + 512) >> 10;
            dst_row[2]    = SaturateCast<Tp>(result);

            dst_row += channel;
            rows0_y += channel;
            rows1_y += channel;
            rows2_y += channel;
            rows3_y += channel;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeCuC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MQVector<Tp, 3>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_S32 *rows = thread_buffer.GetThreadData<MI_S32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_S32 *rows0 = rows;
    MI_S32 *rows1 = rows0 + owidth * channel;
    MI_S32 *rows2 = rows1 + owidth * channel;
    MI_S32 *rows3 = rows2 + owidth * channel;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_S32 *rows0_x  = rows0;
    MI_S32 *rows1_x  = rows1;
    MI_S32 *rows2_x  = rows2;
    MI_S32 *rows3_x  = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 16 + src_c[3] * 19 - src_c[6] * 3;
    rows1_x[0] = src_n0[0] * 16 + src_n0[3] * 19 - src_n0[6] * 3;
    rows2_x[0] = src_n1[0] * 16 + src_n1[3] * 19 - src_n1[6] * 3;
    rows3_x[0] = src_n2[0] * 16 + src_n2[3] * 19 - src_n2[6] * 3;
    rows0_x[1] = src_c[1] * 16 + src_c[4] * 19 - src_c[7] * 3;
    rows1_x[1] = src_n0[1] * 16 + src_n0[4] * 19 - src_n0[7] * 3;
    rows2_x[1] = src_n1[1] * 16 + src_n1[4] * 19 - src_n1[7] * 3;
    rows3_x[1] = src_n2[1] * 16 + src_n2[4] * 19 - src_n2[7] * 3;
    rows0_x[2] = src_c[2] * 16 + src_c[5] * 19 - src_c[8] * 3;
    rows1_x[2] = src_n0[2] * 16 + src_n0[5] * 19 - src_n0[8] * 3;
    rows2_x[2] = src_n1[2] * 16 + src_n1[5] * 19 - src_n1[8] * 3;
    rows3_x[2] = src_n2[2] * 16 + src_n2[5] * 19 - src_n2[8] * 3;

    src_c  += channel;
    src_n0 += channel;
    src_n1 += channel;
    src_n2 += channel;

    rows0_x += channel;
    rows1_x += channel;
    rows2_x += channel;
    rows3_x += channel;

    MI_S32 owidth_align4 = (owidth - 2) & (-4);
    MI_S32 dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        MVType mvq16_cx0  = neon::vload3q(src_c);
        MVType mvq16_n0x0 = neon::vload3q(src_n0);
        MVType mvq16_n1x0 = neon::vload3q(src_n1);
        MVType mvq16_n2x0 = neon::vload3q(src_n2);
        MVType mvq16_cx1  = neon::vload3q(src_c + 2 * channel);
        MVType mvq16_n0x1 = neon::vload3q(src_n0 + 2 * channel);
        MVType mvq16_n1x1 = neon::vload3q(src_n1 + 2 * channel);
        MVType mvq16_n2x1 = neon::vload3q(src_n2 + 2 * channel);

        auto v2q16_cx0_tmp0  = neon::vuzp(mvq16_cx0.val[0],  mvq16_cx0.val[0]);
        auto v2q16_n0x0_tmp0 = neon::vuzp(mvq16_n0x0.val[0], mvq16_n0x0.val[0]);
        auto v2q16_n1x0_tmp0 = neon::vuzp(mvq16_n1x0.val[0], mvq16_n1x0.val[0]);
        auto v2q16_n2x0_tmp0 = neon::vuzp(mvq16_n2x0.val[0], mvq16_n2x0.val[0]);
        auto v2q16_cx1_tmp0  = neon::vuzp(mvq16_cx1.val[0],  mvq16_cx1.val[0]);
        auto v2q16_n0x1_tmp0 = neon::vuzp(mvq16_n0x1.val[0], mvq16_n0x1.val[0]);
        auto v2q16_n1x1_tmp0 = neon::vuzp(mvq16_n1x1.val[0], mvq16_n1x1.val[0]);
        auto v2q16_n2x1_tmp0 = neon::vuzp(mvq16_n2x1.val[0], mvq16_n2x1.val[0]);

        auto v2q16_cx0_tmp1  = neon::vuzp(mvq16_cx0.val[1],  mvq16_cx0.val[1]);
        auto v2q16_n0x0_tmp1 = neon::vuzp(mvq16_n0x0.val[1], mvq16_n0x0.val[1]);
        auto v2q16_n1x0_tmp1 = neon::vuzp(mvq16_n1x0.val[1], mvq16_n1x0.val[1]);
        auto v2q16_n2x0_tmp1 = neon::vuzp(mvq16_n2x0.val[1], mvq16_n2x0.val[1]);
        auto v2q16_cx1_tmp1  = neon::vuzp(mvq16_cx1.val[1],  mvq16_cx1.val[1]);
        auto v2q16_n0x1_tmp1 = neon::vuzp(mvq16_n0x1.val[1], mvq16_n0x1.val[1]);
        auto v2q16_n1x1_tmp1 = neon::vuzp(mvq16_n1x1.val[1], mvq16_n1x1.val[1]);
        auto v2q16_n2x1_tmp1 = neon::vuzp(mvq16_n2x1.val[1], mvq16_n2x1.val[1]);

        auto v2q16_cx0_tmp2  = neon::vuzp(mvq16_cx0.val[2],  mvq16_cx0.val[2]);
        auto v2q16_n0x0_tmp2 = neon::vuzp(mvq16_n0x0.val[2], mvq16_n0x0.val[2]);
        auto v2q16_n1x0_tmp2 = neon::vuzp(mvq16_n1x0.val[2], mvq16_n1x0.val[2]);
        auto v2q16_n2x0_tmp2 = neon::vuzp(mvq16_n2x0.val[2], mvq16_n2x0.val[2]);
        auto v2q16_cx1_tmp2  = neon::vuzp(mvq16_cx1.val[2],  mvq16_cx1.val[2]);
        auto v2q16_n0x1_tmp2 = neon::vuzp(mvq16_n0x1.val[2], mvq16_n0x1.val[2]);
        auto v2q16_n1x1_tmp2 = neon::vuzp(mvq16_n1x1.val[2], mvq16_n1x1.val[2]);
        auto v2q16_n2x1_tmp2 = neon::vuzp(mvq16_n2x1.val[2], mvq16_n2x1.val[2]);

        int32x4_t vqs32_c00_12 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp0.val[1]), neon::vgetlow(v2q16_cx1_tmp0.val[0]));
        int32x4_t vqs32_c00_03 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp0.val[0]), neon::vgetlow(v2q16_cx1_tmp0.val[1]));
        int32x4_t vqs32_c01_12 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp0.val[1]), neon::vgetlow(v2q16_n0x1_tmp0.val[0]));
        int32x4_t vqs32_c01_03 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp0.val[0]), neon::vgetlow(v2q16_n0x1_tmp0.val[1]));
        int32x4_t vqs32_c02_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp0.val[1]), neon::vgetlow(v2q16_n1x1_tmp0.val[0]));
        int32x4_t vqs32_c02_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp0.val[0]), neon::vgetlow(v2q16_n1x1_tmp0.val[1]));
        int32x4_t vqs32_c03_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp0.val[1]), neon::vgetlow(v2q16_n2x1_tmp0.val[0]));
        int32x4_t vqs32_c03_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp0.val[0]), neon::vgetlow(v2q16_n2x1_tmp0.val[1]));

        int32x4_t vqs32_c10_12 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp1.val[1]), neon::vgetlow(v2q16_cx1_tmp1.val[0]));
        int32x4_t vqs32_c10_03 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp1.val[0]), neon::vgetlow(v2q16_cx1_tmp1.val[1]));
        int32x4_t vqs32_c11_12 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp1.val[1]), neon::vgetlow(v2q16_n0x1_tmp1.val[0]));
        int32x4_t vqs32_c11_03 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp1.val[0]), neon::vgetlow(v2q16_n0x1_tmp1.val[1]));
        int32x4_t vqs32_c12_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp1.val[1]), neon::vgetlow(v2q16_n1x1_tmp1.val[0]));
        int32x4_t vqs32_c12_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp1.val[0]), neon::vgetlow(v2q16_n1x1_tmp1.val[1]));
        int32x4_t vqs32_c13_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp1.val[1]), neon::vgetlow(v2q16_n2x1_tmp1.val[0]));
        int32x4_t vqs32_c13_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp1.val[0]), neon::vgetlow(v2q16_n2x1_tmp1.val[1]));

        int32x4_t vqs32_c20_12 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp2.val[1]), neon::vgetlow(v2q16_cx1_tmp2.val[0]));
        int32x4_t vqs32_c20_03 = neon::vaddl(neon::vgetlow(v2q16_cx0_tmp2.val[0]), neon::vgetlow(v2q16_cx1_tmp2.val[1]));
        int32x4_t vqs32_c21_12 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp2.val[1]), neon::vgetlow(v2q16_n0x1_tmp2.val[0]));
        int32x4_t vqs32_c21_03 = neon::vaddl(neon::vgetlow(v2q16_n0x0_tmp2.val[0]), neon::vgetlow(v2q16_n0x1_tmp2.val[1]));
        int32x4_t vqs32_c22_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp2.val[1]), neon::vgetlow(v2q16_n1x1_tmp2.val[0]));
        int32x4_t vqs32_c22_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp2.val[0]), neon::vgetlow(v2q16_n1x1_tmp2.val[1]));
        int32x4_t vqs32_c23_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp2.val[1]), neon::vgetlow(v2q16_n2x1_tmp2.val[0]));
        int32x4_t vqs32_c23_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp2.val[0]), neon::vgetlow(v2q16_n2x1_tmp2.val[1]));

        int32x4_t vqs32_c00_x19 = neon::vmul(vqs32_c00_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c01_x19 = neon::vmul(vqs32_c01_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c02_x19 = neon::vmul(vqs32_c02_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c03_x19 = neon::vmul(vqs32_c03_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c10_x19 = neon::vmul(vqs32_c10_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c11_x19 = neon::vmul(vqs32_c11_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c12_x19 = neon::vmul(vqs32_c12_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c13_x19 = neon::vmul(vqs32_c13_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c20_x19 = neon::vmul(vqs32_c20_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c21_x19 = neon::vmul(vqs32_c21_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c22_x19 = neon::vmul(vqs32_c22_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_c23_x19 = neon::vmul(vqs32_c23_12, static_cast<MI_S32>(19));

        int32x4x3_t v3qs32_result0, v3qs32_result1, v3qs32_result2, v3qs32_result3;
        v3qs32_result0.val[0] = neon::vmls(vqs32_c00_x19, vqs32_c00_03, static_cast<MI_S32>(3));
        v3qs32_result1.val[0] = neon::vmls(vqs32_c01_x19, vqs32_c01_03, static_cast<MI_S32>(3));
        v3qs32_result2.val[0] = neon::vmls(vqs32_c02_x19, vqs32_c02_03, static_cast<MI_S32>(3));
        v3qs32_result3.val[0] = neon::vmls(vqs32_c03_x19, vqs32_c03_03, static_cast<MI_S32>(3));
        v3qs32_result0.val[1] = neon::vmls(vqs32_c10_x19, vqs32_c10_03, static_cast<MI_S32>(3));
        v3qs32_result1.val[1] = neon::vmls(vqs32_c11_x19, vqs32_c11_03, static_cast<MI_S32>(3));
        v3qs32_result2.val[1] = neon::vmls(vqs32_c12_x19, vqs32_c12_03, static_cast<MI_S32>(3));
        v3qs32_result3.val[1] = neon::vmls(vqs32_c13_x19, vqs32_c13_03, static_cast<MI_S32>(3));
        v3qs32_result0.val[2] = neon::vmls(vqs32_c20_x19, vqs32_c20_03, static_cast<MI_S32>(3));
        v3qs32_result1.val[2] = neon::vmls(vqs32_c21_x19, vqs32_c21_03, static_cast<MI_S32>(3));
        v3qs32_result2.val[2] = neon::vmls(vqs32_c22_x19, vqs32_c22_03, static_cast<MI_S32>(3));
        v3qs32_result3.val[2] = neon::vmls(vqs32_c23_x19, vqs32_c23_03, static_cast<MI_S32>(3));

        neon::vstore(rows0_x, v3qs32_result0);
        neon::vstore(rows1_x, v3qs32_result1);
        neon::vstore(rows2_x, v3qs32_result2);
        neon::vstore(rows3_x, v3qs32_result3);

        rows0_x += 12;
        rows1_x += 12;
        rows2_x += 12;
        rows3_x += 12;

        src_c  += 24;
        src_n0 += 24;
        src_n1 += 24;
        src_n2 += 24;
    }

    for (; dx < (owidth - 2); dx++)
    {
        rows0_x[0] = src_c[3] * 19 - src_c[0] * 3 + src_c[6] * 19 - src_c[9] * 3;
        rows1_x[0] = src_n0[3] * 19 - src_n0[0] * 3 + src_n0[6] * 19 - src_n0[9] * 3;
        rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 19 - src_n1[9] * 3;
        rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 19 - src_n2[9] * 3;
        rows0_x[1] = src_c[4] * 19 - src_c[1] * 3 + src_c[7] * 19 - src_c[10] * 3;
        rows1_x[1] = src_n0[4] * 19 - src_n0[1] * 3 + src_n0[7] * 19 - src_n0[10] * 3;
        rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 19 - src_n1[10] * 3;
        rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 19 - src_n2[10] * 3;
        rows0_x[2] = src_c[5] * 19 - src_c[2] * 3 + src_c[8] * 19 - src_c[11] * 3;
        rows1_x[2] = src_n0[5] * 19 - src_n0[2] * 3 + src_n0[8] * 19 - src_n0[11] * 3;
        rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 19 - src_n1[11] * 3;
        rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 19 - src_n2[11] * 3;

        rows0_x += channel;
        rows1_x += channel;
        rows2_x += channel;
        rows3_x += channel;
        src_c  += 2 * channel;
        src_n0 += 2 * channel;
        src_n1 += 2 * channel;
        src_n2 += 2 * channel;
    }

    rows0_x[0] = src_c[3] * 19 - src_c[0] * 3 + src_c[6] * 16;
    rows1_x[0] = src_n0[3] * 19 - src_n0[0] * 3 + src_n0[6] * 16;
    rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 16;
    rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 16;
    rows0_x[1] = src_c[4] * 19 - src_c[1] * 3 + src_c[7] * 16;
    rows1_x[1] = src_n0[4] * 19 - src_n0[1] * 3 + src_n0[7] * 16;
    rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 16;
    rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 16;
    rows0_x[2] = src_c[5] * 19 - src_c[2] * 3 + src_c[8] * 16;
    rows1_x[2] = src_n0[5] * 19 - src_n0[2] * 3 + src_n0[8] * 16;
    rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 16;
    rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 16;

    // vresize
    MI_S32 *rows0_y = rows0;
    MI_S32 *rows1_y = rows1;
    MI_S32 *rows2_y = rows2;
    MI_S32 *rows3_y = rows3;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    MI_S32 owidth_align8 = owidth & (-8);
    dx = 0;
    for (; dx < owidth_align8; dx += 8)
    {
        int32x4x3_t v3qs32_cx0_lo  = neon::vload3q(rows0_y);
        int32x4x3_t v3qs32_n0x0_lo = neon::vload3q(rows1_y);
        int32x4x3_t v3qs32_n1x0_lo = neon::vload3q(rows2_y);
        int32x4x3_t v3qs32_n2x0_lo = neon::vload3q(rows3_y);
        int32x4x3_t v3qs32_cx1_hi  = neon::vload3q(rows0_y + 12);
        int32x4x3_t v3qs32_n0x1_hi = neon::vload3q(rows1_y + 12);
        int32x4x3_t v3qs32_n1x1_hi = neon::vload3q(rows2_y + 12);
        int32x4x3_t v3qs32_n2x1_hi = neon::vload3q(rows3_y + 12);

        int32x4_t vqs32_0_lo12 = neon::vadd(v3qs32_n0x0_lo.val[0], v3qs32_n1x0_lo.val[0]);
        int32x4_t vqs32_0_lo03 = neon::vadd(v3qs32_cx0_lo.val[0], v3qs32_n2x0_lo.val[0]);
        int32x4_t vqs32_1_lo12 = neon::vadd(v3qs32_n0x0_lo.val[1], v3qs32_n1x0_lo.val[1]);
        int32x4_t vqs32_1_lo03 = neon::vadd(v3qs32_cx0_lo.val[1], v3qs32_n2x0_lo.val[1]);
        int32x4_t vqs32_2_lo12 = neon::vadd(v3qs32_n0x0_lo.val[2], v3qs32_n1x0_lo.val[2]);
        int32x4_t vqs32_2_lo03 = neon::vadd(v3qs32_cx0_lo.val[2], v3qs32_n2x0_lo.val[2]);
        int32x4_t vqs32_0_hi12 = neon::vadd(v3qs32_n0x1_hi.val[0], v3qs32_n1x1_hi.val[0]);
        int32x4_t vqs32_0_hi03 = neon::vadd(v3qs32_cx1_hi.val[0], v3qs32_n2x1_hi.val[0]);
        int32x4_t vqs32_1_hi12 = neon::vadd(v3qs32_n0x1_hi.val[1], v3qs32_n1x1_hi.val[1]);
        int32x4_t vqs32_1_hi03 = neon::vadd(v3qs32_cx1_hi.val[1], v3qs32_n2x1_hi.val[1]);
        int32x4_t vqs32_2_hi12 = neon::vadd(v3qs32_n0x1_hi.val[2], v3qs32_n1x1_hi.val[2]);
        int32x4_t vqs32_2_hi03 = neon::vadd(v3qs32_cx1_hi.val[2], v3qs32_n2x1_hi.val[2]);

        int32x4_t vqs32_lo0_x19 = neon::vmul(vqs32_0_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi0_x19 = neon::vmul(vqs32_0_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_lo1_x19 = neon::vmul(vqs32_1_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi1_x19 = neon::vmul(vqs32_1_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_lo2_x19 = neon::vmul(vqs32_2_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi2_x19 = neon::vmul(vqs32_2_hi12, static_cast<MI_S32>(19));
        int32x4_t vqs32_des0_lo = neon::vmls(vqs32_lo0_x19, vqs32_0_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des0_hi = neon::vmls(vqs32_hi0_x19, vqs32_0_hi03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des1_lo = neon::vmls(vqs32_lo1_x19, vqs32_1_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des1_hi = neon::vmls(vqs32_hi1_x19, vqs32_1_hi03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des2_lo = neon::vmls(vqs32_lo2_x19, vqs32_2_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des2_hi = neon::vmls(vqs32_hi2_x19, vqs32_2_hi03, static_cast<MI_S32>(3));

        MVType mvq16_result;
        if (std::is_same<Tp, MI_U16>::value)
        {
            int32x4_t vqs32_zero;
            neon::vdup(vqs32_zero, static_cast<MI_S32>(0));
            vqs32_des0_lo = neon::vmax(vqs32_des0_lo, vqs32_zero);
            vqs32_des0_hi = neon::vmax(vqs32_des0_hi, vqs32_zero);
            vqs32_des1_lo = neon::vmax(vqs32_des1_lo, vqs32_zero);
            vqs32_des1_hi = neon::vmax(vqs32_des1_hi, vqs32_zero);
            vqs32_des2_lo = neon::vmax(vqs32_des2_lo, vqs32_zero);
            vqs32_des2_hi = neon::vmax(vqs32_des2_hi, vqs32_zero);

            uint32x4_t vdu32_des0_lo = neon::vreinterpret(vqs32_des0_lo);
            uint32x4_t vdu32_des0_hi = neon::vreinterpret(vqs32_des0_hi);
            uint32x4_t vdu32_des1_lo = neon::vreinterpret(vqs32_des1_lo);
            uint32x4_t vdu32_des1_hi = neon::vreinterpret(vqs32_des1_hi);
            uint32x4_t vdu32_des2_lo = neon::vreinterpret(vqs32_des2_lo);
            uint32x4_t vdu32_des2_hi = neon::vreinterpret(vqs32_des2_hi);

            uint16x4_t vdu16_des0_lo = neon::vqshrn_n<10>(vdu32_des0_lo);
            uint16x4_t vdu16_des0_hi = neon::vqshrn_n<10>(vdu32_des0_hi);
            uint16x4_t vdu16_des1_lo = neon::vqshrn_n<10>(vdu32_des1_lo);
            uint16x4_t vdu16_des1_hi = neon::vqshrn_n<10>(vdu32_des1_hi);
            uint16x4_t vdu16_des2_lo = neon::vqshrn_n<10>(vdu32_des2_lo);
            uint16x4_t vdu16_des2_hi = neon::vqshrn_n<10>(vdu32_des2_hi);

            mvq16_result.val[0] = neon::vcombine(vdu16_des0_lo, vdu16_des0_hi);
            mvq16_result.val[1] = neon::vcombine(vdu16_des1_lo, vdu16_des1_hi);
            mvq16_result.val[2] = neon::vcombine(vdu16_des2_lo, vdu16_des2_hi);

            neon::vstore(dst_row, mvq16_result);
        }
        else
        {
            int16x4_t vds16_des0_lo = neon::vqshrn_n<10>(vqs32_des0_lo);
            int16x4_t vds16_des0_hi = neon::vqshrn_n<10>(vqs32_des0_hi);
            int16x4_t vds16_des1_lo = neon::vqshrn_n<10>(vqs32_des1_lo);
            int16x4_t vds16_des1_hi = neon::vqshrn_n<10>(vqs32_des1_hi);
            int16x4_t vds16_des2_lo = neon::vqshrn_n<10>(vqs32_des2_lo);
            int16x4_t vds16_des2_hi = neon::vqshrn_n<10>(vqs32_des2_hi);

            mvq16_result.val[0] = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
            mvq16_result.val[1] = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
            mvq16_result.val[2] = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

            neon::vstore(dst_row, mvq16_result);
        }

        dst_row += 24;
        rows0_y += 24;
        rows1_y += 24;
        rows2_y += 24;
        rows3_y += 24;
    }

    for (; dx < owidth; dx++)
    {
        MI_S32 result = (rows1_y[0] * 19 - rows0_y[0] * 3 + rows2_y[0] * 19 - rows3_y[0] * 3 + 512) >> 10;
        dst_row[0]    = SaturateCast<Tp>(result);
        result        = (rows1_y[1] * 19 - rows0_y[1] * 3 + rows2_y[1] * 19 - rows3_y[1] * 3 + 512) >> 10;
        dst_row[1]    = SaturateCast<Tp>(result);
        result        = (rows1_y[2] * 19 - rows0_y[2] * 3 + rows2_y[2] * 19 - rows3_y[2] * 3 + 512) >> 10;
        dst_row[2]    = SaturateCast<Tp>(result);

        dst_row += channel;
        rows0_y += channel;
        rows1_y += channel;
        rows2_y += channel;
        rows3_y += channel;
    }

    // Line 1 ~ h-1
    for (MI_S32 dy = (start_row + 1); dy < end_row; dy++)
    {
        MI_S32 sy = (dy << 1) - 1;

        // hresize two row
        MI_S32 *rows0_tmp = rows0;
        MI_S32 *rows1_tmp = rows1;
        rows0             = rows2;
        rows1             = rows3;
        rows2             = rows0_tmp;
        rows3             = rows1_tmp;

        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        MI_S32 *rows2_x  = rows2;
        MI_S32 *rows3_x  = rows3;

        if ((end_row == oheight) && (dy == (end_row - 1)))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 16 + src_n1[3] * 19 - src_n1[6] * 3;
        rows3_x[0] = src_n2[0] * 16 + src_n2[3] * 19 - src_n2[6] * 3;
        rows2_x[1] = src_n1[1] * 16 + src_n1[4] * 19 - src_n1[7] * 3;
        rows3_x[1] = src_n2[1] * 16 + src_n2[4] * 19 - src_n2[7] * 3;
        rows2_x[2] = src_n1[2] * 16 + src_n1[5] * 19 - src_n1[8] * 3;
        rows3_x[2] = src_n2[2] * 16 + src_n2[5] * 19 - src_n2[8] * 3;

        src_n1  += channel;
        src_n2  += channel;
        rows2_x += channel;
        rows3_x += channel;

        MI_S32 owidth_align4 = (owidth - 2) & (-4);
        MI_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            MVType mvq16_n1x0 = neon::vload3q(src_n1);
            MVType mvq16_n2x0 = neon::vload3q(src_n2);
            MVType mvq16_n1x1 = neon::vload3q(src_n1 + 2 * channel);
            MVType mvq16_n2x1 = neon::vload3q(src_n2 + 2 * channel);

            auto v2q16_n1x0_tmp0 = neon::vuzp(mvq16_n1x0.val[0], mvq16_n1x0.val[0]);
            auto v2q16_n2x0_tmp0 = neon::vuzp(mvq16_n2x0.val[0], mvq16_n2x0.val[0]);
            auto v2q16_n1x1_tmp0 = neon::vuzp(mvq16_n1x1.val[0], mvq16_n1x1.val[0]);
            auto v2q16_n2x1_tmp0 = neon::vuzp(mvq16_n2x1.val[0], mvq16_n2x1.val[0]);
            auto v2q16_n1x0_tmp1 = neon::vuzp(mvq16_n1x0.val[1], mvq16_n1x0.val[1]);
            auto v2q16_n2x0_tmp1 = neon::vuzp(mvq16_n2x0.val[1], mvq16_n2x0.val[1]);
            auto v2q16_n1x1_tmp1 = neon::vuzp(mvq16_n1x1.val[1], mvq16_n1x1.val[1]);
            auto v2q16_n2x1_tmp1 = neon::vuzp(mvq16_n2x1.val[1], mvq16_n2x1.val[1]);
            auto v2q16_n1x0_tmp2 = neon::vuzp(mvq16_n1x0.val[2], mvq16_n1x0.val[2]);
            auto v2q16_n2x0_tmp2 = neon::vuzp(mvq16_n2x0.val[2], mvq16_n2x0.val[2]);
            auto v2q16_n1x1_tmp2 = neon::vuzp(mvq16_n1x1.val[2], mvq16_n1x1.val[2]);
            auto v2q16_n2x1_tmp2 = neon::vuzp(mvq16_n2x1.val[2], mvq16_n2x1.val[2]);

            int32x4_t vqs32_c02_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp0.val[1]), neon::vgetlow(v2q16_n1x1_tmp0.val[0]));
            int32x4_t vqs32_c02_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp0.val[0]), neon::vgetlow(v2q16_n1x1_tmp0.val[1]));
            int32x4_t vqs32_c03_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp0.val[1]), neon::vgetlow(v2q16_n2x1_tmp0.val[0]));
            int32x4_t vqs32_c03_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp0.val[0]), neon::vgetlow(v2q16_n2x1_tmp0.val[1]));
            int32x4_t vqs32_c12_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp1.val[1]), neon::vgetlow(v2q16_n1x1_tmp1.val[0]));
            int32x4_t vqs32_c12_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp1.val[0]), neon::vgetlow(v2q16_n1x1_tmp1.val[1]));
            int32x4_t vqs32_c13_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp1.val[1]), neon::vgetlow(v2q16_n2x1_tmp1.val[0]));
            int32x4_t vqs32_c13_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp1.val[0]), neon::vgetlow(v2q16_n2x1_tmp1.val[1]));
            int32x4_t vqs32_c22_12 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp2.val[1]), neon::vgetlow(v2q16_n1x1_tmp2.val[0]));
            int32x4_t vqs32_c22_03 = neon::vaddl(neon::vgetlow(v2q16_n1x0_tmp2.val[0]), neon::vgetlow(v2q16_n1x1_tmp2.val[1]));
            int32x4_t vqs32_c23_12 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp2.val[1]), neon::vgetlow(v2q16_n2x1_tmp2.val[0]));
            int32x4_t vqs32_c23_03 = neon::vaddl(neon::vgetlow(v2q16_n2x0_tmp2.val[0]), neon::vgetlow(v2q16_n2x1_tmp2.val[1]));

            int32x4_t vqs32_c02_x19 = neon::vmul(vqs32_c02_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c03_x19 = neon::vmul(vqs32_c03_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c12_x19 = neon::vmul(vqs32_c12_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c13_x19 = neon::vmul(vqs32_c13_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c22_x19 = neon::vmul(vqs32_c22_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c23_x19 = neon::vmul(vqs32_c23_12, static_cast<MI_S32>(19));

            int32x4x3_t v3qs32_result2, v3qs32_result3;
            v3qs32_result2.val[0] = neon::vmls(vqs32_c02_x19, vqs32_c02_03, static_cast<MI_S32>(3));
            v3qs32_result3.val[0] = neon::vmls(vqs32_c03_x19, vqs32_c03_03, static_cast<MI_S32>(3));
            v3qs32_result2.val[1] = neon::vmls(vqs32_c12_x19, vqs32_c12_03, static_cast<MI_S32>(3));
            v3qs32_result3.val[1] = neon::vmls(vqs32_c13_x19, vqs32_c13_03, static_cast<MI_S32>(3));
            v3qs32_result2.val[2] = neon::vmls(vqs32_c22_x19, vqs32_c22_03, static_cast<MI_S32>(3));
            v3qs32_result3.val[2] = neon::vmls(vqs32_c23_x19, vqs32_c23_03, static_cast<MI_S32>(3));

            // vst MI_S32
            neon::vstore(rows2_x, v3qs32_result2);
            neon::vstore(rows3_x, v3qs32_result3);

            rows2_x += 12;
            rows3_x += 12;
            src_n1  += 24;
            src_n2  += 24;
        }

        for (; dx < (owidth - 2); dx++)
        {
            rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 19 - src_n1[9] * 3;
            rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 19 - src_n2[9] * 3;
            rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 19 - src_n1[10] * 3;
            rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 19 - src_n2[10] * 3;
            rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 19 - src_n1[11] * 3;
            rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 19 - src_n2[11] * 3;

            rows2_x += channel;
            rows3_x += channel;
            src_n1  += 2 * channel;
            src_n2  += 2 * channel;
        }

        rows2_x[0] = src_n1[3] * 19 - src_n1[0] * 3 + src_n1[6] * 16;
        rows3_x[0] = src_n2[3] * 19 - src_n2[0] * 3 + src_n2[6] * 16;
        rows2_x[1] = src_n1[4] * 19 - src_n1[1] * 3 + src_n1[7] * 16;
        rows3_x[1] = src_n2[4] * 19 - src_n2[1] * 3 + src_n2[7] * 16;
        rows2_x[2] = src_n1[5] * 19 - src_n1[2] * 3 + src_n1[8] * 16;
        rows3_x[2] = src_n2[5] * 19 - src_n2[2] * 3 + src_n2[8] * 16;

        // vresize
        MI_S32 *rows0_y = rows0;
        MI_S32 *rows1_y = rows1;
        MI_S32 *rows2_y = rows2;
        MI_S32 *rows3_y = rows3;

        Tp *dst_row = dst.Ptr<Tp>(dy);

        MI_S32 owidth_align8 = owidth & (-8);
        dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            int32x4x3_t v3qs32_cx0_lo  = neon::vload3q(rows0_y);
            int32x4x3_t v3qs32_n0x0_lo = neon::vload3q(rows1_y);
            int32x4x3_t v3qs32_n1x0_lo = neon::vload3q(rows2_y);
            int32x4x3_t v3qs32_n2x0_lo = neon::vload3q(rows3_y);
            int32x4x3_t v3qs32_cx1_hi  = neon::vload3q(rows0_y + 4 * channel);
            int32x4x3_t v3qs32_n0x1_hi = neon::vload3q(rows1_y + 4 * channel);
            int32x4x3_t v3qs32_n1x1_hi = neon::vload3q(rows2_y + 4 * channel);
            int32x4x3_t v3qs32_n2x1_hi = neon::vload3q(rows3_y + 4 * channel);

            int32x4_t vqs32_0_lo12 = neon::vadd(v3qs32_n0x0_lo.val[0], v3qs32_n1x0_lo.val[0]);
            int32x4_t vqs32_0_lo03 = neon::vadd(v3qs32_cx0_lo.val[0], v3qs32_n2x0_lo.val[0]);
            int32x4_t vqs32_1_lo12 = neon::vadd(v3qs32_n0x0_lo.val[1], v3qs32_n1x0_lo.val[1]);
            int32x4_t vqs32_1_lo03 = neon::vadd(v3qs32_cx0_lo.val[1], v3qs32_n2x0_lo.val[1]);
            int32x4_t vqs32_2_lo12 = neon::vadd(v3qs32_n0x0_lo.val[2], v3qs32_n1x0_lo.val[2]);
            int32x4_t vqs32_2_lo03 = neon::vadd(v3qs32_cx0_lo.val[2], v3qs32_n2x0_lo.val[2]);
            int32x4_t vqs32_0_hi12 = neon::vadd(v3qs32_n0x1_hi.val[0], v3qs32_n1x1_hi.val[0]);
            int32x4_t vqs32_0_hi03 = neon::vadd(v3qs32_cx1_hi.val[0], v3qs32_n2x1_hi.val[0]);
            int32x4_t vqs32_1_hi12 = neon::vadd(v3qs32_n0x1_hi.val[1], v3qs32_n1x1_hi.val[1]);
            int32x4_t vqs32_1_hi03 = neon::vadd(v3qs32_cx1_hi.val[1], v3qs32_n2x1_hi.val[1]);
            int32x4_t vqs32_2_hi12 = neon::vadd(v3qs32_n0x1_hi.val[2], v3qs32_n1x1_hi.val[2]);
            int32x4_t vqs32_2_hi03 = neon::vadd(v3qs32_cx1_hi.val[2], v3qs32_n2x1_hi.val[2]);

            int32x4_t vqs32_lo0_x19 = neon::vmul(vqs32_0_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi0_x19 = neon::vmul(vqs32_0_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_lo1_x19 = neon::vmul(vqs32_1_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi1_x19 = neon::vmul(vqs32_1_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_lo2_x19 = neon::vmul(vqs32_2_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi2_x19 = neon::vmul(vqs32_2_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_des0_lo = neon::vmls(vqs32_lo0_x19, vqs32_0_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des0_hi = neon::vmls(vqs32_hi0_x19, vqs32_0_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des1_lo = neon::vmls(vqs32_lo1_x19, vqs32_1_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des1_hi = neon::vmls(vqs32_hi1_x19, vqs32_1_hi03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des2_lo = neon::vmls(vqs32_lo2_x19, vqs32_2_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des2_hi = neon::vmls(vqs32_hi2_x19, vqs32_2_hi03, static_cast<MI_S32>(3));

            MVType mvq16_result;
            if (std::is_same<Tp, MI_U16>::value)
            {
                int32x4_t vqs32_zero;
                neon::vdup(vqs32_zero, static_cast<MI_S32>(0));
                vqs32_des0_lo = neon::vmax(vqs32_des0_lo, vqs32_zero);
                vqs32_des0_hi = neon::vmax(vqs32_des0_hi, vqs32_zero);
                vqs32_des1_lo = neon::vmax(vqs32_des1_lo, vqs32_zero);
                vqs32_des1_hi = neon::vmax(vqs32_des1_hi, vqs32_zero);
                vqs32_des2_lo = neon::vmax(vqs32_des2_lo, vqs32_zero);
                vqs32_des2_hi = neon::vmax(vqs32_des2_hi, vqs32_zero);

                uint32x4_t vdu32_des0_lo = neon::vreinterpret(vqs32_des0_lo);
                uint32x4_t vdu32_des0_hi = neon::vreinterpret(vqs32_des0_hi);
                uint16x4_t vdu16_des0_lo = neon::vqshrn_n<10>(vdu32_des0_lo);
                uint16x4_t vdu16_des0_hi = neon::vqshrn_n<10>(vdu32_des0_hi);

                uint32x4_t vdu32_des1_lo = neon::vreinterpret(vqs32_des1_lo);
                uint32x4_t vdu32_des1_hi = neon::vreinterpret(vqs32_des1_hi);
                uint16x4_t vdu16_des1_lo = neon::vqshrn_n<10>(vdu32_des1_lo);
                uint16x4_t vdu16_des1_hi = neon::vqshrn_n<10>(vdu32_des1_hi);

                uint32x4_t vdu32_des2_lo = neon::vreinterpret(vqs32_des2_lo);
                uint32x4_t vdu32_des2_hi = neon::vreinterpret(vqs32_des2_hi);
                uint16x4_t vdu16_des2_lo = neon::vqshrn_n<10>(vdu32_des2_lo);
                uint16x4_t vdu16_des2_hi = neon::vqshrn_n<10>(vdu32_des2_hi);

                mvq16_result.val[0] = neon::vcombine(vdu16_des0_lo, vdu16_des0_hi);
                mvq16_result.val[1] = neon::vcombine(vdu16_des1_lo, vdu16_des1_hi);
                mvq16_result.val[2] = neon::vcombine(vdu16_des2_lo, vdu16_des2_hi);

                neon::vstore(dst_row, mvq16_result);
            }
            else
            {
                int16x4_t vds16_des0_lo = neon::vqshrn_n<10>(vqs32_des0_lo);
                int16x4_t vds16_des0_hi = neon::vqshrn_n<10>(vqs32_des0_hi);
                int16x4_t vds16_des1_lo = neon::vqshrn_n<10>(vqs32_des1_lo);
                int16x4_t vds16_des1_hi = neon::vqshrn_n<10>(vqs32_des1_hi);
                int16x4_t vds16_des2_lo = neon::vqshrn_n<10>(vqs32_des2_lo);
                int16x4_t vds16_des2_hi = neon::vqshrn_n<10>(vqs32_des2_hi);

                mvq16_result.val[0] = neon::vcombine(vds16_des0_lo, vds16_des0_hi);
                mvq16_result.val[1] = neon::vcombine(vds16_des1_lo, vds16_des1_hi);
                mvq16_result.val[2] = neon::vcombine(vds16_des2_lo, vds16_des2_hi);

                neon::vstore(dst_row, mvq16_result);
            }

            dst_row += 24;
            rows0_y += 24;
            rows1_y += 24;
            rows2_y += 24;
            rows3_y += 24;
        }

        for (; dx < owidth; dx++)
        {
            MI_S32 result = (rows1_y[0] * 19 - rows0_y[0] * 3 + rows2_y[0] * 19 - rows3_y[0] * 3 + 512) >> 10;
            dst_row[0]    = SaturateCast<Tp>(result);
            result        = (rows1_y[1] * 19 - rows0_y[1] * 3 + rows2_y[1] * 19 - rows3_y[1] * 3 + 512) >> 10;
            dst_row[1]    = SaturateCast<Tp>(result);
            result        = (rows1_y[2] * 19 - rows0_y[2] * 3 + rows2_y[2] * 19 - rows3_y[2] * 3 + 512) >> 10;
            dst_row[2]    = SaturateCast<Tp>(result);

            dst_row += channel;
            rows0_y += channel;
            rows1_y += channel;
            rows2_y += channel;
            rows3_y += channel;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeCuC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MQVector<Tp, 3>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_F32 coef0 = -0.093750;
    MI_F32 coef1 = 0.593750;

    MI_F32 *rows = thread_buffer.GetThreadData<MI_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_F32 *rows0 = rows;
    MI_F32 *rows1 = rows0 + owidth * channel;
    MI_F32 *rows2 = rows1 + owidth * channel;
    MI_F32 *rows3 = rows2 + owidth * channel;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_F32 *rows0_x  = rows0;
    MI_F32 *rows1_x  = rows1;
    MI_F32 *rows2_x  = rows2;
    MI_F32 *rows3_x  = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 0.5f + src_c[3] * coef1 + src_c[6] * coef0;
    rows1_x[0] = src_n0[0] * 0.5f + src_n0[3] * coef1 + src_n0[6] * coef0;
    rows2_x[0] = src_n1[0] * 0.5f + src_n1[3] * coef1 + src_n1[6] * coef0;
    rows3_x[0] = src_n2[0] * 0.5f + src_n2[3] * coef1 + src_n2[6] * coef0;
    rows0_x[1] = src_c[1] * 0.5f + src_c[4] * coef1 + src_c[7] * coef0;
    rows1_x[1] = src_n0[1] * 0.5f + src_n0[4] * coef1 + src_n0[7] * coef0;
    rows2_x[1] = src_n1[1] * 0.5f + src_n1[4] * coef1 + src_n1[7] * coef0;
    rows3_x[1] = src_n2[1] * 0.5f + src_n2[4] * coef1 + src_n2[7] * coef0;
    rows0_x[2] = src_c[2] * 0.5f + src_c[5] * coef1 + src_c[8] * coef0;
    rows1_x[2] = src_n0[2] * 0.5f + src_n0[5] * coef1 + src_n0[8] * coef0;
    rows2_x[2] = src_n1[2] * 0.5f + src_n1[5] * coef1 + src_n1[8] * coef0;
    rows3_x[2] = src_n2[2] * 0.5f + src_n2[5] * coef1 + src_n2[8] * coef0;

    src_c  += channel;
    src_n0 += channel;
    src_n1 += channel;
    src_n2 += channel;

    rows0_x += channel;
    rows1_x += channel;
    rows2_x += channel;
    rows3_x += channel;

    MI_S32 owidth_align4 = (owidth - 2) & (-4);
    MI_S32 dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        MVType mvqf16_cx0  = neon::vload3q(src_c);
        MVType mvqf16_n0x0 = neon::vload3q(src_n0);
        MVType mvqf16_n1x0 = neon::vload3q(src_n1);
        MVType mvqf16_n2x0 = neon::vload3q(src_n2);
        MVType mvqf16_cx1  = neon::vload3q(src_c + 2 * channel);
        MVType mvqf16_n0x1 = neon::vload3q(src_n0 + 2 * channel);
        MVType mvqf16_n1x1 = neon::vload3q(src_n1 + 2 * channel);
        MVType mvqf16_n2x1 = neon::vload3q(src_n2 + 2 * channel);

        float16x8x2_t v2qf16_cx0_tmp0  = neon::vuzp(mvqf16_cx0.val[0],   mvqf16_cx0.val[0]);
        float16x8x2_t v2qf16_n0x0_tmp0 = neon::vuzp(mvqf16_n0x0.val[0],  mvqf16_n0x0.val[0]);
        float16x8x2_t v2qf16_n1x0_tmp0 = neon::vuzp(mvqf16_n1x0.val[0],  mvqf16_n1x0.val[0]);
        float16x8x2_t v2qf16_n2x0_tmp0 = neon::vuzp(mvqf16_n2x0.val[0],  mvqf16_n2x0.val[0]);
        float16x8x2_t v2qf16_cx1_tmp0  = neon::vuzp(mvqf16_cx1.val[0],  mvqf16_cx1.val[0]);
        float16x8x2_t v2qf16_n0x1_tmp0 = neon::vuzp(mvqf16_n0x1.val[0], mvqf16_n0x1.val[0]);
        float16x8x2_t v2qf16_n1x1_tmp0 = neon::vuzp(mvqf16_n1x1.val[0], mvqf16_n1x1.val[0]);
        float16x8x2_t v2qf16_n2x1_tmp0 = neon::vuzp(mvqf16_n2x1.val[0], mvqf16_n2x1.val[0]);

        float16x8x2_t v2qf16_cx0_tmp1  = neon::vuzp(mvqf16_cx0.val[1],   mvqf16_cx0.val[1]);
        float16x8x2_t v2qf16_n0x0_tmp1 = neon::vuzp(mvqf16_n0x0.val[1],  mvqf16_n0x0.val[1]);
        float16x8x2_t v2qf16_n1x0_tmp1 = neon::vuzp(mvqf16_n1x0.val[1],  mvqf16_n1x0.val[1]);
        float16x8x2_t v2qf16_n2x0_tmp1 = neon::vuzp(mvqf16_n2x0.val[1],  mvqf16_n2x0.val[1]);
        float16x8x2_t v2qf16_cx1_tmp1  = neon::vuzp(mvqf16_cx1.val[1],  mvqf16_cx1.val[1]);
        float16x8x2_t v2qf16_n0x1_tmp1 = neon::vuzp(mvqf16_n0x1.val[1], mvqf16_n0x1.val[1]);
        float16x8x2_t v2qf16_n1x1_tmp1 = neon::vuzp(mvqf16_n1x1.val[1], mvqf16_n1x1.val[1]);
        float16x8x2_t v2qf16_n2x1_tmp1 = neon::vuzp(mvqf16_n2x1.val[1], mvqf16_n2x1.val[1]);

        float16x8x2_t v2qf16_cx0_tmp2  = neon::vuzp(mvqf16_cx0.val[2],   mvqf16_cx0.val[2]);
        float16x8x2_t v2qf16_n0x0_tmp2 = neon::vuzp(mvqf16_n0x0.val[2],  mvqf16_n0x0.val[2]);
        float16x8x2_t v2qf16_n1x0_tmp2 = neon::vuzp(mvqf16_n1x0.val[2],  mvqf16_n1x0.val[2]);
        float16x8x2_t v2qf16_n2x0_tmp2 = neon::vuzp(mvqf16_n2x0.val[2],  mvqf16_n2x0.val[2]);
        float16x8x2_t v2qf16_cx1_tmp2  = neon::vuzp(mvqf16_cx1.val[2],  mvqf16_cx1.val[2]);
        float16x8x2_t v2qf16_n0x1_tmp2 = neon::vuzp(mvqf16_n0x1.val[2], mvqf16_n0x1.val[2]);
        float16x8x2_t v2qf16_n1x1_tmp2 = neon::vuzp(mvqf16_n1x1.val[2], mvqf16_n1x1.val[2]);
        float16x8x2_t v2qf16_n2x1_tmp2 = neon::vuzp(mvqf16_n2x1.val[2], mvqf16_n2x1.val[2]);

        float32x4_t vqf32_c00_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp0.val[1])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp0.val[0])));
        float32x4_t vqf32_c00_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp0.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp0.val[1])));
        float32x4_t vqf32_c01_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp0.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp0.val[0])));
        float32x4_t vqf32_c01_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp0.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp0.val[1])));
        float32x4_t vqf32_c02_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp0.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp0.val[0])));
        float32x4_t vqf32_c02_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp0.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp0.val[1])));
        float32x4_t vqf32_c03_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp0.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp0.val[0])));
        float32x4_t vqf32_c03_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp0.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp0.val[1])));

        float32x4_t vqf32_c10_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp1.val[1])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp1.val[0])));
        float32x4_t vqf32_c10_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp1.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp1.val[1])));
        float32x4_t vqf32_c11_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp1.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp1.val[0])));
        float32x4_t vqf32_c11_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp1.val[1])));
        float32x4_t vqf32_c12_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp1.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp1.val[0])));
        float32x4_t vqf32_c12_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp1.val[1])));
        float32x4_t vqf32_c13_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp1.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp1.val[0])));
        float32x4_t vqf32_c13_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp1.val[1])));

        float32x4_t vqf32_c20_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp2.val[1])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp2.val[0])));
        float32x4_t vqf32_c20_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx0_tmp2.val[0])),  neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_cx1_tmp2.val[1])));
        float32x4_t vqf32_c21_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp2.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp2.val[0])));
        float32x4_t vqf32_c21_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x0_tmp2.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n0x1_tmp2.val[1])));
        float32x4_t vqf32_c22_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp2.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp2.val[0])));
        float32x4_t vqf32_c22_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp2.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp2.val[1])));
        float32x4_t vqf32_c23_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp2.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp2.val[0])));
        float32x4_t vqf32_c23_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp2.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp2.val[1])));

        float32x4_t vqf32_c00_x19 = neon::vmul(vqf32_c00_12, coef1);
        float32x4_t vqf32_c01_x19 = neon::vmul(vqf32_c01_12, coef1);
        float32x4_t vqf32_c02_x19 = neon::vmul(vqf32_c02_12, coef1);
        float32x4_t vqf32_c03_x19 = neon::vmul(vqf32_c03_12, coef1);
        float32x4_t vqf32_c10_x19 = neon::vmul(vqf32_c10_12, coef1);
        float32x4_t vqf32_c11_x19 = neon::vmul(vqf32_c11_12, coef1);
        float32x4_t vqf32_c12_x19 = neon::vmul(vqf32_c12_12, coef1);
        float32x4_t vqf32_c13_x19 = neon::vmul(vqf32_c13_12, coef1);
        float32x4_t vqf32_c20_x19 = neon::vmul(vqf32_c20_12, coef1);
        float32x4_t vqf32_c21_x19 = neon::vmul(vqf32_c21_12, coef1);
        float32x4_t vqf32_c22_x19 = neon::vmul(vqf32_c22_12, coef1);
        float32x4_t vqf32_c23_x19 = neon::vmul(vqf32_c23_12, coef1);

        float32x4x3_t v3qf32_result0, v3qf32_result1, v3qf32_result2, v3qf32_result3;
        v3qf32_result0.val[0] = neon::vadd(vqf32_c00_x19, neon::vmul(vqf32_c00_03, coef0));
        v3qf32_result1.val[0] = neon::vadd(vqf32_c01_x19, neon::vmul(vqf32_c01_03, coef0));
        v3qf32_result2.val[0] = neon::vadd(vqf32_c02_x19, neon::vmul(vqf32_c02_03, coef0));
        v3qf32_result3.val[0] = neon::vadd(vqf32_c03_x19, neon::vmul(vqf32_c03_03, coef0));
        v3qf32_result0.val[1] = neon::vadd(vqf32_c10_x19, neon::vmul(vqf32_c10_03, coef0));
        v3qf32_result1.val[1] = neon::vadd(vqf32_c11_x19, neon::vmul(vqf32_c11_03, coef0));
        v3qf32_result2.val[1] = neon::vadd(vqf32_c12_x19, neon::vmul(vqf32_c12_03, coef0));
        v3qf32_result3.val[1] = neon::vadd(vqf32_c13_x19, neon::vmul(vqf32_c13_03, coef0));
        v3qf32_result0.val[2] = neon::vadd(vqf32_c20_x19, neon::vmul(vqf32_c20_03, coef0));
        v3qf32_result1.val[2] = neon::vadd(vqf32_c21_x19, neon::vmul(vqf32_c21_03, coef0));
        v3qf32_result2.val[2] = neon::vadd(vqf32_c22_x19, neon::vmul(vqf32_c22_03, coef0));
        v3qf32_result3.val[2] = neon::vadd(vqf32_c23_x19, neon::vmul(vqf32_c23_03, coef0));

        neon::vstore(rows0_x, v3qf32_result0);
        neon::vstore(rows1_x, v3qf32_result1);
        neon::vstore(rows2_x, v3qf32_result2);
        neon::vstore(rows3_x, v3qf32_result3);

        rows0_x += 4 * channel;
        rows1_x += 4 * channel;
        rows2_x += 4 * channel;
        rows3_x += 4 * channel;

        src_c  += 8 * channel;
        src_n0 += 8 * channel;
        src_n1 += 8 * channel;
        src_n2 += 8 * channel;
    }

    for (; dx < (owidth - 2); dx++)
    {
        rows0_x[0] = (src_c[3] + src_c[6]) * coef1 + (src_c[0] + src_c[9]) * coef0;
        rows1_x[0] = (src_n0[3] + src_n0[6]) * coef1 + (src_n0[0] + src_n0[9]) * coef0;
        rows2_x[0] = (src_n1[3] + src_n1[6]) * coef1 + (src_n1[0] + src_n1[9]) * coef0;
        rows3_x[0] = (src_n2[3] + src_n2[6]) * coef1 + (src_n2[0] + src_n2[9]) * coef0;
        rows0_x[1] = (src_c[4] + src_c[7]) * coef1 + (src_c[1] + src_c[10]) * coef0;
        rows1_x[1] = (src_n0[4] + src_n0[7]) * coef1 + (src_n0[1] + src_n0[10]) * coef0;
        rows2_x[1] = (src_n1[4] + src_n1[7]) * coef1 + (src_n1[1] + src_n1[10]) * coef0;
        rows3_x[1] = (src_n2[4] + src_n2[7]) * coef1 + (src_n2[1] + src_n2[10]) * coef0;
        rows0_x[2] = (src_c[5] + src_c[8]) * coef1 + (src_c[2] + src_c[11]) * coef0;
        rows1_x[2] = (src_n0[5] + src_n0[8]) * coef1 + (src_n0[2] + src_n0[11]) * coef0;
        rows2_x[2] = (src_n1[5] + src_n1[8]) * coef1 + (src_n1[2] + src_n1[11]) * coef0;
        rows3_x[2] = (src_n2[5] + src_n2[8]) * coef1 + (src_n2[2] + src_n2[11]) * coef0;

        rows0_x += channel;
        rows1_x += channel;
        rows2_x += channel;
        rows3_x += channel;

        src_c  += 2 * channel;
        src_n0 += 2 * channel;
        src_n1 += 2 * channel;
        src_n2 += 2 * channel;
    }

    rows0_x[0] = src_c[3] * coef1 + src_c[0] * coef0 + src_c[6] * 0.5f;
    rows1_x[0] = src_n0[3] * coef1 + src_n0[0] * coef0 + src_n0[6] * 0.5f;
    rows2_x[0] = src_n1[3] * coef1 + src_n1[0] * coef0 + src_n1[6] * 0.5f;
    rows3_x[0] = src_n2[3] * coef1 + src_n2[0] * coef0 + src_n2[6] * 0.5f;
    rows0_x[1] = src_c[4] * coef1 + src_c[1] * coef0 + src_c[7] * 0.5f;
    rows1_x[1] = src_n0[4] * coef1 + src_n0[1] * coef0 + src_n0[7] * 0.5f;
    rows2_x[1] = src_n1[4] * coef1 + src_n1[1] * coef0 + src_n1[7] * 0.5f;
    rows3_x[1] = src_n2[4] * coef1 + src_n2[1] * coef0 + src_n2[7] * 0.5f;
    rows0_x[2] = src_c[5] * coef1 + src_c[2] * coef0 + src_c[8] * 0.5f;
    rows1_x[2] = src_n0[5] * coef1 + src_n0[2] * coef0 + src_n0[8] * 0.5f;
    rows2_x[2] = src_n1[5] * coef1 + src_n1[2] * coef0 + src_n1[8] * 0.5f;
    rows3_x[2] = src_n2[5] * coef1 + src_n2[2] * coef0 + src_n2[8] * 0.5f;

    // vresize
    MI_F32 *rows0_y = rows0;
    MI_F32 *rows1_y = rows1;
    MI_F32 *rows2_y = rows2;
    MI_F32 *rows3_y = rows3;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    owidth_align4 = owidth & (-4);
    dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        float32x4x3_t v3qf32_c  = neon::vload3q(rows0_y);
        float32x4x3_t v3qf32_n0 = neon::vload3q(rows1_y);
        float32x4x3_t v3qf32_n1 = neon::vload3q(rows2_y);
        float32x4x3_t v3qf32_n2 = neon::vload3q(rows3_y);

        float32x4_t vqf32_c0_12 = neon::vadd(v3qf32_n0.val[0], v3qf32_n1.val[0]);
        float32x4_t vqf32_c0_03 = neon::vadd(v3qf32_c.val[0], v3qf32_n2.val[0]);
        float32x4_t vqf32_c1_12 = neon::vadd(v3qf32_n0.val[1], v3qf32_n1.val[1]);
        float32x4_t vqf32_c1_03 = neon::vadd(v3qf32_c.val[1], v3qf32_n2.val[1]);
        float32x4_t vqf32_c2_12 = neon::vadd(v3qf32_n0.val[2], v3qf32_n1.val[2]);
        float32x4_t vqf32_c2_03 = neon::vadd(v3qf32_c.val[2], v3qf32_n2.val[2]);

        float32x4_t vqf32_c0_x19 = neon::vmul(vqf32_c0_12, coef1);
        float32x4_t vqf32_c1_x19 = neon::vmul(vqf32_c1_12, coef1);
        float32x4_t vqf32_c2_x19 = neon::vmul(vqf32_c2_12, coef1);

        float16x4x3_t v3df16_result;
        v3df16_result.val[0] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c0_x19, neon::vmul(vqf32_c0_03, coef0)));
        v3df16_result.val[1] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c1_x19, neon::vmul(vqf32_c1_03, coef0)));
        v3df16_result.val[2] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c2_x19, neon::vmul(vqf32_c2_03, coef0)));

        neon::vstore(dst_row, v3df16_result);

        dst_row += 4 * channel;
        rows0_y += 4 * channel;
        rows1_y += 4 * channel;
        rows2_y += 4 * channel;
        rows3_y += 4 * channel;
    }

    for (; dx < owidth; dx++)
    {
        dst_row[0] = SaturateCast<Tp>((rows1_y[0] + rows2_y[0]) * coef1 + (rows0_y[0] + rows3_y[0]) * coef0);
        dst_row[1] = SaturateCast<Tp>((rows1_y[1] + rows2_y[1]) * coef1 + (rows0_y[1] + rows3_y[1]) * coef0);
        dst_row[2] = SaturateCast<Tp>((rows1_y[2] + rows2_y[2]) * coef1 + (rows0_y[2] + rows3_y[2]) * coef0);

        dst_row += channel;
        rows0_y += channel;
        rows1_y += channel;
        rows2_y += channel;
        rows3_y += channel;
    }

    // Line 1 ~ h-1
    for (MI_S32 dy = (start_row + 1); dy < end_row; dy++)
    {
        MI_S32 sy = (dy << 1) - 1;

        // hresize two row
        MI_F32 *rows0_tmp = rows0;
        MI_F32 *rows1_tmp = rows1;
        rows0             = rows2;
        rows1             = rows3;
        rows2             = rows0_tmp;
        rows3             = rows1_tmp;

        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        MI_F32 *rows2_x  = rows2;
        MI_F32 *rows3_x  = rows3;

        if ((end_row == oheight) && (dy == (end_row - 1)))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 0.5f + src_n1[3] * coef1 + src_n1[6] * coef0;
        rows3_x[0] = src_n2[0] * 0.5f + src_n2[3] * coef1 + src_n2[6] * coef0;
        rows2_x[1] = src_n1[1] * 0.5f + src_n1[4] * coef1 + src_n1[7] * coef0;
        rows3_x[1] = src_n2[1] * 0.5f + src_n2[4] * coef1 + src_n2[7] * coef0;
        rows2_x[2] = src_n1[2] * 0.5f + src_n1[5] * coef1 + src_n1[8] * coef0;
        rows3_x[2] = src_n2[2] * 0.5f + src_n2[5] * coef1 + src_n2[8] * coef0;

        src_n1  += channel;
        src_n2  += channel;
        rows2_x += channel;
        rows3_x += channel;

        MI_S32 owidth_align4 = (owidth - 2) & (-4);
        MI_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            MVType mvqf16_n1x0 = neon::vload3q(src_n1);
            MVType mvqf16_n2x0 = neon::vload3q(src_n2);
            MVType mvqf16_n1x1 = neon::vload3q(src_n1 + 2 * channel);
            MVType mvqf16_n2x1 = neon::vload3q(src_n2 + 2 * channel);

            float16x8x2_t v2qf16_n1x0_tmp0 = neon::vuzp(mvqf16_n1x0.val[0], mvqf16_n1x0.val[0]);
            float16x8x2_t v2qf16_n2x0_tmp0 = neon::vuzp(mvqf16_n2x0.val[0], mvqf16_n2x0.val[0]);
            float16x8x2_t v2qf16_n1x1_tmp0 = neon::vuzp(mvqf16_n1x1.val[0], mvqf16_n1x1.val[0]);
            float16x8x2_t v2qf16_n2x1_tmp0 = neon::vuzp(mvqf16_n2x1.val[0], mvqf16_n2x1.val[0]);
            float16x8x2_t v2qf16_n1x0_tmp1 = neon::vuzp(mvqf16_n1x0.val[1], mvqf16_n1x0.val[1]);
            float16x8x2_t v2qf16_n2x0_tmp1 = neon::vuzp(mvqf16_n2x0.val[1], mvqf16_n2x0.val[1]);
            float16x8x2_t v2qf16_n1x1_tmp1 = neon::vuzp(mvqf16_n1x1.val[1], mvqf16_n1x1.val[1]);
            float16x8x2_t v2qf16_n2x1_tmp1 = neon::vuzp(mvqf16_n2x1.val[1], mvqf16_n2x1.val[1]);
            float16x8x2_t v2qf16_n1x0_tmp2 = neon::vuzp(mvqf16_n1x0.val[2], mvqf16_n1x0.val[2]);
            float16x8x2_t v2qf16_n2x0_tmp2 = neon::vuzp(mvqf16_n2x0.val[2], mvqf16_n2x0.val[2]);
            float16x8x2_t v2qf16_n1x1_tmp2 = neon::vuzp(mvqf16_n1x1.val[2], mvqf16_n1x1.val[2]);
            float16x8x2_t v2qf16_n2x1_tmp2 = neon::vuzp(mvqf16_n2x1.val[2], mvqf16_n2x1.val[2]);

            float32x4_t vqf32_c02_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp0.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp0.val[0])));
            float32x4_t vqf32_c02_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp0.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp0.val[1])));
            float32x4_t vqf32_c03_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp0.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp0.val[0])));
            float32x4_t vqf32_c03_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp0.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp0.val[1])));
            float32x4_t vqf32_c12_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp1.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp1.val[0])));
            float32x4_t vqf32_c12_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp1.val[1])));
            float32x4_t vqf32_c13_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp1.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp1.val[0])));
            float32x4_t vqf32_c13_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp1.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp1.val[1])));
            float32x4_t vqf32_c22_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp2.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp2.val[0])));
            float32x4_t vqf32_c22_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x0_tmp2.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n1x1_tmp2.val[1])));
            float32x4_t vqf32_c23_12 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp2.val[1])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp2.val[0])));
            float32x4_t vqf32_c23_03 = neon::vadd(neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x0_tmp2.val[0])), neon::vcvt<MI_F32>(neon::vgetlow(v2qf16_n2x1_tmp2.val[1])));

            float32x4_t vqf32_c02_x19 = neon::vmul(vqf32_c02_12, coef1);
            float32x4_t vqf32_c03_x19 = neon::vmul(vqf32_c03_12, coef1);
            float32x4_t vqf32_c12_x19 = neon::vmul(vqf32_c12_12, coef1);
            float32x4_t vqf32_c13_x19 = neon::vmul(vqf32_c13_12, coef1);
            float32x4_t vqf32_c22_x19 = neon::vmul(vqf32_c22_12, coef1);
            float32x4_t vqf32_c23_x19 = neon::vmul(vqf32_c23_12, coef1);

            float32x4x3_t v3qf32_result2, v3qf32_result3;
            v3qf32_result2.val[0] = neon::vadd(vqf32_c02_x19, neon::vmul(vqf32_c02_03, coef0));
            v3qf32_result3.val[0] = neon::vadd(vqf32_c03_x19, neon::vmul(vqf32_c03_03, coef0));
            v3qf32_result2.val[1] = neon::vadd(vqf32_c12_x19, neon::vmul(vqf32_c12_03, coef0));
            v3qf32_result3.val[1] = neon::vadd(vqf32_c13_x19, neon::vmul(vqf32_c13_03, coef0));
            v3qf32_result2.val[2] = neon::vadd(vqf32_c22_x19, neon::vmul(vqf32_c22_03, coef0));
            v3qf32_result3.val[2] = neon::vadd(vqf32_c23_x19, neon::vmul(vqf32_c23_03, coef0));

            neon::vstore(rows2_x, v3qf32_result2);
            neon::vstore(rows3_x, v3qf32_result3);

            rows2_x += 4 * channel;
            rows3_x += 4 * channel;
            src_n1  += 8 * channel;
            src_n2  += 8 * channel;
        }

        for (; dx < (owidth - 2); dx++)
        {
            rows2_x[0] = (src_n1[3] + src_n1[6]) * coef1 + (src_n1[0] + src_n1[9]) * coef0;
            rows3_x[0] = (src_n2[3] + src_n2[6]) * coef1 + (src_n2[0] + src_n2[9]) * coef0;
            rows2_x[1] = (src_n1[4] + src_n1[7]) * coef1 + (src_n1[1] + src_n1[10]) * coef0;
            rows3_x[1] = (src_n2[4] + src_n2[7]) * coef1 + (src_n2[1] + src_n2[10]) * coef0;
            rows2_x[2] = (src_n1[5] + src_n1[8]) * coef1 + (src_n1[2] + src_n1[11]) * coef0;
            rows3_x[2] = (src_n2[5] + src_n2[8]) * coef1 + (src_n2[2] + src_n2[11]) * coef0;

            rows2_x += channel;
            rows3_x += channel;
            src_n1  += 2 * channel;
            src_n2  += 2 * channel;
        }

        rows2_x[0] = src_n1[3] * coef1 + src_n1[0] * coef0 + src_n1[6] * 0.5f;
        rows3_x[0] = src_n2[3] * coef1 + src_n2[0] * coef0 + src_n2[6] * 0.5f;
        rows2_x[1] = src_n1[4] * coef1 + src_n1[1] * coef0 + src_n1[7] * 0.5f;
        rows3_x[1] = src_n2[4] * coef1 + src_n2[1] * coef0 + src_n2[7] * 0.5f;
        rows2_x[2] = src_n1[5] * coef1 + src_n1[2] * coef0 + src_n1[8] * 0.5f;
        rows3_x[2] = src_n2[5] * coef1 + src_n2[2] * coef0 + src_n2[8] * 0.5f;

        // vresize
        MI_F32 *rows0_y = rows0;
        MI_F32 *rows1_y = rows1;
        MI_F32 *rows2_y = rows2;
        MI_F32 *rows3_y = rows3;

        Tp *dst_row = dst.Ptr<Tp>(dy);

        owidth_align4 = owidth & (-4);
        dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            float32x4x3_t v3qf32_c  = neon::vload3q(rows0_y);
            float32x4x3_t v3qf32_n0 = neon::vload3q(rows1_y);
            float32x4x3_t v3qf32_n1 = neon::vload3q(rows2_y);
            float32x4x3_t v3qf32_n2 = neon::vload3q(rows3_y);

            float32x4_t vqf32_c0_12 = neon::vadd(v3qf32_n0.val[0], v3qf32_n1.val[0]);
            float32x4_t vqf32_c0_03 = neon::vadd(v3qf32_c.val[0], v3qf32_n2.val[0]);
            float32x4_t vqf32_c1_12 = neon::vadd(v3qf32_n0.val[1], v3qf32_n1.val[1]);
            float32x4_t vqf32_c1_03 = neon::vadd(v3qf32_c.val[1], v3qf32_n2.val[1]);
            float32x4_t vqf32_c2_12 = neon::vadd(v3qf32_n0.val[2], v3qf32_n1.val[2]);
            float32x4_t vqf32_c2_03 = neon::vadd(v3qf32_c.val[2], v3qf32_n2.val[2]);

            float32x4_t vqf32_c0_x19 = neon::vmul(vqf32_c0_12, coef1);
            float32x4_t vqf32_c1_x19 = neon::vmul(vqf32_c1_12, coef1);
            float32x4_t vqf32_c2_x19 = neon::vmul(vqf32_c2_12, coef1);

            float16x4x3_t v3df16_result;
            v3df16_result.val[0] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c0_x19, neon::vmul(vqf32_c0_03, coef0)));
            v3df16_result.val[1] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c1_x19, neon::vmul(vqf32_c1_03, coef0)));
            v3df16_result.val[2] = neon::vcvt<MI_F16>(neon::vadd(vqf32_c2_x19, neon::vmul(vqf32_c2_03, coef0)));

            neon::vstore(dst_row, v3df16_result);

            dst_row += 4 * channel;
            rows0_y += 4 * channel;
            rows1_y += 4 * channel;
            rows2_y += 4 * channel;
            rows3_y += 4 * channel;
        }

        for (; dx < owidth; dx++)
        {
            dst_row[0] = SaturateCast<Tp>((rows1_y[0] + rows2_y[0]) * coef1 + (rows0_y[0] + rows3_y[0]) * coef0);
            dst_row[1] = SaturateCast<Tp>((rows1_y[1] + rows2_y[1]) * coef1 + (rows0_y[1] + rows3_y[1]) * coef0);
            dst_row[2] = SaturateCast<Tp>((rows1_y[2] + rows2_y[2]) * coef1 + (rows0_y[2] + rows3_y[2]) * coef0);

            dst_row += channel;
            rows0_y += channel;
            rows1_y += channel;
            rows2_y += channel;
            rows3_y += channel;
        }
    }

    return Status::OK;
}
#endif

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeCuC3DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MQVector<Tp, 3>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_F32 coef0     = 0.093750;
    MI_F32 coef1     = 0.593750;

    MI_F32 *rows = thread_buffer.GetThreadData<MI_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_F32 *rows0 = rows;
    MI_F32 *rows1 = rows0 + owidth * channel;
    MI_F32 *rows2 = rows1 + owidth * channel;
    MI_F32 *rows3 = rows2 + owidth * channel;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_F32 *rows0_x  = rows0;
    MI_F32 *rows1_x  = rows1;
    MI_F32 *rows2_x  = rows2;
    MI_F32 *rows3_x  = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 0.5f + src_c[3] * coef1 - src_c[6] * coef0;
    rows1_x[0] = src_n0[0] * 0.5f + src_n0[3] * coef1 - src_n0[6] * coef0;
    rows2_x[0] = src_n1[0] * 0.5f + src_n1[3] * coef1 - src_n1[6] * coef0;
    rows3_x[0] = src_n2[0] * 0.5f + src_n2[3] * coef1 - src_n2[6] * coef0;
    rows0_x[1] = src_c[1] * 0.5f + src_c[4] * coef1 - src_c[7] * coef0;
    rows1_x[1] = src_n0[1] * 0.5f + src_n0[4] * coef1 - src_n0[7] * coef0;
    rows2_x[1] = src_n1[1] * 0.5f + src_n1[4] * coef1 - src_n1[7] * coef0;
    rows3_x[1] = src_n2[1] * 0.5f + src_n2[4] * coef1 - src_n2[7] * coef0;
    rows0_x[2] = src_c[2] * 0.5f + src_c[5] * coef1 - src_c[8] * coef0;
    rows1_x[2] = src_n0[2] * 0.5f + src_n0[5] * coef1 - src_n0[8] * coef0;
    rows2_x[2] = src_n1[2] * 0.5f + src_n1[5] * coef1 - src_n1[8] * coef0;
    rows3_x[2] = src_n2[2] * 0.5f + src_n2[5] * coef1 - src_n2[8] * coef0;

    src_c  += channel;
    src_n0 += channel;
    src_n1 += channel;
    src_n2 += channel;

    rows0_x += channel;
    rows1_x += channel;
    rows2_x += channel;
    rows3_x += channel;

    MI_S32 owidth_align2 = (owidth - 2) & (-2);
    MI_S32 dx = 0;
    for (; dx < owidth_align2; dx += 2)
    {
        MVType mvqf32_cx0  = neon::vload3q(src_c); //0,3,6,9; 1,4,7,10; 2,5,8,11;
        MVType mvqf32_n0x0 = neon::vload3q(src_n0);
        MVType mvqf32_n1x0 = neon::vload3q(src_n1);
        MVType mvqf32_n2x0 = neon::vload3q(src_n2);
        MVType mvqf32_cx1  = neon::vload3q(src_c + 2 * channel); //6,9,12,15; 7,10,13,16; 8,11,14,17
        MVType mvqf32_n0x1 = neon::vload3q(src_n0 + 2 * channel);
        MVType mvqf32_n1x1 = neon::vload3q(src_n1 + 2 * channel);
        MVType mvqf32_n2x1 = neon::vload3q(src_n2 + 2 * channel);

        auto v2qf32_cx0_tmp0  = neon::vuzp(mvqf32_cx0.val[0], mvqf32_cx0.val[0]);
        auto v2qf32_n0x0_tmp0 = neon::vuzp(mvqf32_n0x0.val[0], mvqf32_n0x0.val[0]);
        auto v2qf32_n1x0_tmp0 = neon::vuzp(mvqf32_n1x0.val[0], mvqf32_n1x0.val[0]);
        auto v2qf32_n2x0_tmp0 = neon::vuzp(mvqf32_n2x0.val[0], mvqf32_n2x0.val[0]);
        auto v2qf32_cx1_tmp0  = neon::vuzp(mvqf32_cx1.val[0], mvqf32_cx1.val[0]);
        auto v2qf32_n0x1_tmp0 = neon::vuzp(mvqf32_n0x1.val[0], mvqf32_n0x1.val[0]);
        auto v2qf32_n1x1_tmp0 = neon::vuzp(mvqf32_n1x1.val[0], mvqf32_n1x1.val[0]);
        auto v2qf32_n2x1_tmp0 = neon::vuzp(mvqf32_n2x1.val[0], mvqf32_n2x1.val[0]);

        auto v2qf32_cx0_tmp1  = neon::vuzp(mvqf32_cx0.val[1], mvqf32_cx0.val[1]);
        auto v2qf32_n0x0_tmp1 = neon::vuzp(mvqf32_n0x0.val[1], mvqf32_n0x0.val[1]);
        auto v2qf32_n1x0_tmp1 = neon::vuzp(mvqf32_n1x0.val[1], mvqf32_n1x0.val[1]);
        auto v2qf32_n2x0_tmp1 = neon::vuzp(mvqf32_n2x0.val[1], mvqf32_n2x0.val[1]);
        auto v2qf32_cx1_tmp1  = neon::vuzp(mvqf32_cx1.val[1], mvqf32_cx1.val[1]);
        auto v2qf32_n0x1_tmp1 = neon::vuzp(mvqf32_n0x1.val[1], mvqf32_n0x1.val[1]);
        auto v2qf32_n1x1_tmp1 = neon::vuzp(mvqf32_n1x1.val[1], mvqf32_n1x1.val[1]);
        auto v2qf32_n2x1_tmp1 = neon::vuzp(mvqf32_n2x1.val[1], mvqf32_n2x1.val[1]);

        auto v2qf32_cx0_tmp2  = neon::vuzp(mvqf32_cx0.val[2], mvqf32_cx0.val[2]);
        auto v2qf32_n0x0_tmp2 = neon::vuzp(mvqf32_n0x0.val[2], mvqf32_n0x0.val[2]);
        auto v2qf32_n1x0_tmp2 = neon::vuzp(mvqf32_n1x0.val[2], mvqf32_n1x0.val[2]);
        auto v2qf32_n2x0_tmp2 = neon::vuzp(mvqf32_n2x0.val[2], mvqf32_n2x0.val[2]);
        auto v2qf32_cx1_tmp2  = neon::vuzp(mvqf32_cx1.val[2], mvqf32_cx1.val[2]);
        auto v2qf32_n0x1_tmp2 = neon::vuzp(mvqf32_n0x1.val[2], mvqf32_n0x1.val[2]);
        auto v2qf32_n1x1_tmp2 = neon::vuzp(mvqf32_n1x1.val[2], mvqf32_n1x1.val[2]);
        auto v2qf32_n2x1_tmp2 = neon::vuzp(mvqf32_n2x1.val[2], mvqf32_n2x1.val[2]);

        float32x2_t vdf32_c00_12 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp0.val[1]), neon::vgetlow(v2qf32_cx1_tmp0.val[0]));
        float32x2_t vdf32_c00_03 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp0.val[0]), neon::vgetlow(v2qf32_cx1_tmp0.val[1]));
        float32x2_t vdf32_c01_12 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp0.val[1]), neon::vgetlow(v2qf32_n0x1_tmp0.val[0]));
        float32x2_t vdf32_c01_03 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp0.val[0]), neon::vgetlow(v2qf32_n0x1_tmp0.val[1]));
        float32x2_t vdf32_c02_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp0.val[1]), neon::vgetlow(v2qf32_n1x1_tmp0.val[0]));
        float32x2_t vdf32_c02_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp0.val[0]), neon::vgetlow(v2qf32_n1x1_tmp0.val[1]));
        float32x2_t vdf32_c03_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp0.val[1]), neon::vgetlow(v2qf32_n2x1_tmp0.val[0]));
        float32x2_t vdf32_c03_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp0.val[0]), neon::vgetlow(v2qf32_n2x1_tmp0.val[1]));

        float32x2_t vdf32_c10_12 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp1.val[1]), neon::vgetlow(v2qf32_cx1_tmp1.val[0]));
        float32x2_t vdf32_c10_03 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp1.val[0]), neon::vgetlow(v2qf32_cx1_tmp1.val[1]));
        float32x2_t vdf32_c11_12 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp1.val[1]), neon::vgetlow(v2qf32_n0x1_tmp1.val[0]));
        float32x2_t vdf32_c11_03 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp1.val[0]), neon::vgetlow(v2qf32_n0x1_tmp1.val[1]));
        float32x2_t vdf32_c12_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp1.val[1]), neon::vgetlow(v2qf32_n1x1_tmp1.val[0]));
        float32x2_t vdf32_c12_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp1.val[0]), neon::vgetlow(v2qf32_n1x1_tmp1.val[1]));
        float32x2_t vdf32_c13_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp1.val[1]), neon::vgetlow(v2qf32_n2x1_tmp1.val[0]));
        float32x2_t vdf32_c13_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp1.val[0]), neon::vgetlow(v2qf32_n2x1_tmp1.val[1]));

        float32x2_t vdf32_c20_12 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp2.val[1]), neon::vgetlow(v2qf32_cx1_tmp2.val[0]));
        float32x2_t vdf32_c20_03 = neon::vadd(neon::vgetlow(v2qf32_cx0_tmp2.val[0]), neon::vgetlow(v2qf32_cx1_tmp2.val[1]));
        float32x2_t vdf32_c21_12 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp2.val[1]), neon::vgetlow(v2qf32_n0x1_tmp2.val[0]));
        float32x2_t vdf32_c21_03 = neon::vadd(neon::vgetlow(v2qf32_n0x0_tmp2.val[0]), neon::vgetlow(v2qf32_n0x1_tmp2.val[1]));
        float32x2_t vdf32_c22_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp2.val[1]), neon::vgetlow(v2qf32_n1x1_tmp2.val[0]));
        float32x2_t vdf32_c22_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp2.val[0]), neon::vgetlow(v2qf32_n1x1_tmp2.val[1]));
        float32x2_t vdf32_c23_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp2.val[1]), neon::vgetlow(v2qf32_n2x1_tmp2.val[0]));
        float32x2_t vdf32_c23_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp2.val[0]), neon::vgetlow(v2qf32_n2x1_tmp2.val[1]));

        float32x2_t vdf32_c00_x19 = neon::vmul(vdf32_c00_12, coef1);
        float32x2_t vdf32_c01_x19 = neon::vmul(vdf32_c01_12, coef1);
        float32x2_t vdf32_c02_x19 = neon::vmul(vdf32_c02_12, coef1);
        float32x2_t vdf32_c03_x19 = neon::vmul(vdf32_c03_12, coef1);
        float32x2_t vdf32_c10_x19 = neon::vmul(vdf32_c10_12, coef1);
        float32x2_t vdf32_c11_x19 = neon::vmul(vdf32_c11_12, coef1);
        float32x2_t vdf32_c12_x19 = neon::vmul(vdf32_c12_12, coef1);
        float32x2_t vdf32_c13_x19 = neon::vmul(vdf32_c13_12, coef1);
        float32x2_t vdf32_c20_x19 = neon::vmul(vdf32_c20_12, coef1);
        float32x2_t vdf32_c21_x19 = neon::vmul(vdf32_c21_12, coef1);
        float32x2_t vdf32_c22_x19 = neon::vmul(vdf32_c22_12, coef1);
        float32x2_t vdf32_c23_x19 = neon::vmul(vdf32_c23_12, coef1);

        float32x2x3_t v3df32_result0, v3df32_result1, v3df32_result2, v3df32_result3;
        v3df32_result0.val[0] = neon::vmls(vdf32_c00_x19, vdf32_c00_03, coef0);
        v3df32_result1.val[0] = neon::vmls(vdf32_c01_x19, vdf32_c01_03, coef0);
        v3df32_result2.val[0] = neon::vmls(vdf32_c02_x19, vdf32_c02_03, coef0);
        v3df32_result3.val[0] = neon::vmls(vdf32_c03_x19, vdf32_c03_03, coef0);
        v3df32_result0.val[1] = neon::vmls(vdf32_c10_x19, vdf32_c10_03, coef0);
        v3df32_result1.val[1] = neon::vmls(vdf32_c11_x19, vdf32_c11_03, coef0);
        v3df32_result2.val[1] = neon::vmls(vdf32_c12_x19, vdf32_c12_03, coef0);
        v3df32_result3.val[1] = neon::vmls(vdf32_c13_x19, vdf32_c13_03, coef0);
        v3df32_result0.val[2] = neon::vmls(vdf32_c20_x19, vdf32_c20_03, coef0);
        v3df32_result1.val[2] = neon::vmls(vdf32_c21_x19, vdf32_c21_03, coef0);
        v3df32_result2.val[2] = neon::vmls(vdf32_c22_x19, vdf32_c22_03, coef0);
        v3df32_result3.val[2] = neon::vmls(vdf32_c23_x19, vdf32_c23_03, coef0);

        neon::vstore(rows0_x, v3df32_result0);
        neon::vstore(rows1_x, v3df32_result1);
        neon::vstore(rows2_x, v3df32_result2);
        neon::vstore(rows3_x, v3df32_result3);

        rows0_x += 2 * channel;
        rows1_x += 2 * channel;
        rows2_x += 2 * channel;
        rows3_x += 2 * channel;

        src_c  += 4 * channel;
        src_n0 += 4 * channel;
        src_n1 += 4 * channel;
        src_n2 += 4 * channel;
    }

    for (; dx < (owidth - 2); dx++)
    {
        rows0_x[0] = (src_c[3] + src_c[6]) * coef1 - (src_c[0] + src_c[9]) * coef0;
        rows1_x[0] = (src_n0[3] + src_n0[6]) * coef1 - (src_n0[0] + src_n0[9]) * coef0;
        rows2_x[0] = (src_n1[3] + src_n1[6]) * coef1 - (src_n1[0] + src_n1[9]) * coef0;
        rows3_x[0] = (src_n2[3] + src_n2[6]) * coef1 - (src_n2[0] + src_n2[9]) * coef0;
        rows0_x[1] = (src_c[4] + src_c[7]) * coef1 - (src_c[1] + src_c[10]) * coef0;
        rows1_x[1] = (src_n0[4] + src_n0[7]) * coef1 - (src_n0[1] + src_n0[10]) * coef0;
        rows2_x[1] = (src_n1[4] + src_n1[7]) * coef1 - (src_n1[1] + src_n1[10]) * coef0;
        rows3_x[1] = (src_n2[4] + src_n2[7]) * coef1 - (src_n2[1] + src_n2[10]) * coef0;
        rows0_x[2] = (src_c[5] + src_c[8]) * coef1 - (src_c[2] + src_c[11]) * coef0;
        rows1_x[2] = (src_n0[5] + src_n0[8]) * coef1 - (src_n0[2] + src_n0[11]) * coef0;
        rows2_x[2] = (src_n1[5] + src_n1[8]) * coef1 - (src_n1[2] + src_n1[11]) * coef0;
        rows3_x[2] = (src_n2[5] + src_n2[8]) * coef1 - (src_n2[2] + src_n2[11]) * coef0;

        rows0_x += channel;
        rows1_x += channel;
        rows2_x += channel;
        rows3_x += channel;

        src_c  += 2 * channel;
        src_n0 += 2 * channel;
        src_n1 += 2 * channel;
        src_n2 += 2 * channel;
    }

    rows0_x[0] = src_c[3] * coef1 - src_c[0] * coef0 + src_c[6] * 0.5f;
    rows1_x[0] = src_n0[3] * coef1 - src_n0[0] * coef0 + src_n0[6] * 0.5f;
    rows2_x[0] = src_n1[3] * coef1 - src_n1[0] * coef0 + src_n1[6] * 0.5f;
    rows3_x[0] = src_n2[3] * coef1 - src_n2[0] * coef0 + src_n2[6] * 0.5f;
    rows0_x[1] = src_c[4] * coef1 - src_c[1] * coef0 + src_c[7] * 0.5f;
    rows1_x[1] = src_n0[4] * coef1 - src_n0[1] * coef0 + src_n0[7] * 0.5f;
    rows2_x[1] = src_n1[4] * coef1 - src_n1[1] * coef0 + src_n1[7] * 0.5f;
    rows3_x[1] = src_n2[4] * coef1 - src_n2[1] * coef0 + src_n2[7] * 0.5f;
    rows0_x[2] = src_c[5] * coef1 - src_c[2] * coef0 + src_c[8] * 0.5f;
    rows1_x[2] = src_n0[5] * coef1 - src_n0[2] * coef0 + src_n0[8] * 0.5f;
    rows2_x[2] = src_n1[5] * coef1 - src_n1[2] * coef0 + src_n1[8] * 0.5f;
    rows3_x[2] = src_n2[5] * coef1 - src_n2[2] * coef0 + src_n2[8] * 0.5f;

    // vresize
    MI_F32 *rows0_y = rows0;
    MI_F32 *rows1_y = rows1;
    MI_F32 *rows2_y = rows2;
    MI_F32 *rows3_y = rows3;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    MI_S32 owidth_align4 = owidth & (-4);
    dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        float32x4x3_t v3qf32_c  = neon::vload3q(rows0_y);
        float32x4x3_t v3qf32_n0 = neon::vload3q(rows1_y);
        float32x4x3_t v3qf32_n1 = neon::vload3q(rows2_y);
        float32x4x3_t v3qf32_n2 = neon::vload3q(rows3_y);

        float32x4_t vqf32_c0_12 = neon::vadd(v3qf32_n0.val[0], v3qf32_n1.val[0]);
        float32x4_t vqf32_c0_03 = neon::vadd(v3qf32_c.val[0], v3qf32_n2.val[0]);
        float32x4_t vqf32_c1_12 = neon::vadd(v3qf32_n0.val[1], v3qf32_n1.val[1]);
        float32x4_t vqf32_c1_03 = neon::vadd(v3qf32_c.val[1], v3qf32_n2.val[1]);
        float32x4_t vqf32_c2_12 = neon::vadd(v3qf32_n0.val[2], v3qf32_n1.val[2]);
        float32x4_t vqf32_c2_03 = neon::vadd(v3qf32_c.val[2], v3qf32_n2.val[2]);

        float32x4_t vqf32_c0_x19 = neon::vmul(vqf32_c0_12, coef1);
        float32x4_t vqf32_c1_x19 = neon::vmul(vqf32_c1_12, coef1);
        float32x4_t vqf32_c2_x19 = neon::vmul(vqf32_c2_12, coef1);

        float32x4x3_t v3qf32_result;
        v3qf32_result.val[0] = neon::vmls(vqf32_c0_x19, vqf32_c0_03, coef0);
        v3qf32_result.val[1] = neon::vmls(vqf32_c1_x19, vqf32_c1_03, coef0);
        v3qf32_result.val[2] = neon::vmls(vqf32_c2_x19, vqf32_c2_03, coef0);

        neon::vstore(dst_row, v3qf32_result);

        dst_row += 4 * channel;
        rows0_y += 4 * channel;
        rows1_y += 4 * channel;
        rows2_y += 4 * channel;
        rows3_y += 4 * channel;
    }

    for (; dx < owidth; dx++)
    {
        dst_row[0] = SaturateCast<Tp>((rows1_y[0] + rows2_y[0]) * coef1 - (rows0_y[0] + rows3_y[0]) * coef0);
        dst_row[1] = SaturateCast<Tp>((rows1_y[1] + rows2_y[1]) * coef1 - (rows0_y[1] + rows3_y[1]) * coef0);
        dst_row[2] = SaturateCast<Tp>((rows1_y[2] + rows2_y[2]) * coef1 - (rows0_y[2] + rows3_y[2]) * coef0);

        dst_row += channel;
        rows0_y += channel;
        rows1_y += channel;
        rows2_y += channel;
        rows3_y += channel;
    }

    // Line 1 ~ h-1
    for (MI_S32 dy = (start_row + 1); dy < end_row; dy++)
    {
        MI_S32 sy = (dy << 1) - 1;

        // hresize two row
        MI_F32 *rows0_tmp = rows0;
        MI_F32 *rows1_tmp = rows1;
        rows0             = rows2;
        rows1             = rows3;
        rows2             = rows0_tmp;
        rows3             = rows1_tmp;

        const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
        const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
        MI_F32 *rows2_x  = rows2;
        MI_F32 *rows3_x  = rows3;

        if ((end_row == oheight) && (dy == (end_row - 1)))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 0.5f + src_n1[3] * coef1 - src_n1[6] * coef0;
        rows3_x[0] = src_n2[0] * 0.5f + src_n2[3] * coef1 - src_n2[6] * coef0;
        rows2_x[1] = src_n1[1] * 0.5f + src_n1[4] * coef1 - src_n1[7] * coef0;
        rows3_x[1] = src_n2[1] * 0.5f + src_n2[4] * coef1 - src_n2[7] * coef0;
        rows2_x[2] = src_n1[2] * 0.5f + src_n1[5] * coef1 - src_n1[8] * coef0;
        rows3_x[2] = src_n2[2] * 0.5f + src_n2[5] * coef1 - src_n2[8] * coef0;

        src_n1  += channel;
        src_n2  += channel;
        rows2_x += channel;
        rows3_x += channel;

        MI_S32 owidth_align2 = (owidth - 2) & (-2);
        MI_S32 dx = 0;
        for (; dx < owidth_align2; dx += 2)
        {
            MVType mvqf32_n1x0 = neon::vload3q(src_n1);
            MVType mvqf32_n2x0 = neon::vload3q(src_n2);
            MVType mvqf32_n1x1 = neon::vload3q(src_n1 + 2 * channel);
            MVType mvqf32_n2x1 = neon::vload3q(src_n2 + 2 * channel);

            auto v2qf32_n1x0_tmp0 = neon::vuzp(mvqf32_n1x0.val[0], mvqf32_n1x0.val[0]);
            auto v2qf32_n2x0_tmp0 = neon::vuzp(mvqf32_n2x0.val[0], mvqf32_n2x0.val[0]);
            auto v2qf32_n1x1_tmp0 = neon::vuzp(mvqf32_n1x1.val[0], mvqf32_n1x1.val[0]);
            auto v2qf32_n2x1_tmp0 = neon::vuzp(mvqf32_n2x1.val[0], mvqf32_n2x1.val[0]);
            auto v2qf32_n1x0_tmp1 = neon::vuzp(mvqf32_n1x0.val[1], mvqf32_n1x0.val[1]);
            auto v2qf32_n2x0_tmp1 = neon::vuzp(mvqf32_n2x0.val[1], mvqf32_n2x0.val[1]);
            auto v2qf32_n1x1_tmp1 = neon::vuzp(mvqf32_n1x1.val[1], mvqf32_n1x1.val[1]);
            auto v2qf32_n2x1_tmp1 = neon::vuzp(mvqf32_n2x1.val[1], mvqf32_n2x1.val[1]);
            auto v2qf32_n1x0_tmp2 = neon::vuzp(mvqf32_n1x0.val[2], mvqf32_n1x0.val[2]);
            auto v2qf32_n2x0_tmp2 = neon::vuzp(mvqf32_n2x0.val[2], mvqf32_n2x0.val[2]);
            auto v2qf32_n1x1_tmp2 = neon::vuzp(mvqf32_n1x1.val[2], mvqf32_n1x1.val[2]);
            auto v2qf32_n2x1_tmp2 = neon::vuzp(mvqf32_n2x1.val[2], mvqf32_n2x1.val[2]);

            float32x2_t vdf32_c02_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp0.val[1]), neon::vgetlow(v2qf32_n1x1_tmp0.val[0]));
            float32x2_t vdf32_c02_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp0.val[0]), neon::vgetlow(v2qf32_n1x1_tmp0.val[1]));
            float32x2_t vdf32_c03_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp0.val[1]), neon::vgetlow(v2qf32_n2x1_tmp0.val[0]));
            float32x2_t vdf32_c03_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp0.val[0]), neon::vgetlow(v2qf32_n2x1_tmp0.val[1]));
            float32x2_t vdf32_c12_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp1.val[1]), neon::vgetlow(v2qf32_n1x1_tmp1.val[0]));
            float32x2_t vdf32_c12_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp1.val[0]), neon::vgetlow(v2qf32_n1x1_tmp1.val[1]));
            float32x2_t vdf32_c13_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp1.val[1]), neon::vgetlow(v2qf32_n2x1_tmp1.val[0]));
            float32x2_t vdf32_c13_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp1.val[0]), neon::vgetlow(v2qf32_n2x1_tmp1.val[1]));
            float32x2_t vdf32_c22_12 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp2.val[1]), neon::vgetlow(v2qf32_n1x1_tmp2.val[0]));
            float32x2_t vdf32_c22_03 = neon::vadd(neon::vgetlow(v2qf32_n1x0_tmp2.val[0]), neon::vgetlow(v2qf32_n1x1_tmp2.val[1]));
            float32x2_t vdf32_c23_12 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp2.val[1]), neon::vgetlow(v2qf32_n2x1_tmp2.val[0]));
            float32x2_t vdf32_c23_03 = neon::vadd(neon::vgetlow(v2qf32_n2x0_tmp2.val[0]), neon::vgetlow(v2qf32_n2x1_tmp2.val[1]));

            float32x2_t vdf32_c02_x19 = neon::vmul(vdf32_c02_12, coef1);
            float32x2_t vdf32_c03_x19 = neon::vmul(vdf32_c03_12, coef1);
            float32x2_t vdf32_c12_x19 = neon::vmul(vdf32_c12_12, coef1);
            float32x2_t vdf32_c13_x19 = neon::vmul(vdf32_c13_12, coef1);
            float32x2_t vdf32_c22_x19 = neon::vmul(vdf32_c22_12, coef1);
            float32x2_t vdf32_c23_x19 = neon::vmul(vdf32_c23_12, coef1);

            float32x2x3_t v3df32_result2, v3df32_result3;
            v3df32_result2.val[0] = neon::vmls(vdf32_c02_x19, vdf32_c02_03, coef0);
            v3df32_result3.val[0] = neon::vmls(vdf32_c03_x19, vdf32_c03_03, coef0);
            v3df32_result2.val[1] = neon::vmls(vdf32_c12_x19, vdf32_c12_03, coef0);
            v3df32_result3.val[1] = neon::vmls(vdf32_c13_x19, vdf32_c13_03, coef0);
            v3df32_result2.val[2] = neon::vmls(vdf32_c22_x19, vdf32_c22_03, coef0);
            v3df32_result3.val[2] = neon::vmls(vdf32_c23_x19, vdf32_c23_03, coef0);

            neon::vstore(rows2_x, v3df32_result2);
            neon::vstore(rows3_x, v3df32_result3);

            rows2_x += 2 * channel;
            rows3_x += 2 * channel;
            src_n1  += 4 * channel;
            src_n2  += 4 * channel;
        }

        for (; dx < (owidth - 2); dx++)
        {
            rows2_x[0] = (src_n1[3] + src_n1[6]) * coef1 - (src_n1[0] + src_n1[9]) * coef0;
            rows3_x[0] = (src_n2[3] + src_n2[6]) * coef1 - (src_n2[0] + src_n2[9]) * coef0;
            rows2_x[1] = (src_n1[4] + src_n1[7]) * coef1 - (src_n1[1] + src_n1[10]) * coef0;
            rows3_x[1] = (src_n2[4] + src_n2[7]) * coef1 - (src_n2[1] + src_n2[10]) * coef0;
            rows2_x[2] = (src_n1[5] + src_n1[8]) * coef1 - (src_n1[2] + src_n1[11]) * coef0;
            rows3_x[2] = (src_n2[5] + src_n2[8]) * coef1 - (src_n2[2] + src_n2[11]) * coef0;

            rows2_x += channel;
            rows3_x += channel;
            src_n1  += 2 * channel;
            src_n2  += 2 * channel;
        }

        rows2_x[0] = src_n1[3] * coef1 - src_n1[0] * coef0 + src_n1[6] * 0.5f;
        rows3_x[0] = src_n2[3] * coef1 - src_n2[0] * coef0 + src_n2[6] * 0.5f;
        rows2_x[1] = src_n1[4] * coef1 - src_n1[1] * coef0 + src_n1[7] * 0.5f;
        rows3_x[1] = src_n2[4] * coef1 - src_n2[1] * coef0 + src_n2[7] * 0.5f;
        rows2_x[2] = src_n1[5] * coef1 - src_n1[2] * coef0 + src_n1[8] * 0.5f;
        rows3_x[2] = src_n2[5] * coef1 - src_n2[2] * coef0 + src_n2[8] * 0.5f;

        // vresize
        MI_F32 *rows0_y = rows0;
        MI_F32 *rows1_y = rows1;
        MI_F32 *rows2_y = rows2;
        MI_F32 *rows3_y = rows3;

        Tp *dst_row = dst.Ptr<Tp>(dy);

        MI_S32 owidth_align4 = owidth & (-4);
        dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            float32x4x3_t v3qf32_c  = neon::vload3q(rows0_y);
            float32x4x3_t v3qf32_n0 = neon::vload3q(rows1_y);
            float32x4x3_t v3qf32_n1 = neon::vload3q(rows2_y);
            float32x4x3_t v3qf32_n2 = neon::vload3q(rows3_y);

            float32x4_t vqf32_c0_12 = neon::vadd(v3qf32_n0.val[0], v3qf32_n1.val[0]);
            float32x4_t vqf32_c0_03 = neon::vadd(v3qf32_c.val[0], v3qf32_n2.val[0]);
            float32x4_t vqf32_c1_12 = neon::vadd(v3qf32_n0.val[1], v3qf32_n1.val[1]);
            float32x4_t vqf32_c1_03 = neon::vadd(v3qf32_c.val[1], v3qf32_n2.val[1]);
            float32x4_t vqf32_c2_12 = neon::vadd(v3qf32_n0.val[2], v3qf32_n1.val[2]);
            float32x4_t vqf32_c2_03 = neon::vadd(v3qf32_c.val[2], v3qf32_n2.val[2]);

            float32x4_t vqf32_c0_x19 = neon::vmul(vqf32_c0_12, coef1);
            float32x4_t vqf32_c1_x19 = neon::vmul(vqf32_c1_12, coef1);
            float32x4_t vqf32_c2_x19 = neon::vmul(vqf32_c2_12, coef1);

            float32x4x3_t v3qf32_result;
            v3qf32_result.val[0] = neon::vmls(vqf32_c0_x19, vqf32_c0_03, coef0);
            v3qf32_result.val[1] = neon::vmls(vqf32_c1_x19, vqf32_c1_03, coef0);
            v3qf32_result.val[2] = neon::vmls(vqf32_c2_x19, vqf32_c2_03, coef0);

            neon::vstore(dst_row, v3qf32_result);

            dst_row += 4 * channel;
            rows0_y += 4 * channel;
            rows1_y += 4 * channel;
            rows2_y += 4 * channel;
            rows3_y += 4 * channel;
        }

        for (; dx < owidth; dx++)
        {
            dst_row[0] = SaturateCast<Tp>((rows1_y[0] + rows2_y[0]) * coef1 - (rows0_y[0] + rows3_y[0]) * coef0);
            dst_row[1] = SaturateCast<Tp>((rows1_y[1] + rows2_y[1]) * coef1 - (rows0_y[1] + rows3_y[1]) * coef0);
            dst_row[2] = SaturateCast<Tp>((rows1_y[2] + rows2_y[2]) * coef1 - (rows0_y[2] + rows3_y[2]) * coef0);

            dst_row += channel;
            rows0_y += channel;
            rows1_y += channel;
            rows2_y += channel;
            rows3_y += channel;
        }
    }

    return Status::OK;
}

template <typename Tp>
Status ResizeCuFastC3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    using BufType  = typename ResizeBnCuTraits<Tp>::BufType;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MI_F32 scale_x = static_cast<MI_F64>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F64>(iheight) / oheight;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 2.f))
    {
        ThreadBuffer thread_buffer(ctx, owidth * 4 * 3 * sizeof(BufType));
        ret = wp->ParallelFor(0, AURA_ALIGN(oheight, 2) / 2, ResizeCuC3DownX2NeonImpl<Tp>, ctx, std::cref(src), std::ref(dst), std::ref(thread_buffer));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeCuC3DownX2NeonImpl run failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.f))
    {
        ret = wp->ParallelFor(0, oheight, ResizeCuC3DownX4NeonImpl<Tp>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeCuC3DownX4NeonImpl run failed");
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "special scale param error");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeCuFastC3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuFastC3NeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuFastC3NeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuFastC3NeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuFastC3NeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeCuFastC3NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeCuFastC3NeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC3NeonHelper run failed, type: MI_F32");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura