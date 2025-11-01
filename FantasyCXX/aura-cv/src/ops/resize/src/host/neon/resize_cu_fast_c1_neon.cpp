#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeCuC1DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align8 = owidth & (-8);
    using VType         = typename neon::DVector<Tp>::VType;

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
            auto v4d8_c  = neon::vload4(src_c);
            auto v4d8_n0 = neon::vload4(src_n0);
            auto v4d8_n1 = neon::vload4(src_n1);
            auto v4d8_n2 = neon::vload4(src_n2);

            int16x8_t vqs16_c_12  = neon::vaddl(v4d8_c.val[1], v4d8_c.val[2]);
            int16x8_t vqs16_c_03  = neon::vaddl(v4d8_c.val[0], v4d8_c.val[3]);
            int16x8_t vqs16_n0_12 = neon::vaddl(v4d8_n0.val[1], v4d8_n0.val[2]);
            int16x8_t vqs16_n0_03 = neon::vaddl(v4d8_n0.val[0], v4d8_n0.val[3]);
            int16x8_t vqs16_n1_12 = neon::vaddl(v4d8_n1.val[1], v4d8_n1.val[2]);
            int16x8_t vqs16_n1_03 = neon::vaddl(v4d8_n1.val[0], v4d8_n1.val[3]);
            int16x8_t vqs16_n2_12 = neon::vaddl(v4d8_n2.val[1], v4d8_n2.val[2]);
            int16x8_t vqs16_n2_03 = neon::vaddl(v4d8_n2.val[0], v4d8_n2.val[3]);

            int16x8_t vqs16_c_x19     = neon::vmul(vqs16_c_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_c_result  = neon::vmls(vqs16_c_x19, vqs16_c_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n0_x19    = neon::vmul(vqs16_n0_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n0_result = neon::vmls(vqs16_n0_x19, vqs16_n0_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n1_x19    = neon::vmul(vqs16_n1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n1_result = neon::vmls(vqs16_n1_x19, vqs16_n1_03, static_cast<MI_S16>(3));
            int16x8_t vqs16_n2_x19    = neon::vmul(vqs16_n2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_n2_result = neon::vmls(vqs16_n2_x19, vqs16_n2_03, static_cast<MI_S16>(3));

            int32x4_t vqs32_result_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_result), neon::vgetlow(vqs16_n1_result));
            int32x4_t vqs32_result_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_result), neon::vgethigh(vqs16_n1_result));
            int32x4_t vqs32_result_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_result), neon::vgetlow(vqs16_n2_result));
            int32x4_t vqs32_result_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_result), neon::vgethigh(vqs16_n2_result));

            int32x4_t vqs32_lo_x19    = neon::vmul(vqs32_result_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_result_lo = neon::vmls(vqs32_lo_x19, vqs32_result_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_hi_x19    = neon::vmul(vqs32_result_hi12, static_cast<MI_S32>(19));
            int32x4_t vqs32_result_hi = neon::vmls(vqs32_hi_x19, vqs32_result_hi03, static_cast<MI_S32>(3));

            int16x4_t vds16_des_lo = neon::vrshrn_n<10>(vqs32_result_lo);
            int16x4_t vds16_des_hi = neon::vrshrn_n<10>(vqs32_result_hi);

            VType vd8_result;
            if (std::is_same<Tp, MI_U8>::value)
            {
                int16x8_t vqs16_zero;
                neon::vdup(vqs16_zero, static_cast<MI_S16>(0));
                int16x8_t vqs16_255;
                neon::vdup(vqs16_255, static_cast<MI_S16>(255));

                int16x8_t vqs16_des  = neon::vcombine(vds16_des_lo, vds16_des_hi);
                vqs16_des            = neon::vmax(vqs16_des, vqs16_zero);
                vqs16_des            = neon::vmin(vqs16_des, vqs16_255);
                uint16x8_t vdu16_des = neon::vreinterpret(vqs16_des);
                vd8_result           = neon::vmovn(vdu16_des);
                neon::vstore(dst_row, vd8_result);
            }
            else
            {
                int16x8_t vqs16_n128;
                neon::vdup(vqs16_n128, static_cast<MI_S16>(-128));
                int16x8_t vqs16_p127;
                neon::vdup(vqs16_p127, static_cast<MI_S16>(127));

                int16x8_t vqs16_des = neon::vcombine(vds16_des_lo, vds16_des_hi);
                vqs16_des           = neon::vmax(vqs16_des, vqs16_n128);
                vqs16_des           = neon::vmin(vqs16_des, vqs16_p127);
                vd8_result          = neon::vmovn(vqs16_des);
                neon::vstore(dst_row, vd8_result);
            }

            dst_row += 8;
            src_c   += 32;
            src_n0  += 32;
            src_n1  += 32;
            src_n2  += 32;
        }

        for (; x < owidth; x++)
        {
            MI_S32 y0 = (src_c[1] + src_c[2]) * 19 - (src_c[0] + src_c[3]) * 3;
            MI_S32 y1 = (src_n0[1] + src_n0[2]) * 19 - (src_n0[0] + src_n0[3]) * 3;
            MI_S32 y2 = (src_n1[1] + src_n1[2]) * 19 - (src_n1[0] + src_n1[3]) * 3;
            MI_S32 y3 = (src_n2[1] + src_n2[2]) * 19 - (src_n2[0] + src_n2[3]) * 3;

            MI_S32 temp = (y1 * 19 - y0 * 3 + y2 * 19 - y3 * 3 + 512) >> 10;
            *dst_row    = SaturateCast<Tp>(temp);

            dst_row++;
            src_c  += 4;
            src_n0 += 4;
            src_n1 += 4;
            src_n2 += 4;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeCuC1DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 owidth       = dst.GetSizes().m_width;
    MI_S32 width_align4 = owidth & (-4);
    using VType         = typename neon::DVector<Tp>::VType;

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
            auto v4d16_c  = neon::vload4(src_c);
            auto v4d16_n0 = neon::vload4(src_n0);
            auto v4d16_n1 = neon::vload4(src_n1);
            auto v4d16_n2 = neon::vload4(src_n2);

            int32x4_t vqs32_c_12  = neon::vaddl(v4d16_c.val[1], v4d16_c.val[2]);
            int32x4_t vqs32_c_03  = neon::vaddl(v4d16_c.val[0], v4d16_c.val[3]);
            int32x4_t vqs32_n0_12 = neon::vaddl(v4d16_n0.val[1], v4d16_n0.val[2]);
            int32x4_t vqs32_n0_03 = neon::vaddl(v4d16_n0.val[0], v4d16_n0.val[3]);
            int32x4_t vqs32_n1_12 = neon::vaddl(v4d16_n1.val[1], v4d16_n1.val[2]);
            int32x4_t vqs32_n1_03 = neon::vaddl(v4d16_n1.val[0], v4d16_n1.val[3]);
            int32x4_t vqs32_n2_12 = neon::vaddl(v4d16_n2.val[1], v4d16_n2.val[2]);
            int32x4_t vqs32_n2_03 = neon::vaddl(v4d16_n2.val[0], v4d16_n2.val[3]);

            int32x4_t vqs32_c_x19     = neon::vmul(vqs32_c_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_c_result  = neon::vmls(vqs32_c_x19, vqs32_c_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n0_x19    = neon::vmul(vqs32_n0_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n0_result = neon::vmls(vqs32_n0_x19, vqs32_n0_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n1_x19    = neon::vmul(vqs32_n1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n1_result = neon::vmls(vqs32_n1_x19, vqs32_n1_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_n2_x19    = neon::vmul(vqs32_n2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_n2_result = neon::vmls(vqs32_n2_x19, vqs32_n2_03, static_cast<MI_S32>(3));

            int32x4_t vqs32_result_12  = neon::vadd(vqs32_n0_result, vqs32_n1_result);
            int32x4_t vqs32_result_03  = neon::vadd(vqs32_c_result, vqs32_n2_result);
            int32x4_t vqs32_result_x19 = neon::vmul(vqs32_result_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_result     = neon::vmls(vqs32_result_x19, vqs32_result_03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des        = neon::vrshr_n<10>(vqs32_result);

            VType vd16_result;
            if (std::is_same<Tp, MI_U16>::value)
            {
                int32x4_t vqs32_zero;
                neon::vdup(vqs32_zero, static_cast<MI_S32>(0));
                int32x4_t vqs32_65535;
                neon::vdup(vqs32_65535, static_cast<MI_S32>(65535));

                vqs32_des      = neon::vmax(vqs32_des, vqs32_zero);
                vqs32_des      = neon::vmin(vqs32_des, vqs32_65535);
                auto vds16_des = neon::vmovn(vqs32_des);
                vd16_result    = neon::vreinterpret(vds16_des);
                neon::vstore(dst_row, vd16_result);
            }
            else
            {
                int32x4_t vqs32_n32768;
                neon::vdup(vqs32_n32768, static_cast<MI_S32>(-32768));
                int32x4_t vqs32_p32767;
                neon::vdup(vqs32_p32767, static_cast<MI_S32>(32767));

                vqs32_des   = neon::vmax(vqs32_des, vqs32_n32768);
                vqs32_des   = neon::vmin(vqs32_des, vqs32_p32767);
                vd16_result = neon::vmovn(vqs32_des);
                neon::vstore(dst_row, vd16_result);
            }

            dst_row += 4;
            src_c   += 16;
            src_n0  += 16;
            src_n1  += 16;
            src_n2  += 16;
        }

        for (; x < owidth; x++)
        {
            MI_S32 y0 = (src_c[1] + src_c[2]) * 19 - (src_c[0] + src_c[3]) * 3;
            MI_S32 y1 = (src_n0[1] + src_n0[2]) * 19 - (src_n0[0] + src_n0[3]) * 3;
            MI_S32 y2 = (src_n1[1] + src_n1[2]) * 19 - (src_n1[0] + src_n1[3]) * 3;
            MI_S32 y3 = (src_n2[1] + src_n2[2]) * 19 - (src_n2[0] + src_n2[3]) * 3;

            MI_S32 temp = (y1 * 19 - y0 * 3 + y2 * 19 - y3 * 3 + 512) >> 10;
            *dst_row    = SaturateCast<Tp>(temp);

            dst_row++;
            src_c  += 4;
            src_n0 += 4;
            src_n1 += 4;
            src_n2 += 4;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeCuC1DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
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
            float16x4x4_t v4df16_c  = neon::vload4(src_c);
            float16x4x4_t v4df16_n0 = neon::vload4(src_n0);
            float16x4x4_t v4df16_n1 = neon::vload4(src_n1);
            float16x4x4_t v4df16_n2 = neon::vload4(src_n2);

            float32x4_t vqf32_c_12  = neon::vadd(neon::vcvt<MI_F32>(v4df16_c.val[1]),  neon::vcvt<MI_F32>(v4df16_c.val[2]));
            float32x4_t vqf32_c_03  = neon::vadd(neon::vcvt<MI_F32>(v4df16_c.val[0]),  neon::vcvt<MI_F32>(v4df16_c.val[3]));
            float32x4_t vqf32_n0_12 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n0.val[1]), neon::vcvt<MI_F32>(v4df16_n0.val[2]));
            float32x4_t vqf32_n0_03 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n0.val[0]), neon::vcvt<MI_F32>(v4df16_n0.val[3]));
            float32x4_t vqf32_n1_12 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n1.val[1]), neon::vcvt<MI_F32>(v4df16_n1.val[2]));
            float32x4_t vqf32_n1_03 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n1.val[0]), neon::vcvt<MI_F32>(v4df16_n1.val[3]));
            float32x4_t vqf32_n2_12 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n2.val[1]), neon::vcvt<MI_F32>(v4df16_n2.val[2]));
            float32x4_t vqf32_n2_03 = neon::vadd(neon::vcvt<MI_F32>(v4df16_n2.val[0]), neon::vcvt<MI_F32>(v4df16_n2.val[3]));

            float32x4_t vqf32_c_x19     = neon::vmul(vqf32_c_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_c_x3      = neon::vmul(vqf32_c_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_c_result  = neon::vadd(vqf32_c_x19, vqf32_c_x3);
            float32x4_t vqf32_n0_x19    = neon::vmul(vqf32_n0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n0_x3     = neon::vmul(vqf32_n0_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n0_result = neon::vadd(vqf32_n0_x19, vqf32_n0_x3);
            float32x4_t vqf32_n1_x19    = neon::vmul(vqf32_n1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n1_x3     = neon::vmul(vqf32_n1_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n1_result = neon::vadd(vqf32_n1_x19, vqf32_n1_x3);
            float32x4_t vqf32_n2_x19    = neon::vmul(vqf32_n2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n2_x3     = neon::vmul(vqf32_n2_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_n2_result = neon::vadd(vqf32_n2_x19, vqf32_n2_x3);

            float32x4_t vqf32_result_12 = neon::vadd(vqf32_n0_result, vqf32_n1_result);
            float32x4_t vqf32_result_03 = neon::vadd(vqf32_c_result,  vqf32_n2_result);

            float32x4_t vqf32_x19 = neon::vmul(vqf32_result_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_x3  = neon::vmul(vqf32_result_03, static_cast<MI_F32>(-0.09375));
            float32x4_t vqf32_des = neon::vadd(vqf32_x19, vqf32_x3);

            float16x4_t vdf16_result = neon::vcvt<MI_F16>(vqf32_des);

            neon::vstore(dst_row, vdf16_result);

            dst_row += 4;
            src_c   += 16;
            src_n0  += 16;
            src_n1  += 16;
            src_n2  += 16;
        }

        for (; x < owidth; x++)
        {
            MI_F32 y0 = (src_c[1] + src_c[2]) * 0.59375f + (src_c[0] + src_c[3]) * (-0.09375f);
            MI_F32 y1 = (src_n0[1] + src_n0[2]) * 0.59375f + (src_n0[0] + src_n0[3]) * (-0.09375f);
            MI_F32 y2 = (src_n1[1] + src_n1[2]) * 0.59375f + (src_n1[0] + src_n1[3]) * (-0.09375f);
            MI_F32 y3 = (src_n2[1] + src_n2[2]) * 0.59375f + (src_n2[0] + src_n2[3]) * (-0.09375f);

            *dst_row = SaturateCast<Tp>((y1 + y2) * 0.59375f + (y0 + y3) * (-0.09375f));

            dst_row++;
            src_c  += 4;
            src_n0 += 4;
            src_n1 += 4;
            src_n2 += 4;
        }
    }

    return Status::OK;
}
#endif

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeCuC1DownX4NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
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
            float32x4x4_t v4qf32_c  = neon::vload4q(src_c);
            float32x4x4_t v4qf32_n0 = neon::vload4q(src_n0);
            float32x4x4_t v4qf32_n1 = neon::vload4q(src_n1);
            float32x4x4_t v4qf32_n2 = neon::vload4q(src_n2);

            float32x4_t vqf32_c_12  = neon::vadd(v4qf32_c.val[1], v4qf32_c.val[2]);
            float32x4_t vqf32_c_03  = neon::vadd(v4qf32_c.val[0], v4qf32_c.val[3]);
            float32x4_t vqf32_n0_12 = neon::vadd(v4qf32_n0.val[1], v4qf32_n0.val[2]);
            float32x4_t vqf32_n0_03 = neon::vadd(v4qf32_n0.val[0], v4qf32_n0.val[3]);
            float32x4_t vqf32_n1_12 = neon::vadd(v4qf32_n1.val[1], v4qf32_n1.val[2]);
            float32x4_t vqf32_n1_03 = neon::vadd(v4qf32_n1.val[0], v4qf32_n1.val[3]);
            float32x4_t vqf32_n2_12 = neon::vadd(v4qf32_n2.val[1], v4qf32_n2.val[2]);
            float32x4_t vqf32_n2_03 = neon::vadd(v4qf32_n2.val[0], v4qf32_n2.val[3]);

            float32x4_t vqf32_c_x19     = neon::vmul(vqf32_c_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_c_result  = neon::vmls(vqf32_c_x19, vqf32_c_03, static_cast<MI_F32>(0.09375));
            float32x4_t vqf32_n0_x19    = neon::vmul(vqf32_n0_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n0_result = neon::vmls(vqf32_n0_x19, vqf32_n0_03, static_cast<MI_F32>(0.09375));
            float32x4_t vqf32_n1_x19    = neon::vmul(vqf32_n1_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n1_result = neon::vmls(vqf32_n1_x19, vqf32_n1_03, static_cast<MI_F32>(0.09375));
            float32x4_t vqf32_n2_x19    = neon::vmul(vqf32_n2_12, static_cast<MI_F32>(0.59375));
            float32x4_t vqf32_n2_result = neon::vmls(vqf32_n2_x19, vqf32_n2_03, static_cast<MI_F32>(0.09375));

            float32x2_t vdf32_result_lo12 = neon::vadd(neon::vgetlow(vqf32_n0_result), neon::vgetlow(vqf32_n1_result));
            float32x2_t vdf32_result_hi12 = neon::vadd(neon::vgethigh(vqf32_n0_result), neon::vgethigh(vqf32_n1_result));
            float32x2_t vdf32_result_lo03 = neon::vadd(neon::vgetlow(vqf32_c_result), neon::vgetlow(vqf32_n2_result));
            float32x2_t vdf32_result_hi03 = neon::vadd(neon::vgethigh(vqf32_c_result), neon::vgethigh(vqf32_n2_result));

            float32x2_t vdf32_lo_x19    = neon::vmul(vdf32_result_lo12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_result_lo = neon::vmls(vdf32_lo_x19, vdf32_result_lo03, static_cast<MI_F32>(0.09375));
            float32x2_t vdf32_hi_x19    = neon::vmul(vdf32_result_hi12, static_cast<MI_F32>(0.59375));
            float32x2_t vdf32_result_hi = neon::vmls(vdf32_hi_x19, vdf32_result_hi03, static_cast<MI_F32>(0.09375));
            float32x4_t vqf32_des       = neon::vcombine(vdf32_result_lo, vdf32_result_hi);
            neon::vstore(dst_row, vqf32_des);

            dst_row += 4;
            src_c   += 16;
            src_n0  += 16;
            src_n1  += 16;
            src_n2  += 16;
        }

        for (; x < owidth; x++)
        {
            MI_F32 y0 = (src_c[1] + src_c[2]) * 0.59375f + (src_c[0] + src_c[3]) * (-0.09375f);
            MI_F32 y1 = (src_n0[1] + src_n0[2]) * 0.59375f + (src_n0[0] + src_n0[3]) * (-0.09375f);
            MI_F32 y2 = (src_n1[1] + src_n1[2]) * 0.59375f + (src_n1[0] + src_n1[3]) * (-0.09375f);
            MI_F32 y3 = (src_n2[1] + src_n2[2]) * 0.59375f + (src_n2[0] + src_n2[3]) * (-0.09375f);

            *dst_row = (y1 + y2) * 0.59375f + (y0 + y3) * (-0.09375f);

            dst_row++;
            src_c  += 4;
            src_n0 += 4;
            src_n1 += 4;
            src_n2 += 4;
        }
    }

    return Status::OK;
}

// Tp = MI_U8, MI_S8
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeCuC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MDVector<Tp, 2>::MVType;
    using VType    = typename neon::DVector<Tp>::VType;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;

    MI_S16 *rows = thread_buffer.GetThreadData<MI_S16>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }
    MI_S16 *rows0 = rows;
    MI_S16 *rows1 = rows0 + owidth;
    MI_S16 *rows2 = rows1 + owidth;
    MI_S16 *rows3 = rows2 + owidth;

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

    rows0_x[0] = src_c[0] * 16 + src_c[1] * 19 - src_c[2] * 3;
    rows1_x[0] = src_n0[0] * 16 + src_n0[1] * 19 - src_n0[2] * 3;
    rows2_x[0] = src_n1[0] * 16 + src_n1[1] * 19 - src_n1[2] * 3;
    rows3_x[0] = src_n2[0] * 16 + src_n2[1] * 19 - src_n2[2] * 3;

    src_c++;
    src_n0++;
    src_n1++;
    src_n2++;

    rows0_x++;
    rows1_x++;
    rows2_x++;
    rows3_x++;

    MI_S32 owidth_align8 = (owidth - 2) & (-8);
    MI_S32 dx = 0;
    for (; dx < owidth_align8; dx += 8)
    {
        MVType mvd8_cx0  = neon::vload2(src_c);
        MVType mvd8_n0x0 = neon::vload2(src_n0);
        MVType mvd8_n1x0 = neon::vload2(src_n1);
        MVType mvd8_n2x0 = neon::vload2(src_n2);
        MVType mvd8_cx1  = neon::vload2(src_c + 2);
        MVType mvd8_n0x1 = neon::vload2(src_n0 + 2);
        MVType mvd8_n1x1 = neon::vload2(src_n1 + 2);
        MVType mvd8_n2x1 = neon::vload2(src_n2 + 2);

        int16x8_t vqs16_c_03  = neon::vaddl(mvd8_cx0.val[0], mvd8_cx1.val[1]);
        int16x8_t vqs16_c_12  = neon::vaddl(mvd8_cx0.val[1], mvd8_cx1.val[0]);
        int16x8_t vqs16_n0_03 = neon::vaddl(mvd8_n0x0.val[0], mvd8_n0x1.val[1]);
        int16x8_t vqs16_n0_12 = neon::vaddl(mvd8_n0x0.val[1], mvd8_n0x1.val[0]);
        int16x8_t vqs16_n1_03 = neon::vaddl(mvd8_n1x0.val[0], mvd8_n1x1.val[1]);
        int16x8_t vqs16_n1_12 = neon::vaddl(mvd8_n1x0.val[1], mvd8_n1x1.val[0]);
        int16x8_t vqs16_n2_03 = neon::vaddl(mvd8_n2x0.val[0], mvd8_n2x1.val[1]);
        int16x8_t vqs16_n2_12 = neon::vaddl(mvd8_n2x0.val[1], mvd8_n2x1.val[0]);

        int16x8_t vqs16_c_x19   = neon::vmul(vqs16_c_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_result0 = neon::vmls(vqs16_c_x19, vqs16_c_03, static_cast<MI_S16>(3));

        int16x8_t vqs16_n0_x19  = neon::vmul(vqs16_n0_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_result1 = neon::vmls(vqs16_n0_x19, vqs16_n0_03, static_cast<MI_S16>(3));

        int16x8_t vqs16_n1_x19  = neon::vmul(vqs16_n1_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_result2 = neon::vmls(vqs16_n1_x19, vqs16_n1_03, static_cast<MI_S16>(3));

        int16x8_t vqs16_n2_x19  = neon::vmul(vqs16_n2_12, static_cast<MI_S16>(19));
        int16x8_t vqs16_result3 = neon::vmls(vqs16_n2_x19, vqs16_n2_03, static_cast<MI_S16>(3));

        neon::vstore(rows0_x, vqs16_result0);
        neon::vstore(rows1_x, vqs16_result1);
        neon::vstore(rows2_x, vqs16_result2);
        neon::vstore(rows3_x, vqs16_result3);

        rows0_x += 8;
        rows1_x += 8;
        rows2_x += 8;
        rows3_x += 8;

        src_c  += 16;
        src_n0 += 16;
        src_n1 += 16;
        src_n2 += 16;
    }

    for (; dx < (owidth - 2); dx++)
    {
        *rows0_x = src_c[1] * 19 - src_c[0] * 3 + src_c[2] * 19 - src_c[3] * 3;
        *rows1_x = src_n0[1] * 19 - src_n0[0] * 3 + src_n0[2] * 19 - src_n0[3] * 3;
        *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 19 - src_n1[3] * 3;
        *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 19 - src_n2[3] * 3;

        rows0_x++;
        rows1_x++;
        rows2_x++;
        rows3_x++;

        src_c  += 2;
        src_n0 += 2;
        src_n1 += 2;
        src_n2 += 2;
    }

    *rows0_x = src_c[1] * 19 - src_c[0] * 3 + src_c[2] * 16;
    *rows1_x = src_n0[1] * 19 - src_n0[0] * 3 + src_n0[2] * 16;
    *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 16;
    *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 16;

    // vresize
    MI_S16 *rows0_y = rows0;
    MI_S16 *rows1_y = rows1;
    MI_S16 *rows2_y = rows2;
    MI_S16 *rows3_y = rows3;

    Tp *dst_row = dst.Ptr<Tp>(start_row);

    int16x4_t vds16_zero;
    neon::vdup(vds16_zero, static_cast<MI_S16>(0));
    int16x4_t vds16_255;
    neon::vdup(vds16_255, static_cast<MI_S16>(255));

    owidth_align8 = owidth & (-8);
    dx = 0;
    for (; dx < owidth_align8; dx += 8)
    {
        int16x8_t vqs16_c_y  = neon::vload1q(rows0_y);
        int16x8_t vqs16_n0_y = neon::vload1q(rows1_y);
        int16x8_t vqs16_n1_y = neon::vload1q(rows2_y);
        int16x8_t vqs16_n2_y = neon::vload1q(rows3_y);

        int32x4_t vqs32_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_y), neon::vgetlow(vqs16_n1_y));
        int32x4_t vqs32_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_y), neon::vgetlow(vqs16_n2_y));
        int32x4_t vqs32_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_y), neon::vgethigh(vqs16_n1_y));
        int32x4_t vqs32_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_y), neon::vgethigh(vqs16_n2_y));

        int32x4_t vqs32_lo_x19 = neon::vmul(vqs32_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi_x19 = neon::vmul(vqs32_hi12, static_cast<MI_S32>(19));

        int32x4_t vqs32_des_lo = neon::vmls(vqs32_lo_x19, vqs32_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des_hi = neon::vmls(vqs32_hi_x19, vqs32_hi03, static_cast<MI_S32>(3));

        int16x4_t vds16_des_lo = neon::vrshrn_n<10>(vqs32_des_lo);
        int16x4_t vds16_des_hi = neon::vrshrn_n<10>(vqs32_des_hi);

        VType vd8_result;
        if (std::is_same<Tp, MI_U8>::value)
        {
            vds16_des_lo = neon::vmax(vds16_des_lo, vds16_zero);
            vds16_des_hi = neon::vmax(vds16_des_hi, vds16_zero);
            vds16_des_lo = neon::vmin(vds16_des_lo, vds16_255);
            vds16_des_hi = neon::vmin(vds16_des_hi, vds16_255);

            uint16x4_t vdu16_des_lo = neon::vreinterpret(vds16_des_lo);
            uint16x4_t vdu16_des_hi = neon::vreinterpret(vds16_des_hi);

            uint16x8_t vqu16_des = neon::vcombine(vdu16_des_lo, vdu16_des_hi);
            vd8_result = neon::vqmovn(vqu16_des);
            neon::vstore(dst_row, vd8_result);
        }
        else
        {
            int16x8_t vqs16_des = neon::vcombine(vds16_des_lo, vds16_des_hi);
            vd8_result = neon::vqmovn(vqs16_des);
            neon::vstore(dst_row, vd8_result);
        }

        dst_row += 8;
        rows0_y += 8;
        rows1_y += 8;
        rows2_y += 8;
        rows3_y += 8;
    }

    for (; dx < owidth; dx++)
    {
        MI_S32 result = (*rows1_y * 19 - *rows0_y * 3 + *rows2_y * 19 - *rows3_y * 3 + 512) >> 10;
        *dst_row = SaturateCast<Tp>(result);

        dst_row++;
        rows0_y++;
        rows1_y++;
        rows2_y++;
        rows3_y++;
    }

    // Line 1 ~ h
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
        MI_S16 *rows2_x = rows2;
        MI_S16 *rows3_x = rows3;

        if (end_row == oheight && dy == (end_row - 1))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 16 + src_n1[1] * 19 - src_n1[2] * 3;
        rows3_x[0] = src_n2[0] * 16 + src_n2[1] * 19 - src_n2[2] * 3;

        src_n1++;
        src_n2++;

        rows2_x++;
        rows3_x++;

        MI_S32 owidth_align8 = (owidth - 2) & (-8);
        MI_S32 dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            MVType mvd8_n1x0 = neon::vload2(src_n1);
            MVType mvd8_n2x0 = neon::vload2(src_n2);
            MVType mvd8_n1x1 = neon::vload2(src_n1 + 2);
            MVType mvd8_n2x1 = neon::vload2(src_n2 + 2);

            int16x8_t vqs16_n1_03 = neon::vaddl(mvd8_n1x0.val[0], mvd8_n1x1.val[1]);
            int16x8_t vqs16_n1_12 = neon::vaddl(mvd8_n1x0.val[1], mvd8_n1x1.val[0]);
            int16x8_t vqs16_n2_03 = neon::vaddl(mvd8_n2x0.val[0], mvd8_n2x1.val[1]);
            int16x8_t vqs16_n2_12 = neon::vaddl(mvd8_n2x0.val[1], mvd8_n2x1.val[0]);

            int16x8_t vqs16_n1_x19  = neon::vmul(vqs16_n1_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_result2 = neon::vmls(vqs16_n1_x19, vqs16_n1_03, static_cast<MI_S16>(3));

            int16x8_t vqs16_n2_x19  = neon::vmul(vqs16_n2_12, static_cast<MI_S16>(19));
            int16x8_t vqs16_result3 = neon::vmls(vqs16_n2_x19, vqs16_n2_03, static_cast<MI_S16>(3));

            neon::vstore(rows2_x, vqs16_result2);
            neon::vstore(rows3_x, vqs16_result3);

            rows2_x += 8;
            rows3_x += 8;
            src_n1 += 16;
            src_n2 += 16;
        }

        for (; dx < (owidth - 2); dx++)
        {
            *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 19 - src_n1[3] * 3;
            *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 19 - src_n2[3] * 3;

            rows2_x++;
            rows3_x++;

            src_n1 += 2;
            src_n2 += 2;
        }

        *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 16;
        *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 16;

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
            int16x8_t vqs16_c_y  = neon::vload1q(rows0_y);
            int16x8_t vqs16_n0_y = neon::vload1q(rows1_y);
            int16x8_t vqs16_n1_y = neon::vload1q(rows2_y);
            int16x8_t vqs16_n2_y = neon::vload1q(rows3_y);

            int32x4_t vqs32_lo12 = neon::vaddl(neon::vgetlow(vqs16_n0_y), neon::vgetlow(vqs16_n1_y));
            int32x4_t vqs32_lo03 = neon::vaddl(neon::vgetlow(vqs16_c_y), neon::vgetlow(vqs16_n2_y));
            int32x4_t vqs32_hi12 = neon::vaddl(neon::vgethigh(vqs16_n0_y), neon::vgethigh(vqs16_n1_y));
            int32x4_t vqs32_hi03 = neon::vaddl(neon::vgethigh(vqs16_c_y), neon::vgethigh(vqs16_n2_y));

            int32x4_t vqs32_lo_x19 = neon::vmul(vqs32_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi_x19 = neon::vmul(vqs32_hi12, static_cast<MI_S32>(19));

            int32x4_t vqs32_des_lo = neon::vmls(vqs32_lo_x19, vqs32_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des_hi = neon::vmls(vqs32_hi_x19, vqs32_hi03, static_cast<MI_S32>(3));

            int16x4_t vds16_des_lo = neon::vrshrn_n<10>(vqs32_des_lo);
            int16x4_t vds16_des_hi = neon::vrshrn_n<10>(vqs32_des_hi);

            VType vd8_result;
            if (std::is_same<Tp, MI_U8>::value)
            {
                vds16_des_lo = neon::vmax(vds16_des_lo, vds16_zero);
                vds16_des_hi = neon::vmax(vds16_des_hi, vds16_zero);
                vds16_des_lo = neon::vmin(vds16_des_lo, vds16_255);
                vds16_des_hi = neon::vmin(vds16_des_hi, vds16_255);

                uint16x4_t vdu16_des_lo = neon::vreinterpret(vds16_des_lo);
                uint16x4_t vdu16_des_hi = neon::vreinterpret(vds16_des_hi);

                uint16x8_t vqu16_des = neon::vcombine(vdu16_des_lo, vdu16_des_hi);
                vd8_result = neon::vqmovn(vqu16_des);
                neon::vstore(dst_row, vd8_result);
            }
            else
            {
                int16x8_t vqs16_des = neon::vcombine(vds16_des_lo, vds16_des_hi);
                vd8_result = neon::vqmovn(vqs16_des);
                neon::vstore(dst_row, vd8_result);
            }

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
            rows2_y += 8;
            rows3_y += 8;
        }

        for (; dx < owidth; dx++)
        {
            MI_S32 result = (*rows1_y * 19 - *rows0_y * 3 + *rows2_y * 19 - *rows3_y * 3 + 512) >> 10;
            *dst_row = SaturateCast<Tp>(result);

            dst_row++;
            rows0_y++;
            rows1_y++;
            rows2_y++;
            rows3_y++;
        }
    }

    return Status::OK;
}

// Tp = MI_U16, MI_S16
template <typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeCuC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MDVector<Tp, 2>::MVType;
    using VType    = typename neon::QVector<Tp>::VType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MI_S32 *rows = thread_buffer.GetThreadData<MI_S32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_S32 *rows0 = rows;
    MI_S32 *rows1 = rows0 + owidth;
    MI_S32 *rows2 = rows1 + owidth;
    MI_S32 *rows3 = rows2 + owidth;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_S32 *rows0_x = rows0;
    MI_S32 *rows1_x = rows1;
    MI_S32 *rows2_x = rows2;
    MI_S32 *rows3_x = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 16 + src_c[1] * 19 - src_c[2] * 3;
    rows1_x[0] = src_n0[0] * 16 + src_n0[1] * 19 - src_n0[2] * 3;
    rows2_x[0] = src_n1[0] * 16 + src_n1[1] * 19 - src_n1[2] * 3;
    rows3_x[0] = src_n2[0] * 16 + src_n2[1] * 19 - src_n2[2] * 3;

    src_c++;
    src_n0++;
    src_n1++;
    src_n2++;

    rows0_x++;
    rows1_x++;
    rows2_x++;
    rows3_x++;

    MI_S32 owidth_align4 = (owidth - 2) & (-4);
    MI_S32 dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        MVType mvd16_cx0  = neon::vload2(src_c);
        MVType mvd16_n0x0 = neon::vload2(src_n0);
        MVType mvd16_n1x0 = neon::vload2(src_n1);
        MVType mvd16_n2x0 = neon::vload2(src_n2);
        MVType mvd16_cx1  = neon::vload2(src_c + 2);
        MVType mvd16_n0x1 = neon::vload2(src_n0 + 2);
        MVType mvd16_n1x1 = neon::vload2(src_n1 + 2);
        MVType mvd16_n2x1 = neon::vload2(src_n2 + 2);

        int32x4_t vqs32_c_03  = neon::vaddl(mvd16_cx0.val[0], mvd16_cx1.val[1]);
        int32x4_t vqs32_c_12  = neon::vaddl(mvd16_cx0.val[1], mvd16_cx1.val[0]);
        int32x4_t vqs32_n0_03 = neon::vaddl(mvd16_n0x0.val[0], mvd16_n0x1.val[1]);
        int32x4_t vqs32_n0_12 = neon::vaddl(mvd16_n0x0.val[1], mvd16_n0x1.val[0]);
        int32x4_t vqs32_n1_03 = neon::vaddl(mvd16_n1x0.val[0], mvd16_n1x1.val[1]);
        int32x4_t vqs32_n1_12 = neon::vaddl(mvd16_n1x0.val[1], mvd16_n1x1.val[0]);
        int32x4_t vqs32_n2_03 = neon::vaddl(mvd16_n2x0.val[0], mvd16_n2x1.val[1]);
        int32x4_t vqs32_n2_12 = neon::vaddl(mvd16_n2x0.val[1], mvd16_n2x1.val[0]);

        int32x4_t vqs32_c_x19   = neon::vmul(vqs32_c_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_result0 = neon::vmls(vqs32_c_x19, vqs32_c_03, static_cast<MI_S32>(3));

        int32x4_t vqs32_n0_x19  = neon::vmul(vqs32_n0_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_result1 = neon::vmls(vqs32_n0_x19, vqs32_n0_03, static_cast<MI_S32>(3));

        int32x4_t vqs32_n1_x19  = neon::vmul(vqs32_n1_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_result2 = neon::vmls(vqs32_n1_x19, vqs32_n1_03, static_cast<MI_S32>(3));

        int32x4_t vqs32_n2_x19  = neon::vmul(vqs32_n2_12, static_cast<MI_S32>(19));
        int32x4_t vqs32_result3 = neon::vmls(vqs32_n2_x19, vqs32_n2_03, static_cast<MI_S32>(3));

        neon::vstore(rows0_x, vqs32_result0);
        neon::vstore(rows1_x, vqs32_result1);
        neon::vstore(rows2_x, vqs32_result2);
        neon::vstore(rows3_x, vqs32_result3);

        rows0_x += 4;
        rows1_x += 4;
        rows2_x += 4;
        rows3_x += 4;

        src_c  += 8;
        src_n0 += 8;
        src_n1 += 8;
        src_n2 += 8;
    }

    for (; dx < (owidth - 2); dx++)
    {
        *rows0_x = src_c[1] * 19 - src_c[0] * 3 + src_c[2] * 19 - src_c[3] * 3;
        *rows1_x = src_n0[1] * 19 - src_n0[0] * 3 + src_n0[2] * 19 - src_n0[3] * 3;
        *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 19 - src_n1[3] * 3;
        *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 19 - src_n2[3] * 3;

        rows0_x++;
        rows1_x++;
        rows2_x++;
        rows3_x++;

        src_c  += 2;
        src_n0 += 2;
        src_n1 += 2;
        src_n2 += 2;
    }

    *rows0_x = src_c[1] * 19 - src_c[0] * 3 + src_c[2] * 16;
    *rows1_x = src_n0[1] * 19 - src_n0[0] * 3 + src_n0[2] * 16;
    *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 16;
    *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 16;

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
        int32x4_t vqs32_c_lo  = neon::vload1q(rows0_y);
        int32x4_t vqs32_n0_lo = neon::vload1q(rows1_y);
        int32x4_t vqs32_n1_lo = neon::vload1q(rows2_y);
        int32x4_t vqs32_n2_lo = neon::vload1q(rows3_y);

        int32x4_t vqs32_c_hi  = neon::vload1q(rows0_y + 4);
        int32x4_t vqs32_n0_hi = neon::vload1q(rows1_y + 4);
        int32x4_t vqs32_n1_hi = neon::vload1q(rows2_y + 4);
        int32x4_t vqs32_n2_hi = neon::vload1q(rows3_y + 4);

        int32x4_t vqs32_lo12 = neon::vadd(vqs32_n0_lo, vqs32_n1_lo);
        int32x4_t vqs32_lo03 = neon::vadd(vqs32_c_lo, vqs32_n2_lo);
        int32x4_t vqs32_hi12 = neon::vadd(vqs32_n0_hi, vqs32_n1_hi);
        int32x4_t vqs32_hi03 = neon::vadd(vqs32_c_hi, vqs32_n2_hi);

        int32x4_t vqs32_lo_x19 = neon::vmul(vqs32_lo12, static_cast<MI_S32>(19));
        int32x4_t vqs32_hi_x19 = neon::vmul(vqs32_hi12, static_cast<MI_S32>(19));

        int32x4_t vqs32_des_lo = neon::vmls(vqs32_lo_x19, vqs32_lo03, static_cast<MI_S32>(3));
        int32x4_t vqs32_des_hi = neon::vmls(vqs32_hi_x19, vqs32_hi03, static_cast<MI_S32>(3));

        VType vq16_result;
        if (std::is_same<Tp, MI_U16>::value)
        {
            int32x4_t vqs32_zero;
            neon::vdup(vqs32_zero, static_cast<MI_S32>(0));

            vqs32_des_lo = neon::vmax(vqs32_des_lo, vqs32_zero);
            vqs32_des_hi = neon::vmax(vqs32_des_hi, vqs32_zero);
            uint32x4_t vdu32_des_lo = neon::vreinterpret(vqs32_des_lo);
            uint32x4_t vdu32_des_hi = neon::vreinterpret(vqs32_des_hi);
            uint16x4_t vdu16_des_lo = neon::vqshrn_n<10>(vdu32_des_lo);
            uint16x4_t vdu16_des_hi = neon::vqshrn_n<10>(vdu32_des_hi);
            vq16_result = neon::vcombine(vdu16_des_lo, vdu16_des_hi);
            neon::vstore(dst_row, vq16_result);
        }
        else
        {
            int16x4_t vds16_des_lo = neon::vqshrn_n<10>(vqs32_des_lo);
            int16x4_t vds16_des_hi = neon::vqshrn_n<10>(vqs32_des_hi);
            vq16_result = neon::vcombine(vds16_des_lo, vds16_des_hi);
            neon::vstore(dst_row, vq16_result);
        }

        dst_row += 8;
        rows0_y += 8;
        rows1_y += 8;
        rows2_y += 8;
        rows3_y += 8;
    }

    for (; dx < owidth; dx++)
    {
        MI_S32 result = (*rows1_y * 19 - *rows0_y * 3 + *rows2_y * 19 - *rows3_y * 3 + 512) >> 10;
        *dst_row = SaturateCast<Tp>(result);

        dst_row++;
        rows0_y++;
        rows1_y++;
        rows2_y++;
        rows3_y++;
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
        MI_S32 *rows2_x = rows2;
        MI_S32 *rows3_x = rows3;

        if (end_row == oheight && dy == (end_row - 1))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 16 + src_n1[1] * 19 - src_n1[2] * 3;
        rows3_x[0] = src_n2[0] * 16 + src_n2[1] * 19 - src_n2[2] * 3;
        src_n1++;
        src_n2++;
        rows2_x++;
        rows3_x++;

        MI_S32 owidth_align4 = (owidth - 2) & (-4);
        MI_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            MVType mvd16_n1x0 = neon::vload2(src_n1);
            MVType mvd16_n2x0 = neon::vload2(src_n2);
            MVType mvd16_n1x1 = neon::vload2(src_n1 + 2);
            MVType mvd16_n2x1 = neon::vload2(src_n2 + 2);

            int32x4_t vqs32_n1_03 = neon::vaddl(mvd16_n1x0.val[0], mvd16_n1x1.val[1]);
            int32x4_t vqs32_n1_12 = neon::vaddl(mvd16_n1x0.val[1], mvd16_n1x1.val[0]);
            int32x4_t vqs32_n2_03 = neon::vaddl(mvd16_n2x0.val[0], mvd16_n2x1.val[1]);
            int32x4_t vqs32_n2_12 = neon::vaddl(mvd16_n2x0.val[1], mvd16_n2x1.val[0]);

            int32x4_t vqs32_n1_x19  = neon::vmul(vqs32_n1_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_result2 = neon::vmls(vqs32_n1_x19, vqs32_n1_03, static_cast<MI_S32>(3));

            int32x4_t vqs32_n2_x19  = neon::vmul(vqs32_n2_12, static_cast<MI_S32>(19));
            int32x4_t vqs32_result3 = neon::vmls(vqs32_n2_x19, vqs32_n2_03, static_cast<MI_S32>(3));

            neon::vstore(rows2_x, vqs32_result2);
            neon::vstore(rows3_x, vqs32_result3);

            rows2_x += 4;
            rows3_x += 4;
            src_n1 += 8;
            src_n2 += 8;
        }

        for (; dx < (owidth - 2); dx++)
        {
            *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 19 - src_n1[3] * 3;
            *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 19 - src_n2[3] * 3;

            rows2_x++;
            rows3_x++;
            src_n1 += 2;
            src_n2 += 2;
        }

        *rows2_x = src_n1[1] * 19 - src_n1[0] * 3 + src_n1[2] * 16;
        *rows3_x = src_n2[1] * 19 - src_n2[0] * 3 + src_n2[2] * 16;

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
            int32x4_t vqs32_c_lo  = neon::vload1q(rows0_y);
            int32x4_t vqs32_n0_lo = neon::vload1q(rows1_y);
            int32x4_t vqs32_n1_lo = neon::vload1q(rows2_y);
            int32x4_t vqs32_n2_lo = neon::vload1q(rows3_y);

            int32x4_t vqs32_c_hi  = neon::vload1q(rows0_y + 4);
            int32x4_t vqs32_n0_hi = neon::vload1q(rows1_y + 4);
            int32x4_t vqs32_n1_hi = neon::vload1q(rows2_y + 4);
            int32x4_t vqs32_n2_hi = neon::vload1q(rows3_y + 4);

            int32x4_t vqs32_lo12 = neon::vadd(vqs32_n0_lo, vqs32_n1_lo);
            int32x4_t vqs32_lo03 = neon::vadd(vqs32_c_lo, vqs32_n2_lo);
            int32x4_t vqs32_hi12 = neon::vadd(vqs32_n0_hi, vqs32_n1_hi);
            int32x4_t vqs32_hi03 = neon::vadd(vqs32_c_hi, vqs32_n2_hi);

            int32x4_t vqs32_lo_x19 = neon::vmul(vqs32_lo12, static_cast<MI_S32>(19));
            int32x4_t vqs32_hi_x19 = neon::vmul(vqs32_hi12, static_cast<MI_S32>(19));

            int32x4_t vqs32_des_lo = neon::vmls(vqs32_lo_x19, vqs32_lo03, static_cast<MI_S32>(3));
            int32x4_t vqs32_des_hi = neon::vmls(vqs32_hi_x19, vqs32_hi03, static_cast<MI_S32>(3));

            VType vq16_result;
            if (std::is_same<Tp, MI_U16>::value)
            {
                int32x4_t vqs32_zero;
                neon::vdup(vqs32_zero, static_cast<MI_S32>(0));

                vqs32_des_lo = neon::vmax(vqs32_des_lo, vqs32_zero);
                vqs32_des_hi = neon::vmax(vqs32_des_hi, vqs32_zero);
                uint32x4_t vdu32_des_lo = neon::vreinterpret(vqs32_des_lo);
                uint32x4_t vdu32_des_hi = neon::vreinterpret(vqs32_des_hi);
                uint16x4_t vdu16_des_lo = neon::vqshrn_n<10>(vdu32_des_lo);
                uint16x4_t vdu16_des_hi = neon::vqshrn_n<10>(vdu32_des_hi);
                vq16_result = neon::vcombine(vdu16_des_lo, vdu16_des_hi);
                neon::vstore(dst_row, vq16_result);
            }
            else
            {
                int16x4_t vds16_des_lo = neon::vqshrn_n<10>(vqs32_des_lo);
                int16x4_t vds16_des_hi = neon::vqshrn_n<10>(vqs32_des_hi);
                vq16_result = neon::vcombine(vds16_des_lo, vds16_des_hi);
                neon::vstore(dst_row, vq16_result);
            }

            dst_row += 8;
            rows0_y += 8;
            rows1_y += 8;
            rows2_y += 8;
            rows3_y += 8;
        }

        for (; dx < owidth; dx++)
        {
            MI_S32 result = (*rows1_y * 19 - *rows0_y * 3 + *rows2_y * 19 - *rows3_y * 3 + 512) >> 10;
            *dst_row = SaturateCast<Tp>(result);

            dst_row++;
            rows0_y++;
            rows1_y++;
            rows2_y++;
            rows3_y++;
        }
    }

    return Status::OK;
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeCuC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MDVector<Tp, 2>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MI_F32 coef0 = -0.093750; // GetCuCoef(1.5f);
    MI_F32 coef1 = 0.593750; // GetCuCoef(0.5f);

    MI_F32 *rows = thread_buffer.GetThreadData<MI_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_F32 *rows0 = rows;
    MI_F32 *rows1 = rows0 + owidth;
    MI_F32 *rows2 = rows1 + owidth;
    MI_F32 *rows3 = rows2 + owidth;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    // Line 0
    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_F32 *rows0_x = rows0;
    MI_F32 *rows1_x = rows1;
    MI_F32 *rows2_x = rows2;
    MI_F32 *rows3_x = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 0.5f + src_c[1] * coef1 + src_c[2] * coef0;
    rows1_x[0] = src_n0[0] * 0.5f + src_n0[1] * coef1 + src_n0[2] * coef0;
    rows2_x[0] = src_n1[0] * 0.5f + src_n1[1] * coef1 + src_n1[2] * coef0;
    rows3_x[0] = src_n2[0] * 0.5f + src_n2[1] * coef1 + src_n2[2] * coef0;

    src_c++;
    src_n0++;
    src_n1++;
    src_n2++;
    rows0_x++;
    rows1_x++;
    rows2_x++;
    rows3_x++;

    MI_S32 owidth_align4 = (owidth - 2) & (-4);
    MI_S32 dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        MVType mvdf16_cx0  = neon::vload2(src_c);
        MVType mvdf16_n0x0 = neon::vload2(src_n0);
        MVType mvdf16_n1x0 = neon::vload2(src_n1);
        MVType mvdf16_n2x0 = neon::vload2(src_n2);
        MVType mvdf16_cx1  = neon::vload2(src_c + 2);
        MVType mvdf16_n0x1 = neon::vload2(src_n0 + 2);
        MVType mvdf16_n1x1 = neon::vload2(src_n1 + 2);
        MVType mvdf16_n2x1 = neon::vload2(src_n2 + 2);

        // vmul 3||19
        float32x4_t vqf32_cc_x3    = neon::vmul(neon::vcvt<MI_F32>(mvdf16_cx0.val[0]), coef0);
        float32x4_t vqf32_cc_x19   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_cx0.val[1]), coef1);
        float32x4_t vqf32_cr0_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_cx1.val[1]), coef0);
        float32x4_t vqf32_cr0_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_cx1.val[0]), coef1);
        float32x4_t vqf32_n0c_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n0x0.val[0]), coef0);
        float32x4_t vqf32_n0c_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n0x0.val[1]), coef1);
        float32x4_t vqf32_n0r0_x3  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n0x1.val[1]), coef0);
        float32x4_t vqf32_n0r0_x19 = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n0x1.val[0]), coef1);
        float32x4_t vqf32_n1c_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x0.val[0]), coef0);
        float32x4_t vqf32_n1c_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x0.val[1]), coef1);
        float32x4_t vqf32_n1r0_x3  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x1.val[1]), coef0);
        float32x4_t vqf32_n1r0_x19 = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x1.val[0]), coef1);
        float32x4_t vqf32_n2c_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x0.val[0]), coef0);
        float32x4_t vqf32_n2c_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x0.val[1]), coef1);
        float32x4_t vqf32_n2r0_x3  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x1.val[1]), coef0);
        float32x4_t vqf32_n2r0_x19 = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x1.val[0]), coef1);

        // vsub x19 - x3
        float32x4_t vqf32_c_result0  = neon::vadd(vqf32_cc_x19,   vqf32_cc_x3);
        float32x4_t vqf32_r0_result0 = neon::vadd(vqf32_cr0_x19,  vqf32_cr0_x3);
        float32x4_t vqf32_c_result1  = neon::vadd(vqf32_n0c_x19,  vqf32_n0c_x3);
        float32x4_t vqf32_r0_result1 = neon::vadd(vqf32_n0r0_x19, vqf32_n0r0_x3);
        float32x4_t vqf32_c_result2  = neon::vadd(vqf32_n1c_x19,  vqf32_n1c_x3);
        float32x4_t vqf32_r0_result2 = neon::vadd(vqf32_n1r0_x19, vqf32_n1r0_x3);
        float32x4_t vqf32_c_result3  = neon::vadd(vqf32_n2c_x19,  vqf32_n2c_x3);
        float32x4_t vqf32_r0_result3 = neon::vadd(vqf32_n2r0_x19, vqf32_n2r0_x3);

        float32x4_t vqf32_result0 = neon::vadd(vqf32_c_result0, vqf32_r0_result0);
        float32x4_t vqf32_result1 = neon::vadd(vqf32_c_result1, vqf32_r0_result1);
        float32x4_t vqf32_result2 = neon::vadd(vqf32_c_result2, vqf32_r0_result2);
        float32x4_t vqf32_result3 = neon::vadd(vqf32_c_result3, vqf32_r0_result3);

        neon::vstore(rows0_x, vqf32_result0);
        neon::vstore(rows1_x, vqf32_result1);
        neon::vstore(rows2_x, vqf32_result2);
        neon::vstore(rows3_x, vqf32_result3);

        rows0_x += 4;
        rows1_x += 4;
        rows2_x += 4;
        rows3_x += 4;

        src_c  += 8;
        src_n0 += 8;
        src_n1 += 8;
        src_n2 += 8;
    }

    for (; dx < (owidth - 2); dx++)
    {
        *rows0_x = src_c[0] * coef0 + src_c[1] * coef1 + src_c[2] * coef1 + src_c[3] * coef0;
        *rows1_x = src_n0[0] * coef0 + src_n0[1] * coef1 + src_n0[2] * coef1 + src_n0[3] * coef0;
        *rows2_x = src_n1[0] * coef0 + src_n1[1] * coef1 + src_n1[2] * coef1 + src_n1[3] * coef0;
        *rows3_x = src_n2[0] * coef0 + src_n2[1] * coef1 + src_n2[2] * coef1 + src_n2[3] * coef0;

        rows0_x++;
        rows1_x++;
        rows2_x++;
        rows3_x++;

        src_c  += 2;
        src_n0 += 2;
        src_n1 += 2;
        src_n2 += 2;
    }

    *rows0_x = src_c[1] * coef1 + src_c[0] * coef0 + src_c[2] * 0.5f;
    *rows1_x = src_n0[1] * coef1 + src_n0[0] * coef0 + src_n0[2] * 0.5f;
    *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * 0.5f;
    *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * 0.5f;

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
        float32x4_t vqf32_c  = neon::vload1q(rows0_y);
        float32x4_t vqf32_n0 = neon::vload1q(rows1_y);
        float32x4_t vqf32_n1 = neon::vload1q(rows2_y);
        float32x4_t vqf32_n2 = neon::vload1q(rows3_y);

        float32x4_t vqf32_c_x3   = neon::vmul(vqf32_c, coef0);
        float32x4_t vqf32_n0_x19 = neon::vmul(vqf32_n0, coef1);
        float32x4_t vqf32_n1_x19 = neon::vmul(vqf32_n1, coef1);
        float32x4_t vqf32_n2_x3  = neon::vmul(vqf32_n2, coef0);

        float16x4_t vdf16_result = neon::vcvt<MI_F16>(neon::vadd(neon::vadd(vqf32_n0_x19, vqf32_c_x3), neon::vadd(vqf32_n1_x19, vqf32_n2_x3)));
        neon::vstore(dst_row, vdf16_result);

        dst_row += 4;
        rows0_y += 4;
        rows1_y += 4;
        rows2_y += 4;
        rows3_y += 4;
    }

    for (; dx < owidth; dx++)
    {
        *dst_row = SaturateCast<Tp>((*rows0_y) * coef0 + (*rows1_y) * coef1 + (*rows2_y) * coef1 + (*rows3_y) * coef0);

        dst_row++;
        rows0_y++;
        rows1_y++;
        rows2_y++;
        rows3_y++;
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
        MI_F32 *rows2_x = rows2;
        MI_F32 *rows3_x = rows3;

        if (end_row == oheight && dy == (end_row - 1))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 0.5f + src_n1[1] * coef1 + src_n1[2] * coef0;
        rows3_x[0] = src_n2[0] * 0.5f + src_n2[1] * coef1 + src_n2[2] * coef0;

        src_n1++;
        src_n2++;

        rows2_x++;
        rows3_x++;

        MI_S32 owidth_align4 = (owidth - 2) & (-4);
        MI_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            MVType mvdf16_n1x0 = neon::vload2(src_n1);
            MVType mvdf16_n2x0 = neon::vload2(src_n2);
            MVType mvdf16_n1x1 = neon::vload2(src_n1 + 2);
            MVType mvdf16_n2x1 = neon::vload2(src_n2 + 2);

            // vmul 3||19
            float32x4_t vqf32_n1c_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x0.val[0]), coef0);
            float32x4_t vqf32_n1c_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x0.val[1]), coef1);
            float32x4_t vqf32_n1r0_x19 = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x1.val[0]), coef1);
            float32x4_t vqf32_n1r0_x3  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n1x1.val[1]), coef0);
            float32x4_t vqf32_n2c_x3   = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x0.val[0]), coef0);
            float32x4_t vqf32_n2c_x19  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x0.val[1]), coef1);
            float32x4_t vqf32_n2r0_x19 = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x1.val[0]), coef1);
            float32x4_t vqf32_n2r0_x3  = neon::vmul(neon::vcvt<MI_F32>(mvdf16_n2x1.val[1]), coef0);

            // vsub x19 - x3
            float32x4_t vqf32_c_result2  = neon::vadd(vqf32_n1c_x19,  vqf32_n1c_x3);
            float32x4_t vqf32_r0_result2 = neon::vadd(vqf32_n1r0_x19, vqf32_n1r0_x3);
            float32x4_t vqf32_c_result3  = neon::vadd(vqf32_n2c_x19,  vqf32_n2c_x3);
            float32x4_t vqf32_r0_result3 = neon::vadd(vqf32_n2r0_x19, vqf32_n2r0_x3);

            float32x4_t vqf32_result2 = neon::vadd(vqf32_c_result2, vqf32_r0_result2);
            float32x4_t vqf32_result3 = neon::vadd(vqf32_c_result3, vqf32_r0_result3);

            neon::vstore(rows2_x, vqf32_result2);
            neon::vstore(rows3_x, vqf32_result3);

            rows2_x += 4;
            rows3_x += 4;
            src_n1 += 8;
            src_n2 += 8;
        }

        for (; dx < (owidth - 2); dx++)
        {
            *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * coef1 + src_n1[3] * coef0;
            *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * coef1 + src_n2[3] * coef0;

            rows2_x++;
            rows3_x++;
            src_n1 += 2;
            src_n2 += 2;
        }

        *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * 0.5f;
        *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * 0.5f;

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
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);
            float32x4_t vqf32_n1 = neon::vload1q(rows2_y);
            float32x4_t vqf32_n2 = neon::vload1q(rows3_y);

            float32x4_t vqf32_c_x3   = neon::vmul(vqf32_c, coef0);
            float32x4_t vqf32_n0_x19 = neon::vmul(vqf32_n0, coef1);
            float32x4_t vqf32_n1_x19 = neon::vmul(vqf32_n1, coef1);
            float32x4_t vqf32_n2_x3  = neon::vmul(vqf32_n2, coef0);

            float16x4_t vdf16_result = neon::vcvt<MI_F16>(neon::vadd(neon::vadd(vqf32_n0_x19, vqf32_c_x3), neon::vadd(vqf32_n1_x19, vqf32_n2_x3)));
            neon::vstore(dst_row, vdf16_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
            rows2_y += 4;
            rows3_y += 4;
        }

        for (; dx < owidth; dx++)
        {
            *dst_row = SaturateCast<Tp>(*rows1_y * coef1 + *rows0_y * coef0 + *rows2_y * coef1 + *rows3_y * coef0);

            dst_row++;
            rows0_y++;
            rows1_y++;
            rows2_y++;
            rows3_y++;
        }
    }

    return Status::OK;
}
#endif

// Tp = MI_F32
template <typename Tp>
static typename std::enable_if<std::is_same<MI_F32, Tp>::value, Status>::type
ResizeCuC1DownX2NeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType   = typename neon::MQVector<Tp, 2>::MVType;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;

    MI_F32 coef0 = -0.093750;//GetCuCoef(1.5f);
    MI_F32 coef1 = 0.593750;//GetCuCoef(0.5f);

    MI_F32 *rows = thread_buffer.GetThreadData<MI_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_F32 *rows0 = rows;
    MI_F32 *rows1 = rows0 + owidth;
    MI_F32 *rows2 = rows1 + owidth;
    MI_F32 *rows3 = rows2 + owidth;

    start_row = start_row << 1;
    end_row   = Min(end_row << 1, oheight);

    // Line 0
    const Tp *src_c  = src.Ptr<Tp>((start_row << 1) - 1);
    const Tp *src_n0 = src.Ptr<Tp>(start_row << 1);
    const Tp *src_n1 = src.Ptr<Tp>((start_row << 1) + 1);
    const Tp *src_n2 = src.Ptr<Tp>((start_row << 1) + 2);
    MI_F32 *rows0_x = rows0;
    MI_F32 *rows1_x = rows1;
    MI_F32 *rows2_x = rows2;
    MI_F32 *rows3_x = rows3;

    // Line 0
    if (0 == start_row)
    {
        src_c  = src.Ptr<Tp>(0);
        src_n0 = src.Ptr<Tp>(0);
        src_n1 = src.Ptr<Tp>(1);
        src_n2 = src.Ptr<Tp>(2);
    }

    rows0_x[0] = src_c[0] * 0.5f + src_c[1] * coef1 + src_c[2] * coef0;
    rows1_x[0] = src_n0[0] * 0.5f + src_n0[1] * coef1 + src_n0[2] * coef0;
    rows2_x[0] = src_n1[0] * 0.5f + src_n1[1] * coef1 + src_n1[2] * coef0;
    rows3_x[0] = src_n2[0] * 0.5f + src_n2[1] * coef1 + src_n2[2] * coef0;

    src_c++;
    src_n0++;
    src_n1++;
    src_n2++;
    rows0_x++;
    rows1_x++;
    rows2_x++;
    rows3_x++;

    MI_S32 owidth_align4 = (owidth - 2) & (-4);
    MI_S32 dx = 0;
    for (; dx < owidth_align4; dx += 4)
    {
        MVType mvqf32_cx0  = neon::vload2q(src_c);
        MVType mvqf32_n0x0 = neon::vload2q(src_n0);
        MVType mvqf32_n1x0 = neon::vload2q(src_n1);
        MVType mvqf32_n2x0 = neon::vload2q(src_n2);
        MVType mvqf32_cx1  = neon::vload2q(src_c + 2);
        MVType mvqf32_n0x1 = neon::vload2q(src_n0 + 2);
        MVType mvqf32_n1x1 = neon::vload2q(src_n1 + 2);
        MVType mvqf32_n2x1 = neon::vload2q(src_n2 + 2);

        // vmul 3||19
        float32x4_t vqf32_cc_x3    = neon::vmul(mvqf32_cx0.val[0], coef0);
        float32x4_t vqf32_cc_x19   = neon::vmul(mvqf32_cx0.val[1], coef1);
        float32x4_t vqf32_cr0_x3   = neon::vmul(mvqf32_cx1.val[1], coef0);
        float32x4_t vqf32_cr0_x19  = neon::vmul(mvqf32_cx1.val[0], coef1);
        float32x4_t vqf32_n0c_x3   = neon::vmul(mvqf32_n0x0.val[0], coef0);
        float32x4_t vqf32_n0c_x19  = neon::vmul(mvqf32_n0x0.val[1], coef1);
        float32x4_t vqf32_n0r0_x3  = neon::vmul(mvqf32_n0x1.val[1], coef0);
        float32x4_t vqf32_n0r0_x19 = neon::vmul(mvqf32_n0x1.val[0], coef1);
        float32x4_t vqf32_n1c_x3   = neon::vmul(mvqf32_n1x0.val[0], coef0);
        float32x4_t vqf32_n1c_x19  = neon::vmul(mvqf32_n1x0.val[1], coef1);
        float32x4_t vqf32_n1r0_x3  = neon::vmul(mvqf32_n1x1.val[1], coef0);
        float32x4_t vqf32_n1r0_x19 = neon::vmul(mvqf32_n1x1.val[0], coef1);
        float32x4_t vqf32_n2c_x3   = neon::vmul(mvqf32_n2x0.val[0], coef0);
        float32x4_t vqf32_n2c_x19  = neon::vmul(mvqf32_n2x0.val[1], coef1);
        float32x4_t vqf32_n2r0_x3  = neon::vmul(mvqf32_n2x1.val[1], coef0);
        float32x4_t vqf32_n2r0_x19 = neon::vmul(mvqf32_n2x1.val[0], coef1);

        // vsub x19 - x3
        float32x4_t vqf32_c_result0  = neon::vadd(vqf32_cc_x19, vqf32_cc_x3);
        float32x4_t vqf32_r0_result0 = neon::vadd(vqf32_cr0_x19, vqf32_cr0_x3);
        float32x4_t vqf32_c_result1  = neon::vadd(vqf32_n0c_x19, vqf32_n0c_x3);
        float32x4_t vqf32_r0_result1 = neon::vadd(vqf32_n0r0_x19, vqf32_n0r0_x3);
        float32x4_t vqf32_c_result2  = neon::vadd(vqf32_n1c_x19, vqf32_n1c_x3);
        float32x4_t vqf32_r0_result2 = neon::vadd(vqf32_n1r0_x19, vqf32_n1r0_x3);
        float32x4_t vqf32_c_result3  = neon::vadd(vqf32_n2c_x19, vqf32_n2c_x3);
        float32x4_t vqf32_r0_result3 = neon::vadd(vqf32_n2r0_x19, vqf32_n2r0_x3);

        float32x4_t vqf32_result0 = neon::vadd(vqf32_c_result0, vqf32_r0_result0);
        float32x4_t vqf32_result1 = neon::vadd(vqf32_c_result1, vqf32_r0_result1);
        float32x4_t vqf32_result2 = neon::vadd(vqf32_c_result2, vqf32_r0_result2);
        float32x4_t vqf32_result3 = neon::vadd(vqf32_c_result3, vqf32_r0_result3);

        neon::vstore(rows0_x, vqf32_result0);
        neon::vstore(rows1_x, vqf32_result1);
        neon::vstore(rows2_x, vqf32_result2);
        neon::vstore(rows3_x, vqf32_result3);

        rows0_x += 4;
        rows1_x += 4;
        rows2_x += 4;
        rows3_x += 4;

        src_c  += 8;
        src_n0 += 8;
        src_n1 += 8;
        src_n2 += 8;
    }

    for (; dx < (owidth - 2); dx++)
    {
        *rows0_x = src_c[0] * coef0 + src_c[1] * coef1 + src_c[2] * coef1 + src_c[3] * coef0;
        *rows1_x = src_n0[0] * coef0 + src_n0[1] * coef1 + src_n0[2] * coef1 + src_n0[3] * coef0;
        *rows2_x = src_n1[0] * coef0 + src_n1[1] * coef1 + src_n1[2] * coef1 + src_n1[3] * coef0;
        *rows3_x = src_n2[0] * coef0 + src_n2[1] * coef1 + src_n2[2] * coef1 + src_n2[3] * coef0;
        rows0_x++;
        rows1_x++;
        rows2_x++;
        rows3_x++;

        src_c  += 2;
        src_n0 += 2;
        src_n1 += 2;
        src_n2 += 2;
    }

    *rows0_x = src_c[1] * coef1 + src_c[0] * coef0 + src_c[2] * 0.5f;
    *rows1_x = src_n0[1] * coef1 + src_n0[0] * coef0 + src_n0[2] * 0.5f;
    *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * 0.5f;
    *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * 0.5f;

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
        float32x4_t vqf32_c  = neon::vload1q(rows0_y);
        float32x4_t vqf32_n0 = neon::vload1q(rows1_y);
        float32x4_t vqf32_n1 = neon::vload1q(rows2_y);
        float32x4_t vqf32_n2 = neon::vload1q(rows3_y);

        float32x4_t vqf32_c_x3   = neon::vmul(vqf32_c, coef0);
        float32x4_t vqf32_n0_x19 = neon::vmul(vqf32_n0, coef1);
        float32x4_t vqf32_n1_x19 = neon::vmul(vqf32_n1, coef1);
        float32x4_t vqf32_n2_x3  = neon::vmul(vqf32_n2, coef0);

        float32x4_t vqf32_result = neon::vadd(neon::vadd(vqf32_n0_x19, vqf32_c_x3), neon::vadd(vqf32_n1_x19, vqf32_n2_x3));
        neon::vstore(dst_row, vqf32_result);

        dst_row += 4;
        rows0_y += 4;
        rows1_y += 4;
        rows2_y += 4;
        rows3_y += 4;
    }

    for (; dx < owidth; dx++)
    {
        *dst_row = SaturateCast<Tp>((*rows0_y) * coef0 + (*rows1_y) * coef1 + (*rows2_y) * coef1 + (*rows3_y) * coef0);

        dst_row++;
        rows0_y++;
        rows1_y++;
        rows2_y++;
        rows3_y++;
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
        MI_F32 *rows2_x = rows2;
        MI_F32 *rows3_x = rows3;

        if (end_row == oheight && dy == (end_row - 1))
        {
            src_n2 = src.Ptr<Tp>(sy + 2);
        }

        rows2_x[0] = src_n1[0] * 0.5f + src_n1[1] * coef1 + src_n1[2] * coef0;
        rows3_x[0] = src_n2[0] * 0.5f + src_n2[1] * coef1 + src_n2[2] * coef0;

        src_n1++;
        src_n2++;

        rows2_x++;
        rows3_x++;

        MI_S32 owidth_align4 = (owidth - 2) & (-4);
        MI_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            MVType mvqf32_n1x0 = neon::vload2q(src_n1);
            MVType mvqf32_n2x0 = neon::vload2q(src_n2);
            MVType mvqf32_n1x1 = neon::vload2q(src_n1 + 2);
            MVType mvqf32_n2x1 = neon::vload2q(src_n2 + 2);

            // vmul 3||19
            float32x4_t vqf32_n1c_x3   = neon::vmul(mvqf32_n1x0.val[0], coef0);
            float32x4_t vqf32_n1c_x19  = neon::vmul(mvqf32_n1x0.val[1], coef1);
            float32x4_t vqf32_n1r0_x19 = neon::vmul(mvqf32_n1x1.val[0], coef1);
            float32x4_t vqf32_n1r0_x3  = neon::vmul(mvqf32_n1x1.val[1], coef0);
            float32x4_t vqf32_n2c_x3   = neon::vmul(mvqf32_n2x0.val[0], coef0);
            float32x4_t vqf32_n2c_x19  = neon::vmul(mvqf32_n2x0.val[1], coef1);
            float32x4_t vqf32_n2r0_x19 = neon::vmul(mvqf32_n2x1.val[0], coef1);
            float32x4_t vqf32_n2r0_x3  = neon::vmul(mvqf32_n2x1.val[1], coef0);

            // vsub x19 - x3
            float32x4_t vqf32_c_result2  = neon::vadd(vqf32_n1c_x19, vqf32_n1c_x3);
            float32x4_t vqf32_r0_result2 = neon::vadd(vqf32_n1r0_x19, vqf32_n1r0_x3);
            float32x4_t vqf32_c_result3  = neon::vadd(vqf32_n2c_x19, vqf32_n2c_x3);
            float32x4_t vqf32_r0_result3 = neon::vadd(vqf32_n2r0_x19, vqf32_n2r0_x3);

            float32x4_t vqf32_result2 = neon::vadd(vqf32_c_result2, vqf32_r0_result2);
            float32x4_t vqf32_result3 = neon::vadd(vqf32_c_result3, vqf32_r0_result3);

            neon::vstore(rows2_x, vqf32_result2);
            neon::vstore(rows3_x, vqf32_result3);

            rows2_x += 4;
            rows3_x += 4;
            src_n1 += 8;
            src_n2 += 8;
        }

        for (; dx < (owidth - 2); dx++)
        {
            *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * coef1 + src_n1[3] * coef0;
            *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * coef1 + src_n2[3] * coef0;

            rows2_x++;
            rows3_x++;
            src_n1 += 2;
            src_n2 += 2;
        }

        *rows2_x = src_n1[1] * coef1 + src_n1[0] * coef0 + src_n1[2] * 0.5f;
        *rows3_x = src_n2[1] * coef1 + src_n2[0] * coef0 + src_n2[2] * 0.5f;

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
            float32x4_t vqf32_c  = neon::vload1q(rows0_y);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_y);
            float32x4_t vqf32_n1 = neon::vload1q(rows2_y);
            float32x4_t vqf32_n2 = neon::vload1q(rows3_y);

            float32x4_t vqf32_c_x3   = neon::vmul(vqf32_c, coef0);
            float32x4_t vqf32_n0_x19 = neon::vmul(vqf32_n0, coef1);
            float32x4_t vqf32_n1_x19 = neon::vmul(vqf32_n1, coef1);
            float32x4_t vqf32_n2_x3  = neon::vmul(vqf32_n2, coef0);

            float32x4_t vqf32_result = neon::vadd(neon::vadd(vqf32_n0_x19, vqf32_c_x3), neon::vadd(vqf32_n1_x19, vqf32_n2_x3));
            neon::vstore(dst_row, vqf32_result);

            dst_row += 4;
            rows0_y += 4;
            rows1_y += 4;
            rows2_y += 4;
            rows3_y += 4;
        }

        for (; dx < owidth; dx++)
        {
            *dst_row = SaturateCast<Tp>(*rows1_y * coef1 + *rows0_y * coef0 + *rows2_y * coef1 + *rows3_y * coef0);

            dst_row++;
            rows0_y++;
            rows1_y++;
            rows2_y++;
            rows3_y++;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ResizeCuFastC1NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
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
        ThreadBuffer thread_buffer(ctx, owidth * 4 * sizeof(BufType));
        ret = wp->ParallelFor(0, AURA_ALIGN(oheight, 2) / 2, ResizeCuC1DownX2NeonImpl<Tp>, ctx, std::cref(src), std::ref(dst), std::ref(thread_buffer));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeCuC1DownX2NeonImpl run failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.f))
    {
        ret = wp->ParallelFor(0, oheight, ResizeCuC1DownX4NeonImpl<Tp>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeCuC1DownX4NeonImpl run failed");
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "special scale param error");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeCuFastC1Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuFastC1NeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuFastC1NeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuFastC1NeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuFastC1NeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeCuFastC1NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeCuFastC1NeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastC1NeonHelper run failed, type: MI_F32");
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