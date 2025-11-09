#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = DT_U8, DT_S8
template <typename Tp>
AURA_ALWAYS_INLINE typename neon::DVector<Tp>::VType
ResizeCuCommNeonVerCore(int32x4_t &vqs32_cx0_rows, int32x4_t &vqs32_n0x0_rows, int32x4_t &vqs32_n1x0_rows, int32x4_t &vqs32_n2x0_rows,
                        int32x4_t &vqs32_cx1_rows, int32x4_t &vqs32_n0x1_rows, int32x4_t &vqs32_n1x1_rows, int32x4_t &vqs32_n2x1_rows,
                        int32x4_t &vqs32_c_beta_mov)
{
    using VType = typename neon::DVector<Tp>::VType;

    int32x4_t vqs32_cx0_tmp0  = neon::vmul(vqs32_cx0_rows, neon::vgetlane<0>(vqs32_c_beta_mov));
    int32x4_t vqs32_n0x0_tmp0 = neon::vmul(vqs32_n0x0_rows, neon::vgetlane<1>(vqs32_c_beta_mov));
    int32x4_t vqs32_n1x0_tmp0 = neon::vmul(vqs32_n1x0_rows, neon::vgetlane<2>(vqs32_c_beta_mov));
    int32x4_t vqs32_n2x0_tmp0 = neon::vmul(vqs32_n2x0_rows, neon::vgetlane<3>(vqs32_c_beta_mov));
    int32x4_t vqs32_cx1_tmp0  = neon::vmul(vqs32_cx1_rows, neon::vgetlane<0>(vqs32_c_beta_mov));
    int32x4_t vqs32_n0x1_tmp0 = neon::vmul(vqs32_n0x1_rows, neon::vgetlane<1>(vqs32_c_beta_mov));
    int32x4_t vqs32_n1x1_tmp0 = neon::vmul(vqs32_n1x1_rows, neon::vgetlane<2>(vqs32_c_beta_mov));
    int32x4_t vqs32_n2x1_tmp0 = neon::vmul(vqs32_n2x1_rows, neon::vgetlane<3>(vqs32_c_beta_mov));

    int32x4_t vqs32_c_sum = neon::vadd(neon::vadd(vqs32_cx0_tmp0, vqs32_n0x0_tmp0), neon::vadd(vqs32_n1x0_tmp0, vqs32_n2x0_tmp0));
    int32x4_t vqs32_n_sum = neon::vadd(neon::vadd(vqs32_cx1_tmp0, vqs32_n0x1_tmp0), neon::vadd(vqs32_n1x1_tmp0, vqs32_n2x1_tmp0));

    int32x4_t vqs32_zero;
    neon::vdup(vqs32_zero, static_cast<DT_S32>(0));

    VType vd8_result;
    if (std::is_same<Tp, DT_U8>::value)
    {
        vqs32_c_sum = neon::vmax(vqs32_c_sum, vqs32_zero);
        vqs32_n_sum = neon::vmax(vqs32_n_sum, vqs32_zero);

        uint32x4_t vqu32_c_sum0   = neon::vreinterpret(vqs32_c_sum);
        uint32x4_t vqu32_n_sum0   = neon::vreinterpret(vqs32_n_sum);
        uint16x8_t vqu16_sum_tmp0 = neon::vcombine(neon::vqshrn_n<14>(vqu32_c_sum0), neon::vqshrn_n<14>(vqu32_n_sum0));

        vd8_result = neon::vqrshrn_n<8>(vqu16_sum_tmp0);
    }
    else
    {
        int16x8_t vqs16_sum_tmp0 = neon::vcombine(neon::vqshrn_n<14>(vqs32_c_sum), neon::vqshrn_n<14>(vqs32_n_sum));
        vd8_result = neon::vqrshrn_n<8>(vqs16_sum_tmp0);
    }

    return vd8_result;
}

// Tp = DT_U16, DT_S16
template <typename Tp>
AURA_ALWAYS_INLINE typename neon:: QVector<Tp>::VType
ResizeCuCommNeonVerCore(float32x4_t &vqf32_cx0_rows, float32x4_t &vqf32_n0x0_rows, float32x4_t &vqf32_n1x0_rows, float32x4_t &vqf32_n2x0_rows,
                        float32x4_t &vqf32_cx1_rows, float32x4_t &vqf32_n0x1_rows, float32x4_t &vqf32_n1x1_rows, float32x4_t &vqf32_n2x1_rows,
                        float32x4_t &vqf32_c_beta)
{
    using VType = typename neon:: QVector<Tp>::VType;
    using SType = typename std::conditional<std::is_same<Tp, DT_U16>::value, DT_U32, DT_S32>::type;

    float32x4_t vqf32_cx0_tmp0  = neon::vmul(vqf32_cx0_rows, neon::vgetlane<0>(vqf32_c_beta));
    float32x4_t vqf32_n0x0_tmp0 = neon::vmul(vqf32_n0x0_rows, neon::vgetlane<1>(vqf32_c_beta));
    float32x4_t vqf32_n1x0_tmp0 = neon::vmul(vqf32_n1x0_rows, neon::vgetlane<2>(vqf32_c_beta));
    float32x4_t vqf32_n2x0_tmp0 = neon::vmul(vqf32_n2x0_rows, neon::vgetlane<3>(vqf32_c_beta));
    float32x4_t vqf32_cx1_tmp0  = neon::vmul(vqf32_cx1_rows, neon::vgetlane<0>(vqf32_c_beta));
    float32x4_t vqf32_n0x1_tmp0 = neon::vmul(vqf32_n0x1_rows, neon::vgetlane<1>(vqf32_c_beta));
    float32x4_t vqf32_n1x1_tmp0 = neon::vmul(vqf32_n1x1_rows, neon::vgetlane<2>(vqf32_c_beta));
    float32x4_t vqf32_n2x1_tmp0 = neon::vmul(vqf32_n2x1_rows, neon::vgetlane<3>(vqf32_c_beta));

    float32x4_t vqf32_c_sum0 = neon::vadd(neon::vadd(vqf32_cx0_tmp0, vqf32_n0x0_tmp0), neon::vadd(vqf32_n1x0_tmp0, vqf32_n2x0_tmp0));
    float32x4_t vqf32_n_sum0 = neon::vadd(neon::vadd(vqf32_cx1_tmp0, vqf32_n0x1_tmp0), neon::vadd(vqf32_n1x1_tmp0, vqf32_n2x1_tmp0));

    auto vq32_c_sum0 = neon::vcvt<SType>(vqf32_c_sum0);
    auto vq32_n_sum0 = neon::vcvt<SType>(vqf32_n_sum0);

    VType vq16_result;
    vq16_result = neon::vcombine(neon::vmovn(vq32_c_sum0), neon::vmovn(vq32_n_sum0));

    return vq16_result;
}

// Tp = DT_F32
template <typename Tp>
AURA_ALWAYS_INLINE float32x4_t ResizeCuCommNeonVerCore(float32x4_t &vqf32_cx0_rows, float32x4_t &vqf32_n0x0_rows, float32x4_t &vqf32_n1x0_rows, float32x4_t &vqf32_n2x0_rows, float32x4_t &vqf32_c_beta)
{
    float32x4_t vqf32_cx0_tmp0  = neon::vmul(vqf32_cx0_rows, neon::vgetlane<0>(vqf32_c_beta));
    float32x4_t vqf32_n0x0_tmp0 = neon::vmul(vqf32_n0x0_rows, neon::vgetlane<1>(vqf32_c_beta));
    float32x4_t vqf32_n1x0_tmp0 = neon::vmul(vqf32_n1x0_rows, neon::vgetlane<2>(vqf32_c_beta));
    float32x4_t vqf32_n2x0_tmp0 = neon::vmul(vqf32_n2x0_rows, neon::vgetlane<3>(vqf32_c_beta));

    float32x4_t vqf32_result;
    vqf32_result = neon::vadd(neon::vadd(vqf32_cx0_tmp0, vqf32_n0x0_tmp0), neon::vadd(vqf32_n1x0_tmp0, vqf32_n2x0_tmp0));

    return vqf32_result;
}

// SType = DT_U8, DT_S8
// VType = unit8x8_t, int8x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, SType>::value || std::is_same<DT_S8, SType>::value, int32x2_t>::type
ResizeCuCommNeonHorCore(VType &vd8_x0_src, VType &vd8_x1_src, int16x4_t &vds16_x0_alpha, int16x4_t &vds16_x1_alpha)
{
    int32x2_t vds32_row_result;

    int16x8_t vqs16_x0_src_mov0    = neon::vmovl(vd8_x0_src);
    int16x8_t vqs16_x1_src_mov0    = neon::vmovl(vd8_x1_src);
    int32x4_t vqs32_x0_buffer_tmp0 = neon::vmull(neon::vgetlow(vqs16_x0_src_mov0), vds16_x0_alpha);
    int32x4_t vqs32_x1_buffer_tmp0 = neon::vmull(neon::vgetlow(vqs16_x1_src_mov0), vds16_x1_alpha);
    int32x2_t vds32_x0_row_tmp0    = neon::vpadd(neon::vgetlow(vqs32_x0_buffer_tmp0), neon::vgethigh(vqs32_x0_buffer_tmp0));
    int32x2_t vds32_x1_row_tmp0    = neon::vpadd(neon::vgetlow(vqs32_x1_buffer_tmp0), neon::vgethigh(vqs32_x1_buffer_tmp0));
    vds32_row_result               = neon::vpadd(vds32_x0_row_tmp0, vds32_x1_row_tmp0);

    return vds32_row_result;
}

// SType = DT_U16, DT_S16
// VType = unit16x4_t, int16x4_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, SType>::value || std::is_same<DT_S16, SType>::value, float32x2_t>::type
ResizeCuCommNeonHorCore(VType &vd16_x0_src, VType &vd16_x1_src, float32x4_t &vqf32_x0_alpha, float32x4_t &vqf32_x1_alpha)
{
    float32x2_t vdf32_row_result;
    auto vq32_x0_src_mov0            = neon::vmovl(vd16_x0_src);
    auto vq32_x1_src_mov0            = neon::vmovl(vd16_x1_src);
    float32x4_t vqf32_x0_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vq32_x0_src_mov0), vqf32_x0_alpha);
    float32x4_t vqf32_x1_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vq32_x1_src_mov0), vqf32_x1_alpha);
    float32x2_t vdf32_x0_row_tmp0    = neon::vpadd(neon::vgetlow(vqf32_x0_buffer_tmp0), neon::vgethigh(vqf32_x0_buffer_tmp0));
    float32x2_t vdf32_x1_row_tmp0    = neon::vpadd(neon::vgetlow(vqf32_x1_buffer_tmp0), neon::vgethigh(vqf32_x1_buffer_tmp0));
    vdf32_row_result                 = neon::vpadd(vdf32_x0_row_tmp0, vdf32_x1_row_tmp0);

    return vdf32_row_result;
}

// SType = MI_F16
// VType = float16x4_t
#if defined(AURA_ENABLE_NEON_FP16)
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_F16, SType>::value, float32x4_t>::type
ResizeCuCommNeonHorCore(VType &vdf16_x0_src, VType &vdf16_x1_src, VType &vdf16_x2_src, VType &vdf16_x3_src,
                        float32x4_t &vqf32_x0_alpha, float32x4_t &vqf32_x1_alpha, float32x4_t &vqf32_x2_alpha, float32x4_t &vqf32_x3_alpha)
{
    float32x4_t vqf32_row_result;
    float32x4_t vqf32_x0_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vdf16_x0_src), vqf32_x0_alpha);
    float32x4_t vqf32_x1_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vdf16_x1_src), vqf32_x1_alpha);
    float32x4_t vqf32_x2_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vdf16_x2_src), vqf32_x2_alpha);
    float32x4_t vqf32_x3_buffer_tmp0 = neon::vmul(neon::vcvt<DT_F32>(vdf16_x3_src), vqf32_x3_alpha);

    float32x4_t vqf32_x0_row_tmp01 = neon::vcombine(neon::vpadd(neon::vgetlow(vqf32_x0_buffer_tmp0), neon::vgethigh(vqf32_x0_buffer_tmp0)),
                                                    neon::vpadd(neon::vgetlow(vqf32_x1_buffer_tmp0), neon::vgethigh(vqf32_x1_buffer_tmp0)));
    float32x4_t vqf32_x0_row_tmp23 = neon::vcombine(neon::vpadd(neon::vgetlow(vqf32_x2_buffer_tmp0), neon::vgethigh(vqf32_x2_buffer_tmp0)),
                                                    neon::vpadd(neon::vgetlow(vqf32_x3_buffer_tmp0), neon::vgethigh(vqf32_x3_buffer_tmp0)));
    vqf32_row_result               = neon::vcombine(neon::vpadd(neon::vgetlow(vqf32_x0_row_tmp01), neon::vgethigh(vqf32_x0_row_tmp01)),
                                                    neon::vpadd(neon::vgetlow(vqf32_x0_row_tmp23), neon::vgethigh(vqf32_x0_row_tmp23)));

    return vqf32_row_result;
}
#endif

// SType = DT_F32
// VType = float32x4_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_F32, SType>::value, float32x2_t>::type
ResizeCuCommNeonHorCore(VType &vqf32_x0_src, VType &vqf32_x1_src, float32x4_t &vqf32_x0_alpha, float32x4_t &vqf32_x1_alpha)
{
    float32x2_t vdf32_row_result;
    float32x4_t vqf32_x0_buffer_tmp0 = neon::vmul(vqf32_x0_src, vqf32_x0_alpha);
    float32x4_t vqf32_x1_buffer_tmp0 = neon::vmul(vqf32_x1_src, vqf32_x1_alpha);
    float32x2_t vdf32_x0_row_tmp0    = neon::vpadd(neon::vgetlow(vqf32_x0_buffer_tmp0), neon::vgethigh(vqf32_x0_buffer_tmp0));
    float32x2_t vdf32_x1_row_tmp0    = neon::vpadd(neon::vgetlow(vqf32_x1_buffer_tmp0), neon::vgethigh(vqf32_x1_buffer_tmp0));
    vdf32_row_result                 = neon::vpadd(vdf32_x0_row_tmp0, vdf32_x1_row_tmp0);

    return vdf32_row_result;
}

// Tp = DT_U8, DT_S8
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeCuCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    using MVTypeD8   = typename neon::MDVector<Tp, C>::MVType;
    using VTypeD8    = typename neon::DVector<Tp>::VType;
    using MVTypeS32  = typename neon::MDVector<DT_S32, C>::MVType;
    using WMVTypeS32 = typename neon::WMVectorNums<MVTypeS32>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;

    DT_S32 *xofs = buffer;
    DT_S32 *yofs = xofs + owidth;
    DT_S16 *alpha = reinterpret_cast<DT_S16*>(yofs + oheight);
    DT_S16 *beta  = reinterpret_cast<DT_S16*>(alpha + (owidth * 4));

    DT_S32 *rows = thread_buffer.GetThreadData<DT_S32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_S32 *rows0 = rows;
    DT_S32 *rows1 = rows0 + owidth * C;
    DT_S32 *rows2 = rows1 + owidth * C;
    DT_S32 *rows3 = rows2 + owidth * C;

    DT_S32 prev_sy1 = -5;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 width_align2 = loop_end_owidth & (-2);

    const DT_S16 *beta_ptr = beta + start_row * 4;

    for (DT_S32 dy = start_row; dy < end_row; dy++)
    {
        DT_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            DT_S32 *rows0_tmp = rows0;
            rows0             = rows1;
            rows1             = rows2;
            rows2             = rows3;
            rows3             = rows0_tmp;

            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_S16 *alpha_ptr = reinterpret_cast<DT_S16*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                int16x4_t vds16_x0_alpha = neon::vload1(alpha_ptr);     // 16x4
                int16x4_t vds16_x1_alpha = neon::vload1(alpha_ptr + 4); // 16x4

                MVTypeS32 mvds32_row_result;
                MVTypeD8  mvd8_x0_src, mvd8_x1_src;
                neon::vload(src_n2_c, mvd8_x0_src);
                neon::vload(src_n2_n, mvd8_x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvds32_row_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_x0_src.val[ch], mvd8_x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                }
                neon::vstore(rows3 + dx * C, mvds32_row_result);
                alpha_ptr += 8;
            }

            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_S16 a0 = alpha_ptr[0];
                DT_S16 a1 = alpha_ptr[1];
                DT_S16 a2 = alpha_ptr[2];
                DT_S16 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize two row
            DT_S32 *rows0_tmp = rows0;
            DT_S32 *rows1_tmp = rows1;
            rows0             = rows2;
            rows1             = rows3;
            rows2             = rows0_tmp;
            rows3             = rows1_tmp;

            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_S16 *alpha_ptr = reinterpret_cast<DT_S16*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                int16x4_t vds16_x0_alpha = neon::vload1(alpha_ptr);     // 16x4
                int16x4_t vds16_x1_alpha = neon::vload1(alpha_ptr + 4); // 16x4

                MVTypeS32 mvds32_row2_result, mvds32_row3_result;
                MVTypeD8  mvd8_n1x0_src, mvd8_n1x1_src, mvd8_n2x0_src, mvd8_n2x1_src;
                neon::vload(src_n1_c, mvd8_n1x0_src);
                neon::vload(src_n1_n, mvd8_n1x1_src);
                neon::vload(src_n2_c, mvd8_n2x0_src);
                neon::vload(src_n2_n, mvd8_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvds32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n1x0_src.val[ch], mvd8_n1x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n2x0_src.val[ch], mvd8_n2x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                }

                neon::vstore(rows2 + dx * C, mvds32_row2_result);
                neon::vstore(rows3 + dx * C, mvds32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_S16 a0 = alpha_ptr[0];
                DT_S16 a1 = alpha_ptr[1];
                DT_S16 a2 = alpha_ptr[2];
                DT_S16 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize three row
            DT_S32 *rows0_tmp = rows0;
            rows0             = rows3;
            rows3             = rows2;
            rows2             = rows1;
            rows1             = rows0_tmp;

            const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_S16 *alpha_ptr = reinterpret_cast<DT_S16*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                int16x4_t vds16_x0_alpha = neon::vload1(alpha_ptr);     // 16x4
                int16x4_t vds16_x1_alpha = neon::vload1(alpha_ptr + 4); // 16x4

                MVTypeS32 mvds32_row1_result, mvds32_row2_result, mvds32_row3_result;
                MVTypeD8  mvd8_n0x0_src, mvd8_n0x1_src, mvd8_n1x0_src, mvd8_n1x1_src, mvd8_n2x0_src, mvd8_n2x1_src;
                neon::vload(src_n0_c, mvd8_n0x0_src);
                neon::vload(src_n0_n, mvd8_n0x1_src);
                neon::vload(src_n1_c, mvd8_n1x0_src);
                neon::vload(src_n1_n, mvd8_n1x1_src);
                neon::vload(src_n2_c, mvd8_n2x0_src);
                neon::vload(src_n2_n, mvd8_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvds32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n0x0_src.val[ch], mvd8_n0x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n1x0_src.val[ch], mvd8_n1x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n2x0_src.val[ch], mvd8_n2x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                }

                neon::vstore(rows1 + dx * C, mvds32_row1_result);
                neon::vstore(rows2 + dx * C, mvds32_row2_result);
                neon::vstore(rows3 + dx * C, mvds32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_S16 a0 = alpha_ptr[0];
                DT_S16 a1 = alpha_ptr[1];
                DT_S16 a2 = alpha_ptr[2];
                DT_S16 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy > prev_sy1 + 2)
        {
            // hresize four rows
            const Tp *src_c         = src.Ptr<Tp>(sy);
            const Tp *src_n0        = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1        = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2        = src.Ptr<Tp>(sy + 3);
            const DT_S16 *alpha_ptr = reinterpret_cast<DT_S16*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_c_c  = src_c + sx_c;
                const Tp *src_c_n  = src_c + sx_n;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                int16x4_t vds16_x0_alpha = neon::vload1(alpha_ptr);     // 16x4
                int16x4_t vds16_x1_alpha = neon::vload1(alpha_ptr + 4); // 16x4

                MVTypeS32 mvds32_row0_result, mvds32_row1_result, mvds32_row2_result, mvds32_row3_result;
                MVTypeD8  mvd8_cx0_src, mvd8_cx1_src, mvd8_n0x0_src, mvd8_n0x1_src, mvd8_n1x0_src, mvd8_n1x1_src, mvd8_n2x0_src, mvd8_n2x1_src;
                neon::vload(src_c_c,  mvd8_cx0_src);
                neon::vload(src_c_n,  mvd8_cx1_src);
                neon::vload(src_n0_c, mvd8_n0x0_src);
                neon::vload(src_n0_n, mvd8_n0x1_src);
                neon::vload(src_n1_c, mvd8_n1x0_src);
                neon::vload(src_n1_n, mvd8_n1x1_src);
                neon::vload(src_n2_c, mvd8_n2x0_src);
                neon::vload(src_n2_n, mvd8_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvds32_row0_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_cx0_src.val[ch], mvd8_cx1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n0x0_src.val[ch], mvd8_n0x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n1x0_src.val[ch], mvd8_n1x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                    mvds32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD8>(mvd8_n2x0_src.val[ch], mvd8_n2x1_src.val[ch], vds16_x0_alpha, vds16_x1_alpha);
                }

                neon::vstore(rows0 + dx * C, mvds32_row0_result);
                neon::vstore(rows1 + dx * C, mvds32_row1_result);
                neon::vstore(rows2 + dx * C, mvds32_row2_result);
                neon::vstore(rows3 + dx * C, mvds32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_c_ptr  = src_c + sx;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_S16 a0 = alpha_ptr[0];
                DT_S16 a1 = alpha_ptr[1];
                DT_S16 a2 = alpha_ptr[2];
                DT_S16 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows0[x_id + ch] = src_c_ptr[ch] * a0 + src_c_ptr[ch + C] * a1 + src_c_ptr[ch + C * 2] * a2 + src_c_ptr[ch + C * 3] * a3;
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        Tp *dst_row = dst.Ptr<Tp>(dy);
        int16x4_t vds16_c_beta = neon::vload1(beta_ptr);
        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            WMVTypeS32 wmvqs32_cx0_rows, wmvqs32_n0x0_rows, wmvqs32_n1x0_rows, wmvqs32_n2x0_rows;
            WMVTypeS32 wmvqs32_cx1_rows, wmvqs32_n0x1_rows, wmvqs32_n1x1_rows, wmvqs32_n2x1_rows;

            neon::vload(rows0 + dx * C, wmvqs32_cx0_rows);
            neon::vload(rows1 + dx * C, wmvqs32_n0x0_rows);
            neon::vload(rows2 + dx * C, wmvqs32_n1x0_rows);
            neon::vload(rows3 + dx * C, wmvqs32_n2x0_rows);
            neon::vload(rows0 + (dx + 4) * C, wmvqs32_cx1_rows);
            neon::vload(rows1 + (dx + 4) * C, wmvqs32_n0x1_rows);
            neon::vload(rows2 + (dx + 4) * C, wmvqs32_n1x1_rows);
            neon::vload(rows3 + (dx + 4) * C, wmvqs32_n2x1_rows);

            int32x4_t vqs32_c_beta_mov = neon::vmovl(vds16_c_beta);

            MVTypeD8 mvd8_result;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd8_result.val[ch] = ResizeCuCommNeonVerCore<Tp>(wmvqs32_cx0_rows.val[ch], wmvqs32_n0x0_rows.val[ch], wmvqs32_n1x0_rows.val[ch], wmvqs32_n2x0_rows.val[ch],
                                                                  wmvqs32_cx1_rows.val[ch], wmvqs32_n0x1_rows.val[ch], wmvqs32_n1x1_rows.val[ch], wmvqs32_n2x1_rows.val[ch], vqs32_c_beta_mov);
            }
            neon::vstore(dst_row + dx * C, mvd8_result);
        }

        for (; dx < owidth; dx++)
        {
            DT_S32 x_id = dx * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                DT_S32 result = (rows0[x_id + ch] * beta_ptr[0] + rows1[x_id + ch] * beta_ptr[1] + rows2[x_id + ch] * beta_ptr[2] + rows3[x_id + ch] * beta_ptr[3]);
                dst_row[x_id + ch] = SaturateCast<Tp>((result + (1 << 21)) >> 22);
            }
        }
        beta_ptr += 4;
    }

    return Status::OK;
}

// Tp = DT_U16, DT_S16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeCuCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    using MVTypeD16  = typename neon::MDVector<Tp, C>::MVType;
    using WMVTypeD16 = typename neon::WMVectorNums<MVTypeD16>::MVType;
    using VTypeD16   = typename neon::DVector<Tp>::VType;
    using MVTypeF32  = typename neon::MDVector<DT_F32, C>::MVType;
    using WMVTypeF32 = typename neon::WMVectorNums<MVTypeF32>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;

    DT_S32 *xofs = buffer;
    DT_S32 *yofs = xofs + owidth;
    DT_F32 *alpha = reinterpret_cast<DT_F32*>(yofs + oheight);
    DT_F32 *beta  = reinterpret_cast<DT_F32*>(alpha + (owidth * 4));

    DT_F32 *rows = thread_buffer.GetThreadData<DT_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_F32 *rows0 = rows;
    DT_F32 *rows1 = rows0 + owidth * C;
    DT_F32 *rows2 = rows1 + owidth * C;
    DT_F32 *rows3 = rows2 + owidth * C;

    DT_S32 prev_sy1 = -5;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 width_align2 = loop_end_owidth & (-2);

    const DT_F32 *beta_ptr = beta + start_row * 4;

    for (DT_S32 dy = start_row; dy < end_row; dy++)
    {
        DT_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows1;
            rows1             = rows2;
            rows2             = rows3;
            rows3             = rows0_tmp;

            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // f32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // f32x4

                MVTypeF32 mvdf32_row_result;
                MVTypeD16 mvd16_n2x0_src, mvd16_n2x1_src;
                neon::vload(src_n2_c, mvd16_n2x0_src);
                neon::vload(src_n2_n, mvd16_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n2x0_src.val[ch], mvd16_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }
                neon::vstore(rows3 + dx * C, mvdf32_row_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize two row
            DT_F32 *rows0_tmp = rows0;
            DT_F32 *rows1_tmp = rows1;
            rows0             = rows2;
            rows1             = rows3;
            rows2             = rows0_tmp;
            rows3             = rows1_tmp;

            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c = xofs[dx] * C;
                DT_S32 sx_n = xofs[dx + 1] * C;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVTypeF32 mvdf32_row2_result, mvdf32_row3_result;
                MVTypeD16 mvd16_n1x0_src, mvd16_n1x1_src, mvd16_n2x0_src, mvd16_n2x1_src;
                neon::vload(src_n1_c, mvd16_n1x0_src);
                neon::vload(src_n1_n, mvd16_n1x1_src);
                neon::vload(src_n2_c, mvd16_n2x0_src);
                neon::vload(src_n2_n, mvd16_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n1x0_src.val[ch], mvd16_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n2x0_src.val[ch], mvd16_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize three row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows3;
            rows3             = rows2;
            rows2             = rows1;
            rows1             = rows0_tmp;

            const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVTypeF32 mvdf32_row1_result, mvdf32_row2_result, mvdf32_row3_result;
                MVTypeD16 mvd16_n0x0_src, mvd16_n0x1_src, mvd16_n1x0_src, mvd16_n1x1_src, mvd16_n2x0_src, mvd16_n2x1_src;
                neon::vload(src_n0_c, mvd16_n0x0_src);
                neon::vload(src_n0_n, mvd16_n0x1_src);
                neon::vload(src_n1_c, mvd16_n1x0_src);
                neon::vload(src_n1_n, mvd16_n1x1_src);
                neon::vload(src_n2_c, mvd16_n2x0_src);
                neon::vload(src_n2_n, mvd16_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n0x0_src.val[ch], mvd16_n0x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n1x0_src.val[ch], mvd16_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n2x0_src.val[ch], mvd16_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows1 + dx * C, mvdf32_row1_result);
                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy > prev_sy1 + 2)
        {
            // hresize four rows
            const Tp *src_c         = src.Ptr<Tp>(sy);
            const Tp *src_n0        = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1        = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2        = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_c_c  = src_c + sx_c;
                const Tp *src_c_n  = src_c + sx_n;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVTypeF32 mvdf32_row0_result, mvdf32_row1_result, mvdf32_row2_result, mvdf32_row3_result;
                MVTypeD16 mvd16_cx0_src, mvd16_cx1_src, mvd16_n0x0_src, mvd16_n0x1_src, mvd16_n1x0_src, mvd16_n1x1_src, mvd16_n2x0_src, mvd16_n2x1_src;
                neon::vload(src_c_c,  mvd16_cx0_src);
                neon::vload(src_c_n,  mvd16_cx1_src);
                neon::vload(src_n0_c, mvd16_n0x0_src);
                neon::vload(src_n0_n, mvd16_n0x1_src);
                neon::vload(src_n1_c, mvd16_n1x0_src);
                neon::vload(src_n1_n, mvd16_n1x1_src);
                neon::vload(src_n2_c, mvd16_n2x0_src);
                neon::vload(src_n2_n, mvd16_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row0_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_cx0_src.val[ch], mvd16_cx1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n0x0_src.val[ch], mvd16_n0x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n1x0_src.val[ch], mvd16_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VTypeD16>(mvd16_n2x0_src.val[ch], mvd16_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows0 + dx * C, mvdf32_row0_result);
                neon::vstore(rows1 + dx * C, mvdf32_row1_result);
                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_c_ptr  = src_c + sx;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows0[x_id + ch] = src_c_ptr[ch] * a0 + src_c_ptr[ch + C] * a1 + src_c_ptr[ch + C * 2] * a2 + src_c_ptr[ch + C * 3] * a3;
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        Tp *dst_row = dst.Ptr<Tp>(dy);

        float32x4_t vqf32_c_beta = neon::vload1q(beta_ptr);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            WMVTypeF32 wmvqf32_cx0_rows, wmvqf32_n0x0_rows, wmvqf32_n1x0_rows, wmvqf32_n2x0_rows;
            WMVTypeF32 wmvqf32_cx1_rows, wmvqf32_n0x1_rows, wmvqf32_n1x1_rows, wmvqf32_n2x1_rows;

            neon::vload(rows0 + dx * C, wmvqf32_cx0_rows);
            neon::vload(rows1 + dx * C, wmvqf32_n0x0_rows);
            neon::vload(rows2 + dx * C, wmvqf32_n1x0_rows);
            neon::vload(rows3 + dx * C, wmvqf32_n2x0_rows);
            neon::vload(rows0 + (dx + 4) * C, wmvqf32_cx1_rows);
            neon::vload(rows1 + (dx + 4) * C, wmvqf32_n0x1_rows);
            neon::vload(rows2 + (dx + 4) * C, wmvqf32_n1x1_rows);
            neon::vload(rows3 + (dx + 4) * C, wmvqf32_n2x1_rows);

            WMVTypeD16 wmvq16_result;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                wmvq16_result.val[ch] = ResizeCuCommNeonVerCore<Tp>(wmvqf32_cx0_rows.val[ch], wmvqf32_n0x0_rows.val[ch], wmvqf32_n1x0_rows.val[ch], wmvqf32_n2x0_rows.val[ch],
                                                                    wmvqf32_cx1_rows.val[ch], wmvqf32_n0x1_rows.val[ch], wmvqf32_n1x1_rows.val[ch], wmvqf32_n2x1_rows.val[ch], vqf32_c_beta);
            }
            neon::vstore(dst_row + dx * C, wmvq16_result);
        }

        for (; dx < owidth; dx++)
        {
            DT_S32 x_id = dx * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x_id + ch] = SaturateCast<Tp>(rows0[x_id + ch] * beta_ptr[0] + rows1[x_id + ch] * beta_ptr[1] + rows2[x_id + ch] * beta_ptr[2] + rows3[x_id + ch] * beta_ptr[3]);
            }
        }

        beta_ptr += 4;
    }

    return Status::OK;
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeCuCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    using MVType  = typename neon::MDVector<Tp, C>::MVType;
    using WMVType = typename neon::WMVectorBits<MVType>::MVType;
    using VType   = typename neon::DVector<Tp>::VType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;

    DT_S32 *xofs = buffer;
    DT_S32 *yofs = xofs + owidth;
    DT_F32 *alpha = reinterpret_cast<DT_F32*>(yofs + oheight);
    DT_F32 *beta  = reinterpret_cast<DT_F32*>(alpha + (owidth * 4));

    DT_F32 *rows = thread_buffer.GetThreadData<DT_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_F32 *rows0 = rows;
    DT_F32 *rows1 = rows0 + owidth * C;
    DT_F32 *rows2 = rows1 + owidth * C;
    DT_F32 *rows3 = rows2 + owidth * C;

    DT_S32 prev_sy1 = -5;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 width_align4 = loop_end_owidth & (-4);

    const DT_F32 *beta_ptr = beta + start_row * 4;

    for (DT_S32 dy = start_row; dy < end_row; dy++)
    {
        DT_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows1;
            rows1             = rows2;
            rows2             = rows3;
            rows3             = rows0_tmp;

            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align4; dx += 4)
            {
                DT_S32 sx_c         = xofs[dx] * C;
                DT_S32 sx_r0        = xofs[dx + 1] * C;
                DT_S32 sx_r1        = xofs[dx + 2] * C;
                DT_S32 sx_r2        = xofs[dx + 3] * C;
                const Tp *src_n2_c  = src_n2 + sx_c;
                const Tp *src_n2_r0 = src_n2 + sx_r0;
                const Tp *src_n2_r1 = src_n2 + sx_r1;
                const Tp *src_n2_r2 = src_n2 + sx_r2;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4);
                float32x4_t vqf32_x2_alpha = neon::vload1q(alpha_ptr + 8);
                float32x4_t vqf32_x3_alpha = neon::vload1q(alpha_ptr + 12);

                MVType  mvdf16_n2x0_src, mvdf16_n2x1_src, mvdf16_n2x2_src, mvdf16_n2x3_src;
                WMVType mvqf32_row_result;
                neon::vload(src_n2_c,  mvdf16_n2x0_src);
                neon::vload(src_n2_r0, mvdf16_n2x1_src);
                neon::vload(src_n2_r1, mvdf16_n2x2_src);
                neon::vload(src_n2_r2, mvdf16_n2x3_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvqf32_row_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n2x0_src.val[ch], mvdf16_n2x1_src.val[ch], mvdf16_n2x2_src.val[ch], mvdf16_n2x3_src.val[ch],
                                                                                vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                }
                neon::vstore(rows3 + dx * C, mvqf32_row_result);

                alpha_ptr += 16;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize two row
            DT_F32 *rows0_tmp = rows0;
            DT_F32 *rows1_tmp = rows1;
            rows0             = rows2;
            rows1             = rows3;
            rows2             = rows0_tmp;
            rows3             = rows1_tmp;

            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align4; dx += 4)
            {
                DT_S32 sx_c         = xofs[dx] * C;
                DT_S32 sx_r0        = xofs[dx + 1] * C;
                DT_S32 sx_r1        = xofs[dx + 2] * C;
                DT_S32 sx_r2        = xofs[dx + 3] * C;
                const Tp *src_n1_c  = src_n1 + sx_c;
                const Tp *src_n1_r0 = src_n1 + sx_r0;
                const Tp *src_n1_r1 = src_n1 + sx_r1;
                const Tp *src_n1_r2 = src_n1 + sx_r2;
                const Tp *src_n2_c  = src_n2 + sx_c;
                const Tp *src_n2_r0 = src_n2 + sx_r0;
                const Tp *src_n2_r1 = src_n2 + sx_r1;
                const Tp *src_n2_r2 = src_n2 + sx_r2;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4);
                float32x4_t vqf32_x2_alpha = neon::vload1q(alpha_ptr + 8);
                float32x4_t vqf32_x3_alpha = neon::vload1q(alpha_ptr + 12);

                MVType  mvdf16_n1x0_src, mvdf16_n1x1_src, mvdf16_n1x2_src, mvdf16_n1x3_src;
                MVType  mvdf16_n2x0_src, mvdf16_n2x1_src, mvdf16_n2x2_src, mvdf16_n2x3_src;
                WMVType mvqf32_row2_result, mvqf32_row3_result;

                neon::vload(src_n1_c,  mvdf16_n1x0_src);
                neon::vload(src_n1_r0, mvdf16_n1x1_src);
                neon::vload(src_n1_r1, mvdf16_n1x2_src);
                neon::vload(src_n1_r2, mvdf16_n1x3_src);

                neon::vload(src_n2_c,  mvdf16_n2x0_src);
                neon::vload(src_n2_r0, mvdf16_n2x1_src);
                neon::vload(src_n2_r1, mvdf16_n2x2_src);
                neon::vload(src_n2_r2, mvdf16_n2x3_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvqf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n1x0_src.val[ch], mvdf16_n1x1_src.val[ch], mvdf16_n1x2_src.val[ch], mvdf16_n1x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n2x0_src.val[ch], mvdf16_n2x1_src.val[ch], mvdf16_n2x2_src.val[ch], mvdf16_n2x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                }
                neon::vstore(rows2 + dx * C, mvqf32_row2_result);
                neon::vstore(rows3 + dx * C, mvqf32_row3_result);

                alpha_ptr += 16;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx   = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize three row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows3;
            rows3             = rows2;
            rows2             = rows1;
            rows1             = rows0_tmp;

            const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align4; dx += 4)
            {
                DT_S32 sx_c         = xofs[dx] * C;
                DT_S32 sx_r0        = xofs[dx + 1] * C;
                DT_S32 sx_r1        = xofs[dx + 2] * C;
                DT_S32 sx_r2        = xofs[dx + 3] * C;
                const Tp *src_n0_c  = src_n0 + sx_c;
                const Tp *src_n0_r0 = src_n0 + sx_r0;
                const Tp *src_n0_r1 = src_n0 + sx_r1;
                const Tp *src_n0_r2 = src_n0 + sx_r2;
                const Tp *src_n1_c  = src_n1 + sx_c;
                const Tp *src_n1_r0 = src_n1 + sx_r0;
                const Tp *src_n1_r1 = src_n1 + sx_r1;
                const Tp *src_n1_r2 = src_n1 + sx_r2;
                const Tp *src_n2_c  = src_n2 + sx_c;
                const Tp *src_n2_r0 = src_n2 + sx_r0;
                const Tp *src_n2_r1 = src_n2 + sx_r1;
                const Tp *src_n2_r2 = src_n2 + sx_r2;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4);
                float32x4_t vqf32_x2_alpha = neon::vload1q(alpha_ptr + 8);
                float32x4_t vqf32_x3_alpha = neon::vload1q(alpha_ptr + 12);

                MVType  mvdf16_n0x0_src, mvdf16_n0x1_src, mvdf16_n0x2_src, mvdf16_n0x3_src;
                MVType  mvdf16_n1x0_src, mvdf16_n1x1_src, mvdf16_n1x2_src, mvdf16_n1x3_src;
                MVType  mvdf16_n2x0_src, mvdf16_n2x1_src, mvdf16_n2x2_src, mvdf16_n2x3_src;
                WMVType mvqf32_row1_result, mvqf32_row2_result, mvqf32_row3_result;

                neon::vload(src_n0_c,  mvdf16_n0x0_src);
                neon::vload(src_n0_r0, mvdf16_n0x1_src);
                neon::vload(src_n0_r1, mvdf16_n0x2_src);
                neon::vload(src_n0_r2, mvdf16_n0x3_src);

                neon::vload(src_n1_c,  mvdf16_n1x0_src);
                neon::vload(src_n1_r0, mvdf16_n1x1_src);
                neon::vload(src_n1_r1, mvdf16_n1x2_src);
                neon::vload(src_n1_r2, mvdf16_n1x3_src);

                neon::vload(src_n2_c,  mvdf16_n2x0_src);
                neon::vload(src_n2_r0, mvdf16_n2x1_src);
                neon::vload(src_n2_r1, mvdf16_n2x2_src);
                neon::vload(src_n2_r2, mvdf16_n2x3_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvqf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n0x0_src.val[ch], mvdf16_n0x1_src.val[ch], mvdf16_n0x2_src.val[ch], mvdf16_n0x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n1x0_src.val[ch], mvdf16_n1x1_src.val[ch], mvdf16_n1x2_src.val[ch], mvdf16_n1x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n2x0_src.val[ch], mvdf16_n2x1_src.val[ch], mvdf16_n2x2_src.val[ch], mvdf16_n2x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                }

                neon::vstore(rows1 + dx * C, mvqf32_row1_result);
                neon::vstore(rows2 + dx * C, mvqf32_row2_result);
                neon::vstore(rows3 + dx * C, mvqf32_row3_result);

                alpha_ptr += 16;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy > prev_sy1 + 2)
        {
            // hresize four rows
            const Tp *src_c         = src.Ptr<Tp>(sy);
            const Tp *src_n0        = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1        = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2        = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align4; dx += 4)
            {
                DT_S32 sx_c         = xofs[dx] * C;
                DT_S32 sx_r0        = xofs[dx + 1] * C;
                DT_S32 sx_r1        = xofs[dx + 2] * C;
                DT_S32 sx_r2        = xofs[dx + 3] * C;
                const Tp *src_c_c   = src_c + sx_c;
                const Tp *src_c_r0  = src_c + sx_r0;
                const Tp *src_c_r1  = src_c + sx_r1;
                const Tp *src_c_r2  = src_c + sx_r2;
                const Tp *src_n0_c  = src_n0 + sx_c;
                const Tp *src_n0_r0 = src_n0 + sx_r0;
                const Tp *src_n0_r1 = src_n0 + sx_r1;
                const Tp *src_n0_r2 = src_n0 + sx_r2;
                const Tp *src_n1_c  = src_n1 + sx_c;
                const Tp *src_n1_r0 = src_n1 + sx_r0;
                const Tp *src_n1_r1 = src_n1 + sx_r1;
                const Tp *src_n1_r2 = src_n1 + sx_r2;
                const Tp *src_n2_c  = src_n2 + sx_c;
                const Tp *src_n2_r0 = src_n2 + sx_r0;
                const Tp *src_n2_r1 = src_n2 + sx_r1;
                const Tp *src_n2_r2 = src_n2 + sx_r2;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4);
                float32x4_t vqf32_x2_alpha = neon::vload1q(alpha_ptr + 8);
                float32x4_t vqf32_x3_alpha = neon::vload1q(alpha_ptr + 12);

                MVType  mvdf16_cx0_src,  mvdf16_cx1_src,  mvdf16_cx2_src,  mvdf16_cx3_src;
                MVType  mvdf16_n0x0_src, mvdf16_n0x1_src, mvdf16_n0x2_src, mvdf16_n0x3_src;
                MVType  mvdf16_n1x0_src, mvdf16_n1x1_src, mvdf16_n1x2_src, mvdf16_n1x3_src;
                MVType  mvdf16_n2x0_src, mvdf16_n2x1_src, mvdf16_n2x2_src, mvdf16_n2x3_src;
                WMVType mvqf32_row0_result, mvqf32_row1_result, mvqf32_row2_result, mvqf32_row3_result;

                neon::vload(src_c_c,  mvdf16_cx0_src);
                neon::vload(src_c_r0, mvdf16_cx1_src);
                neon::vload(src_c_r1, mvdf16_cx2_src);
                neon::vload(src_c_r2, mvdf16_cx3_src);

                neon::vload(src_n0_c,  mvdf16_n0x0_src);
                neon::vload(src_n0_r0, mvdf16_n0x1_src);
                neon::vload(src_n0_r1, mvdf16_n0x2_src);
                neon::vload(src_n0_r2, mvdf16_n0x3_src);

                neon::vload(src_n1_c,  mvdf16_n1x0_src);
                neon::vload(src_n1_r0, mvdf16_n1x1_src);
                neon::vload(src_n1_r1, mvdf16_n1x2_src);
                neon::vload(src_n1_r2, mvdf16_n1x3_src);

                neon::vload(src_n2_c,  mvdf16_n2x0_src);
                neon::vload(src_n2_r0, mvdf16_n2x1_src);
                neon::vload(src_n2_r1, mvdf16_n2x2_src);
                neon::vload(src_n2_r2, mvdf16_n2x3_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvqf32_row0_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_cx0_src.val[ch], mvdf16_cx1_src.val[ch], mvdf16_cx2_src.val[ch], mvdf16_cx3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n0x0_src.val[ch], mvdf16_n0x1_src.val[ch], mvdf16_n0x2_src.val[ch], mvdf16_n0x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n1x0_src.val[ch], mvdf16_n1x1_src.val[ch], mvdf16_n1x2_src.val[ch], mvdf16_n1x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                    mvqf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(mvdf16_n2x0_src.val[ch], mvdf16_n2x1_src.val[ch], mvdf16_n2x2_src.val[ch], mvdf16_n2x3_src.val[ch],
                                                                                 vqf32_x0_alpha, vqf32_x1_alpha, vqf32_x2_alpha, vqf32_x3_alpha);
                }

                neon::vstore(rows0 + dx * C, mvqf32_row0_result);
                neon::vstore(rows1 + dx * C, mvqf32_row1_result);
                neon::vstore(rows2 + dx * C, mvqf32_row2_result);
                neon::vstore(rows3 + dx * C, mvqf32_row3_result);

                alpha_ptr += 16;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_c_ptr  = src_c + sx;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows0[x_id + ch] = src_c_ptr[ch] * a0 + src_c_ptr[ch + C] * a1 + src_c_ptr[ch + C * 2] * a2 + src_c_ptr[ch + C * 3] * a3;
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        Tp *dst_row = dst.Ptr<Tp>(dy);

        float32x4_t vqf32_c_beta = neon::vload1q(beta_ptr);

        DT_S32 owidth_align4 = owidth & (-4);
        DT_S32 dx = 0;
        for (; dx < owidth_align4; dx += 4)
        {
            WMVType wmvqf32_cx0_rows, wmvqf32_n0x0_rows, wmvqf32_n1x0_rows, wmvqf32_n2x0_rows;

            neon::vload(rows0 + dx * C, wmvqf32_cx0_rows);
            neon::vload(rows1 + dx * C, wmvqf32_n0x0_rows);
            neon::vload(rows2 + dx * C, wmvqf32_n1x0_rows);
            neon::vload(rows3 + dx * C, wmvqf32_n2x0_rows);

            MVType mvdf16_result;
            WMVType wmvqf32_result;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                wmvqf32_result.val[ch] = ResizeCuCommNeonVerCore<DT_F32>(wmvqf32_cx0_rows.val[ch], wmvqf32_n0x0_rows.val[ch], wmvqf32_n1x0_rows.val[ch], wmvqf32_n2x0_rows.val[ch], vqf32_c_beta);
                mvdf16_result.val[ch]  = neon::vcvt<MI_F16>(wmvqf32_result.val[ch]);
            }
            neon::vstore(dst_row + dx * C, mvdf16_result);
        }
        for (; dx < owidth; dx++)
        {
            DT_S32 x_id = dx * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x_id + ch] = SaturateCast<Tp>(rows0[x_id + ch] * beta_ptr[0] + rows1[x_id + ch] * beta_ptr[1] + rows2[x_id + ch] * beta_ptr[2] + rows3[x_id + ch] * beta_ptr[3]);
            }
        }

        beta_ptr += 4;
    }

    return Status::OK;
}
#endif

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeCuCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    using MVType  = typename neon::MDVector<Tp, C>::MVType;
    using WMVType = typename neon::WMVectorNums<MVType>::MVType;
    using VType   = typename neon::QVector<Tp>::VType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;

    DT_S32 *xofs = buffer;
    DT_S32 *yofs = xofs + owidth;
    DT_F32 *alpha = reinterpret_cast<DT_F32*>(yofs + oheight);
    DT_F32 *beta  = reinterpret_cast<DT_F32*>(alpha + (owidth * 4));

    DT_F32 *rows = thread_buffer.GetThreadData<DT_F32>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_F32 *rows0 = rows;
    DT_F32 *rows1 = rows0 + owidth * C;
    DT_F32 *rows2 = rows1 + owidth * C;
    DT_F32 *rows3 = rows2 + owidth * C;

    DT_S32 prev_sy1 = -5;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 width_align2 = loop_end_owidth & (-2);

    const DT_F32 *beta_ptr = beta + start_row * 4;

    for (DT_S32 dy = start_row; dy < end_row; dy++)
    {
        DT_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows1;
            rows1             = rows2;
            rows2             = rows3;
            rows3             = rows0_tmp;

            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // f32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // f32x4

                MVType mvdf32_row_result;
                WMVType wmvqf32_n2x0_src, wmvqf32_n2x1_src;
                neon::vload(src_n2_c, wmvqf32_n2x0_src);
                neon::vload(src_n2_n, wmvqf32_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n2x0_src.val[ch], wmvqf32_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }
                neon::vstore(rows3 + dx * C, mvdf32_row_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize two row
            DT_F32 *rows0_tmp = rows0;
            DT_F32 *rows1_tmp = rows1;
            rows0             = rows2;
            rows1             = rows3;
            rows2             = rows0_tmp;
            rows3             = rows1_tmp;

            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c = xofs[dx] * C;
                DT_S32 sx_n = xofs[dx + 1] * C;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVType  mvdf32_row2_result, mvdf32_row3_result;
                WMVType wmvqf32_n1x0_src, wmvqf32_n1x1_src, wmvqf32_n2x0_src, wmvqf32_n2x1_src;
                neon::vload(src_n1_c, wmvqf32_n1x0_src);
                neon::vload(src_n1_n, wmvqf32_n1x1_src);
                neon::vload(src_n2_c, wmvqf32_n2x0_src);
                neon::vload(src_n2_n, wmvqf32_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n1x0_src.val[ch], wmvqf32_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n2x0_src.val[ch], wmvqf32_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx   = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize three row
            DT_F32 *rows0_tmp = rows0;
            rows0             = rows3;
            rows3             = rows2;
            rows2             = rows1;
            rows1             = rows0_tmp;

            const Tp *src_n0 = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1 = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2 = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVType mvdf32_row1_result, mvdf32_row2_result, mvdf32_row3_result;
                WMVType wmvqf32_n0x0_src, wmvqf32_n0x1_src, wmvqf32_n1x0_src, wmvqf32_n1x1_src, wmvqf32_n2x0_src, wmvqf32_n2x1_src;
                neon::vload(src_n0_c, wmvqf32_n0x0_src);
                neon::vload(src_n0_n, wmvqf32_n0x1_src);
                neon::vload(src_n1_c, wmvqf32_n1x0_src);
                neon::vload(src_n1_n, wmvqf32_n1x1_src);
                neon::vload(src_n2_c, wmvqf32_n2x0_src);
                neon::vload(src_n2_n, wmvqf32_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n0x0_src.val[ch], wmvqf32_n0x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n1x0_src.val[ch], wmvqf32_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n2x0_src.val[ch], wmvqf32_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows1 + dx * C, mvdf32_row1_result);
                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }
        else if (sy > prev_sy1 + 2)
        {
            // hresize four rows
            const Tp *src_c         = src.Ptr<Tp>(sy);
            const Tp *src_n0        = src.Ptr<Tp>(sy + 1);
            const Tp *src_n1        = src.Ptr<Tp>(sy + 2);
            const Tp *src_n2        = src.Ptr<Tp>(sy + 3);
            const DT_F32 *alpha_ptr = reinterpret_cast<DT_F32*>(alpha);

            DT_S32 dx = 0;
            for (; dx < width_align2; dx += 2)
            {
                DT_S32 sx_c        = xofs[dx] * C;
                DT_S32 sx_n        = xofs[dx + 1] * C;
                const Tp *src_c_c  = src_c + sx_c;
                const Tp *src_c_n  = src_c + sx_n;
                const Tp *src_n0_c = src_n0 + sx_c;
                const Tp *src_n0_n = src_n0 + sx_n;
                const Tp *src_n1_c = src_n1 + sx_c;
                const Tp *src_n1_n = src_n1 + sx_n;
                const Tp *src_n2_c = src_n2 + sx_c;
                const Tp *src_n2_n = src_n2 + sx_n;

                float32x4_t vqf32_x0_alpha = neon::vload1q(alpha_ptr);     // 32x4
                float32x4_t vqf32_x1_alpha = neon::vload1q(alpha_ptr + 4); // 32x4

                MVType mvdf32_row0_result, mvdf32_row1_result, mvdf32_row2_result, mvdf32_row3_result;
                WMVType wmvqf32_cx0_src, wmvqf32_cx1_src, wmvqf32_n0x0_src, wmvqf32_n0x1_src, wmvqf32_n1x0_src, wmvqf32_n1x1_src, wmvqf32_n2x0_src, wmvqf32_n2x1_src;
                neon::vload(src_c_c,  wmvqf32_cx0_src);
                neon::vload(src_c_n,  wmvqf32_cx1_src);
                neon::vload(src_n0_c, wmvqf32_n0x0_src);
                neon::vload(src_n0_n, wmvqf32_n0x1_src);
                neon::vload(src_n1_c, wmvqf32_n1x0_src);
                neon::vload(src_n1_n, wmvqf32_n1x1_src);
                neon::vload(src_n2_c, wmvqf32_n2x0_src);
                neon::vload(src_n2_n, wmvqf32_n2x1_src);

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    mvdf32_row0_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_cx0_src.val[ch], wmvqf32_cx1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row1_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n0x0_src.val[ch], wmvqf32_n0x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row2_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n1x0_src.val[ch], wmvqf32_n1x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                    mvdf32_row3_result.val[ch] = ResizeCuCommNeonHorCore<Tp, VType>(wmvqf32_n2x0_src.val[ch], wmvqf32_n2x1_src.val[ch], vqf32_x0_alpha, vqf32_x1_alpha);
                }

                neon::vstore(rows0 + dx * C, mvdf32_row0_result);
                neon::vstore(rows1 + dx * C, mvdf32_row1_result);
                neon::vstore(rows2 + dx * C, mvdf32_row2_result);
                neon::vstore(rows3 + dx * C, mvdf32_row3_result);

                alpha_ptr += 8;
            }
            for (; dx < owidth; dx++)
            {
                DT_S32 sx = xofs[dx] * C;
                DT_S32 x_id = dx * C;
                const Tp *src_c_ptr  = src_c + sx;
                const Tp *src_n0_ptr = src_n0 + sx;
                const Tp *src_n1_ptr = src_n1 + sx;
                const Tp *src_n2_ptr = src_n2 + sx;

                DT_F32 a0 = alpha_ptr[0];
                DT_F32 a1 = alpha_ptr[1];
                DT_F32 a2 = alpha_ptr[2];
                DT_F32 a3 = alpha_ptr[3];

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    rows0[x_id + ch] = src_c_ptr[ch] * a0 + src_c_ptr[ch + C] * a1 + src_c_ptr[ch + C * 2] * a2 + src_c_ptr[ch + C * 3] * a3;
                    rows1[x_id + ch] = src_n0_ptr[ch] * a0 + src_n0_ptr[ch + C] * a1 + src_n0_ptr[ch + C * 2] * a2 + src_n0_ptr[ch + C * 3] * a3;
                    rows2[x_id + ch] = src_n1_ptr[ch] * a0 + src_n1_ptr[ch + C] * a1 + src_n1_ptr[ch + C * 2] * a2 + src_n1_ptr[ch + C * 3] * a3;
                    rows3[x_id + ch] = src_n2_ptr[ch] * a0 + src_n2_ptr[ch + C] * a1 + src_n2_ptr[ch + C * 2] * a2 + src_n2_ptr[ch + C * 3] * a3;
                }

                alpha_ptr += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        Tp *dst_row = dst.Ptr<Tp>(dy);

        float32x4_t vqf32_c_beta = neon::vload1q(beta_ptr);

        DT_S32 owidth_align8 = owidth & (-8);
        DT_S32 dx = 0;
        for (; dx < owidth_align8; dx += 8)
        {
            WMVType wmvqf32_cx0_rows, wmvqf32_n0x0_rows, wmvqf32_n1x0_rows, wmvqf32_n2x0_rows;
            WMVType wmvqf32_cx1_rows, wmvqf32_n0x1_rows, wmvqf32_n1x1_rows, wmvqf32_n2x1_rows;

            neon::vload(rows0 + dx * C, wmvqf32_cx0_rows);
            neon::vload(rows1 + dx * C, wmvqf32_n0x0_rows);
            neon::vload(rows2 + dx * C, wmvqf32_n1x0_rows);
            neon::vload(rows3 + dx * C, wmvqf32_n2x0_rows);
            neon::vload(rows0 + (dx + 4) * C, wmvqf32_cx1_rows);
            neon::vload(rows1 + (dx + 4) * C, wmvqf32_n0x1_rows);
            neon::vload(rows2 + (dx + 4) * C, wmvqf32_n1x1_rows);
            neon::vload(rows3 + (dx + 4) * C, wmvqf32_n2x1_rows);

            WMVType wmvqf32_x0_result, wmvqf32_x1_result;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                wmvqf32_x0_result.val[ch] = ResizeCuCommNeonVerCore<Tp>(wmvqf32_cx0_rows.val[ch], wmvqf32_n0x0_rows.val[ch], wmvqf32_n1x0_rows.val[ch], wmvqf32_n2x0_rows.val[ch], vqf32_c_beta);
                wmvqf32_x1_result.val[ch] = ResizeCuCommNeonVerCore<Tp>(wmvqf32_cx1_rows.val[ch], wmvqf32_n0x1_rows.val[ch], wmvqf32_n1x1_rows.val[ch], wmvqf32_n2x1_rows.val[ch], vqf32_c_beta);
            }
            neon::vstore(dst_row + dx * C, wmvqf32_x0_result);
            neon::vstore(dst_row + (dx + 4) * C, wmvqf32_x1_result);
        }
        for (; dx < owidth; dx++)
        {
            DT_S32 x_id = dx * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x_id + ch] = rows0[x_id + ch] * beta_ptr[0] + rows1[x_id + ch] * beta_ptr[1] + rows2[x_id + ch] * beta_ptr[2] + rows3[x_id + ch] * beta_ptr[3];
            }
        }

        beta_ptr += 4;
    }

    return Status::OK;
}

template <typename Tp>
static Status ResizeCuCommNeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    DT_S32 iwidth   = src.GetSizes().m_width;
    DT_S32 iheight  = src.GetSizes().m_height;
    DT_S32 owidth   = dst.GetSizes().m_width;
    DT_S32 oheight  = dst.GetSizes().m_height;
    DT_S32 channel  = dst.GetSizes().m_channel;

    DT_S32 buffer_size = (owidth + oheight) * sizeof(DT_S32) + (owidth * 4 + oheight * 4) * sizeof(AlphaType);
    DT_S32 *buffer = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, buffer_size, 0));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    GetCuOffset<Tp>(buffer, iwidth, owidth, iheight, oheight);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    ThreadBuffer thread_buffer(ctx, owidth * 4 * channel * sizeof(DT_S32));

    Status ret = Status::ERROR;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, oheight, ResizeCuCommNeonImpl<Tp, 1>, ctx, std::cref(src), std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonImpl run failed, channel: 1");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, oheight, ResizeCuCommNeonImpl<Tp, 2>, ctx, std::cref(src), std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonImpl run failed, channel: 2");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, oheight, ResizeCuCommNeonImpl<Tp, 3>, ctx, std::cref(src), std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonImpl run failed, channel: 3");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport channel more than 3");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_FREE(ctx, buffer);

    AURA_RETURN(ctx, ret);
}

Status ResizeCuCommNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuCommNeonHelper<DT_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuCommNeonHelper<DT_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuCommNeonHelper<DT_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuCommNeonHelper<DT_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: DT_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeCuCommNeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeCuCommNeonHelper<DT_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommNeonHelper run failed, type: DT_F32");
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