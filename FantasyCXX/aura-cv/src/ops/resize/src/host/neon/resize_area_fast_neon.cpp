#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = DT_U8, DT_S8, DT_U16, DT_S16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value || std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeAreaFastNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 scale_x, DT_S32 scale_y, DT_S32 start_row, DT_S32 end_row)
{
    using Type = typename Promote<Tp>::Type;

    DT_S32 iwidth      = src.GetSizes().m_width;
    DT_S32 iwidth_x_cn = iwidth * C;
    DT_S32 istride     = src.GetRowPitch();
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 istride_t   = istride / sizeof(Tp);
    DT_S32 elem_counts = 16 / sizeof(Tp);
    DT_S32 width_align = iwidth_x_cn & (-elem_counts);
    DT_F32 area_div    = 1.f / (scale_x * scale_y);

    Type *buffer_data = thread_buffer.GetThreadData<Type>();

    if (!buffer_data)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y * scale_y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        const Tp *src_tmp_row = src_row;
        Type *buffer_tmp_row  = buffer_data;

        for (; x < width_align; x += elem_counts)
        {
            auto vq_src = neon::vload1q(src_tmp_row);
            auto vq_x0_sum = neon::vmovl(neon::vgetlow(vq_src));
            auto vq_x1_sum = neon::vmovl(neon::vgethigh(vq_src));

            for (DT_S32 z = 1; z < scale_y; z++)
            {
                vq_src = neon::vload1q(src_tmp_row + istride_t * z);
                vq_x0_sum = neon::vaddw(vq_x0_sum, neon::vgetlow(vq_src));
                vq_x1_sum = neon::vaddw(vq_x1_sum, neon::vgethigh(vq_src));
            }

            neon::vstore(buffer_tmp_row, vq_x0_sum);
            neon::vstore(buffer_tmp_row + elem_counts / 2, vq_x1_sum);
            src_tmp_row += elem_counts;
            buffer_tmp_row += elem_counts;
        }

        for (; x < iwidth_x_cn; x++)
        {
            Type sum = 0;
            for (DT_S32 z = 0; z < scale_y; z++)
            {
                sum += static_cast<Type>(*(src_tmp_row + istride_t * z));
            }
            *buffer_tmp_row++ = sum;
            src_tmp_row++;
        }

        for (x = 0; x < owidth; x++)
        {
            DT_S32 start = C * x;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                DT_F32 sum = 0.f;
                for (DT_S32 z = 0; z < scale_x; z++)
                {
                    sum += static_cast<DT_F32>(buffer_data[start * scale_x + C * z + ch]);
                }

                dst_row[start + ch] = SaturateCast<Tp>(sum * area_div);
            }
        }
    }

    return Status::OK;
}

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeAreaFastNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 scale_x, DT_S32 scale_y, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth      = src.GetSizes().m_width;
    DT_S32 iwidth_x_cn = iwidth * C;
    DT_S32 istride     = src.GetRowPitch();
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 istride_t   = istride / sizeof(Tp);
    DT_S32 elem_counts = 32 / sizeof(Tp);
    DT_S32 width_align = iwidth_x_cn & (-elem_counts);
    DT_F32 area_div    = 1.f / (scale_x * scale_y);

    Tp *buffer_data = thread_buffer.GetThreadData<Tp>();

    if (!buffer_data)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y * scale_y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        const Tp *src_tmp_row = src_row;
        Tp *buffer_tmp_row = buffer_data;

        for (; x < width_align; x += elem_counts)
        {
            auto vq_x0_sum = neon::vload1q(src_tmp_row);
            auto vq_x1_sum = neon::vload1q(src_tmp_row + elem_counts / 2);

            for (DT_S32 z = 1; z < scale_y; z++)
            {
                auto vq_x0_src = neon::vload1q(src_tmp_row + istride_t * z);
                auto vq_x1_src = neon::vload1q(src_tmp_row + istride_t * z + elem_counts / 2);
                vq_x0_sum = neon::vadd(vq_x0_sum, vq_x0_src);
                vq_x1_sum = neon::vadd(vq_x1_sum, vq_x1_src);
            }

            neon::vstore(buffer_tmp_row, vq_x0_sum);
            neon::vstore(buffer_tmp_row + elem_counts / 2, vq_x1_sum);
            src_tmp_row += elem_counts;
            buffer_tmp_row += elem_counts;
        }

        for (; x < iwidth_x_cn; x++)
        {
            Tp sum = *src_tmp_row;
            for (DT_S32 z = 1; z < scale_y; z++)
            {
                sum += *(src_tmp_row + istride_t * z);
            }
            *buffer_tmp_row++ = sum;
            src_tmp_row++;
        }

        for (x = 0; x < owidth; x++)
        {
            DT_S32 xc = C * x;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Tp sum = *(buffer_data + xc * scale_x + ch);

                for (DT_S32 z = 1; z < scale_x; z++)
                {
                    sum += *(buffer_data + xc * scale_x + C * z + ch);
                }

                *(dst_row + xc + ch) = sum * area_div;
            }
        }
    }

    return Status::OK;
}

#if defined(AURA_ENABLE_NEON_FP16)
// Tp = MI_F16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeAreaFastNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 scale_x, DT_S32 scale_y, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth      = src.GetSizes().m_width;
    DT_S32 iwidth_x_cn = iwidth * C;
    DT_S32 istride     = src.GetRowPitch();
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 istride_t   = istride / sizeof(Tp);
    DT_S32 elem_counts = 16 / sizeof(Tp);
    DT_S32 width_align = iwidth_x_cn & (-elem_counts);
    DT_F32 area_div    = static_cast<DT_F32>(1.f / (scale_x * scale_y));

    DT_F32 *buffer_data = thread_buffer.GetThreadData<DT_F32>();

    if (!buffer_data)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y * scale_y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        const Tp *src_tmp_row = src_row;
        DT_F32 *buffer_tmp_row = buffer_data;

        for (; x < width_align; x += elem_counts)
        {
            auto vqf16_src = neon::vload1q(src_tmp_row);
            float32x4_t vqf32_x0_sum = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src));
            float32x4_t vqf32_x1_sum = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src));

            for (DT_S32 z = 1; z < scale_y; z++)
            {
                vqf16_src = neon::vload1q(src_tmp_row + istride_t * z);
                float32x4_t vqf32_x0_tmp = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src));
                float32x4_t vqf32_x1_tmp = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src));

                vqf32_x0_sum = neon::vadd(vqf32_x0_sum, vqf32_x0_tmp);
                vqf32_x1_sum = neon::vadd(vqf32_x1_sum, vqf32_x1_tmp);
            }

            neon::vstore(buffer_tmp_row, vqf32_x0_sum);
            neon::vstore(buffer_tmp_row + elem_counts / 2, vqf32_x1_sum);
            src_tmp_row += elem_counts;
            buffer_tmp_row += elem_counts;
        }

        for (; x < iwidth_x_cn; x++)
        {
            DT_F32 sum = static_cast<DT_F32>(*src_tmp_row);
            for (DT_S32 z = 1; z < scale_y; z++)
            {
                sum += static_cast<DT_F32>(*(src_tmp_row + istride_t * z));
            }
            *buffer_tmp_row++ = sum;
            src_tmp_row++;
        }

        for (x = 0; x < owidth; x++)
        {
            DT_S32 xc = C * x;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                DT_F32 sum = *(buffer_data + xc * scale_x + ch);

                for (DT_S32 z = 1; z < scale_x; z++)
                {
                    sum += *(buffer_data + xc * scale_x + C * z + ch);
                }

                *(dst_row + xc + ch) = static_cast<Tp>(sum * area_div);
            }
        }
    }

    return Status::OK;
}
#endif

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, DT_F32, MI_F16
template <typename Tp, DT_S32 C>
static Status ResizeAreaUpX2Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using MVTypeInter = typename neon::MQVector<Tp, 2>::MVType;
    using MVType = typename neon::MQVector<Tp, C>::MVType;

    DT_S32 elem_counts = 16 / sizeof(Tp);
    DT_S32 iwidth = src.GetSizes().m_width;
    DT_S32 width_align = iwidth & (-elem_counts);

    start_row >>= 1; //iheight is half of oheight
    end_row >>= 1;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_c = dst.Ptr<Tp>(2 * y);
        Tp *dst_n = dst.Ptr<Tp>(2 * y + 1);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            MVType mvq_src;
            MVTypeInter v2q_tmp;
            MVType mvq_x0_result, mvq_x1_result;

            neon::vload(src_row + x * C, mvq_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                v2q_tmp = neon::vzip(mvq_src.val[ch], mvq_src.val[ch]);
                mvq_x0_result.val[ch] = v2q_tmp.val[0];
                mvq_x1_result.val[ch] = v2q_tmp.val[1];
            }

            neon::vstore(dst_c + x * C * 2, mvq_x0_result);
            neon::vstore(dst_c + x * C * 2 + elem_counts * C, mvq_x1_result);

            neon::vstore(dst_n + x * C * 2, mvq_x0_result);
            neon::vstore(dst_n + x * C * 2 + elem_counts * C, mvq_x1_result);
        }

        for (; x < iwidth; x++)
        {
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Tp result = src_row[x * C + ch];

                dst_c[x * 2 * C + ch] = result;
                dst_c[x * 2 * C + 1 * C + ch] = result;
                dst_n[x * 2 * C + ch] = result;
                dst_n[x * 2 * C + 1 * C + ch] = result;
            }
        }
    }

    return Status::OK;
}

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, DT_F32, MI_F16
template <typename Tp, DT_S32 C>
static Status ResizeAreaUpX4Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using MVTypeInter = typename neon::MQVector<Tp, 2>::MVType;
    using MVType = typename neon::MQVector<Tp, C>::MVType;
    DT_S32 elem_counts = 16 / sizeof(Tp);

    DT_S32 iwidth = src.GetSizes().m_width;
    DT_S32 width_align = iwidth & (-elem_counts);

    start_row >>= 2; //iheight is 1/4 of oheight
    end_row >>= 2;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_c  = dst.Ptr<Tp>(4 * y);
        Tp *dst_n0 = dst.Ptr<Tp>(4 * y + 1);
        Tp *dst_n1 = dst.Ptr<Tp>(4 * y + 2);
        Tp *dst_n2 = dst.Ptr<Tp>(4 * y + 3);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            MVType mvq_src;
            MVTypeInter v2q_tmp, v2q_l, v2q_h;
            MVType mvq_x0_result, mvq_x1_result, mvq_x2_result, mvq_x3_result;

            neon::vload(src_row + x * C, mvq_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                v2q_tmp = neon::vzip(mvq_src.val[ch], mvq_src.val[ch]);

                v2q_l = neon::vzip(v2q_tmp.val[0], v2q_tmp.val[0]);
                v2q_h = neon::vzip(v2q_tmp.val[1], v2q_tmp.val[1]);

                mvq_x0_result.val[ch] = v2q_l.val[0];
                mvq_x1_result.val[ch] = v2q_l.val[1];
                mvq_x2_result.val[ch] = v2q_h.val[0];
                mvq_x3_result.val[ch] = v2q_h.val[1];
            }

            neon::vstore(dst_c + x * C * 4, mvq_x0_result);
            neon::vstore(dst_c + x * C * 4 + elem_counts * C, mvq_x1_result);
            neon::vstore(dst_c + x * C * 4 + 2 * elem_counts * C, mvq_x2_result);
            neon::vstore(dst_c + x * C * 4 + 3 * elem_counts * C, mvq_x3_result);

            neon::vstore(dst_n0 + x * C * 4, mvq_x0_result);
            neon::vstore(dst_n0 + x * C * 4 + elem_counts * C, mvq_x1_result);
            neon::vstore(dst_n0 + x * C * 4 + 2 * elem_counts * C, mvq_x2_result);
            neon::vstore(dst_n0 + x * C * 4 + 3 * elem_counts * C, mvq_x3_result);

            neon::vstore(dst_n1 + x * C * 4, mvq_x0_result);
            neon::vstore(dst_n1 + x * C * 4 + elem_counts * C, mvq_x1_result);
            neon::vstore(dst_n1 + x * C * 4 + 2 * elem_counts * C, mvq_x2_result);
            neon::vstore(dst_n1 + x * C * 4 + 3 * elem_counts * C, mvq_x3_result);

            neon::vstore(dst_n2 + x * C * 4, mvq_x0_result);
            neon::vstore(dst_n2 + x * C * 4 + elem_counts * C, mvq_x1_result);
            neon::vstore(dst_n2 + x * C * 4 + 2 * elem_counts * C, mvq_x2_result);
            neon::vstore(dst_n2 + x * C * 4 + 3 * elem_counts * C, mvq_x3_result);
        }

        for (; x < iwidth; x++)
        {
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Tp result = src_row[x * C + ch];

                dst_c[x * 4 * C + ch] = result;
                dst_c[x * 4 * C + 1 * C + ch] = result;
                dst_c[x * 4 * C + 2 * C + ch] = result;
                dst_c[x * 4 * C + 3 * C + ch] = result;

                dst_n0[x * 4 * C + ch] = result;
                dst_n0[x * 4 * C + 1 * C + ch] = result;
                dst_n0[x * 4 * C + 2 * C + ch] = result;
                dst_n0[x * 4 * C + 3 * C + ch] = result;

                dst_n1[x * 4 * C + ch] = result;
                dst_n1[x * 4 * C + 1 * C + ch] = result;
                dst_n1[x * 4 * C + 2 * C + ch] = result;
                dst_n1[x * 4 * C + 3 * C + ch] = result;

                dst_n2[x * 4 * C + ch] = result;
                dst_n2[x * 4 * C + 1 * C + ch] = result;
                dst_n2[x * 4 * C + 2 * C + ch] = result;
                dst_n2[x * 4 * C + 3 * C + ch] = result;
            }
        }
    }

    return Status::OK;
}

// SType = DT_U8, DT_S8
// VType = uint8x16_t, int8x16_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, SType>::value || std::is_same<DT_S8, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX2NeonCore(VType &vq8_c_src, VType &vq8_n0_src)
{
    auto vq16_c_src  = neon::vpaddl(vq8_c_src);
    auto vq16_n0_src = neon::vpaddl(vq8_n0_src);

    vq16_c_src      = neon::vadd(vq16_c_src, vq16_n0_src);
    auto vd8_result = neon::vrshrn_n<2>(vq16_c_src);

    return vd8_result;
}

// Tp = DT_U8, DT_S8
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeAreaDownX2Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 width_align = owidth & (-8);
    using MVType       = typename neon::MDVector<Tp, C>::MVType;
    using VType        = typename neon::QVector<Tp>::VType;
    using WMVType      = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y << 1);
        const Tp *src_n = src.Ptr<Tp>((y << 1) + 1);
        Tp *dst_row     = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 8)
        {
            WMVType wmvq8_c_src, wmvq8_n0_src;
            neon::vload(src_c + (x * 2 * C), wmvq8_c_src);
            neon::vload(src_n + (x * 2 * C), wmvq8_n0_src);
            MVType mvd8_result;

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd8_result.val[ch] = ResizeAreaDownX2NeonCore<Tp, VType>(wmvq8_c_src.val[ch], wmvq8_n0_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd8_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 2 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x * C + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C] + src_n[x * step + ch]
                                       + src_n[(x * step + ch) + C] + (1 << 1)) >> 2;
            }
        }
    }

    return Status::OK;
}

// SType = DT_U16, DT_S16
// VType = uint16x8_t, int16x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, SType>::value || std::is_same<DT_S16, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX2NeonCore(VType &vq16_c_src, VType &vq16_n0_src)
{
    auto vq32_c_src  = neon::vpaddl(vq16_c_src);
    auto vq32_n0_src = neon::vpaddl(vq16_n0_src);

    vq32_c_src       = neon::vadd(vq32_c_src, vq32_n0_src);
    auto vd16_result = neon::vrshrn_n<2>(vq32_c_src);

    return vd16_result;
}

// Tp = DT_U16, DT_S16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeAreaDownX2Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 width_align = owidth & (-4);
    using MVType       = typename neon::MDVector<Tp, C>::MVType;
    using VType        = typename neon::QVector<Tp>::VType;
    using WMVType      = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y << 1);
        const Tp *src_n = src.Ptr<Tp>((y << 1) + 1);
        Tp *dst_row     = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 4)
        {
            WMVType wmvq16_c_src, wmvq16_n0_src;
            neon::vload(src_c + (x * 2 * C), wmvq16_c_src);
            neon::vload(src_n + (x * 2 * C), wmvq16_n0_src);
            MVType mvd16_result;

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd16_result.val[ch] = ResizeAreaDownX2NeonCore<Tp, VType>(wmvq16_c_src.val[ch], wmvq16_n0_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd16_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 2 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x * C + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C] + src_n[x * step + ch]
                                       + src_n[(x * step + ch) + C] + (1 << 1)) >> 2;
            }
        }
    }

    return Status::OK;
}

// SType = DT_F32
// VType = float32x4_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_F32, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX2NeonCore(VType &vq_c_src, VType &vq_n0_src)
{
    auto vd_c_tmp  = neon::vpadd(neon::vgetlow(vq_c_src), neon::vgethigh(vq_c_src));
    auto vd_n0_tmp = neon::vpadd(neon::vgetlow(vq_n0_src), neon::vgethigh(vq_n0_src));

    auto vd_result = neon::vadd(vd_c_tmp, vd_n0_tmp);
    vd_result      = neon::vmul(vd_result, static_cast<SType>(0.25f));

    return vd_result;
}

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeAreaDownX2Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth            = dst.GetSizes().m_width;
    const DT_S32 elem_counts = 8 / sizeof(Tp);
    DT_S32 width_align       = owidth & (-elem_counts);
    using MVType             = typename neon::MDVector<Tp, C>::MVType;
    using VType              = typename neon::QVector<Tp>::VType;
    using WMVType            = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y << 1);
        const Tp *src_n = src.Ptr<Tp>((y << 1) + 1);
        Tp *dst_row     = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            WMVType wmvq_c_src, wmvq_n0_src;
            neon::vload(src_c + (x * 2 * C), wmvq_c_src);
            neon::vload(src_n + (x * 2 * C), wmvq_n0_src);
            MVType mvd_result;

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd_result.val[ch] = ResizeAreaDownX2NeonCore<Tp, VType>(wmvq_c_src.val[ch], wmvq_n0_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 2 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x * C + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C]
                                       + src_n[x * step + ch] + src_n[(x * step + ch) + C]) / 4;
            }
        }
    }

    return Status::OK;
}

#if defined(AURA_ENABLE_NEON_FP16)
// Tp = MI_F16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeAreaDownX2Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth            = dst.GetSizes().m_width;
    const DT_S32 elem_counts = 8 / sizeof(Tp);
    DT_S32 width_align       = owidth & (-elem_counts);
    using MVType             = typename neon::MDVector<Tp, C>::MVType;
    using VType              = typename neon::QVector<DT_F32>::VType;
    using WMVType            = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y << 1);
        const Tp *src_n = src.Ptr<Tp>((y << 1) + 1);
        Tp *dst_row     = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            WMVType wmvq_c_src, wmvq_n0_src;
            neon::vload(src_c + (x * 2 * C), wmvq_c_src);
            neon::vload(src_n + (x * 2 * C), wmvq_n0_src);
            MVType mvd_result;

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                VType vqf32_cx0_src = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_c_src.val[ch]));
                VType vqf32_cx1_src = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_c_src.val[ch]));
                VType vqf32_n0x0_src = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n0_src.val[ch]));
                VType vqf32_n0x1_src = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n0_src.val[ch]));
                VType vqf32_tmp = neon::vcombine(ResizeAreaDownX2NeonCore<DT_F32, VType>(vqf32_cx0_src, vqf32_n0x0_src),
                                                 ResizeAreaDownX2NeonCore<DT_F32, VType>(vqf32_cx1_src, vqf32_n0x1_src));

                mvd_result.val[ch] = neon::vcvt<Tp>(vqf32_tmp);
            }

            neon::vstore(dst_row + (x * C), mvd_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 2 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[x * C + ch] = (static_cast<DT_F32>(src_c[x * step + ch]) + static_cast<DT_F32>(src_c[(x * step + ch) + C]) +
                                       static_cast<DT_F32>(src_n[x * step + ch]) + static_cast<DT_F32>(src_n[(x * step + ch) + C])) * 0.25f;
            }
        }
    }

    return Status::OK;
}
#endif

// SType = DT_U8, DT_S8
// VType = uint8x16_t, int8x16_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, SType>::value || std::is_same<DT_S8, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX4NeonCore(VType &vq8_cx0_src, VType &vq8_cx1_src, VType &vq8_n0x0_src, VType &vq8_n0x1_src, VType &vq8_n1x0_src,
                         VType &vq8_n1x1_src, VType &vq8_n2x0_src, VType &vq8_n2x1_src)
{
    auto vq_cx0_tmp  = neon::vpaddl(vq8_cx0_src);
    auto vq_cx1_tmp  = neon::vpaddl(vq8_cx1_src);
    auto vq_n0x0_tmp = neon::vpaddl(vq8_n0x0_src);
    auto vq_n0x1_tmp = neon::vpaddl(vq8_n0x1_src);
    auto vq_n1x0_tmp = neon::vpaddl(vq8_n1x0_src);
    auto vq_n1x1_tmp = neon::vpaddl(vq8_n1x1_src);
    auto vq_n2x0_tmp = neon::vpaddl(vq8_n2x0_src);
    auto vq_n2x1_tmp = neon::vpaddl(vq8_n2x1_src);

    vq_cx0_tmp  = neon::vcombine(neon::vpadd(neon::vgetlow(vq_cx0_tmp), neon::vgethigh(vq_cx0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_cx1_tmp), neon::vgethigh(vq_cx1_tmp)));
    vq_n0x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n0x0_tmp), neon::vgethigh(vq_n0x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n0x1_tmp), neon::vgethigh(vq_n0x1_tmp)));
    vq_n1x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n1x0_tmp), neon::vgethigh(vq_n1x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n1x1_tmp), neon::vgethigh(vq_n1x1_tmp)));
    vq_n2x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n2x0_tmp), neon::vgethigh(vq_n2x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n2x1_tmp), neon::vgethigh(vq_n2x1_tmp)));

    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n0x0_tmp);
    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n1x0_tmp);
    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n2x0_tmp);

    auto vq_result = neon::vrshrn_n<4>(vq_cx0_tmp);

    return vq_result;
}

// Tp = DT_U8, DT_S8
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeAreaDownX4Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 width_align = owidth & (-8);
    using MVType       = typename neon::MDVector<Tp, C>::MVType;
    using VType        = typename neon::QVector<Tp>::VType;
    using WMVType      = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c  = src.Ptr<Tp>(y << 2);
        const Tp *src_n0 = src.Ptr<Tp>((y << 2) + 1);
        const Tp *src_n1 = src.Ptr<Tp>((y << 2) + 2);
        const Tp *src_n2 = src.Ptr<Tp>((y << 2) + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 8)
        {
            WMVType wmvq8_cx0_src, wmvq8_cx1_src, wmvq8_n0x0_src, wmvq8_n0x1_src, wmvq8_n1x0_src, wmvq8_n1x1_src, wmvq8_n2x0_src, wmvq8_n2x1_src;
            MVType mvd8_result;

            neon::vload(src_c + (x * 4 * C), wmvq8_cx0_src);
            neon::vload(src_c + (x * 4 * C) + (16 * C), wmvq8_cx1_src);
            neon::vload(src_n0 + (x * 4 * C), wmvq8_n0x0_src);
            neon::vload(src_n0 + (x * 4 * C) + (16 * C), wmvq8_n0x1_src);
            neon::vload(src_n1 + (x * 4 * C), wmvq8_n1x0_src);
            neon::vload(src_n1 + (x * 4 * C) + (16 * C), wmvq8_n1x1_src);
            neon::vload(src_n2 + (x * 4 * C), wmvq8_n2x0_src);
            neon::vload(src_n2 + (x * 4 * C) + (16 * C), wmvq8_n2x1_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd8_result.val[ch] = ResizeAreaDownX4NeonCore<Tp, VType>(wmvq8_cx0_src.val[ch], wmvq8_cx1_src.val[ch],
                                                                          wmvq8_n0x0_src.val[ch], wmvq8_n0x1_src.val[ch],
                                                                          wmvq8_n1x0_src.val[ch], wmvq8_n1x1_src.val[ch],
                                                                          wmvq8_n2x0_src.val[ch], wmvq8_n2x1_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd8_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 4 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[(x * C) + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C] + src_c[(x * step + ch) + 2 * C] + src_c[(x * step + ch) + 3 * C]
                                        + src_n0[x * step + ch] + src_n0[(x * step + ch) + C] + src_n0[(x * step + ch) + 2 * C] + src_n0[(x * step + ch) + 3 * C]
                                        + src_n1[x * step + ch] + src_n1[(x * step + ch) + C] + src_n1[(x * step + ch) + 2 * C] + src_n1[(x * step + ch) + 3 * C]
                                        + src_n2[x * step + ch] + src_n2[(x * step + ch) + C] + src_n2[(x * step + ch) + 2 * C] + src_n2[(x * step + ch) + 3 * C]
                                        + (1 << 3)) >> 4;
            }
        }
    }

    return Status::OK;
}

// SType = DT_U16, DT_S16
// VType = uint16x8_t, int16x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, SType>::value || std::is_same<DT_S16, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX4NeonCore(VType &vq16_cx0_src, VType &vq16_cx1_src, VType &vq16_n0x0_src, VType &vq16_n0x1_src,
                         VType &vq16_n1x0_src, VType &vq16_n1x1_src, VType &vq16_n2x0_src, VType &vq16_n2x1_src)
{
    auto vq_cx0_tmp  = neon::vpaddl(vq16_cx0_src);
    auto vq_cx1_tmp  = neon::vpaddl(vq16_cx1_src);
    auto vq_n0x0_tmp = neon::vpaddl(vq16_n0x0_src);
    auto vq_n0x1_tmp = neon::vpaddl(vq16_n0x1_src);
    auto vq_n1x0_tmp = neon::vpaddl(vq16_n1x0_src);
    auto vq_n1x1_tmp = neon::vpaddl(vq16_n1x1_src);
    auto vq_n2x0_tmp = neon::vpaddl(vq16_n2x0_src);
    auto vq_n2x1_tmp = neon::vpaddl(vq16_n2x1_src);

    vq_cx0_tmp  = neon::vcombine(neon::vpadd(neon::vgetlow(vq_cx0_tmp), neon::vgethigh(vq_cx0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_cx1_tmp), neon::vgethigh(vq_cx1_tmp)));
    vq_n0x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n0x0_tmp), neon::vgethigh(vq_n0x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n0x1_tmp), neon::vgethigh(vq_n0x1_tmp)));
    vq_n1x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n1x0_tmp), neon::vgethigh(vq_n1x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n1x1_tmp), neon::vgethigh(vq_n1x1_tmp)));
    vq_n2x0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n2x0_tmp), neon::vgethigh(vq_n2x0_tmp)),
                                 neon::vpadd(neon::vgetlow(vq_n2x1_tmp), neon::vgethigh(vq_n2x1_tmp)));

    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n0x0_tmp);
    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n1x0_tmp);
    vq_cx0_tmp = neon::vadd(vq_cx0_tmp, vq_n2x0_tmp);

    auto vq_result = neon::vrshrn_n<4>(vq_cx0_tmp);

    return vq_result;
}

// Tp = DT_U16, DT_S16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeAreaDownX4Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 width_align = owidth & (-4);

    using MVType  = typename neon::MDVector<Tp, C>::MVType;
    using VType   = typename neon::QVector<Tp>::VType;
    using WMVType = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c  = src.Ptr<Tp>(y << 2);
        const Tp *src_n0 = src.Ptr<Tp>((y << 2) + 1);
        const Tp *src_n1 = src.Ptr<Tp>((y << 2) + 2);
        const Tp *src_n2 = src.Ptr<Tp>((y << 2) + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 4)
        {
            WMVType wmvq16_cx0_src, wmvq16_cx1_src, wmvq16_n0x0_src, wmvq16_n0x1_src;
            WMVType wmvq16_n1x0_src, wmvq16_n1x1_src, wmvq16_n2x0_src, wmvq16_n2x1_src;
            MVType mvd16_result;

            neon::vload(src_c + (x * 4 * C), wmvq16_cx0_src);
            neon::vload(src_c + (x * 4 * C) + (8 * C), wmvq16_cx1_src);
            neon::vload(src_n0 + (x * 4 * C), wmvq16_n0x0_src);
            neon::vload(src_n0 + (x * 4 * C) + (8 * C), wmvq16_n0x1_src);
            neon::vload(src_n1 + (x * 4 * C), wmvq16_n1x0_src);
            neon::vload(src_n1 + (x * 4 * C) + (8 * C), wmvq16_n1x1_src);
            neon::vload(src_n2 + (x * 4 * C), wmvq16_n2x0_src);
            neon::vload(src_n2 + (x * 4 * C) + (8 * C), wmvq16_n2x1_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd16_result.val[ch] = ResizeAreaDownX4NeonCore<Tp, VType>(wmvq16_cx0_src.val[ch],  wmvq16_cx1_src.val[ch],
                                                                           wmvq16_n0x0_src.val[ch], wmvq16_n0x1_src.val[ch],
                                                                           wmvq16_n1x0_src.val[ch], wmvq16_n1x1_src.val[ch],
                                                                           wmvq16_n2x0_src.val[ch], wmvq16_n2x1_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd16_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 4 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[(x * C) + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C] + src_c[(x * step + ch) + 2 * C] + src_c[(x * step + ch) + 3 * C]
                                        + src_n0[x * step + ch] + src_n0[(x * step + ch) + C] + src_n0[(x * step + ch) + 2 * C] + src_n0[(x * step + ch) + 3 * C]
                                        + src_n1[x * step + ch] + src_n1[(x * step + ch) + C] + src_n1[(x * step + ch) + 2 * C] + src_n1[(x * step + ch) + 3 * C]
                                        + src_n2[x * step + ch] + src_n2[(x * step + ch) + C] + src_n2[(x * step + ch) + 2 * C] + src_n2[(x * step + ch) + 3 * C]
                                        + (1 << 3)) >> 4;
            }
        }
    }

    return Status::OK;
}

// SType = DT_F32
// VType = float32x4_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_F32, SType>::value, typename neon::DVector<SType>::VType>::type
ResizeAreaDownX4NeonCore(VType &vq_cx0_src, VType &vq_cx1_src, VType &vq_n0x0_src, VType &vq_n0x1_src,
                         VType &vq_n1x0_src, VType &vq_n1x1_src, VType &vq_n2x0_src, VType &vq_n2x1_src)
{
    auto vq_c_tmp  = neon::vcombine(neon::vpadd(neon::vgetlow(vq_cx0_src), neon::vgethigh(vq_cx0_src)),
                                    neon::vpadd(neon::vgetlow(vq_cx1_src), neon::vgethigh(vq_cx1_src)));
    auto vq_n0_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n0x0_src), neon::vgethigh(vq_n0x0_src)),
                                    neon::vpadd(neon::vgetlow(vq_n0x1_src), neon::vgethigh(vq_n0x1_src)));
    auto vq_n1_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n1x0_src), neon::vgethigh(vq_n1x0_src)),
                                    neon::vpadd(neon::vgetlow(vq_n1x1_src), neon::vgethigh(vq_n1x1_src)));
    auto vq_n2_tmp = neon::vcombine(neon::vpadd(neon::vgetlow(vq_n2x0_src), neon::vgethigh(vq_n2x0_src)),
                                    neon::vpadd(neon::vgetlow(vq_n2x1_src), neon::vgethigh(vq_n2x1_src)));

    auto vd_c_tmp  = neon::vpadd(neon::vgetlow(vq_c_tmp), neon::vgethigh(vq_c_tmp));
    auto vd_n0_tmp = neon::vpadd(neon::vgetlow(vq_n0_tmp), neon::vgethigh(vq_n0_tmp));
    auto vd_n1_tmp = neon::vpadd(neon::vgetlow(vq_n1_tmp), neon::vgethigh(vq_n1_tmp));
    auto vd_n2_tmp = neon::vpadd(neon::vgetlow(vq_n2_tmp), neon::vgethigh(vq_n2_tmp));

    vd_c_tmp = neon::vadd(vd_c_tmp, vd_n0_tmp);
    vd_c_tmp = neon::vadd(vd_c_tmp, vd_n1_tmp);
    vd_c_tmp = neon::vadd(vd_c_tmp, vd_n2_tmp);

    auto vd_result = neon::vmul(vd_c_tmp, static_cast<SType>(0.0625f));

    return vd_result;
}

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeAreaDownX4Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth            = dst.GetSizes().m_width;
    const DT_S32 elem_counts = 8 / sizeof(Tp);
    DT_S32 width_align       = owidth & (-elem_counts);
    using MVType             = typename neon::MDVector<Tp, C>::MVType;
    using VType              = typename neon::QVector<Tp>::VType;
    using WMVType            = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c  = src.Ptr<Tp>(y << 2);
        const Tp *src_n0 = src.Ptr<Tp>((y << 2) + 1);
        const Tp *src_n1 = src.Ptr<Tp>((y << 2) + 2);
        const Tp *src_n2 = src.Ptr<Tp>((y << 2) + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            WMVType wmvq_cx0_src, wmvq_cx1_src, wmvq_n0x0_src, wmvq_n0x1_src;
            WMVType wmvq_n1x0_src, wmvq_n1x1_src, wmvq_n2x0_src, wmvq_n2x1_src;
            MVType mvd_result;

            neon::vload(src_c + (x * 4 * C), wmvq_cx0_src);
            neon::vload(src_c + (x * 4 * C) + 2 * elem_counts * C, wmvq_cx1_src);
            neon::vload(src_n0 + (x * 4 * C), wmvq_n0x0_src);
            neon::vload(src_n0 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n0x1_src);
            neon::vload(src_n1 + (x * 4 * C), wmvq_n1x0_src);
            neon::vload(src_n1 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n1x1_src);
            neon::vload(src_n2 + (x * 4 * C), wmvq_n2x0_src);
            neon::vload(src_n2 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n2x1_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mvd_result.val[ch] = ResizeAreaDownX4NeonCore<Tp, VType>(wmvq_cx0_src.val[ch], wmvq_cx1_src.val[ch],
                                                                         wmvq_n0x0_src.val[ch], wmvq_n0x1_src.val[ch],
                                                                         wmvq_n1x0_src.val[ch], wmvq_n1x1_src.val[ch],
                                                                         wmvq_n2x0_src.val[ch], wmvq_n2x1_src.val[ch]);
            }

            neon::vstore(dst_row + (x * C), mvd_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 4 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[(x * C) + ch] = (src_c[x * step + ch] + src_c[(x * step + ch) + C] + src_c[(x * step + ch) + 2 * C] + src_c[(x * step + ch) + 3 * C]
                                        + src_n0[x * step + ch] + src_n0[(x * step + ch) + C] + src_n0[(x * step + ch) + 2 * C] + src_n0[(x * step + ch) + 3 * C]
                                        + src_n1[x * step + ch] + src_n1[(x * step + ch) + C] + src_n1[(x * step + ch) + 2 * C] + src_n1[(x * step + ch) + 3 * C]
                                        + src_n2[x * step + ch] + src_n2[(x * step + ch) + C] + src_n2[(x * step + ch) + 2 * C] + src_n2[(x * step + ch) + 3 * C]
                                        ) / 16;
            }
        }
    }

    return Status::OK;
}

#if defined(AURA_ENABLE_NEON_FP16)
// Tp = MI_F16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeAreaDownX4Neon(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth            = dst.GetSizes().m_width;
    const DT_S32 elem_counts = 8 / sizeof(Tp);
    DT_S32 width_align       = owidth & (-elem_counts);
    using MVType             = typename neon::MDVector<Tp, C>::MVType;
    using VType              = typename neon::QVector<DT_F32>::VType;
    using WMVType            = typename neon::WMVectorNums<MVType>::MVType;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c  = src.Ptr<Tp>(y << 2);
        const Tp *src_n0 = src.Ptr<Tp>((y << 2) + 1);
        const Tp *src_n1 = src.Ptr<Tp>((y << 2) + 2);
        const Tp *src_n2 = src.Ptr<Tp>((y << 2) + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += elem_counts)
        {
            WMVType wmvq_cx0_src, wmvq_cx1_src, wmvq_n0x0_src, wmvq_n0x1_src;
            WMVType wmvq_n1x0_src, wmvq_n1x1_src, wmvq_n2x0_src, wmvq_n2x1_src;
            MVType mvd_result;

            neon::vload(src_c + (x * 4 * C), wmvq_cx0_src);
            neon::vload(src_c + (x * 4 * C) + 2 * elem_counts * C, wmvq_cx1_src);
            neon::vload(src_n0 + (x * 4 * C), wmvq_n0x0_src);
            neon::vload(src_n0 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n0x1_src);
            neon::vload(src_n1 + (x * 4 * C), wmvq_n1x0_src);
            neon::vload(src_n1 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n1x1_src);
            neon::vload(src_n2 + (x * 4 * C), wmvq_n2x0_src);
            neon::vload(src_n2 + (x * 4 * C) + 2 * elem_counts * C, wmvq_n2x1_src);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                VType vqf32_cx0 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_cx0_src.val[ch]));
                VType vqf32_cx1 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_cx0_src.val[ch]));
                VType vqf32_cx2 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_cx1_src.val[ch]));
                VType vqf32_cx3 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_cx1_src.val[ch]));

                VType vqf32_n0x0 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n0x0_src.val[ch]));
                VType vqf32_n0x1 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n0x0_src.val[ch]));
                VType vqf32_n0x2 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n0x1_src.val[ch]));
                VType vqf32_n0x3 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n0x1_src.val[ch]));

                VType vqf32_n1x0 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n1x0_src.val[ch]));
                VType vqf32_n1x1 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n1x0_src.val[ch]));
                VType vqf32_n1x2 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n1x1_src.val[ch]));
                VType vqf32_n1x3 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n1x1_src.val[ch]));

                VType vqf32_n2x0 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n2x0_src.val[ch]));
                VType vqf32_n2x1 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n2x0_src.val[ch]));
                VType vqf32_n2x2 = neon::vcvt<DT_F32>(neon::vgetlow(wmvq_n2x1_src.val[ch]));
                VType vqf32_n2x3 = neon::vcvt<DT_F32>(neon::vgethigh(wmvq_n2x1_src.val[ch]));

                VType vqf32_tmp = neon::vcombine(ResizeAreaDownX4NeonCore<DT_F32, VType>(vqf32_cx0, vqf32_cx1, vqf32_n0x0, vqf32_n0x1,
                                                                                         vqf32_n1x0, vqf32_n1x1, vqf32_n2x0, vqf32_n2x1),
                                                 ResizeAreaDownX4NeonCore<DT_F32, VType>(vqf32_cx2, vqf32_cx3, vqf32_n0x2, vqf32_n0x3,
                                                                                         vqf32_n1x2, vqf32_n1x3, vqf32_n2x2, vqf32_n2x3));
                mvd_result.val[ch] = neon::vcvt<MI_F16>(vqf32_tmp);
            }

            neon::vstore(dst_row + (x * C), mvd_result);
        }

        for (; x < owidth; x++)
        {
            DT_S32 step = 4 * C;
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                dst_row[(x * C) + ch] = (static_cast<DT_F32>(src_c[x * step + ch]) + static_cast<DT_F32>(src_c[(x * step + ch) + C]) +
                                         static_cast<DT_F32>(src_c[(x * step + ch) + 2 * C]) + static_cast<DT_F32>(src_c[(x * step + ch) + 3 * C]) +
                                         static_cast<DT_F32>(src_n0[x * step + ch] + src_n0[(x * step + ch) + C]) + static_cast<DT_F32>(src_n0[(x * step + ch) + 2 * C] + src_n0[(x * step + ch) + 3 * C]) +
                                         static_cast<DT_F32>(src_n1[x * step + ch] + src_n1[(x * step + ch) + C]) + static_cast<DT_F32>(src_n1[(x * step + ch) + 2 * C] + src_n1[(x * step + ch) + 3 * C]) +
                                         static_cast<DT_F32>(src_n2[x * step + ch] + src_n2[(x * step + ch) + C]) + static_cast<DT_F32>(src_n2[(x * step + ch) + 2 * C] + src_n2[(x * step + ch) + 3 * C])
                                        ) / 16.f;
            }
        }
    }

    return Status::OK;
}
#endif

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, DT_F32, MI_F16
template <typename Tp, DT_S32 C>
static Status ResizeAreaFastNeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "null workerpool ptr");
        return Status::ERROR;
    }

    DT_F32 scale_x = static_cast<DT_F64>(src.GetSizes().m_width) / dst.GetSizes().m_width;
    DT_F32 scale_y = static_cast<DT_F64>(src.GetSizes().m_height) / dst.GetSizes().m_height;
    DT_S32 int_scale_x = src.GetSizes().m_width / dst.GetSizes().m_width;
    DT_S32 int_scale_y = src.GetSizes().m_height / dst.GetSizes().m_height;

    Status ret = Status::ERROR;

    if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 2.0f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeAreaDownX2Neon<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDownX2Neon failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.0f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeAreaDownX4Neon<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDownX4Neon failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.5f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeAreaUpX2Neon<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaUpX2Neon failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.25f))
    {
        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeAreaUpX4Neon<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaUpX4Neon failed");
        }
    }
    else if (NearlyEqual(scale_x, int_scale_x) && NearlyEqual(scale_y, int_scale_y))
    {
        DT_S32 iwidth = src.GetSizes().m_width;
        DT_S32 type_size = Min(static_cast<DT_S32>(sizeof(Tp) * 2), 4);

        ThreadBuffer thread_buffer(ctx, type_size * iwidth * C);

        ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeAreaFastNeonImpl<Tp, C>, ctx, std::cref(src),
                              std::ref(dst), std::ref(thread_buffer), scale_x, scale_y);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonImpl failed");
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "not x2, x4 or int scale");
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeAreaFastNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    DT_S32 channel = src.GetSizes().m_channel;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), channel);

    Status ret = Status::ERROR;

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, 1):
        {
            ret = ResizeAreaFastNeonHelper<DT_U8, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U8, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 1):
        {
            ret = ResizeAreaFastNeonHelper<DT_S8, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S8, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 1):
        {
            ret = ResizeAreaFastNeonHelper<DT_U16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U16, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 1):
        {
            ret = ResizeAreaFastNeonHelper<DT_S16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S16, C1");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 1):
        {
            ret = ResizeAreaFastNeonHelper<MI_F16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: MI_F16, C1");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 1):
        {
            ret = ResizeAreaFastNeonHelper<DT_F32, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_F32, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, 2):
        {
            ret = ResizeAreaFastNeonHelper<DT_U8, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U8, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 2):
        {
            ret = ResizeAreaFastNeonHelper<DT_S8, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S8, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 2):
        {
            ret = ResizeAreaFastNeonHelper<DT_U16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U16, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 2):
        {
            ret = ResizeAreaFastNeonHelper<DT_S16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S16, C2");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 2):
        {
            ret = ResizeAreaFastNeonHelper<MI_F16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: MI_F16, C2");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 2):
        {
            ret = ResizeAreaFastNeonHelper<DT_F32, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_F32, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, 3):
        {
            ret = ResizeAreaFastNeonHelper<DT_U8, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U8, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 3):
        {
            ret = ResizeAreaFastNeonHelper<DT_S8, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S8, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 3):
        {
            ret = ResizeAreaFastNeonHelper<DT_U16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_U16, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 3):
        {
            ret = ResizeAreaFastNeonHelper<DT_S16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_S16, C3");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 3):
        {
            ret = ResizeAreaFastNeonHelper<MI_F16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: MI_F16, C3");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 3):
        {
            ret = ResizeAreaFastNeonHelper<DT_F32, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastNeonHelper failed, type: DT_F32, C3");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel number or data type");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura