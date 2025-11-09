#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <DT_U32 MODE>
AURA_ALWAYS_INLINE DT_VOID Uv2Rgbuv(uint8x8_t vdu8_u, uint8x8_t vdu8_v,
                                    int32x4x2_t &v2qs32_ruv, int32x4x2_t &v2qs32_guv, int32x4x2_t &v2qs32_buv)
{
   
    /// 
    int16x8_t vqs16_val128;
    int32x4_t vqs32_round;
    neon::vdup(vqs16_val128, static_cast<DT_S16>(128));
    neon::vdup(vqs32_round, static_cast<DT_S32>(1 << (CVTCOLOR_COEF_BITS - 1)));

    constexpr DT_S32 V2R = Yuv2RgbParamTraits<MODE>::V2R;
    constexpr DT_S32 V2G = Yuv2RgbParamTraits<MODE>::V2G;
    constexpr DT_S32 U2G = Yuv2RgbParamTraits<MODE>::U2G;
    constexpr DT_S32 U2B = Yuv2RgbParamTraits<MODE>::U2B;

    int16x8_t vqs16_u = neon::vreinterpret(neon::vmovl(vdu8_u));
    int16x8_t vqs16_v = neon::vreinterpret(neon::vmovl(vdu8_v));
    vqs16_u           = neon::vsub(vqs16_u, vqs16_val128);
    vqs16_v           = neon::vsub(vqs16_v, vqs16_val128);

    int32x4_t vqs32_u_lo = neon::vmovl(neon::vgetlow(vqs16_u));
    int32x4_t vqs32_u_hi = neon::vmovl(neon::vgethigh(vqs16_u));
    int32x4_t vqs32_v_lo = neon::vmovl(neon::vgetlow(vqs16_v));
    int32x4_t vqs32_v_hi = neon::vmovl(neon::vgethigh(vqs16_v));

    v2qs32_ruv.val[0] = neon::vmla(vqs32_round, vqs32_v_lo, V2R);
    v2qs32_ruv.val[1] = neon::vmla(vqs32_round, vqs32_v_hi, V2R);

    v2qs32_guv.val[0] = neon::vmla(vqs32_round, vqs32_v_lo, V2G);
    v2qs32_guv.val[1] = neon::vmla(vqs32_round, vqs32_v_hi, V2G);
    v2qs32_guv.val[0] = neon::vmla(v2qs32_guv.val[0], vqs32_u_lo, U2G);
    v2qs32_guv.val[1] = neon::vmla(v2qs32_guv.val[1], vqs32_u_hi, U2G);

    v2qs32_buv.val[0] = neon::vmla(vqs32_round, vqs32_u_lo, U2B);
    v2qs32_buv.val[1] = neon::vmla(vqs32_round, vqs32_u_hi, U2B);
}

template <DT_U32 MODE>
AURA_ALWAYS_INLINE DT_VOID Yrgbuv2Rgb(uint8x8_t vdu8_y, int32x4x2_t v2qs32_ruv, int32x4x2_t v2qs32_guv, int32x4x2_t v2qs32_buv,
                                      uint8x8_t &vdu8_r, uint8x8_t &vdu8_g, uint8x8_t &vdu8_b)
{
    constexpr DT_S32 Y2RGB = Yuv2RgbParamTraits<MODE>::Y2RGB;

    int16x8_t vqs16_y;
    if (!MODE)
    {
        uint8x8_t vu8_val16;
        neon::vdup(vu8_val16, static_cast<DT_U8>(16));
        vqs16_y = neon::vreinterpret(neon::vmovl(neon::vqsub(vdu8_y, vu8_val16)));
    }
    else
    {
        vqs16_y = neon::vreinterpret(neon::vmovl(vdu8_y));
    }

    int32x4_t vqs32_y_lo = neon::vmovl(neon::vgetlow(vqs16_y));
    int32x4_t vqs32_y_hi = neon::vmovl(neon::vgethigh(vqs16_y));
    vqs32_y_lo           = neon::vmul(vqs32_y_lo, Y2RGB);
    vqs32_y_hi           = neon::vmul(vqs32_y_hi, Y2RGB);

    int16x4_t vds16_sum_lo = neon::vshrn_n<16>(neon::vadd(vqs32_y_lo, v2qs32_ruv.val[0]));
    int16x4_t vds16_sum_hi = neon::vshrn_n<16>(neon::vadd(vqs32_y_hi, v2qs32_ruv.val[1]));
    vds16_sum_lo           = neon::vshr_n<4>(vds16_sum_lo);
    vds16_sum_hi           = neon::vshr_n<4>(vds16_sum_hi);
    vdu8_r                 = neon::vqmovun(neon::vcombine(vds16_sum_lo, vds16_sum_hi));

    vds16_sum_lo = neon::vshrn_n<16>(neon::vadd(vqs32_y_lo, v2qs32_guv.val[0]));
    vds16_sum_hi = neon::vshrn_n<16>(neon::vadd(vqs32_y_hi, v2qs32_guv.val[1]));
    vds16_sum_lo = neon::vshr_n<4>(vds16_sum_lo);
    vds16_sum_hi = neon::vshr_n<4>(vds16_sum_hi);
    vdu8_g       = neon::vqmovun(neon::vcombine(vds16_sum_lo, vds16_sum_hi));

    vds16_sum_lo = neon::vshrn_n<16>(neon::vadd(vqs32_y_lo, v2qs32_buv.val[0]));
    vds16_sum_hi = neon::vshrn_n<16>(neon::vadd(vqs32_y_hi, v2qs32_buv.val[1]));
    vds16_sum_lo = neon::vshr_n<4>(vds16_sum_lo);
    vds16_sum_hi = neon::vshr_n<4>(vds16_sum_hi);
    vdu8_b       = neon::vqmovun(neon::vcombine(vds16_sum_lo, vds16_sum_hi));
}

template <DT_U32 MODE>
static Status CvtNv2RgbNeonImpl(const Mat &src_y, const Mat &src_uv, Mat &dst, DT_BOOL swapuv, DT_S32 start_row, DT_S32 end_row)
{
    /// 获取输出的RGB的的width
    const DT_S32 width       = dst.GetSizes().m_width;
    /// 将这个图像的宽度对齐到16的整数倍。后续进行
    const DT_S32 width_align = width & (-16);

    const DT_S32 uidx = swapuv;
    const DT_S32 vidx = 1 - uidx;
    // 注意：这个地方传入的end_row 是UV分量的
    for (DT_S32 y = start_row; y < end_row; y += 1)
    {
        /// 找到Y分量的对应的当前行和下一行
        const DT_U8 *src_y_c  = src_y.Ptr<DT_U8>(2 * y);
        const DT_U8 *src_y_n  = src_y.Ptr<DT_U8>(2 * y + 1);
        /// 找到UV分量的当前行
        const DT_U8 *src_uv_c = src_uv.Ptr<DT_U8>(y);

        DT_U8 *dst_c = dst.Ptr<DT_U8>(2 * y);
        DT_U8 *dst_n = dst.Ptr<DT_U8>(2 * y + 1);

        // TODO 这个地方DT_S32 x = 0放在外部，是不是故意为之？？有什么特殊意义？？
        DT_S32 x = 0;
        /// 对齐到16的整数倍之后，我们一次性处理16个元素
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            // uint8数据的类型，这个就是加载交错加载16个8位数据
            // vld2_u8(ptr); 它的作用是：从内存中加载交错的 uint8 数据，分别放入两个 8 字节(8个)的 NEON 向量中。
            // 
            uint8x8x2_t v2du8_src_y_c = neon::vload2(src_y_c + x);
            uint8x8x2_t v2du8_src_y_n = neon::vload2(src_y_n + x);
            /// 
            uint8x8x2_t v2du8_src_uv  = neon::vload2(src_uv_c + x);

            int32x4x2_t  v2qs32_ruv, v2qs32_guv, v2qs32_buv;
            uint8x8x2_t  v2du8_r_c, v2du8_g_c, v2du8_b_c;
            uint8x8x2_t  v2du8_r_n, v2du8_g_n, v2du8_b_n;
            uint8x16x3_t v3qu8_dst_c, v3qu8_dst_n;
            /// 因为我们加载的v2du8_src_uv一个二维向量，我们可以根据
            Uv2Rgbuv<MODE>(v2du8_src_uv.val[uidx], v2du8_src_uv.val[vidx], v2qs32_ruv, v2qs32_guv, v2qs32_buv);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_c.val[0], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_c.val[0], v2du8_g_c.val[0], v2du8_b_c.val[0]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_c.val[1], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_c.val[1], v2du8_g_c.val[1], v2du8_b_c.val[1]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_n.val[0], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_n.val[0], v2du8_g_n.val[0], v2du8_b_n.val[0]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_n.val[1], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_n.val[1], v2du8_g_n.val[1], v2du8_b_n.val[1]);

            v2du8_r_c = neon::vzip(v2du8_r_c.val[0], v2du8_r_c.val[1]);
            v2du8_g_c = neon::vzip(v2du8_g_c.val[0], v2du8_g_c.val[1]);
            v2du8_b_c = neon::vzip(v2du8_b_c.val[0], v2du8_b_c.val[1]);

            v3qu8_dst_c.val[0] = neon::vcombine(v2du8_r_c.val[0], v2du8_r_c.val[1]);
            v3qu8_dst_c.val[1] = neon::vcombine(v2du8_g_c.val[0], v2du8_g_c.val[1]);
            v3qu8_dst_c.val[2] = neon::vcombine(v2du8_b_c.val[0], v2du8_b_c.val[1]);

            v2du8_r_n = neon::vzip(v2du8_r_n.val[0], v2du8_r_n.val[1]);
            v2du8_g_n = neon::vzip(v2du8_g_n.val[0], v2du8_g_n.val[1]);
            v2du8_b_n = neon::vzip(v2du8_b_n.val[0], v2du8_b_n.val[1]);

            v3qu8_dst_n.val[0] = neon::vcombine(v2du8_r_n.val[0], v2du8_r_n.val[1]);
            v3qu8_dst_n.val[1] = neon::vcombine(v2du8_g_n.val[0], v2du8_g_n.val[1]);
            v3qu8_dst_n.val[2] = neon::vcombine(v2du8_b_n.val[0], v2du8_b_n.val[1]);

            neon::vstore(dst_c + 3 * x, v3qu8_dst_c);
            neon::vstore(dst_n + 3 * x, v3qu8_dst_n);
        }

        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtNv2RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_uv.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_uv.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_uv.GetSizes().m_channel != 2 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    DT_S32 height = src_uv.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtNv2RgbNeonImpl<0>, std::cref(src_y), std::cref(src_uv), std::ref(dst), swapuv);
            break;
        }

        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtNv2RgbNeonImpl<1>, std::cref(src_y), std::cref(src_uv), std::ref(dst), swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
static Status CvtY4202RgbNeonImpl(const Mat &imat0, const Mat &imat1, const Mat &imat2, Mat &omat, DT_BOOL swapuv, DT_S32 start_row, DT_S32 end_row)
{
    /// 
    const DT_S32 width       = omat.GetSizes().m_width;
    const DT_S32 width_align = width & (-16);

    for (DT_S32 y = start_row; y < end_row; y += 1)
    {
        const DT_U8 *src_y_c = imat0.Ptr<DT_U8>(2 * y);
        const DT_U8 *src_y_n = imat0.Ptr<DT_U8>(2 * y + 1);
        const DT_U8 *src_u_c = swapuv ? imat2.Ptr<DT_U8>(y) : imat1.Ptr<DT_U8>(y);
        const DT_U8 *src_v_c = swapuv ? imat1.Ptr<DT_U8>(y) : imat2.Ptr<DT_U8>(y);

        DT_U8 *dst_c = omat.Ptr<DT_U8>(2 * y);
        DT_U8 *dst_n = omat.Ptr<DT_U8>(2 * y + 1);

        DT_S32 x = 0;
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            uint8x8x2_t v2du8_src_y_c = neon::vload2(src_y_c + x);
            uint8x8x2_t v2du8_src_y_n = neon::vload2(src_y_n + x);
            uint8x8_t   vdu8_src_u    = neon::vload1(src_u_c + (x >> 1));
            uint8x8_t   vdu8_src_v    = neon::vload1(src_v_c + (x >> 1));

            int32x4x2_t  v2qs32_ruv, v2qs32_guv, v2qs32_buv;
            uint8x8x2_t  v2du8_r_c, v2du8_g_c, v2du8_b_c;
            uint8x8x2_t  v2du8_r_n, v2du8_g_n, v2du8_b_n;
            uint8x16x3_t v3qu8_dst_c, v3qu8_dst_n;

            Uv2Rgbuv<MODE>(vdu8_src_u, vdu8_src_v, v2qs32_ruv, v2qs32_guv, v2qs32_buv);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_c.val[0], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_c.val[0], v2du8_g_c.val[0], v2du8_b_c.val[0]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_c.val[1], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_c.val[1], v2du8_g_c.val[1], v2du8_b_c.val[1]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_n.val[0], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_n.val[0], v2du8_g_n.val[0], v2du8_b_n.val[0]);
            Yrgbuv2Rgb<MODE>(v2du8_src_y_n.val[1], v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r_n.val[1], v2du8_g_n.val[1], v2du8_b_n.val[1]);

            v2du8_r_c = neon::vzip(v2du8_r_c.val[0], v2du8_r_c.val[1]);
            v2du8_g_c = neon::vzip(v2du8_g_c.val[0], v2du8_g_c.val[1]);
            v2du8_b_c = neon::vzip(v2du8_b_c.val[0], v2du8_b_c.val[1]);

            v3qu8_dst_c.val[0] = neon::vcombine(v2du8_r_c.val[0], v2du8_r_c.val[1]);
            v3qu8_dst_c.val[1] = neon::vcombine(v2du8_g_c.val[0], v2du8_g_c.val[1]);
            v3qu8_dst_c.val[2] = neon::vcombine(v2du8_b_c.val[0], v2du8_b_c.val[1]);

            v2du8_r_n = neon::vzip(v2du8_r_n.val[0], v2du8_r_n.val[1]);
            v2du8_g_n = neon::vzip(v2du8_g_n.val[0], v2du8_g_n.val[1]);
            v2du8_b_n = neon::vzip(v2du8_b_n.val[0], v2du8_b_n.val[1]);

            v3qu8_dst_n.val[0] = neon::vcombine(v2du8_r_n.val[0], v2du8_r_n.val[1]);
            v3qu8_dst_n.val[1] = neon::vcombine(v2du8_g_n.val[0], v2du8_g_n.val[1]);
            v3qu8_dst_n.val[2] = neon::vcombine(v2du8_b_n.val[0], v2du8_b_n.val[1]);

            neon::vstore(dst_c + 3 * x, v3qu8_dst_c);
            neon::vstore(dst_n + 3 * x, v3qu8_dst_n);
        }

        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4202RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != src_v.GetSizes().m_width || src_u.GetSizes().m_height != src_v.GetSizes().m_height ||
        src_u.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_u.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    DT_S32 height = src_u.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4202RgbNeonImpl<0>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst), swapuv);
            break;
        }

        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4202RgbNeonImpl<1>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst), swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
static Status CvtY4222RgbNeonImpl(const Mat &src, Mat &dst, DT_BOOL swapuv, DT_BOOL swapy, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width       = dst.GetSizes().m_width;
    const DT_S32 width_align = width & (-16);

    const DT_S32 uidx = 1 - swapy + swapuv * 2;
    const DT_S32 vidx = (2 + uidx) % 4;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_c = src.Ptr<DT_U8>(y);
        DT_U8       *dst_c = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 16)
        {
LOOP_BODY:
            uint8x8x4_t v4du8_src = neon::vload4(src_c + 2 * x);
            uint8x8_t   vdu8_u    = v4du8_src.val[uidx];
            uint8x8_t   vdu8_v    = v4du8_src.val[vidx];
            uint8x8_t   vdu8_y0   = v4du8_src.val[swapy];
            uint8x8_t   vdu8_y1   = v4du8_src.val[swapy + 2];

            int32x4x2_t  v2qs32_ruv, v2qs32_guv, v2qs32_buv;
            uint8x8x2_t  v2du8_r, v2du8_g, v2du8_b;
            uint8x16x3_t v3qu8_dst;

            Uv2Rgbuv<MODE>(vdu8_u, vdu8_v, v2qs32_ruv, v2qs32_guv, v2qs32_buv);
            Yrgbuv2Rgb<MODE>(vdu8_y0, v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r.val[0], v2du8_g.val[0], v2du8_b.val[0]);
            Yrgbuv2Rgb<MODE>(vdu8_y1, v2qs32_ruv, v2qs32_guv, v2qs32_buv, v2du8_r.val[1], v2du8_g.val[1], v2du8_b.val[1]);

            v2du8_r          = neon::vzip(v2du8_r.val[0], v2du8_r.val[1]);
            v2du8_g          = neon::vzip(v2du8_g.val[0], v2du8_g.val[1]);
            v2du8_b          = neon::vzip(v2du8_b.val[0], v2du8_b.val[1]);
            v3qu8_dst.val[0] = neon::vcombine(v2du8_r.val[0], v2du8_r.val[1]);
            v3qu8_dst.val[1] = neon::vcombine(v2du8_g.val[0], v2du8_g.val[1]);
            v3qu8_dst.val[2] = neon::vcombine(v2du8_b.val[0], v2du8_b.val[1]);

            neon::vstore(dst_c + 3 * x, v3qu8_dst);
        }

        if (x < width)
        {
            x = width - 16;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4222RgbNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapuv, DT_BOOL swapy, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (src.GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src width only support even");
        return ret;
    }

    if ((src.GetSizes().m_width != dst.GetSizes().m_width) || (src.GetSizes().m_height != dst.GetSizes().m_height) ||
        (src.GetSizes().m_channel != 2) || (dst.GetSizes().m_channel != 3))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    DT_S32 height = src.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4222RgbNeonImpl<0>, std::cref(src), std::ref(dst), swapuv, swapy);
            break;
        }

        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4222RgbNeonImpl<1>, std::cref(src), std::ref(dst), swapuv, swapy);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <DT_U32 MODE>
static Status CvtY4442RgbNeonImpl(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width       = dst.GetSizes().m_width;
    const DT_S32 width_align = width & (-8);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_y_c = src_y.Ptr<DT_U8>(y);
        const DT_U8 *src_u_c = src_u.Ptr<DT_U8>(y);
        const DT_U8 *src_v_c = src_v.Ptr<DT_U8>(y);

        DT_U8 *dst_c = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 8)
        {
LOOP_BODY:
            uint8x8_t   vdu8_src_y = neon::vload1(src_y_c + x);
            uint8x8_t   vdu8_src_u = neon::vload1(src_u_c + x);
            uint8x8_t   vdu8_src_v = neon::vload1(src_v_c + x);
            int32x4x2_t v2qs32_ruv, v2qs32_guv, v2qs32_buv;
            uint8x8x3_t v3du8_dst;

            Uv2Rgbuv<MODE>(vdu8_src_u, vdu8_src_v, v2qs32_ruv, v2qs32_guv, v2qs32_buv);
            Yrgbuv2Rgb<MODE>(vdu8_src_y, v2qs32_ruv, v2qs32_guv, v2qs32_buv, v3du8_dst.val[0], v3du8_dst.val[1], v3du8_dst.val[2]);

            neon::vstore(dst_c + 3 * x, v3du8_dst);
        }

        if (x < width)
        {
            x = width - 8;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4442RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != dst.GetSizes().m_width || src_u.GetSizes().m_height != dst.GetSizes().m_height ||
        src_v.GetSizes().m_width != dst.GetSizes().m_width || src_v.GetSizes().m_height != dst.GetSizes().m_height ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    DT_S32 height = dst.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y444:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4442RgbNeonImpl<0>, std::cref(src_y), std::cref(src_u), std::cref(src_v), std::ref(dst));
            break;
        }

        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtY4442RgbNeonImpl<1>, std::cref(src_y), std::cref(src_u), std::cref(src_v), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
