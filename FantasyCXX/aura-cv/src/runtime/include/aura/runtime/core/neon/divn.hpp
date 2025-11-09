#ifndef AURA_RUNTIME_CORE_NEON_DIVN_HPP__
#define AURA_RUNTIME_CORE_NEON_DIVN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{
/************************************ u8 vdvin ************************************/
template <DT_U8 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_171 = vdup_n_u8(171);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_171);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint8x8_t  vdu8_result = vshr_n_u8(vdu8_shr, 1);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_205 = vdup_n_u8(205);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_205);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint8x8_t  vdu8_result = vshr_n_u8(vdu8_shr, 2);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_37 = vdup_n_u8(37);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_37);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint16x8_t vqu16_add   = vaddl_u8(vdu8_shr, vdu8_src);
    uint8x8_t  vdu8_result = vshrn_n_u16(vqu16_add, 3);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_57 = vdup_n_u8(57);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_57);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint8x8_t  vdu8_result = vshr_n_u8(vdu8_shr, 1);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_41 = vdup_n_u8(41);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_41);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint8x8_t  vdu8_result = vshr_n_u8(vdu8_shr, 2);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    const static uint8x8_t vdu8_79 = vdup_n_u8(79);

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_79);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint16x8_t vqu8_add    = vaddl_u8(vdu8_shr, vdu8_src);
    uint8x8_t  vdu8_result = vshrn_n_u16(vqu8_add, 6);

    return vdu8_result;
}

template <DT_U8 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                               (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline uint8x8_t vdiv_n(const uint8x8_t &vdu8_src)
{
    constexpr DT_U8 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U8 MUL    = (0 == DENOM) ? 0 : ((DT_U16)256 * ((1 << LENGTH) - DENOM)) / DENOM + 1;

    const static uint8x8_t vdu8_mul    = vdup_n_u8(MUL);
    const static int8x8_t  vds8_shift0 = vdup_n_s8((DT_S8)(-(Min<DT_U8>(LENGTH, (DT_U8)1))));
    const static int8x8_t  vds8_shift1 = vdup_n_s8((DT_S8)(-(Max<DT_S8>((DT_S8)LENGTH - 1, (DT_S8)0))));

    uint16x8_t vqu16_mul   = vmull_u8(vdu8_src, vdu8_mul);
    uint8x8_t  vdu8_shr    = vshrn_n_u16(vqu16_mul, 8);
    uint8x8_t  vdu8_sls    = vshl_u8(vsub_u8(vdu8_src, vdu8_shr), vds8_shift0);
    uint8x8_t  vdu8_result = vshl_u8(vadd_u8(vdu8_sls, vdu8_shr), vds8_shift1);

    return vdu8_result;
}

template <DT_U8 DENOM>
inline uint8x16_t vdiv_n(const uint8x16_t &vqu8_src)
{
    uint8x8_t vdu8_result_lo = vdiv_n<DENOM>(vget_low_u8(vqu8_src));
    uint8x8_t vdu8_result_hi = vdiv_n<DENOM>(vget_high_u8(vqu8_src));

    return vcombine(vdu8_result_lo, vdu8_result_hi);
}

/************************************ s8 vdvin ************************************/
template <DT_S8 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_511 = vdupq_n_s16(-511);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 171);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_511);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-9)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_1023 = vdupq_n_s16(-1023);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 205);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_1023);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-10)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_1023 = vdupq_n_s16(-1023);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 147);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_1023);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-10)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_511 = vdupq_n_s16(-511);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 57);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_511);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-9)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_1023 = vdupq_n_s16(-1023);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 41);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_1023);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-10)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    const static int16x8_t vqs16_2047 = vdupq_n_s16(-2047);

    int16x8_t vqs16_mul   = vmulq_n_s16(vmovl_s8(vds8_src), 42);
    int16x8_t vqs16_flag  = vcgtq_s16(vdupq_n_s16(0), vqs16_mul);
    int16x8_t vqs16_mla   = vmlaq_s16(vqs16_mul, vqs16_flag, vqs16_2047);
    int8x8_t  vds8_result = vqmovn_s16(vshlq_s16(vqs16_mla, vdupq_n_s16(-11)));

    return vds8_result;
}

template <DT_S8 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                               (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline int8x8_t vdiv_n(const int8x8_t &vds8_src)
{
    constexpr DT_U8  ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S8  BITS   = static_cast<DT_S8>(32 - __builtin_clz(ABS_D - 1));
    constexpr DT_S8  LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S16 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U16(1)) << (8 + LENGTH - 1)) / ABS_D - ((DT_U16)256);

    const static int8x8_t vds8_mul   = vdup_n_s8(MUL);
    const static int8x8_t vds8_sign  = vdup_n_s8(DENOM >> 7);
    const static int8x8_t vds8_shift = vdup_n_s8(-(LENGTH - 1));

    int16x8_t vqs16_mul   = vmull_s8(vds8_src, vds8_mul);
    int8x8_t  vds8_shr    = vshrn_n_s16(vqs16_mul, 8);
    int8x8_t  vds8_add    = vadd_s8(vds8_src, vds8_shr);
    int8x8_t  vds8_sub    = vsub_s8(vshl_s8(vds8_add, vds8_shift), vshl_s8(vds8_src, vdup_n_s8(-7)));
    int8x8_t  vds8_result = vsub_s8(veor_s8(vds8_sub, vds8_sign), vds8_sign);

    return vds8_result;
}

template <DT_S8 DENOM>
inline int8x16_t vdiv_n(const int8x16_t &vqs8_src)
{
    int8x8_t vds8_result_lo = vdiv_n<DENOM>(vget_low_s8(vqs8_src));
    int8x8_t vds8_result_hi = vdiv_n<DENOM>(vget_high_s8(vqs8_src));

    return vcombine(vds8_result_lo, vds8_result_hi);
}

/************************************ u16 vdvin ************************************/
template <DT_U16 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_43691 = vdup_n_u16(43691);

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_43691);
    uint16x4_t vdu16_h      = vshrn_n_u32(vqu32_mul, 16);
    uint16x4_t vdu16_result = vshr_n_u16(vdu16_h, 1);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_43691 = vdup_n_u16(43691);

    uint32x4_t vqu32_mul_lo = vmull_u16(vget_low_u16(vqu16_src),  vdu16_43691);
    uint32x4_t vqu32_mul_hi = vmull_u16(vget_high_u16(vqu16_src), vdu16_43691);
    uint16x8_t vqu16_shrn   = vcombine_u16(vshrn_n_u32(vqu32_mul_lo, 16), vshrn_n_u32(vqu32_mul_hi, 16));
    uint8x8_t  vdu8_result  = vqshrn_n_u16(vqu16_shrn, 1);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_52429 = vdup_n_u16(52429);

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_52429);
    uint16x4_t vdu16_h      = vshrn_n_u32(vqu32_mul, 16);
    uint16x4_t vdu16_result = vshr_n_u16(vdu16_h, 2);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_52429 = vdup_n_u16(52429);

    uint32x4_t vqu32_mul_lo = vmull_u16(vget_low_u16(vqu16_src), vdu16_52429);
    uint32x4_t vqu32_mul_hi = vmull_u16(vget_high_u16(vqu16_src), vdu16_52429);
    uint16x8_t vqu16_shrn   = vcombine_u16(vshrn_n_u32(vqu32_mul_lo, 16), vshrn_n_u32(vqu32_mul_hi, 16));
    uint8x8_t  vdu8_result  = vqshrn_n_u16(vqu16_shrn, 2);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_9363 = vdup_n_u16(9363);

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_9363);
    uint16x4_t vdu16_shr    = vshrn_n_u32(vqu32_mul, 16);
    uint32x4_t vqu32_add    = vaddl_u16(vdu16_shr, vdu16_src);
    uint16x4_t vdu16_result = vshrn_n_u32(vqu32_add, 3);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_9363 = vdup_n_u16(9363);

    uint16x4_t vdu16_src_lo = vget_low_u16(vqu16_src);
    uint16x4_t vdu16_src_hi = vget_high_u16(vqu16_src);

    uint32x4_t vqu32_mul_lo = vmull_u16(vdu16_src_lo, vdu16_9363);
    uint32x4_t vqu32_mul_hi = vmull_u16(vdu16_src_hi, vdu16_9363);

    uint32x4_t vqu32_sra_lo = vaddl_u16(vshrn_n_u32(vqu32_mul_lo, 16), vdu16_src_lo);
    uint32x4_t vqu32_sra_hi = vaddl_u16(vshrn_n_u32(vqu32_mul_hi, 16), vdu16_src_hi);

    uint16x8_t vqu16_result = vcombine_u16(vqshrn_n_u32(vqu32_sra_lo, 3),
                                           vqshrn_n_u32(vqu32_sra_hi, 3));

    uint8x8_t vdu8_result   = vqmovn_u16(vqu16_result);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_58255 = vdup_n_u16(58255);

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_58255);
    uint16x4_t vdu16_shr    = vshrn_n_u32(vqu32_mul, 16);
    uint16x4_t vdu16_result = vshr_n_u16(vdu16_shr, 3);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_58255 = vdup_n_u16(58255);

    uint32x4_t vqu32_mul_lo = vmull_u16(vget_low_u16(vqu16_src), vdu16_58255);
    uint32x4_t vqu32_mul_hi = vmull_u16(vget_high_u16(vqu16_src), vdu16_58255);

    uint16x8_t vqu16_shrn   = vcombine_u16(vshrn_n_u32(vqu32_mul_lo, 16), vshrn_n_u32(vqu32_mul_hi, 16));
    uint8x8_t  vdu8_result  = vqshrn_n_u16(vqu16_shrn, 3);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_18351 = vdup_n_u16(18351);

    uint16x4_t vdu16_shrn   = vshrn_n_u32(vmull_u16(vdu16_src,  vdu16_18351), 16);
    uint16x4_t vdu16_result = vshrn_n_u32(vaddl_u16(vdu16_shrn, vdu16_src),   5);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_18351 = vdup_n_u16(18351);

    uint16x4_t vqu32_mul_lo = vshrn_n_u32(vmull_u16(vget_low_u16(vqu16_src),  vdu16_18351), 16);
    uint16x4_t vqu32_mul_hi = vshrn_n_u32(vmull_u16(vget_high_u16(vqu16_src), vdu16_18351), 16);
    uint32x4_t vqu32_add_lo = vaddl_u16(vqu32_mul_lo, vget_low_u16(vqu16_src));
    uint32x4_t vqu32_add_hi = vaddl_u16(vqu32_mul_hi, vget_high_u16(vqu16_src));
    uint16x8_t vqu16_result = vcombine_u16(vshrn_n_u32(vqu32_add_lo, 5), vshrn_n_u32(vqu32_add_hi, 5));
    uint8x8_t vdu8_result   = vqmovn_u16(vqu16_result);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    const static uint16x4_t vdu16_20063 = vdup_n_u16(20063);

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_20063);
    uint16x4_t vdu16_shr    = vshrn_n_u32(vqu32_mul, 16);
    uint32x4_t vqu32_add    = vaddl_u16(vdu16_shr, vdu16_src);
    uint16x4_t vdu16_result = vshrn_n_u32(vqu32_add, 6);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    const static uint16x4_t vdu16_20063 = vdup_n_u16(20063);

    uint16x4_t vqu32_mul_lo = vshrn_n_u32(vmull_u16(vget_low_u16(vqu16_src),  vdu16_20063), 16);
    uint16x4_t vqu32_mul_hi = vshrn_n_u32(vmull_u16(vget_high_u16(vqu16_src), vdu16_20063), 16);
    uint32x4_t vqu32_add_lo = vaddl_u16(vqu32_mul_lo, vget_low_u16(vqu16_src));
    uint32x4_t vqu32_add_hi = vaddl_u16(vqu32_mul_hi, vget_high_u16(vqu16_src));
    uint16x8_t vqu16_result = vcombine_u16(vshrn_n_u32(vqu32_add_lo, 6), vshrn_n_u32(vqu32_add_hi, 6));
    uint8x8_t vdu8_result   = vqmovn_u16(vqu16_result);

    return vdu8_result;
}

template <DT_U16 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline uint16x4_t vdiv_n(const uint16x4_t &vdu16_src)
{
    constexpr DT_U16 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U16 MUL    = (0 == DENOM) ? 0 : ((65536) * ((1 << LENGTH) - DENOM)) / DENOM + 1;

    const static uint16x4_t vdu16_mul    = vdup_n_u16(MUL);
    const static int16x4_t  vds16_shift0 = vdup_n_s16((DT_S16)(-(Min<DT_U16>(LENGTH, (DT_U16)1))));
    const static int16x4_t  vds16_shift1 = vdup_n_s16((DT_S16)(-(Max<DT_S16>((DT_S16)LENGTH - 1, (DT_S16)0))));

    uint32x4_t vqu32_mul    = vmull_u16(vdu16_src, vdu16_mul);
    uint16x4_t vdu16_h      = vshrn_n_u32(vqu32_mul, 16);
    uint16x4_t vdu16_sub_t  = vsub_u16(vdu16_src, vdu16_h);
               vdu16_sub_t  = vshl_u16(vdu16_sub_t,  vds16_shift0);
    uint16x4_t vdu16_result = vadd_u16(vdu16_sub_t,  vdu16_h);
               vdu16_result = vshl_u16(vdu16_result, vds16_shift1);

    return vdu16_result;
}

template <DT_U16 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline uint8x8_t vqdivn_n(const uint16x8_t &vqu16_src)
{
    constexpr DT_U16 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U16 MUL    = (0 == DENOM) ? 0 : ((65536) * ((1 << LENGTH) - DENOM)) / DENOM + 1;

    const static uint16x4_t vdu16_mul    = vdup_n_u16(MUL);
    const static int16x8_t  vqs16_shift0 = vdupq_n_s16((DT_S16)(-(Min<DT_U16>(LENGTH, (DT_U16)1))));
    const static int16x8_t  vqs16_shift1 = vdupq_n_s16((DT_S16)(-(Max<DT_S16>((DT_S16)LENGTH - 1, (DT_S16)0))));

    uint32x4_t vqu32_mul_lo = vmull_u16(vget_low_u16(vqu16_src), vdu16_mul);
    uint32x4_t vqu32_mul_hi = vmull_u16(vget_high_u16(vqu16_src), vdu16_mul);

    uint16x8_t vqu16_shr    = vcombine_u16(vshrn_n_u32(vqu32_mul_lo, 16), vshrn_n_u32(vqu32_mul_hi, 16));
    uint16x8_t vqu16_sls    = vshlq_u16(vsubq_u16(vqu16_src, vqu16_shr), vqs16_shift0);
    uint16x8_t vqu16_result = vshlq_u16(vaddq_u16(vqu16_sls, vqu16_shr), vqs16_shift1);

    uint8x8_t  vdu8_result  = vqmovn_u16(vqu16_result);

    return vdu8_result;
}

template <DT_U16 DENOM>
inline uint16x8_t vdiv_n(const uint16x8_t &vqu16_src)
{
    uint16x4_t vdu16_result_lo = vdiv_n<DENOM>(vget_low_u16(vqu16_src));
    uint16x4_t vdu16_result_hi = vdiv_n<DENOM>(vget_high_u16(vqu16_src));

    return vcombine(vdu16_result_lo, vdu16_result_hi);
}
/************************************ s16 vdvin ************************************/

template <DT_S16 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul    = vmulq_n_s32(vmovl_s16(vds16_src), 43691);
    uint32x4_t vqu32_flag   = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla    = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_131071);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-17)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul_lo  = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)),  43691);
    int32x4_t  vqs32_mul_hi  = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 43691);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_131071);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_131071);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 1);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_262143 = vdupq_n_s32(-262143);

    int32x4_t  vqs32_mul    = vmulq_n_s32(vmovl_s16(vds16_src), 52429);
    uint32x4_t vqu32_flag   = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla    = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_262143);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-18)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_262143 = vdupq_n_s32(-262143);

    int32x4_t  vqs32_mul_lo  = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)),  52429);
    int32x4_t  vqs32_mul_hi  = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 52429);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_262143);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_262143);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 2);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul    = vmulq_n_s32(vmovl_s16(vds16_src), 18725);
    uint32x4_t vqu32_flag   = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla    = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_131071);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-17)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul_lo  = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)),  18725);
    int32x4_t  vqs32_mul_hi  = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 18725);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_131071);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_131071);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 1);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_524287 = vdupq_n_s32(-524287);

    int32x4_t  vqs32_mul    = vmulq_n_s32(vmovl_s16(vds16_src), 58255);
    uint32x4_t vqu32_flag   = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla    = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_524287);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-19)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_524287 = vdupq_n_s32(-524287);

    int32x4_t  vqs32_mul_lo  = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)),  58255);
    int32x4_t  vqs32_mul_hi  = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 58255);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_524287);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_524287);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 3);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_524287 = vdupq_n_s32(-524287);

    int32x4_t  vqs32_mul    = vmulq_n_s32(vmovl_s16(vds16_src), 20972);
    uint32x4_t vqu32_flag   = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla    = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_524287);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-19)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_524287 = vdupq_n_s32(-524287);

    int32x4_t  vqs32_mul_lo  = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)), 20972);
    int32x4_t  vqs32_mul_hi  = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 20972);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_524287);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_524287);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 3);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul   = vmulq_n_s32(vmovl_s16(vds16_src), 2675);
    uint32x4_t vqu32_flag  = vcgtq_s32(vdupq_n_s32(0), vqs32_mul);
    int32x4_t  vqs32_mla   = vmlaq_s32(vqs32_mul, vqu32_flag, vqs32_131071);
    int16x4_t  vds16_result = vqmovn_s32(vshlq_s32(vqs32_mla, vdupq_n_s32(-17)));

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    const static int32x4_t vqs32_131071 = vdupq_n_s32(-131071);

    int32x4_t  vqs32_mul_lo = vmulq_n_s32(vmovl_s16(vget_low_s16(vqs16_src)), 2675);
    int32x4_t  vqs32_mul_hi = vmulq_n_s32(vmovl_s16(vget_high_s16(vqs16_src)), 2675);

    uint32x4_t vqu32_flag_lo = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_lo);
    uint32x4_t vqu32_flag_hi = vcgtq_s32(vdupq_n_s32(0), vqs32_mul_hi);

    int32x4_t  vqs32_mla_lo  = vmlaq_s32(vqs32_mul_lo, vqu32_flag_lo, vqs32_131071);
    int32x4_t  vqs32_mla_hi  = vmlaq_s32(vqs32_mul_hi, vqu32_flag_hi, vqs32_131071);

    int16x8_t  vqs16_result  = vcombine_s16(vshrn_n_s32(vqs32_mla_lo, 16), vshrn_n_s32(vqs32_mla_hi, 16));
    int8x8_t   vds8_result   = vqshrn_n_s16(vqs16_result, 1);

    return vds8_result;
}

template <DT_S16 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline int16x4_t vdiv_n(const int16x4_t &vds16_src)
{
    constexpr DT_U16 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S16 BITS   = static_cast<DT_S16>(32 - __builtin_clz(ABS_D - 1));
    constexpr DT_S16 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S32 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U32(1)) << (16 + LENGTH - 1)) / ABS_D - (65536);

    const static int16x4_t vds16_mul   = vdup_n_s16(MUL);
    const static int16x4_t vds16_sign  = vdup_n_s16(DENOM >> 15);
    const static int16x4_t vds16_shift = vdup_n_s16(-(LENGTH - 1));

    int32x4_t vqs32_mul   = vmull_s16(vds16_src, vds16_mul);
    int16x4_t vds16_shr    = vshrn_n_s32(vqs32_mul, 16);
    int16x4_t vds16_add    = vqadd_s16(vds16_src, vds16_shr);
    int16x4_t vds16_sub    = vsub_s16(vshl_s16(vds16_add, vds16_shift), vshl_s16(vds16_src, vdup_n_s16(-15)));
    int16x4_t vds16_result = vsub_s16(veor_s16(vds16_sub, vds16_sign), vds16_sign);

    return vds16_result;
}

template <DT_S16 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline int8x8_t vqdivn_n(const int16x8_t &vqs16_src)
{
    constexpr DT_U16 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S16 BITS   = static_cast<DT_S16>(32 - __builtin_clz(ABS_D - 1));
    constexpr DT_S16 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S32 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U32(1)) << (16 + LENGTH - 1)) / ABS_D - (65536);

    const static int16x4_t vds16_mul   = vdup_n_s16(MUL);
    const static int16x8_t vqs16_sign  = vdupq_n_s16(DENOM >> 15);
    const static int16x8_t vqs16_shift = vdupq_n_s16(-(LENGTH - 1));

    int32x4_t vqs32_mul_lo = vmull_s16(vget_low_s16(vqs16_src),  vds16_mul);
    int32x4_t vqs32_mul_hi = vmull_s16(vget_high_s16(vqs16_src), vds16_mul);

    int16x4_t vds16_shr_lo = vshrn_n_s32(vqs32_mul_lo, 16);
    int16x4_t vds16_shr_hi = vshrn_n_s32(vqs32_mul_hi, 16);

    int16x8_t vqs16_add    = vqaddq_s16(vqs16_src, vcombine_s16(vds16_shr_lo, vds16_shr_hi));
    int16x8_t vqs16_sub    = vqsubq_s16(vshlq_s16(vqs16_add, vqs16_shift), vshlq_s16(vqs16_src, vdupq_n_s16(-15)));
    int16x8_t vqs16_result = vqsubq_s16(veorq_s16(vqs16_sub, vqs16_sign), vqs16_sign);

    int8x8_t vds8_result   = vqmovn_s16(vqs16_result);

    return vds8_result;
}

template <DT_S16 DENOM>
inline int16x8_t vdiv_n(const int16x8_t &vqs16_src)
{
    int16x4_t vds16_result_lo = vdiv_n<DENOM>(vget_low_s16(vqs16_src));
    int16x4_t vds16_result_hi = vdiv_n<DENOM>(vget_high_s16(vqs16_src));

    return vcombine_s16(vds16_result_lo, vds16_result_hi);
}

/************************************ u32 vdvin ************************************/
template <DT_U32 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 2863311531), 32);
    uint32x2_t vdu32_result = vshr_n_u32(vdu32_shr, 1);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 3>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_n_u32(vget_low_u16(vqu32_src),  2863311531), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_n_u32(vget_high_u16(vqu32_src), 2863311531), 32);

    uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);
    uint16x4_t vdu16_result = vqshrn_n_u32(vqu32_shr, 1);

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 3435973837), 32);
    uint32x2_t vdu32_result = vshr_n_u32(vdu32_shr, 2);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 5>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_n_u32(vget_low_u16(vqu32_src),  3435973837), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_n_u32(vget_high_u16(vqu32_src), 3435973837), 32);

    uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);
    uint16x4_t vdu16_result = vqshrn_n_u32(vqu32_shr, 2);

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 613566757), 32);
    uint32x2_t vdu32_result = vshrn_n_u64(vaddl_u32(vdu32_shr, vdu32_src), 3);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 7>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vqu32_shrn_lo = vshrn_n_u64(vmull_n_u32(vget_low_u16(vqu32_src),  613566757), 32);
    uint32x2_t vqu32_shrn_hi = vshrn_n_u64(vmull_n_u32(vget_high_u16(vqu32_src), 613566757), 32);

    uint32x2_t vdu32_result_lo = vqshrn_n_u32(vaddl_u32(vqu32_shrn_lo, vget_low_u16(vqu32_src)),  3);
    uint32x2_t vdu32_result_hi = vqshrn_n_u32(vaddl_u32(vqu32_shrn_hi, vget_high_u16(vqu32_src)), 3);

    uint16x4_t vdu16_result = vqmovn_u32(vcombine_u32(vdu32_result_lo, vdu32_result_hi));

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 1908874354), 32);
    uint32x2_t vdu32_result = vshr_n_u32(vdu32_shr, 2);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 9>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_n_u32(vget_low_u32(vqu32_src),  1908874354), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_n_u32(vget_high_u32(vqu32_src), 1908874354), 32);

    uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);
    uint16x4_t vdu16_result = vqshrn_n_u32(vqu32_shr, 2);

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 1374389535), 32);
    uint32x2_t vdu32_result = vshr_n_u32(vdu32_shr, 3);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 25>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_n_u32(vget_low_u32(vqu32_src),  1374389535), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_n_u32(vget_high_u32(vqu32_src), 1374389535), 32);

    uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);
    uint16x4_t vdu16_result = vqshrn_n_u32(vqu32_shr, 3);

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_n_u32(vdu32_src, 1402438301), 32);
    uint32x2_t vdu32_result = vshr_n_u32(vdu32_shr, 4);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<DENOM == 49>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_n_u32(vget_low_u32(vqu32_src),  1402438301), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_n_u32(vget_high_u32(vqu32_src), 1402438301), 32);

    uint32x4_t vqu32_shrn   = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);
    uint16x4_t vdu16_result = vqshrn_n_u32(vqu32_shrn, 4);

    return vdu16_result;
}

template <DT_U32 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline uint32x2_t vdiv_n(const uint32x2_t &vdu32_src)
{
    constexpr DT_U32 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U32 MUL    = (0 == DENOM) ? 0 : (DT_U64)(4294967296) * (((DT_U64)1 << LENGTH) - (DT_U64)DENOM) / DENOM + 1;

    const static uint32x2_t vdu32_mul    = vdup_n_u32(MUL);
    const static int32x2_t  vds32_shift0 = vdup_n_s32(-(Min<DT_U32>(LENGTH, (DT_U32)1)));
    const static int32x2_t  vds32_shift1 = vdup_n_s32(-(Max<DT_S32>((DT_S32)LENGTH - 1, (DT_S32)0)));

    uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_u32(vdu32_src, vdu32_mul), 32);
    uint32x2_t vdu32_sls    = vshl_u32(vsub_u32(vdu32_src, vdu32_shr), vds32_shift0);
    uint32x2_t vdu32_result = vshl_u32(vadd_u32(vdu32_sls, vdu32_shr), vds32_shift1);

    return vdu32_result;
}

template <DT_U32 DENOM, typename std::enable_if<(DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                                (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline uint16x4_t vqdivn_n(const uint32x4_t &vqu32_src)
{
    constexpr DT_U32 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U32 MUL    = (0 == DENOM) ? 0 : (DT_U64)(4294967296) * (((DT_U64)1 << LENGTH) - (DT_U64)DENOM) / DENOM + 1;

    const static uint32x2_t vdu32_mul    = vdup_n_u32(MUL);
    const static int32x4_t  vqs32_shift0 = vdupq_n_s32(-(Min<DT_U32>(LENGTH, (DT_U32)1)));
    const static int32x4_t  vqs32_shift1 = vdupq_n_s32(-(Max<DT_S32>((DT_S32)LENGTH - 1, (DT_S32)0)));

    uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_u32(vget_low_u16(vqu32_src),  vdu32_mul), 32);
    uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_u32(vget_high_u16(vqu32_src), vdu32_mul), 32);
    uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);

    uint32x4_t vqu32_sls    = vshlq_u32(vsubq_u32(vqu32_src, vqu32_shr), vqs32_shift0);
    uint32x4_t vqu32_result = vshlq_u32(vaddq_u32(vqu32_sls, vqu32_shr), vqs32_shift1);
    uint16x4_t vdu16_result = vqmovn_u32(vqu32_result);

    return vdu16_result;
}

template <DT_U32 DENOM>
inline uint32x4_t vdiv_n(const uint32x4_t &vqu32_src)
{
    uint32x2_t vdu32_result_lo = vdiv_n<DENOM>(vget_low_u32(vqu32_src));
    uint32x2_t vdu32_result_hi = vdiv_n<DENOM>(vget_high_u32(vqu32_src));

    return vcombine(vdu32_result_lo, vdu32_result_hi);
}

/************************************ s32 vdvin ************************************/
template <DT_S32 DENOM>
inline int32x2_t vdiv_n(const int32x2_t &vds32_src)
{
    constexpr DT_U32 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S32 BITS   = 32 - __builtin_clz(ABS_D - 1);
    constexpr DT_S32 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S64 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U64(1)) << (32 + LENGTH - 1)) / ABS_D - (4294967296);

    const static int32x2_t vds32_mul   = vdup_n_s32(MUL);
    const static int32x2_t vds32_sign  = vdup_n_s32(DENOM >> 31);
    const static int32x2_t vds32_shift = vdup_n_s32(-(LENGTH - 1));

    int32x2_t vds32_shr = vshrn_n_s64(vmull_s32(vds32_src, vds32_mul), 32);
    int32x2_t vds32_sla = vshl_s32(vqadd_s32(vds32_src, vds32_shr), vds32_shift);

    int32x2_t vds32_sub    = vsub_s32(vds32_sla, vshl_s32(vds32_src, vdup_n_s32(-31)));
    int32x2_t vds32_result = vsub_s32(veor_s32(vds32_sub, vds32_sign), vds32_sign);

    return vds32_result;
}

template <DT_S32 DENOM>
inline int16x4_t vqdivn_n(const int32x4_t &vqs32_src)
{
    constexpr DT_U32 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S32 BITS   = 32 - __builtin_clz(ABS_D - 1);
    constexpr DT_S32 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S64 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U64(1)) << (32 + LENGTH - 1)) / ABS_D - (4294967296);

    const static int32x2_t vds32_mul    = vdup_n_s32(MUL);
    const static int32x4_t vqs32_sign  = vdupq_n_s32(DENOM >> 31);
    const static int32x4_t vqs32_shift = vdupq_n_s32(-(LENGTH - 1));

    int32x2_t vds32_shr_lo = vshrn_n_s64(vmull_s32(vget_low_u16(vqs32_src),  vds32_mul), 32);
    int32x2_t vds32_shr_hi = vshrn_n_s64(vmull_s32(vget_high_u16(vqs32_src), vds32_mul), 32);

    int32x4_t vqs32_shr = vcombine_s32(vds32_shr_lo, vds32_shr_hi);
    int32x4_t vqs32_sla = vshlq_s32(vqaddq_s32(vqs32_src, vqs32_shr), vqs32_shift);
    int32x4_t vqs32_sub = vsubq_s32(vqs32_sla, vshlq_s32(vqs32_src, vdupq_n_s32(-31)));

    int32x4_t vqs32_result = vsubq_s32(veorq_s32(vqs32_sub, vqs32_sign), vqs32_sign);
    int16x4_t vds16_result = vqmovn_s32(vqs32_result);

    return vds16_result;
}

template <DT_S32 DENOM>
inline int32x4_t vdiv_n(const int32x4_t &vqs32_src)
{
    int32x2_t vds32_result_lo = vdiv_n<DENOM>(vget_low_s32(vqs32_src));
    int32x2_t vds32_result_hi = vdiv_n<DENOM>(vget_high_s32(vqs32_src));

    return vcombine(vds32_result_lo, vds32_result_hi);
}

#if defined(__aarch64__)
/************************************ u64 vdvin ************************************/
template <DT_U64 DENOM>
inline uint64x1_t vdiv_n(const uint64x1_t &vdu64_src)
{
    DT_U64 u64_src[1], u64_result[1];
    vst1_u64(u64_src, vdu64_src);
    u64_result[0] = u64_src[0] / DENOM;

    return vld1_u64(u64_result);
}

template <DT_U64 DENOM>
inline uint64x2_t vdiv_n(const uint64x2_t &vqu64_src)
{
    DT_U64 u64_src[2], u64_result[2];
    vst1q_u64(u64_src, vqu64_src);

    u64_result[0] = u64_src[0] / DENOM;
    u64_result[1] = u64_src[1] / DENOM;

    return vld1q_u64(u64_result);
}

/************************************ s64 vdvin ************************************/
template <DT_S64 DENOM>
inline int64x1_t vdiv_n(const int64x1_t &vds64_src)
{
    DT_S64 s64_src[1], s64_result[1];
    vst1_s64(s64_src, vds64_src);
    s64_result[0] = s64_src[0] / DENOM;

    return vld1_s64(s64_result);
}

template <DT_S64 DENOM>
inline int64x2_t vdiv_n(const int64x2_t &vqs64_src)
{
    DT_S64 s64_src[2], s64_result[2];
    vst1q_s64(s64_src, vqs64_src);

    s64_result[0] = s64_src[0] / DENOM;
    s64_result[1] = s64_src[1] / DENOM;

    return vld1q_s64(s64_result);
}
#endif // __aarch64__

/************************************ common vdvin ************************************/
#define VQ_VDIVN_COMM(v_type, vq_type)                              \
    vq_type operator()(const vq_type &vq_src)                       \
    {                                                               \
        v_type v_result_lo = operator()(neon::vgetlow(vq_src));     \
        v_type v_result_hi = operator()(neon::vgethigh(vq_src));    \
        return neon::vcombine(v_result_lo, v_result_hi);            \
    }

template <typename Tp> class VdivNHelper{};
template <typename Tp> class VqdivnNHelper{};

template <>
class VdivNHelper<DT_U8>
{
public:
    explicit VdivNHelper(DT_U8 denom) : m_denom(denom)
    {
        DT_U8 length = 32 - __builtin_clz(m_denom - 1);

        m_vdu8_mul    = vdup_n_u8(((DT_U16)256 * ((1 << length) - m_denom)) / m_denom + 1);
        m_vds8_shift0 = vdup_n_s8((DT_S8)(-(Min<DT_U8>(length, (DT_U8)1))));
        m_vds8_shift1 = vdup_n_s8((DT_S8)(-(Max<DT_S8>((DT_S8)length - 1, (DT_S8)0))));
    }

    uint8x8_t operator()(const uint8x8_t &vdu8_src)
    {
        uint8x8_t  vdu8_shr    = vshrn_n_u16(vmull_u8(vdu8_src, m_vdu8_mul), 8);
        uint8x8_t  vdu8_sub    = vshl_u8(vsub_u8(vdu8_src, vdu8_shr), m_vds8_shift0);
        uint8x8_t  vdu8_result = vshl_u8(vadd_u8(vdu8_sub, vdu8_shr), m_vds8_shift1);

        return vdu8_result;
    }

    VQ_VDIVN_COMM(uint8x8_t, uint8x16_t);

private:
    DT_U8     m_denom;
    uint8x8_t m_vdu8_mul;
    int8x8_t  m_vds8_shift0;
    int8x8_t  m_vds8_shift1;
};

template <>
class VdivNHelper<DT_S8>
{
public:
    explicit VdivNHelper(DT_S8 denom) : m_denom(denom)
    {
        DT_U8 abs_d  = Abs(m_denom);
        DT_S8 length = 32 - __builtin_clz(abs_d - 1);
        length       = Max<DT_U8>(length, (DT_S8)1);
        DT_S16 mul   = 1 + ((DT_U16(1)) << (8 + length - 1)) / abs_d;
        mul          = mul - ((DT_U16)256);
        mul          = (0 == m_denom) ? 0 : mul;

        m_vds8_mul    = vdup_n_s8(mul);
        m_vds8_sign   = vdup_n_s8(m_denom >> 7);
        m_vds8_shift  = vdup_n_s8(-(length - 1));
    }

    int8x8_t operator()(const int8x8_t &vds8_src)
    {
        int8x8_t  vds8_shr    = vshrn_n_s16(vmull_s8(vds8_src, m_vds8_mul), 8);
        int8x8_t  vds8_sla    = vshl_s8(vadd_s8(vds8_src, vds8_shr), m_vds8_shift);
        int8x8_t  vds8_sub    = vsub_s8(vds8_sla, vshl_s8(vds8_src, vdup_n_s8(-7)));
        int8x8_t  vds8_result = vsub_s8(veor_s8(vds8_sub, m_vds8_sign), m_vds8_sign);

        return vds8_result;
    }

    VQ_VDIVN_COMM(int8x8_t, int8x16_t);

private:
    DT_S8    m_denom;
    int8x8_t m_vds8_mul;
    int8x8_t m_vds8_sign;
    int8x8_t m_vds8_shift;
};

template <>
class VdivNHelper<DT_U16>
{
public:
    explicit VdivNHelper(DT_U16 denom) : m_denom(denom)
    {
        DT_U16 length = 32 - __builtin_clz(m_denom - 1);

        m_vdu16_mul    = vdup_n_u16(((65536) * ((1 << length) - m_denom)) / m_denom + 1);
        m_vds16_shift0 = vdup_n_s16((DT_S16)(-(Min<DT_U16>(length, (DT_U16)1))));
        m_vds16_shift1 = vdup_n_s16((DT_S16)(-(Max<DT_S16>((DT_S16)length - 1, (DT_S16)0))));
    }

    uint16x4_t operator()(const uint16x4_t &vdu16_src)
    {
        uint16x4_t vdu16_shr    = vshrn_n_u32(vmull_u16(vdu16_src, m_vdu16_mul), 16);
        uint16x4_t vdu16_sls    = vshl_u16(vsub_u16(vdu16_src,  vdu16_shr), m_vds16_shift0);
        uint16x4_t vdu16_result = vshl_u16(vadd_u16(vdu16_sls,  vdu16_shr), m_vds16_shift1);

        return vdu16_result;
    }

    VQ_VDIVN_COMM(uint16x4_t, uint16x8_t);

private:
    DT_U16     m_denom;
    uint16x4_t m_vdu16_mul;
    int16x4_t  m_vds16_shift0;
    int16x4_t  m_vds16_shift1;
};

template <>
class VqdivnNHelper<DT_U16>
{
    public:
    explicit VqdivnNHelper(DT_U16 denom) : m_denom(denom)
    {
        DT_U16 length  = 32 - __builtin_clz(m_denom - 1);

        m_vdu16_mul     = vdup_n_u16(((65536) * ((1 << length) - m_denom)) / m_denom + 1);
        m_vqs16_shift0 = vdupq_n_s16((DT_S16)(-(Min<DT_U16>(length, (DT_U16)1))));
        m_vqs16_shift1 = vdupq_n_s16((DT_S16)(-(Max<DT_S16>((DT_S16)length - 1, (DT_S16)0))));
    }

    uint8x8_t operator()(const uint16x8_t &vqu16_src)
    {
        uint16x4_t vdu16_shr_lo  = vshrn_n_u32(vmull_u16(vget_low_u16(vqu16_src),  m_vdu16_mul), 16);
        uint16x4_t vdu16_shr_hi  = vshrn_n_u32(vmull_u16(vget_high_u16(vqu16_src), m_vdu16_mul), 16);
        uint16x8_t vqu16_shr     = vcombine_u16(vdu16_shr_lo, vdu16_shr_hi);

        uint16x8_t vqu16_sls    = vshlq_u16(vsubq_u16(vqu16_src, vqu16_shr), m_vqs16_shift0);
        uint16x8_t vqu16_result = vshlq_u16(vaddq_u16(vqu16_sls, vqu16_shr), m_vqs16_shift1);
        uint8x8_t  vdu8_result  = vqmovn_u16(vqu16_result);

        return vdu8_result;
    }

private:
    DT_U16     m_denom;
    uint16x4_t m_vdu16_mul;
    int16x8_t  m_vqs16_shift0;
    int16x8_t  m_vqs16_shift1;
};

template <>
class VdivNHelper<DT_S16>
{
public:
    explicit VdivNHelper(DT_S16 denom) : m_denom(denom)
    {
        DT_U16 abs_d  = Abs(m_denom);
        DT_S16 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U16>(length, (DT_S16)1);
        DT_S32 mul    = 1 + ((DT_U32(1)) << (16 + length - 1)) / abs_d;
               mul    = mul - (65536);
               mul    = (0 == m_denom) ? 0 : mul;

        m_vds16_mul   = vdup_n_s16(mul);
        m_vds16_sign  = vdup_n_s16(m_denom >> 15);
        m_vds16_shift = vdup_n_s16(-(length - 1));
    }

    int16x4_t operator()(const int16x4_t &vds16_src)
    {
        int16x4_t vds16_shr    = vshrn_n_s32(vmull_s16(vds16_src, m_vds16_mul), 16);
        int16x4_t vds16_sla    = vshl_s16(vadd_s16(vds16_src, vds16_shr), m_vds16_shift);
        int16x4_t vds16_sub    = vsub_s16(vds16_sla, vshl_s16(vds16_src, vdup_n_s16(-15)));
        int16x4_t vds16_result = vsub_s16(veor_s16(vds16_sub, m_vds16_sign), m_vds16_sign);

        return vds16_result;
    }

    VQ_VDIVN_COMM(int16x4_t, int16x8_t);

private:
    DT_S16    m_denom;
    int16x4_t m_vds16_mul;
    int16x4_t m_vds16_sign;
    int16x4_t m_vds16_shift;
};

template <>
class VqdivnNHelper<DT_S16>
{
public:
    explicit VqdivnNHelper(DT_S16 denom) : m_denom(denom)
    {
        DT_U16 abs_d  = Abs(m_denom);
        DT_S16 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U16>(length, (DT_S16)1);
        DT_S32 mul    = 1 + ((DT_U32(1)) << (16 + length - 1)) / abs_d;
               mul    = mul - (65536);
               mul    = (0 == m_denom) ? 0 : mul;

        m_vds16_mul   = vdup_n_s16(mul);
        m_vqs16_sign  = vdupq_n_s16(m_denom >> 15);
        m_vqs16_shift = vdupq_n_s16(-(length - 1));
    }

    int8x8_t operator()(const int16x8_t &vqs16_src)
    {
        int16x4_t vds16_shr_lo = vshrn_n_s32(vmull_s16(vget_low_s16(vqs16_src),  m_vds16_mul), 16);
        int16x4_t vds16_shr_hi = vshrn_n_s32(vmull_s16(vget_high_s16(vqs16_src), m_vds16_mul), 16);
        int16x8_t vds16_shr    = vcombine_s16(vds16_shr_lo, vds16_shr_hi);

        int16x8_t vqs16_sla    = vshlq_s16(vqaddq_s16(vqs16_src, vds16_shr), m_vqs16_shift);
        int16x8_t vqs16_sub    = vsubq_s16(vqs16_sla, vshlq_s16(vqs16_src, vdupq_n_s16(-15)));
        int16x8_t vqs16_result = vsubq_s16(veorq_s16(vqs16_sub, m_vqs16_sign), m_vqs16_sign);
        int8x8_t vds8_result   = vqmovn_s16(vqs16_result);

        return vds8_result;
    }

private:
    DT_S16    m_denom;
    int16x4_t m_vds16_mul;
    int16x8_t m_vqs16_sign;
    int16x8_t m_vqs16_shift;
};

template <>
class VdivNHelper<DT_U32>
{
public:
    explicit VdivNHelper(DT_U32 denom) : m_denom(denom)
    {
        DT_U32 length = 32 - __builtin_clz(m_denom - 1);
        m_mul         = (DT_U64)(4294967296) * (((DT_U64)1 << length) - (DT_U64)m_denom) / m_denom + 1;

        m_vdu32_mul    = vdup_n_u32(m_mul);
        m_vds32_shift0 = vdup_n_s32(-(Min<DT_U32>(length, (DT_U32)1)));
        m_vds32_shift1 = vdup_n_s32(-(Max<DT_S32>((DT_S32)length - 1, (DT_S32)0)));
    }

    uint32x2_t operator()(const uint32x2_t &vdu32_src)
    {
        uint32x2_t vdu32_shr    = vshrn_n_u64(vmull_u32(vdu32_src, m_vdu32_mul), 32);
        uint32x2_t vdu32_sls    = vshl_u32(vsub_u32(vdu32_src, vdu32_shr), m_vds32_shift0);
        uint32x2_t vdu32_result = vshl_u32(vadd_u32(vdu32_sls, vdu32_shr), m_vds32_shift1);

        return vdu32_result;
    }

    VQ_VDIVN_COMM(uint32x2_t, uint32x4_t);

private:
    DT_U32     m_denom;
    DT_U32     m_mul;
    uint32x2_t m_vdu32_mul;
    int32x2_t  m_vds32_shift0;
    int32x2_t  m_vds32_shift1;
};

template <>
class VqdivnNHelper<DT_U32>
{
public:
    explicit VqdivnNHelper(DT_U32 denom) : m_denom(denom)
    {
        DT_U32 length = 32 - __builtin_clz(m_denom - 1);
        m_mul         = (DT_U64)(4294967296) * (((DT_U64)1 << length) - (DT_U64)m_denom) / m_denom + 1;

        m_vdu32_mul    = vdup_n_u32(m_mul);
        m_vqs32_shift0 = vdupq_n_s32(-(Min<DT_U32>(length, (DT_U32)1)));
        m_vqs32_shift1 = vdupq_n_s32(-(Max<DT_S32>((DT_S32)length - 1, (DT_S32)0)));
    }

    uint16x4_t operator()(const uint32x4_t &vqu32_src)
    {
        uint32x2_t vdu32_shr_lo = vshrn_n_u64(vmull_u32(vget_low_u16(vqu32_src),  m_vdu32_mul), 32);
        uint32x2_t vdu32_shr_hi = vshrn_n_u64(vmull_u32(vget_high_u16(vqu32_src), m_vdu32_mul), 32);
        uint32x4_t vqu32_shr    = vcombine_u32(vdu32_shr_lo, vdu32_shr_hi);

        uint32x4_t vqu32_sls    = vshlq_u32(vsubq_u32(vqu32_src, vqu32_shr), m_vqs32_shift0);
        uint32x4_t vqu32_result = vshlq_u32(vaddq_u32(vqu32_sls, vqu32_shr), m_vqs32_shift1);
        uint16x4_t vdu16_result = vqmovn_u32(vqu32_result);

        return vdu16_result;
    }

private:
    DT_U32     m_denom;
    DT_U32     m_mul;
    uint32x2_t m_vdu32_mul;
    int32x4_t  m_vqs32_shift0;
    int32x4_t  m_vqs32_shift1;
};

template <>
class VdivNHelper<DT_S32>
{
public:
    explicit VdivNHelper(DT_S32 denom) : m_denom(denom)
    {
        DT_U32 abs_d  = Abs(m_denom);
        DT_S32 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U32>(length, (DT_S32)1);

        DT_S64 mul = 1 + ((DT_U64(1)) << (32 + length - 1)) / abs_d;
               mul = mul - (4294967296);
               mul = (0 == m_denom) ? 0 : mul;

        m_vds32_mul   = vdup_n_s32(mul);
        m_vds32_sign  = vdup_n_s32(m_denom >> 31);
        m_vds32_shift = vdup_n_s32(-(length - 1));
    }

    int32x2_t operator()(const int32x2_t &vds32_src)
    {
        int32x2_t vds32_shr    = vshrn_n_s64(vmull_s32(vds32_src, m_vds32_mul), 32);
        int32x2_t vds32_sla    = vshl_s32(vadd_s32(vds32_src, vds32_shr), m_vds32_shift);
        int32x2_t vds32_sub    = vsub_s32(vds32_sla, vshl_s32(vds32_src, vdup_n_s32(-31)));
        int32x2_t vds32_result = vsub_s32(veor_s32(vds32_sub, m_vds32_sign), m_vds32_sign);

        return vds32_result;
    }

    VQ_VDIVN_COMM(int32x2_t, int32x4_t);

private:
    DT_S32    m_denom;
    int32x2_t m_vds32_mul;
    int32x2_t m_vds32_sign;
    int32x2_t m_vds32_shift;
};

template <>
class VqdivnNHelper<DT_S32>
{
public:
    explicit VqdivnNHelper(DT_S32 denom) : m_denom(denom)
    {
        DT_U32 abs_d  = Abs(m_denom);
        DT_S32 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U32>(length, (DT_S32)1);

        DT_S64 mul = 1 + ((DT_U64(1)) << (32 + length - 1)) / abs_d;
               mul = mul - (4294967296);
               mul = (0 == m_denom) ? 0 : mul;

        m_vds32_mul   = vdup_n_s32(mul);
        m_vqs32_sign  = vdupq_n_s32(m_denom >> 31);
        m_vqs32_shift = vdupq_n_s32(-(length - 1));
    }

    int16x4_t operator()(const int32x4_t &vqs32_src)
    {
        int32x2_t vds32_shr_lo = vshrn_n_s64(vmull_s32(vget_low_u16(vqs32_src),  m_vds32_mul), 32);
        int32x2_t vds32_shr_hi = vshrn_n_s64(vmull_s32(vget_high_u16(vqs32_src), m_vds32_mul), 32);
        int32x4_t vqs32_shr    = vcombine_s32(vds32_shr_lo, vds32_shr_hi);

        int32x4_t vqs32_sla    = vshlq_s32(vqaddq_s32(vqs32_src, vqs32_shr), m_vqs32_shift);
        int32x4_t vqs32_sub    = vqsubq_s32(vqs32_sla, vshlq_s32(vqs32_src, vdupq_n_s32(-31)));
        int32x4_t vqs32_result = vqsubq_s32(veorq_s32(vqs32_sub, m_vqs32_sign), m_vqs32_sign);
        int16x4_t vds16_result = vqmovn_s32(vqs32_result);

        return vds16_result;
    }

private:
    DT_S32    m_denom;
    int32x2_t m_vds32_mul;
    int32x4_t m_vqs32_sign;
    int32x4_t m_vqs32_shift;
};

#if defined(__aarch64__)
template <>
class VdivNHelper<DT_U64>
{
public:
    explicit VdivNHelper(DT_U64 denom) : m_denom(denom)
    {}

    uint64x1_t operator()(const uint64x1_t &vdu64_src)
    {
        DT_U64 u64_src[1], u64_result[1];
        vst1_u64(u64_src, vdu64_src);
        u64_result[0] = u64_src[0] / m_denom;

        return vld1_u64(u64_result);
    }

    uint64x2_t operator()(const uint64x2_t &vqu64_src)
    {
        DT_U64 u64_src[2], u64_result[2];
        vst1q_u64(u64_src, vqu64_src);

        u64_result[0] = u64_src[0] / m_denom;
        u64_result[1] = u64_src[1] / m_denom;

        return vld1q_u64(u64_result);
    }

private:
    DT_U64 m_denom;
};

template <>
class VdivNHelper<DT_S64>
{
public:
    explicit VdivNHelper(DT_S64 denom) : m_denom(denom)
    {}

    int64x1_t operator()(const int64x1_t &vs64_src)
    {
        DT_S64 s64_src[1], s64_result[1];
        vst1_s64(s64_src, vs64_src);
        s64_result[0] = s64_src[0] / m_denom;

        return vld1_s64(s64_result);
    }

    int64x2_t operator()(const int64x2_t &vqs64_src)
    {
        DT_S64 s64_src[2], s64_result[2];
        vst1q_s64(s64_src, vqs64_src);

        s64_result[0] = s64_src[0] / m_denom;
        s64_result[1] = s64_src[1] / m_denom;

        return vld1q_s64(s64_result);
    }

private:
    DT_S64 m_denom;
};
#endif // defined(__aarch64__)

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_DIVN_HPP__