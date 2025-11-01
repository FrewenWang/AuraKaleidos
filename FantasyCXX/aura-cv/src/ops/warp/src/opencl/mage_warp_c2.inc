#include "aura_warp.inc"

/*==========================================  NEAREST =============================================================*/

#define NEAREST_LOAD1(vtp_result, offset)                   vtp_result     = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);

#define NEAREST_LOAD2(vtp_result, vs32_offset)              vtp_result.s01 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.lo).lo, V2Tp);                        \
                                                            vtp_result.s23 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.hi).lo, V2Tp);

#define NEAREST_LOAD4(vtp_result, vs32_offset)              vtp_result.s01 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s01).lo, V2Tp);                       \
                                                            vtp_result.s23 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s23).lo, V2Tp);                       \
                                                            vtp_result.s45 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s45).lo, V2Tp);                       \
                                                            vtp_result.s67 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s67).lo, V2Tp);

#define NEAREST_LOAD8(vtp_result, vs32_offset)              vtp_result.s01 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s01).lo, V2Tp);                       \
                                                            vtp_result.s23 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s23).lo, V2Tp);                       \
                                                            vtp_result.s45 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s45).lo, V2Tp);                       \
                                                            vtp_result.s67 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s67).lo, V2Tp);                       \
                                                            vtp_result.s89 = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.s89).lo, V2Tp);                       \
                                                            vtp_result.sab = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.sab).lo, V2Tp);                       \
                                                            vtp_result.scd = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.scd).lo, V2Tp);                       \
                                                            vtp_result.sef = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset.sef).lo, V2Tp);

#define NEAREST_LOAD_STR(vtp_result, vs32_offset, size)     NEAREST_LOAD##size(vtp_result, vs32_offset)
#define NEAREST_LOAD(vtp_result, vs32_offset, size)         NEAREST_LOAD_STR(vtp_result, vs32_offset, size)

/*==========================================  LINEAR =============================================================*/

#define LINEAR_CONSTANT_READ_IAURA(vtp_src0, vtp_src1, vtp_src2, vtp_src3, vs32_offset, label, v2tp_border_val)                                                     \
    {                                                                                                                                                               \
        vtp_src0        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s0 += 1;                                                                                                                                        \
        vtp_src1        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s1 += 1;                                                                                                                                        \
        vtp_src3        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        vtp_src2        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        int2 v2s32_flag = (int2)(iwidth - 1, iheight);                                                                                                              \
        v2s32_flag      = (v2s32_flag == vs32_offset) - (int2)label;                                                                                                \
        v2s32_flag      = (-1) * (v2s32_flag != 0);                                                                                                                 \
        int flag        = (v2s32_flag.s0 + v2s32_flag.s1) == 0;                                                                                                     \
        vtp_src3        = (V2Tp)(1 - flag) * v2tp_border_val + (V2Tp)flag * vtp_src3;                                                                               \
        flag            = (label != 0) | (vs32_offset.s0 == -1) | (vs32_offset.s1 == 0);                                                                            \
        vtp_src0        = (V2Tp)flag * v2tp_border_val + (V2Tp)(1 - flag) * vtp_src0;                                                                               \
        flag            = (v2s32_flag.s0 == 1) || (vs32_offset.s1 == 0);                                                                                            \
        vtp_src1        = (V2Tp)flag * v2tp_border_val + (V2Tp)(1 - flag) * vtp_src1;                                                                               \
        flag            = (v2s32_flag.s1 == 1) || (vs32_offset.s0 == -1);                                                                                           \
        vtp_src2        = (V2Tp)flag * v2tp_border_val + (V2Tp)(1 - flag) * vtp_src2;                                                                               \
    }

#define LINEAR_REPLICATE_READ_IAURA(vtp_src0, vtp_src1, vtp_src2, vtp_src3, vs32_offset)                                                                            \
    {                                                                                                                                                               \
        vtp_src0        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s0 += 1;                                                                                                                                        \
        vtp_src1        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s1 += 1;                                                                                                                                        \
        vs32_offset.s1  = min(vs32_offset.s1, iheight - 1);                                                                                                         \
        vtp_src3        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        vtp_src2        = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, V2Tp);                                                                          \
    }

#define LINEAR_REFLECT_101_READ_IAURA1(vtp_src, vs32_offset)                                                                                                        \
    {                                                                                                                                                               \
        int8 v8s32_offset;                                                                                                                                          \
        v8s32_offset.even    = vs32_offset.s0;                                                                                                                      \
        v8s32_offset.odd     = vs32_offset.s1;                                                                                                                      \
        v8s32_offset.s2567  += 1;                                                                                                                                   \
        v8s32_offset         = CONVERT(abs(v8s32_offset), int8);                                                                                                    \
        int8 v8s32_quotient  = v8s32_offset / v8s32_src_border;                                                                                                     \
        int8 v8s32_remainder = v8s32_offset - v8s32_quotient * v8s32_src_border;                                                                                    \
        v8s32_offset         = select(v8s32_src_border - v8s32_remainder, v8s32_remainder, (v8s32_quotient & 1) - 1);                                               \
        vtp_src[0]           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v8s32_offset.s01).lo, V2Tp);                                                                \
        vtp_src[1]           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v8s32_offset.s23).lo, V2Tp);                                                                \
        vtp_src[2]           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v8s32_offset.s45).lo, V2Tp);                                                                \
        vtp_src[3]           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v8s32_offset.s67).lo, V2Tp);                                                                \
    }

#define LINEAR_REFLECT_101_READ_IAURA2(vtp_src, vs32_offset)                                                                                                        \
    {                                                                                                                                                               \
        int16 v16s32_offset;                                                                                                                                        \
        v16s32_offset.s0246      = vs32_offset.s0;                                                                                                                  \
        v16s32_offset.s1357      = vs32_offset.s1;                                                                                                                  \
        v16s32_offset.s8ace      = vs32_offset.s2;                                                                                                                  \
        v16s32_offset.s9bdf      = vs32_offset.s3;                                                                                                                  \
        v16s32_offset.s2567adef += 1;                                                                                                                               \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                              \
        int16 v16s32_quotient    = v16s32_offset / v16s32_src_border;                                                                                               \
        int16 v16s32_remainder   = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                             \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                                       \
        vtp_src[0].lo            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).lo, V2Tp);                                                           \
        vtp_src[1].lo            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).lo, V2Tp);                                                           \
        vtp_src[2].lo            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).lo, V2Tp);                                                           \
        vtp_src[3].lo            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).lo, V2Tp);                                                           \
        vtp_src[0].hi            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).lo, V2Tp);                                                           \
        vtp_src[1].hi            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).lo, V2Tp);                                                           \
        vtp_src[2].hi            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).lo, V2Tp);                                                           \
        vtp_src[3].hi            = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).lo, V2Tp);                                                           \
    }

#define LINEAR_REFLECT_101_READ_IAURA4(vtp_src, vs32_offset)                                                                                                        \
    {                                                                                                                                                               \
        int16 v16s32_offset;                                                                                                                                        \
        v16s32_offset.s0246      = vs32_offset.s0;                                                                                                                  \
        v16s32_offset.s1357      = vs32_offset.s1;                                                                                                                  \
        v16s32_offset.s8ace      = vs32_offset.s2;                                                                                                                  \
        v16s32_offset.s9bdf      = vs32_offset.s3;                                                                                                                  \
        v16s32_offset.s2567adef += 1;                                                                                                                               \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                              \
        int16 v16s32_quotient    = v16s32_offset / v16s32_src_border;                                                                                               \
        int16 v16s32_remainder   = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                             \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                                       \
        vtp_src[0].s01           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).lo, V2Tp);                                                           \
        vtp_src[1].s01           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).lo, V2Tp);                                                           \
        vtp_src[2].s01           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).lo, V2Tp);                                                           \
        vtp_src[3].s01           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).lo, V2Tp);                                                           \
        vtp_src[0].s23           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).lo, V2Tp);                                                           \
        vtp_src[1].s23           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).lo, V2Tp);                                                           \
        vtp_src[2].s23           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).lo, V2Tp);                                                           \
        vtp_src[3].s23           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).lo, V2Tp);                                                           \
        v16s32_offset.s0246      = vs32_offset.s4;                                                                                                                  \
        v16s32_offset.s1357      = vs32_offset.s5;                                                                                                                  \
        v16s32_offset.s8ace      = vs32_offset.s6;                                                                                                                  \
        v16s32_offset.s9bdf      = vs32_offset.s7;                                                                                                                  \
        v16s32_offset.s2567adef += 1;                                                                                                                               \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                              \
        v16s32_quotient          = v16s32_offset / v16s32_src_border;                                                                                               \
        v16s32_remainder         = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                             \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                                       \
        vtp_src[0].s45           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).lo, V2Tp);                                                           \
        vtp_src[1].s45           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).lo, V2Tp);                                                           \
        vtp_src[2].s45           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).lo, V2Tp);                                                           \
        vtp_src[3].s45           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).lo, V2Tp);                                                           \
        vtp_src[0].s67           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).lo, V2Tp);                                                           \
        vtp_src[1].s67           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).lo, V2Tp);                                                           \
        vtp_src[2].s67           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).lo, V2Tp);                                                           \
        vtp_src[3].s67           = CONVERT_SAT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).lo, V2Tp);                                                           \
    }

#define LINEAR_CONSTANT_LOAD1(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val)                                                                               \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0], vtp_src[1], vtp_src[2], vtp_src[3], vs32_offset, vs8_border_flag.s0, v2tp_border_val)

#define LINEAR_CONSTANT_LOAD2(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val)                                                                               \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vtp_src[2].s01, vtp_src[3].s01, vs32_offset.s01, vs8_border_flag.s0, v2tp_border_val)                \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vtp_src[2].s23, vtp_src[3].s23, vs32_offset.s23, vs8_border_flag.s1, v2tp_border_val)

#define LINEAR_CONSTANT_LOAD4(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val)                                                                               \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vtp_src[2].s01, vtp_src[3].s01, vs32_offset.s01, vs8_border_flag.s0, v2tp_border_val)                \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vtp_src[2].s23, vtp_src[3].s23, vs32_offset.s23, vs8_border_flag.s1, v2tp_border_val)                \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s45, vtp_src[1].s45, vtp_src[2].s45, vtp_src[3].s45, vs32_offset.s45, vs8_border_flag.s2, v2tp_border_val)                \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s67, vtp_src[1].s67, vtp_src[2].s67, vtp_src[3].s67, vs32_offset.s67, vs8_border_flag.s3, v2tp_border_val)

#define LINEAR_CONSTANT_LOAD_STR(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val, size)                                                                      \
    LINEAR_CONSTANT_LOAD##size(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val)
#define LINEAR_CONSTANT_LOAD(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val, size)                                                                          \
    LINEAR_CONSTANT_LOAD_STR(vtp_src, vs32_offset, vs8_border_flag, v2tp_border_val, size)

#define LINEAR_REPLICATE_LOAD1(vtp_src, vs32_offset)                                                                                                                \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0], vtp_src[1], vtp_src[2], vtp_src[3], vs32_offset)

#define LINEAR_REPLICATE_LOAD2(vtp_src, vs32_offset)                                                                                                                \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vtp_src[2].s01, vtp_src[3].s01, vs32_offset.s01)                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vtp_src[2].s23, vtp_src[3].s23, vs32_offset.s23)

#define LINEAR_REPLICATE_LOAD4(vtp_src, vs32_offset)                                                                                                                \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vtp_src[2].s01, vtp_src[3].s01, vs32_offset.s01)                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vtp_src[2].s23, vtp_src[3].s23, vs32_offset.s23)                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s45, vtp_src[1].s45, vtp_src[2].s45, vtp_src[3].s45, vs32_offset.s45)                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s67, vtp_src[1].s67, vtp_src[2].s67, vtp_src[3].s67, vs32_offset.s67)

#define LINEAR_REPLICATE_LOAD_STR(vtp_src, vs32_offset, size)                       LINEAR_REPLICATE_LOAD##size(vtp_src, vs32_offset)
#define LINEAR_REPLICATE_LOAD(vtp_src, vs32_offset, size)                           LINEAR_REPLICATE_LOAD_STR(vtp_src, vs32_offset, size)

#define LINEAR_REFLECT_101_READ_IAURA_STR(vtp_src, vs32_offset, size)               LINEAR_REFLECT_101_READ_IAURA##size(vtp_src, vs32_offset)
#define LINEAR_REFLECT_101_LOAD(vtp_src, vs32_offset, size)                         LINEAR_REFLECT_101_READ_IAURA_STR(vtp_src, vs32_offset, size)

/*==========================================  CUBIC =============================================================*/

#define CUBIC_CONSTANT_READ_IAURA(v2tp_result, v2f32_border_val, vs32_offset, vf32_weight0, vf32_weight1, outof_border, vs8_label_border)                           \
    {                                                                                                                                                               \
        if (outof_border == 0)                                                                                                                                      \
        {                                                                                                                                                           \
            float4 v4f32_weight_xy = (float4)vf32_weight1.s0 * vf32_weight0;                                                                                        \
            float4 v4f32_flag_x    = (float4)((float)(vs8_label_border.s0 < 0),                                                                                     \
                                             (float)((abs(vs8_label_border.s0) == 3) + (vs8_label_border.s0 == -2)),                                                \
                                             (float)((abs(vs8_label_border.s0) == 3) + (vs8_label_border.s0 == 2)),                                                 \
                                             (float)(vs8_label_border.s0 > 0));                                                                                     \
            float flag_y           = (float)(vs8_label_border.s1 < 0);                                                                                              \
            float2 v2f32_result    = ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s0) + (v4f32_flag_x.s0 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s0;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s1) + (v4f32_flag_x.s1 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s1;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s2) + (v4f32_flag_x.s2 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s2;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s3) + (v4f32_flag_x.s3 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s3;                               \
            v4f32_weight_xy        = (float4)vf32_weight1.s1 * vf32_weight0;                                                                                        \
            flag_y                 = (float)((abs(vs8_label_border.s1) == 3) + (vs8_label_border.s1 == -2));                                                        \
            vs32_offset.s1        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s3) + (v4f32_flag_x.s3 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s3;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s2) + (v4f32_flag_x.s2 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s2;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s1) + (v4f32_flag_x.s1 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s1;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s0) + (v4f32_flag_x.s0 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s0;                               \
            v4f32_weight_xy        = (float4)vf32_weight1.s2 * vf32_weight0;                                                                                        \
            flag_y                 = (float)((abs(vs8_label_border.s1) == 3) + (vs8_label_border.s1 == 2));                                                         \
            vs32_offset.s1        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s0) + (v4f32_flag_x.s0 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s0;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s1) + (v4f32_flag_x.s1 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s1;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s2) + (v4f32_flag_x.s2 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s2;                               \
            vs32_offset.s0        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s3) + (v4f32_flag_x.s3 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s3;                               \
            v4f32_weight_xy        = (float4)vf32_weight1.s3 * vf32_weight0;                                                                                        \
            flag_y                 = (float)(vs8_label_border.s1 > 0);                                                                                              \
            vs32_offset.s1        += 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s3) + (v4f32_flag_x.s3 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s3;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s2) + (v4f32_flag_x.s2 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s2;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s1) + (v4f32_flag_x.s1 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s1;                               \
            vs32_offset.s0        -= 1;                                                                                                                             \
            v2f32_result          += ((1.f - flag_y) * CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) *                                              \
                                      (1.f - v4f32_flag_x.s0) + (v4f32_flag_x.s0 || flag_y) * v2f32_border_val) * v4f32_weight_xy.s0;                               \
            v2tp_result            = WARP_CONVERT_SRC(v2f32_result, V2Tp, rte);                                                                                     \
        }                                                                                                                                                           \
        else                                                                                                                                                        \
        {                                                                                                                                                           \
            v2tp_result = WARP_CONVERT_SRC(v2f32_border_val, V2Tp, rte);                                                                                            \
        }                                                                                                                                                           \
    }

#define CUBIC_REPLICATE_READ_IAURA(v2tp_result, vs32_offset, vf32_weight0, vf32_weight1)                                                                            \
    {                                                                                                                                                               \
        float4 v4f32_weight_xy = (float4)vf32_weight1.s0 * vf32_weight0;                                                                                            \
        float2 v2f32_result = CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s0;                                                   \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s1;                                                       \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s2;                                                       \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s3;                                                       \
        v4f32_weight_xy = (float4)vf32_weight1.s1 * vf32_weight0;                                                                                                   \
        vs32_offset.s1 += 1;                                                                                                                                        \
        vs32_offset.s1  = min(vs32_offset.s1, iheight - 1);                                                                                                         \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s3;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s2;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s1;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s0;                                                       \
        v4f32_weight_xy = (float4)vf32_weight1.s2 * vf32_weight0;                                                                                                   \
        vs32_offset.s1 += 1;                                                                                                                                        \
        vs32_offset.s1  = min(vs32_offset.s1, iheight - 1);                                                                                                         \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s0;                                                       \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s1;                                                       \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s2;                                                       \
        vs32_offset.s0 += 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s3;                                                       \
        v4f32_weight_xy = (float4)vf32_weight1.s3 * vf32_weight0;                                                                                                   \
        vs32_offset.s1 += 1;                                                                                                                                        \
        vs32_offset.s1  = min(vs32_offset.s1, iheight - 1);                                                                                                         \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s3;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s2;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s1;                                                       \
        vs32_offset.s0 -= 1;                                                                                                                                        \
        v2f32_result   += CONVERT(READ_IAURA(Tp)(src, sampler, vs32_offset).lo, float2) * v4f32_weight_xy.s0;                                                       \
        v2tp_result     = WARP_CONVERT_SRC(v2f32_result, V2Tp, rte);                                                                                                \
    }

#define CUBIC_REFLECT_101_READ_IAURA1(vtp_result, vs32_offset, vf32_weight0, vf32_weight1)                                                                          \
    {                                                                                                                                                               \
        int16 v16s32_offset;                                                                                                                                        \
        v16s32_offset.even     = vs32_offset.s0;                                                                                                                    \
        v16s32_offset.odd      = vs32_offset.s1;                                                                                                                    \
        v16s32_offset.s9bdf   += 1;                                                                                                                                 \
        v16s32_offset.s2a     += 1;                                                                                                                                 \
        v16s32_offset.s4c     += 2;                                                                                                                                 \
        v16s32_offset.s6e     += 3;                                                                                                                                 \
        float4 v4f32_weight_xy = (float4)vf32_weight1.s0 * vf32_weight0;                                                                                            \
        v16s32_offset          = CONVERT(abs(v16s32_offset), int16);                                                                                                \
        int16 v16s32_quotient  = v16s32_offset / v16s32_src_border;                                                                                                 \
        int16 v16s32_remainder = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                               \
        v16s32_offset          = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                                         \
        float2 v2f32_result    = CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).lo, float2) * v4f32_weight_xy.s0;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).lo, float2) * v4f32_weight_xy.s1;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).lo, float2) * v4f32_weight_xy.s2;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).lo, float2) * v4f32_weight_xy.s3;                                          \
        v4f32_weight_xy        = (float4)vf32_weight1.s1 * vf32_weight0;                                                                                            \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).lo, float2) * v4f32_weight_xy.s0;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).lo, float2) * v4f32_weight_xy.s1;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).lo, float2) * v4f32_weight_xy.s2;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).lo, float2) * v4f32_weight_xy.s3;                                          \
        v16s32_offset.odd      = vs32_offset.s1;                                                                                                                    \
        v16s32_offset.s1357   += 2;                                                                                                                                 \
        v16s32_offset.s9bdf   += 3;                                                                                                                                 \
        v16s32_offset.odd      = CONVERT(abs(v16s32_offset.odd), int8);                                                                                             \
        v16s32_quotient.odd    = v16s32_offset.odd / v16s32_src_border.odd;                                                                                         \
        v16s32_remainder.odd   = v16s32_offset.odd - v16s32_src_border.odd * v16s32_quotient.odd;                                                                   \
        v16s32_offset.odd      = select(v16s32_src_border.odd - v16s32_remainder.odd, v16s32_remainder.odd, (v16s32_quotient.odd & 1) - 1);                         \
        v4f32_weight_xy        = (float4)vf32_weight1.s2 * vf32_weight0;                                                                                            \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).lo, float2) * v4f32_weight_xy.s0;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).lo, float2) * v4f32_weight_xy.s1;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).lo, float2) * v4f32_weight_xy.s2;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).lo, float2) * v4f32_weight_xy.s3;                                          \
        v4f32_weight_xy        = (float4)vf32_weight1.s3 * vf32_weight0;                                                                                            \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).lo, float2) * v4f32_weight_xy.s0;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).lo, float2) * v4f32_weight_xy.s1;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).lo, float2) * v4f32_weight_xy.s2;                                          \
        v2f32_result          += CONVERT(READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).lo, float2) * v4f32_weight_xy.s3;                                          \
        vtp_result             = WARP_CONVERT_SRC(v2f32_result, V2Tp, rte);                                                                                         \
    }

#define CUBIC_REFLECT_101_READ_IAURA2(vtp_result, vs32_offset, vf32_weight0, vf32_weight1)                                                                          \
    CUBIC_REFLECT_101_READ_IAURA1(vtp_result.lo, vs32_offset.lo, vf32_weight0.lo, vf32_weight1.lo)                                                                  \
    CUBIC_REFLECT_101_READ_IAURA1(vtp_result.hi, vs32_offset.hi, vf32_weight0.hi, vf32_weight1.hi)

#define CUBIC_CONSTANT_LOAD1(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)                                            \
    CUBIC_CONSTANT_READ_IAURA(vtp_result, v2f32_border_val, vs32_offset, vf32_weight[0], vf32_weight[1], vs8_outof_border, vs8_label_border)

#define CUBIC_CONSTANT_LOAD2(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)                                            \
    CUBIC_CONSTANT_READ_IAURA(vtp_result.s01, v2f32_border_val, vs32_offset.s01, vf32_weight[0].lo, vf32_weight[1].lo, vs8_outof_border.s0, vs8_label_border.s01)   \
    CUBIC_CONSTANT_READ_IAURA(vtp_result.s23, v2f32_border_val, vs32_offset.s23, vf32_weight[0].hi, vf32_weight[1].hi, vs8_outof_border.s1, vs8_label_border.s23)

#define CUBIC_CONSTANT_LOAD_STR(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)                                   \
    CUBIC_CONSTANT_LOAD##size(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)

#define CUBIC_CONSTANT_LOAD(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)                                       \
    CUBIC_CONSTANT_LOAD_STR(vtp_result, v2f32_border_val, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)
    
#define CUBIC_REPLICATE_LOAD1(vtp_result, vs32_offset, vf32_weight)                                                                                                 \
    CUBIC_REPLICATE_READ_IAURA(vtp_result, vs32_offset, vf32_weight[0], vf32_weight[1])

#define CUBIC_REPLICATE_LOAD2(vtp_result, vs32_offset, vf32_weight)                                                                                                 \
    CUBIC_REPLICATE_READ_IAURA(vtp_result.s01, vs32_offset.s01, vf32_weight[0].lo, vf32_weight[1].lo)                                                               \
    CUBIC_REPLICATE_READ_IAURA(vtp_result.s23, vs32_offset.s23, vf32_weight[0].hi, vf32_weight[1].hi)

#define CUBIC_REPLICATE_LOAD_STR(vtp_result, vs32_offset, vf32_weight, size)                                                                                        \
    CUBIC_REPLICATE_LOAD##size(vtp_result, vs32_offset, vf32_weight)
#define CUBIC_REPLICATE_LOAD(vtp_result, vs32_offset, vf32_weight, size)                                                                                            \
    CUBIC_REPLICATE_LOAD_STR(vtp_result, vs32_offset, vf32_weight, size)

#define CUBIC_REFLECT_101_READ_IAURA_STR(vtp_result, vs32_offset, vf32_weight, size)                                                                                \
    CUBIC_REFLECT_101_READ_IAURA##size(vtp_result, vs32_offset, vf32_weight[0], vf32_weight[1])

#define CUBIC_REFLECT_101_LOAD(vtp_result, vs32_offset, vf32_weight, size)                                                                                          \
    CUBIC_REFLECT_101_READ_IAURA_STR(vtp_result, vs32_offset, vf32_weight, size)
