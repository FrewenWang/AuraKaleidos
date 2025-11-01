#include "aura_warp.inc"
/*==========================================  NEAREST =============================================================*/

#define NEAREST_LOAD1(vtp_result, offset)                   vtp_result    = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;

#define NEAREST_LOAD2(vtp_result, vs32_offset)              vtp_result.s0 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.lo).s0;                        \
                                                            vtp_result.s1 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.hi).s0;

#define NEAREST_LOAD4(vtp_result, vs32_offset)              vtp_result.s0 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s01).s0;                       \
                                                            vtp_result.s1 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s23).s0;                       \
                                                            vtp_result.s2 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s45).s0;                       \
                                                            vtp_result.s3 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s67).s0;

#define NEAREST_LOAD8(vtp_result, vs32_offset)              vtp_result.s0 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s01).s0;                       \
                                                            vtp_result.s1 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s23).s0;                       \
                                                            vtp_result.s2 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s45).s0;                       \
                                                            vtp_result.s3 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s67).s0;                       \
                                                            vtp_result.s4 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.s89).s0;                       \
                                                            vtp_result.s5 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.sab).s0;                       \
                                                            vtp_result.s6 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.scd).s0;                       \
                                                            vtp_result.s7 = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset.sef).s0;

#define NEAREST_LOAD_STR(vtp_result, vs32_offset, size)     NEAREST_LOAD##size(vtp_result, vs32_offset)
#define NEAREST_LOAD(vtp_result, vs32_offset, size)         NEAREST_LOAD_STR(vtp_result, vs32_offset, size)

/*==========================================  LINEAR =============================================================*/

#define LINEAR_CONSTANT_READ_IAURA(vtp_src0, vtp_src1, vs32_offset, label)                                                                              \
    {                                                                                                                                                   \
        vtp_src0.s0     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s0 += 1;                                                                                                                            \
        vtp_src0.s1     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s1 += 1;                                                                                                                            \
        vtp_src1.s1     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s0 -= 1;                                                                                                                            \
        vtp_src1.s0     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        int flag        = vs32_offset.s0 == (iwidth - 1);                                                                                               \
        vtp_src0.s1     = flag * border_value.val[0] + (1 - flag) * vtp_src0.s1;                                                                        \
        vtp_src1.s1     = flag * border_value.val[0] + (1 - flag) * vtp_src1.s1;                                                                        \
        flag            = vs32_offset.s0 == -1;                                                                                                         \
        vtp_src0.s0     = flag * border_value.val[0] + (1 - flag) * vtp_src0.s0;                                                                        \
        vtp_src1.s0     = flag * border_value.val[0] + (1 - flag) * vtp_src1.s0;                                                                        \
        flag            = (vs32_offset.s1 == iheight) | (label != 0);                                                                                   \
        vtp_src1        = (V2Tp)((Tp)flag * border_value.val[0]) + (V2Tp)(1 - flag) * vtp_src1;                                                         \
        flag            = (label != 0) | (vs32_offset.s1 == 0);                                                                                         \
        vtp_src0        = (V2Tp)((Tp)flag * border_value.val[0]) + (V2Tp)(1 - flag) * vtp_src0;                                                         \
    }

#define LINEAR_REPLICATE_READ_IAURA(vtp_src0, vtp_src1, vs32_offset)                                                                                    \
    {                                                                                                                                                   \
        vtp_src0.s0     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s0 += 1;                                                                                                                            \
        vtp_src0.s1     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s1 += 1;                                                                                                                            \
        vs32_offset.s1  = min(vs32_offset.s1, iheight - 1);                                                                                             \
        vtp_src1.s1     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
        vs32_offset.s0 -= 1;                                                                                                                            \
        vtp_src1.s0     = (Tp)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                             \
    }

#define LINEAR_REFLECT_101_READ_IAURA1(vtp_src, vs32_offset)                                                                                            \
    {                                                                                                                                                   \
        int8 v8s32_offset;                                                                                                                              \
        v8s32_offset.even    = vs32_offset.s0;                                                                                                          \
        v8s32_offset.odd     = vs32_offset.s1;                                                                                                          \
        v8s32_offset.s2567  += 1;                                                                                                                       \
        v8s32_offset         = CONVERT(abs(v8s32_offset), int8);                                                                                        \
        int8 v8s32_quotient  = v8s32_offset / v8s32_src_border;                                                                                         \
        int8 v8s32_remainder = v8s32_offset - v8s32_quotient * v8s32_src_border;                                                                        \
        v8s32_offset         = select(v8s32_src_border - v8s32_remainder, v8s32_remainder, (v8s32_quotient & 1) - 1);                                   \
        vtp_src[0].s0        = (Tp)READ_IAURA(Tp)(src, sampler, v8s32_offset.s01).s0;                                                                   \
        vtp_src[0].s1        = (Tp)READ_IAURA(Tp)(src, sampler, v8s32_offset.s23).s0;                                                                   \
        vtp_src[1].s0        = (Tp)READ_IAURA(Tp)(src, sampler, v8s32_offset.s45).s0;                                                                   \
        vtp_src[1].s1        = (Tp)READ_IAURA(Tp)(src, sampler, v8s32_offset.s67).s0;                                                                   \
    }

#define LINEAR_REFLECT_101_READ_IAURA2(vtp_src, vs32_offset)                                                                                            \
    {                                                                                                                                                   \
        int16 v16s32_offset;                                                                                                                            \
        v16s32_offset.s0246      = vs32_offset.s0;                                                                                                      \
        v16s32_offset.s1357      = vs32_offset.s1;                                                                                                      \
        v16s32_offset.s8ace      = vs32_offset.s2;                                                                                                      \
        v16s32_offset.s9bdf      = vs32_offset.s3;                                                                                                      \
        v16s32_offset.s2567adef += 1;                                                                                                                   \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                  \
        int16 v16s32_quotient    = v16s32_offset / v16s32_src_border;                                                                                   \
        int16 v16s32_remainder   = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                 \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                           \
        vtp_src[0].s0            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).s0;                                                              \
        vtp_src[0].s1            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).s0;                                                              \
        vtp_src[1].s0            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).s0;                                                              \
        vtp_src[1].s1            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).s0;                                                              \
        vtp_src[0].s2            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).s0;                                                              \
        vtp_src[0].s3            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).s0;                                                              \
        vtp_src[1].s2            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).s0;                                                              \
        vtp_src[1].s3            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).s0;                                                              \
    }

#define LINEAR_REFLECT_101_READ_IAURA4(vtp_src, vs32_offset)                                                                                            \
    {                                                                                                                                                   \
        int16 v16s32_offset;                                                                                                                            \
        v16s32_offset.s0246      = vs32_offset.s0;                                                                                                      \
        v16s32_offset.s1357      = vs32_offset.s1;                                                                                                      \
        v16s32_offset.s8ace      = vs32_offset.s2;                                                                                                      \
        v16s32_offset.s9bdf      = vs32_offset.s3;                                                                                                      \
        v16s32_offset.s2567adef += 1;                                                                                                                   \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                  \
        int16 v16s32_quotient    = v16s32_offset / v16s32_src_border;                                                                                   \
        int16 v16s32_remainder   = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                 \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                           \
        vtp_src[0].s0            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).s0;                                                              \
        vtp_src[0].s1            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).s0;                                                              \
        vtp_src[1].s0            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).s0;                                                              \
        vtp_src[1].s1            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).s0;                                                              \
        vtp_src[0].s2            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).s0;                                                              \
        vtp_src[0].s3            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).s0;                                                              \
        vtp_src[1].s2            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).s0;                                                              \
        vtp_src[1].s3            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).s0;                                                              \
        v16s32_offset.s0246      = vs32_offset.s4;                                                                                                      \
        v16s32_offset.s1357      = vs32_offset.s5;                                                                                                      \
        v16s32_offset.s8ace      = vs32_offset.s6;                                                                                                      \
        v16s32_offset.s9bdf      = vs32_offset.s7;                                                                                                      \
        v16s32_offset.s2567adef += 1;                                                                                                                   \
        v16s32_offset            = CONVERT(abs(v16s32_offset), int16);                                                                                  \
        v16s32_quotient          = v16s32_offset / v16s32_src_border;                                                                                   \
        v16s32_remainder         = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                 \
        v16s32_offset            = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                           \
        vtp_src[0].s4            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).s0;                                                              \
        vtp_src[0].s5            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).s0;                                                              \
        vtp_src[1].s4            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).s0;                                                              \
        vtp_src[1].s5            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).s0;                                                              \
        vtp_src[0].s6            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).s0;                                                              \
        vtp_src[0].s7            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).s0;                                                              \
        vtp_src[1].s6            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).s0;                                                              \
        vtp_src[1].s7            = (Tp)READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).s0;                                                              \
    }

#define LINEAR_CONSTANT_LOAD1(vtp_src, vs32_offset, label)                                                                                              \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0], vtp_src[1], vs32_offset, label)

#define LINEAR_CONSTANT_LOAD2(vtp_src, vs32_offset, vs32_border_flag)                                                                                   \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vs32_offset.s01, vs32_border_flag.s0)                                                    \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vs32_offset.s23, vs32_border_flag.s1)

#define LINEAR_CONSTANT_LOAD4(vtp_src, vs32_offset, vs32_border_flag)                                                                                   \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vs32_offset.s01, vs32_border_flag.s0)                                                    \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vs32_offset.s23, vs32_border_flag.s1)                                                    \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s45, vtp_src[1].s45, vs32_offset.s45, vs32_border_flag.s2)                                                    \
    LINEAR_CONSTANT_READ_IAURA(vtp_src[0].s67, vtp_src[1].s67, vs32_offset.s67, vs32_border_flag.s3)

#define LINEAR_CONSTANT_LOAD_STR(vtp_src, vs32_offset, vs32_border_flag, size)                                                                          \
    LINEAR_CONSTANT_LOAD##size(vtp_src, vs32_offset, vs32_border_flag)
#define LINEAR_CONSTANT_LOAD(vtp_src, vs32_offset, vs32_border_flag, size)                                                                              \
    LINEAR_CONSTANT_LOAD_STR(vtp_src, vs32_offset, vs32_border_flag, size)

#define LINEAR_REPLICATE_LOAD1(vtp_src, vs32_offset)                                                                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0], vtp_src[1], vs32_offset)

#define LINEAR_REPLICATE_LOAD2(vtp_src, vs32_offset)                                                                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vs32_offset.s01)                                                                        \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vs32_offset.s23)

#define LINEAR_REPLICATE_LOAD4(vtp_src, vs32_offset)                                                                                                    \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s01, vtp_src[1].s01, vs32_offset.s01)                                                                        \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s23, vtp_src[1].s23, vs32_offset.s23)                                                                        \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s45, vtp_src[1].s45, vs32_offset.s45)                                                                        \
    LINEAR_REPLICATE_READ_IAURA(vtp_src[0].s67, vtp_src[1].s67, vs32_offset.s67)

#define LINEAR_REPLICATE_LOAD_STR(vtp_src, vs32_offset, size)               LINEAR_REPLICATE_LOAD##size(vtp_src, vs32_offset)
#define LINEAR_REPLICATE_LOAD(vtp_src, vs32_offset, size)                   LINEAR_REPLICATE_LOAD_STR(vtp_src, vs32_offset, size)

#define LINEAR_REFLECT_101_READ_IAURA_STR(vtp_src, vs32_offset, size)       LINEAR_REFLECT_101_READ_IAURA##size(vtp_src, vs32_offset)
#define LINEAR_REFLECT_101_LOAD(vtp_src, vs32_offset, size)                 LINEAR_REFLECT_101_READ_IAURA_STR(vtp_src, vs32_offset, size)

/*==========================================  CUBIC =============================================================*/

#define CUBIC_CONSTANT_READ_IAURA(result_tp, vs32_offset, vf32_weight0, vf32_weight1, vs8_outof_border, vs8_label_border)                               \
    {                                                                                                                                                   \
        float4 vf32_src[4];                                                                                                                             \
        if (vs8_outof_border == 0)                                                                                                                      \
        {                                                                                                                                               \
            vf32_src[0].s0  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[0].s1  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[0].s2  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[0].s3  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s1 += 1;                                                                                                                        \
            vf32_src[1].s3  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[1].s2  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[1].s1  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[1].s0  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s1 += 1;                                                                                                                        \
            vf32_src[2].s0  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[2].s1  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[2].s2  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 += 1;                                                                                                                        \
            vf32_src[2].s3  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s1 += 1;                                                                                                                        \
            vf32_src[3].s3  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[3].s2  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[3].s1  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            vs32_offset.s0 -= 1;                                                                                                                        \
            vf32_src[3].s0  = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0;                                                                      \
            float flag      = (float)(vs8_label_border.s1 < 0);                                                                                         \
            vf32_src[0]     = (float4)(flag * border_value.val[0]) + (1.f - flag) * vf32_src[0];                                                        \
            flag            = (float)((abs(vs8_label_border.s1) == 3) + (vs8_label_border.s1 == -2));                                                   \
            vf32_src[1]     = (float4)(flag * border_value.val[0]) + (1.f - flag) * vf32_src[1];                                                        \
            flag            = (float)((abs(vs8_label_border.s1) == 3) + (vs8_label_border.s1 == 2));                                                    \
            vf32_src[2]     = (float4)(flag * border_value.val[0]) + (1.f - flag) * vf32_src[2];                                                        \
            flag            = (float)(vs8_label_border.s1 > 0);                                                                                         \
            vf32_src[3]     = (float4)(flag * border_value.val[0]) + (1.f - flag) * vf32_src[3];                                                        \
            flag            = (float)(vs8_label_border.s0 < 0);                                                                                         \
            vf32_src[0].s0  = flag * border_value.val[0] + (1.f - flag) * vf32_src[0].s0;                                                               \
            vf32_src[1].s0  = flag * border_value.val[0] + (1.f - flag) * vf32_src[1].s0;                                                               \
            vf32_src[2].s0  = flag * border_value.val[0] + (1.f - flag) * vf32_src[2].s0;                                                               \
            vf32_src[3].s0  = flag * border_value.val[0] + (1.f - flag) * vf32_src[3].s0;                                                               \
            flag            = (float)((abs(vs8_label_border.s0) == 3) + (vs8_label_border.s0 == -2));                                                   \
            vf32_src[0].s1  = flag * border_value.val[0] + (1.f - flag) * vf32_src[0].s1;                                                               \
            vf32_src[1].s1  = flag * border_value.val[0] + (1.f - flag) * vf32_src[1].s1;                                                               \
            vf32_src[2].s1  = flag * border_value.val[0] + (1.f - flag) * vf32_src[2].s1;                                                               \
            vf32_src[3].s1  = flag * border_value.val[0] + (1.f - flag) * vf32_src[3].s1;                                                               \
            flag            = (float)((abs(vs8_label_border.s0) == 3) + (vs8_label_border.s0 == 2));                                                    \
            vf32_src[0].s2  = flag * border_value.val[0] + (1.f - flag) * vf32_src[0].s2;                                                               \
            vf32_src[1].s2  = flag * border_value.val[0] + (1.f - flag) * vf32_src[1].s2;                                                               \
            vf32_src[2].s2  = flag * border_value.val[0] + (1.f - flag) * vf32_src[2].s2;                                                               \
            vf32_src[3].s2  = flag * border_value.val[0] + (1.f - flag) * vf32_src[3].s2;                                                               \
            flag            = (float)(vs8_label_border.s0 > 0);                                                                                         \
            vf32_src[0].s3  = flag * border_value.val[0] + (1.f - flag) * vf32_src[0].s3;                                                               \
            vf32_src[1].s3  = flag * border_value.val[0] + (1.f - flag) * vf32_src[1].s3;                                                               \
            vf32_src[2].s3  = flag * border_value.val[0] + (1.f - flag) * vf32_src[2].s3;                                                               \
            vf32_src[3].s3  = flag * border_value.val[0] + (1.f - flag) * vf32_src[3].s3;                                                               \
            vf32_src[0]     = vf32_src[0] * vf32_weight1.s0;                                                                                            \
            vf32_src[1]     = mad(vf32_src[1], vf32_weight1.s1, vf32_src[0]);                                                                           \
            vf32_src[2]     = mad(vf32_src[2], vf32_weight1.s2, vf32_src[1]);                                                                           \
            vf32_src[3]     = mad(vf32_src[3], vf32_weight1.s3, vf32_src[2]);                                                                           \
            vf32_src[3]     = vf32_src[3] * vf32_weight0;                                                                                               \
            result_tp       = WARP_CONVERT_SRC(vf32_src[3].s0 + vf32_src[3].s1 + vf32_src[3].s2 + vf32_src[3].s3, Tp, rte);                             \
        }                                                                                                                                               \
        else                                                                                                                                            \
        {                                                                                                                                               \
            result_tp = WARP_CONVERT_SRC(border_value.val[0], Tp, rte);                                                                                 \
        }                                                                                                                                               \
    }

#define CUBIC_REPLICATE_READ_IAURA(result_tp, vs32_offset, vf32_weight0, vf32_weight1)                                                                  \
    {                                                                                                                                                   \
        float4 v4f32_weight_xy = (float4)vf32_weight1.s0 * vf32_weight0;                                                                                \
        float result_f32 = (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s0;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s1;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s2;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s3;                                                    \
        v4f32_weight_xy  = (float4)vf32_weight1.s1 * vf32_weight0;                                                                                      \
        vs32_offset.s1  += 1;                                                                                                                           \
        vs32_offset.s1   = min(vs32_offset.s1, iheight - 1);                                                                                            \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s3;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s2;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s1;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s0;                                                    \
        v4f32_weight_xy  = (float4)vf32_weight1.s2 * vf32_weight0;                                                                                      \
        vs32_offset.s1  += 1;                                                                                                                           \
        vs32_offset.s1   = min(vs32_offset.s1, iheight - 1);                                                                                            \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s0;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s1;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s2;                                                    \
        vs32_offset.s0  += 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s3;                                                    \
        v4f32_weight_xy  = (float4)vf32_weight1.s3 * vf32_weight0;                                                                                      \
        vs32_offset.s1  += 1;                                                                                                                           \
        vs32_offset.s1   = min(vs32_offset.s1, iheight - 1);                                                                                            \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s3;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s2;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s1;                                                    \
        vs32_offset.s0  -= 1;                                                                                                                           \
        result_f32      += (float)READ_IAURA(Tp)(src, sampler, vs32_offset).s0 * v4f32_weight_xy.s0;                                                    \
        result_tp        = WARP_CONVERT_SRC(result_f32, Tp, rte);                                                                                       \
    }

#define CUBIC_REFLECT_101_READ_IAURA1(vtp_result, vs32_offset, vf32_weight0, vf32_weight1)                                                              \
    {                                                                                                                                                   \
        int16 v16s32_offset;                                                                                                                            \
        v16s32_offset.even     = vs32_offset.s0;                                                                                                        \
        v16s32_offset.odd      = vs32_offset.s1;                                                                                                        \
        v16s32_offset.s9bdf   += 1;                                                                                                                     \
        v16s32_offset.s2a     += 1;                                                                                                                     \
        v16s32_offset.s4c     += 2;                                                                                                                     \
        v16s32_offset.s6e     += 3;                                                                                                                     \
        float4 v4f32_weight_xy = (float4)vf32_weight1.s0 * vf32_weight0;                                                                                \
        v16s32_offset          = CONVERT(abs(v16s32_offset), int16);                                                                                    \
        int16 v16s32_quotient  = v16s32_offset / v16s32_src_border;                                                                                     \
        int16 v16s32_remainder = v16s32_offset - v16s32_src_border * v16s32_quotient;                                                                   \
        v16s32_offset          = select(v16s32_src_border - v16s32_remainder, v16s32_remainder, (v16s32_quotient & 1) - 1);                             \
        float result_f32       = (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).s0 * v4f32_weight_xy.s0;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).s0 * v4f32_weight_xy.s1;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).s0 * v4f32_weight_xy.s2;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).s0 * v4f32_weight_xy.s3;                                        \
        v4f32_weight_xy        = (float4)vf32_weight1.s1 * vf32_weight0;                                                                                \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).s0 * v4f32_weight_xy.s0;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).s0 * v4f32_weight_xy.s1;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).s0 * v4f32_weight_xy.s2;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).s0 * v4f32_weight_xy.s3;                                        \
        v16s32_offset.odd      = vs32_offset.s1;                                                                                                        \
        v16s32_offset.s1357   += 2;                                                                                                                     \
        v16s32_offset.s9bdf   += 3;                                                                                                                     \
        v16s32_offset.odd      = CONVERT(abs(v16s32_offset.odd), int8);                                                                                 \
        v16s32_quotient.odd    = v16s32_offset.odd / v16s32_src_border.odd;                                                                             \
        v16s32_remainder.odd   = v16s32_offset.odd - v16s32_src_border.odd * v16s32_quotient.odd;                                                       \
        v16s32_offset.odd      = select(v16s32_src_border.odd - v16s32_remainder.odd, v16s32_remainder.odd, (v16s32_quotient.odd & 1) - 1);             \
        v4f32_weight_xy        = (float4)vf32_weight1.s2 * vf32_weight0;                                                                                \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s01).s0 * v4f32_weight_xy.s0;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s23).s0 * v4f32_weight_xy.s1;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s45).s0 * v4f32_weight_xy.s2;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s67).s0 * v4f32_weight_xy.s3;                                        \
        v4f32_weight_xy        = (float4)vf32_weight1.s3 * vf32_weight0;                                                                                \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.s89).s0 * v4f32_weight_xy.s0;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.sab).s0 * v4f32_weight_xy.s1;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.scd).s0 * v4f32_weight_xy.s2;                                        \
        result_f32            += (float)READ_IAURA(Tp)(src, sampler, v16s32_offset.sef).s0 * v4f32_weight_xy.s3;                                        \
        vtp_result             = WARP_CONVERT_SRC(result_f32, Tp, rte);                                                                                 \
    }

#define CUBIC_REFLECT_101_READ_IAURA2(vtp_result, vs32_offset, vf32_weight0, vf32_weight1)                                                              \
    CUBIC_REFLECT_101_READ_IAURA1(vtp_result.lo, vs32_offset.lo, vf32_weight0.lo, vf32_weight1.lo)                                                      \
    CUBIC_REFLECT_101_READ_IAURA1(vtp_result.hi, vs32_offset.hi, vf32_weight0.hi, vf32_weight1.hi)

#define CUBIC_CONSTANT_LOAD1(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)                                                  \
    CUBIC_CONSTANT_READ_IAURA(vtp_result, vs32_offset, vf32_weight[0], vf32_weight[1], vs8_outof_border, vs8_label_border)

#define CUBIC_CONSTANT_LOAD2(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)                                                  \
    CUBIC_CONSTANT_READ_IAURA(vtp_result.s0, vs32_offset.s01, vf32_weight[0].lo, vf32_weight[1].lo, vs8_outof_border.s0, vs8_label_border.s01)          \
    CUBIC_CONSTANT_READ_IAURA(vtp_result.s1, vs32_offset.s23, vf32_weight[0].hi, vf32_weight[1].hi, vs8_outof_border.s1, vs8_label_border.s23)

#define CUBIC_CONSTANT_LOAD_STR(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)                                         \
    CUBIC_CONSTANT_LOAD##size(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border)

#define CUBIC_CONSTANT_LOAD(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)                                             \
    CUBIC_CONSTANT_LOAD_STR(vtp_result, vs32_offset, vf32_weight, vs8_outof_border, vs8_label_border, size)

#define CUBIC_REPLICATE_LOAD1(vtp_result, vs32_offset, vf32_weight)                                                                                     \
    CUBIC_REPLICATE_READ_IAURA(vtp_result, vs32_offset, vf32_weight[0], vf32_weight[1])

#define CUBIC_REPLICATE_LOAD2(vtp_result, vs32_offset, vf32_weight)                                                                                     \
    CUBIC_REPLICATE_READ_IAURA(vtp_result.s0, vs32_offset.s01, vf32_weight[0].lo, vf32_weight[1].lo)                                                    \
    CUBIC_REPLICATE_READ_IAURA(vtp_result.s1, vs32_offset.s23, vf32_weight[0].hi, vf32_weight[1].hi)

#define CUBIC_REPLICATE_LOAD_STR(vtp_result, vs32_offset, vf32_weight, size)                                                                            \
    CUBIC_REPLICATE_LOAD##size(vtp_result, vs32_offset, vf32_weight)
#define CUBIC_REPLICATE_LOAD(vtp_result, vs32_offset, vf32_weight, size)                                                                                \
    CUBIC_REPLICATE_LOAD_STR(vtp_result, vs32_offset, vf32_weight, size)

#define CUBIC_REFLECT_101_READ_IAURA_STR(vtp_result, vs32_offset, vf32_weight, size)                                                                    \
    CUBIC_REFLECT_101_READ_IAURA##size(vtp_result, vs32_offset, vf32_weight[0], vf32_weight[1])
#define CUBIC_REFLECT_101_LOAD(vtp_result, vs32_offset, vf32_weight, size)                                                                              \
    CUBIC_REFLECT_101_READ_IAURA_STR(vtp_result, vs32_offset, vf32_weight, size)
