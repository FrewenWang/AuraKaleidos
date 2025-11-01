#include "cl_helper.inc"

#define V4Tp   VTYPE(Tp, 4)
#define V8Tp   VTYPE(Tp, 8)

#define ROW_DFT_BUTTERFLY_1X16(v8f32_src0, v8f32_src1)                                                                              \
    {                                                                                                                               \
        float8 v8f32_temp0, v8f32_temp1;                                                                                            \
                                                                                                                                    \
        v8f32_temp0 = v8f32_src0;                                                                                                   \
        v8f32_temp1 = v8f32_src1;                                                                                                   \
        v8f32_src0.odd  = v8f32_src0.even - v8f32_temp0.odd;                                                                        \
        v8f32_src0.even = v8f32_src0.even + v8f32_temp0.odd;                                                                        \
        v8f32_src1.odd  = v8f32_src1.even - v8f32_temp1.odd;                                                                        \
        v8f32_src1.even = v8f32_src1.even + v8f32_temp1.odd;                                                                        \
                                                                                                                                    \
        v8f32_temp0.s26 =  v8f32_src0.s26;                                                                                          \
        v8f32_temp0.s37 =  v8f32_src1.s37;                                                                                          \
        v8f32_temp1.s26 =  v8f32_src1.s26;                                                                                          \
        v8f32_temp1.s37 = -v8f32_src0.s37;                                                                                          \
        v8f32_src0.s2367 = v8f32_src0.s0145 - v8f32_temp0.s2367;                                                                    \
        v8f32_src0.s0145 = v8f32_src0.s0145 + v8f32_temp0.s2367;                                                                    \
        v8f32_src1.s2367 = v8f32_src1.s0145 - v8f32_temp1.s2367;                                                                    \
        v8f32_src1.s0145 = v8f32_src1.s0145 + v8f32_temp1.s2367;                                                                    \
                                                                                                                                    \
        float8 v8f32_exp_table = (float8)(1.0f,  0.0f,  0.70710677f, -0.70710677f,                                                  \
                                          0.0f, -1.0f, -0.70710677f, -0.70710677f);                                                 \
        v8f32_temp0.hi = v8f32_src0.hi * v8f32_exp_table.even - v8f32_src1.hi * v8f32_exp_table.odd;                                \
        v8f32_temp1.hi = v8f32_src1.hi * v8f32_exp_table.even + v8f32_src0.hi * v8f32_exp_table.odd;                                \
        v8f32_src0.hi  = v8f32_src0.lo - v8f32_temp0.hi;                                                                            \
        v8f32_src0.lo  = v8f32_src0.lo + v8f32_temp0.hi;                                                                            \
        v8f32_src1.hi  = v8f32_src1.lo - v8f32_temp1.hi;                                                                            \
        v8f32_src1.lo  = v8f32_src1.lo + v8f32_temp1.hi;                                                                            \
    }

#define ROW_IDFT_BUTTERFLY_1X4(v8f32_src, v8f32_row_dft, v2f32_exp_table)                                                           \
    {                                                                                                                               \
        float4 v4f32_temp0, v4f32_temp1;                                                                                            \
        v4f32_temp0   = v8f32_src.lo + v8f32_src.hi;                                                                                \
        v4f32_temp1   = v8f32_src.lo - v8f32_src.hi;                                                                                \
        v8f32_row_dft = (float8)(v4f32_temp0.lo, v4f32_temp1.lo, v4f32_temp0.hi, v4f32_temp1.hi);                                   \
                                                                                                                                    \
        v4f32_temp0 = (float4)(v8f32_row_dft.s45, v8f32_row_dft.s76 * v2f32_exp_table);                                             \
        v8f32_row_dft.hi = v8f32_row_dft.lo - v4f32_temp0;                                                                          \
        v8f32_row_dft.lo = v8f32_row_dft.lo + v4f32_temp0;                                                                          \
    }

#define ROW_IDFT_BUTTERFLY_1X8(v16f32_dst)                                                                                          \
    {                                                                                                                               \
        float8 v8f32_temp0, v8f32_temp1;                                                                                            \
        float8 v8f32_dst0 = v16f32_dst.even;                                                                                        \
        float8 v8f32_dst1 = v16f32_dst.odd;                                                                                         \
                                                                                                                                    \
        v8f32_temp0.s26 =  v8f32_dst0.s26;                                                                                          \
        v8f32_temp1.s26 =  v8f32_dst1.s26;                                                                                          \
        v8f32_temp0.s37 = -v8f32_dst1.s37;                                                                                          \
        v8f32_temp1.s37 =  v8f32_dst0.s37;                                                                                          \
        v8f32_dst0.s2367 = v8f32_dst0.s0145 - v8f32_temp0.s2367;                                                                    \
        v8f32_dst0.s0145 = v8f32_dst0.s0145 + v8f32_temp0.s2367;                                                                    \
        v8f32_dst1.s2367 = v8f32_dst1.s0145 - v8f32_temp1.s2367;                                                                    \
        v8f32_dst1.s0145 = v8f32_dst1.s0145 + v8f32_temp1.s2367;                                                                    \
                                                                                                                                    \
        float8 v8f32_exp_table = (float8)(1.0f, 0.0f, 0.70710677f, 0.70710677f,                                                     \
                                          0.0f, 1.0f, -0.70710677f, 0.70710677f);                                                   \
        v8f32_temp0.hi = v8f32_dst0.hi * v8f32_exp_table.even - v8f32_dst1.hi * v8f32_exp_table.odd;                                \
        v8f32_temp1.hi = v8f32_dst1.hi * v8f32_exp_table.even + v8f32_dst0.hi * v8f32_exp_table.odd;                                \
        v8f32_dst0.hi  = v8f32_dst0.lo - v8f32_temp0.hi;                                                                            \
        v8f32_dst0.lo  = v8f32_dst0.lo + v8f32_temp0.hi;                                                                            \
        v8f32_dst1.hi  = v8f32_dst1.lo - v8f32_temp1.hi;                                                                            \
        v8f32_dst1.lo  = v8f32_dst1.lo + v8f32_temp1.hi;                                                                            \
                                                                                                                                    \
        v16f32_dst.even = v8f32_dst0;                                                                                               \
        v16f32_dst.odd  = v8f32_dst1;                                                                                               \
    }

#define ROW_IDFT_BUTTERFLY_1X16(v16f32_dst0, v16f32_dst1, v8f32_exp_table0, v8f32_exp_table1)                                       \
    {                                                                                                                               \
        ROW_IDFT_BUTTERFLY_1X8(v16f32_dst0);                                                                                        \
        ROW_IDFT_BUTTERFLY_1X8(v16f32_dst1);                                                                                        \
                                                                                                                                    \
        float16 v16f32_temp0;                                                                                                       \
        v16f32_temp0.even = v16f32_dst1.even * v8f32_exp_table0 - v16f32_dst1.odd  * v8f32_exp_table1;                              \
        v16f32_temp0.odd  = v16f32_dst1.odd  * v8f32_exp_table0 + v16f32_dst1.even * v8f32_exp_table1;                              \
                                                                                                                                    \
        v16f32_dst1.even = v16f32_dst0.even - v16f32_temp0.even;                                                                    \
        v16f32_dst1.odd  = v16f32_dst0.odd  - v16f32_temp0.odd;                                                                     \
        v16f32_dst0.even = v16f32_dst0.even + v16f32_temp0.even;                                                                    \
        v16f32_dst0.odd  = v16f32_dst0.odd  + v16f32_temp0.odd;                                                                     \
    }

#define ROW_IDFT_BUTTERFLY_1X32(v16f32_dst0, v16f32_dst1, v16f32_dst2, v16f32_dst3, v16f32_exp_table0, v16f32_exp_table1)           \
    {                                                                                                                               \
        ROW_IDFT_BUTTERFLY_1X16(v16f32_dst0, v16f32_dst1, v16f32_exp_table0.even, v16f32_exp_table1.even);                          \
        ROW_IDFT_BUTTERFLY_1X16(v16f32_dst2, v16f32_dst3, v16f32_exp_table0.even, v16f32_exp_table1.even);                          \
                                                                                                                                    \
        float16 v16f32_temp0, v16f32_temp1;                                                                                         \
        float16 v16f32_real = (float16)(v16f32_dst2.even, v16f32_dst3.even);                                                        \
        float16 v16f32_imag = (float16)(v16f32_dst2.odd, v16f32_dst3.odd);                                                          \
                                                                                                                                    \
        v16f32_temp0 = v16f32_real * v16f32_exp_table0 - v16f32_imag * v16f32_exp_table1;                                           \
        v16f32_temp1 = v16f32_imag * v16f32_exp_table0 + v16f32_real * v16f32_exp_table1;                                           \
                                                                                                                                    \
        v16f32_dst2.even = v16f32_dst0.even - v16f32_temp0.lo;                                                                      \
        v16f32_dst2.odd  = v16f32_dst0.odd  - v16f32_temp1.lo;                                                                      \
        v16f32_dst0.even = v16f32_dst0.even + v16f32_temp0.lo;                                                                      \
        v16f32_dst0.odd  = v16f32_dst0.odd  + v16f32_temp1.lo;                                                                      \
                                                                                                                                    \
        v16f32_dst3.even = v16f32_dst1.even - v16f32_temp0.hi;                                                                      \
        v16f32_dst3.odd  = v16f32_dst1.odd  - v16f32_temp1.hi;                                                                      \
        v16f32_dst1.even = v16f32_dst1.even + v16f32_temp0.hi;                                                                      \
        v16f32_dst1.odd  = v16f32_dst1.odd  + v16f32_temp1.hi;                                                                      \
    }