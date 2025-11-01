#include "cl_helper.inc"

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define Dt                             St

#define V2St                           VTYPE(St, 2)
#define V3St                           VTYPE(St, 3)
#define V4St                           VTYPE(St, 4)
#define V8St                           VTYPE(St, 8)
#define V16St                          VTYPE(St, 16)

#define V2Dt                           VTYPE(Dt,  2)
#define V3Dt                           VTYPE(Dt,  3)
#define V4Dt                           VTYPE(Dt,  4)
#define V8Dt                           VTYPE(Dt,  8)
#define V16Dt                          VTYPE(Dt,  16)


#if IS_FLOAT(St)
# define BILATERAL_ABS_CONVERT_F32(data0, data1, dtype)     (fabs(data0 - data1))
#elif IS_HALF(St)
# define BILATERAL_ABS_CONVERT_F32(data0, data1, dtype)     (fabs(CONVERT(data0, dtype) - CONVERT(data1, dtype)))
#else
# define BILATERAL_ABS_CONVERT_F32(data0, data1, dtype)     (CONVERT(abs_diff(data0, data1), dtype))
#endif

#if IS_UCHAR(St)
# define BILATERAL_CONVERT_CLAMP(data, dtype, min, max)     (CONVERT(data, dtype))
# define BILATERAL_CONVERT_ROUND(data, dtype)               (CONVERT_SAT_ROUND(data, dtype, rte))
#else
# define BILATERAL_CONVERT_CLAMP(data, dtype, min, max)     (clamp(CONVERT(data, dtype), min, max))
# define BILATERAL_CONVERT_ROUND(data, dtype)               (CONVERT_SAT(data, dtype))
#endif

#define BILATERAL_READ_IAURA_V4(iaura, sampler, v4s32_idx)  ((float4){READ_IAURA(float)(iaura, sampler, (int2)((v4s32_idx).s0, 0)).s0, \
                                                                      READ_IAURA(float)(iaura, sampler, (int2)((v4s32_idx).s1, 0)).s0, \
                                                                      READ_IAURA(float)(iaura, sampler, (int2)((v4s32_idx).s2, 0)).s0, \
                                                                      READ_IAURA(float)(iaura, sampler, (int2)((v4s32_idx).s3, 0)).s0})