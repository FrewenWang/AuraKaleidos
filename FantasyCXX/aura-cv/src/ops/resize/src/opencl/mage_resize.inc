#include "cl_helper.inc"

#define V2Tp                                            VTYPE(Tp, 2)
#define V3Tp                                            VTYPE(Tp, 3)
#define V4Tp                                            VTYPE(Tp, 4)
#define V8Tp                                            VTYPE(Tp, 8)
#define V16Tp                                           VTYPE(Tp, 16)

#define V2InterType                                     VTYPE(InterType, 2)
#define V3InterType                                     VTYPE(InterType, 3)
#define V4InterType                                     VTYPE(InterType, 4)
#define V8InterType                                     VTYPE(InterType, 8)
#define V16InterType                                    VTYPE(InterType, 16)

#if IS_FLOAT(Tp)
#  define RESIZE_CONVERT(data, type)                    (data)
#  define RESIZE_CONVERT_SAT_ROUND(data, type, rule)    (data)
#else
#  define RESIZE_CONVERT                                CONVERT
#  if IS_HALF(Tp)
#    define RESIZE_CONVERT_SAT_ROUND(data, type, rule)  CONVERT_SAT(data, type)
#  else
#    define RESIZE_CONVERT_SAT_ROUND                    CONVERT_SAT_ROUND
#  endif
#endif

inline float GetBicubicCoef(float x)
{
    float xx          = x * x;
    float xxx         = xx * x;
    float2 v2f32_coef = (float2)(xx, xxx);

    return (x) <= (1.f)
                ? (1.0f + dot((float2)(-2.25f, 1.25f), v2f32_coef))
                : (3 + (-6 * x) + dot((float2)(3.75f, -0.75f), v2f32_coef));
}

inline void ResizeBnCoefScalar(const int isize, const int osize, const int idx, int *src_idx, float *b0, float *b1)
{
    float scale = isize / (float)osize;
    float idx_f = (idx + 0.5f) * scale - 0.5f;
    int   idx_s = (int)floor(idx_f);

    idx_f -= idx_s;
    idx_f = select(idx_f, 0.0f, idx_s < 0);
    idx_f = select(idx_f, 1.0f, idx_s > isize - 2);
    idx_s = clamp(idx_s, 0, isize - 2);

    *src_idx = idx_s;
    *b0 = 1.0f - idx_f;
    *b1 = idx_f;
}

inline void ResizeBnCoefVector(const int isize, const int osize, const int idx, int4 *v4s32_src_idx, float4 *v4f32_a0, float4 *v4f32_a1)
{
    float4 v4f32_scale = (float4)(isize / (float)osize);

    float4 v4f32_fx = ((float4)(idx) + (float4)(0.5f, 1.5f, 2.5f, 3.5f)) * v4f32_scale - (float4)0.5f;
    int4 v4s32_idx_s = CONVERT(floor(v4f32_fx), int4);
    v4f32_fx = v4f32_fx - CONVERT(v4s32_idx_s, float4);
    v4f32_fx = select(v4f32_fx, (float4)(0.0f), v4s32_idx_s < (int4)0);
    v4f32_fx = select(v4f32_fx, (float4)(1.0f), v4s32_idx_s > (int4)(isize - 2));
    v4s32_idx_s = clamp(v4s32_idx_s, (int4)(0), (int4)(isize - 2));

    *v4s32_src_idx = v4s32_idx_s;
    *v4f32_a0 = (float4)1.0f - v4f32_fx;
    *v4f32_a1 = v4f32_fx;
}