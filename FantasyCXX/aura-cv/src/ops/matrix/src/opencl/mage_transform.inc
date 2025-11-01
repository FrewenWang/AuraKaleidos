#include "cl_helper.inc"

#define V2Dt                       VTYPE(Dt, 2)

#if IS_FLOAT(Dt)
#  define IDFT_CONVERT(data, type) (data)
#else
#  define IDFT_CONVERT             CONVERT
#endif

inline float2 MulComplex(float2 a, float2 b)
{
    return (float2)(mad(a.x, b.x, -a.y * b.y), mad(a.x, b.y, a.y * b.x));
}

inline float4 MulComplex2(float4 a, float4 b)
{
    float2 real = mad(a.xz, b.xz, -a.yw * b.yw);
    float2 imag = mad(a.xz, b.yw, a.yw * b.xz);
    return (float4)(real.x, imag.x, real.y, imag.y);
}