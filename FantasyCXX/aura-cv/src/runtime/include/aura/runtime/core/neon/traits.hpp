#ifndef AURA_RUNTIME_CORE_NEON_TRAITS_HPP__
#define AURA_RUNTIME_CORE_NEON_TRAITS_HPP__

#include <arm_neon.h>

namespace aura
{

struct AURA_EXPORTS uint8x8x1_t
{
    uint8x8_t val[1];
};

struct AURA_EXPORTS uint8x16x1_t
{
    uint8x16_t val[1];
};

struct AURA_EXPORTS uint16x4x1_t
{
    uint16x4_t val[1];
};

struct AURA_EXPORTS uint16x8x1_t
{
    uint16x8_t val[1];
};

struct AURA_EXPORTS uint32x2x1_t
{
    uint32x2_t val[1];
};

struct AURA_EXPORTS uint32x4x1_t
{
    uint32x4_t val[1];
};

struct AURA_EXPORTS uint64x1x1_t
{
    uint64x1_t val[1];
};

struct AURA_EXPORTS uint64x2x1_t
{
    uint64x2_t val[1];
};

struct AURA_EXPORTS int8x8x1_t
{
    int8x8_t val[1];
};

struct AURA_EXPORTS int8x16x1_t
{
    int8x16_t val[1];
};

struct AURA_EXPORTS int16x4x1_t
{
    int16x4_t val[1];
};

struct AURA_EXPORTS int16x8x1_t
{
    int16x8_t val[1];
};

struct AURA_EXPORTS int32x2x1_t
{
    int32x2_t val[1];
};

struct AURA_EXPORTS int32x4x1_t
{
    int32x4_t val[1];
};

struct AURA_EXPORTS int64x1x1_t
{
    int64x1_t val[1];
};

struct AURA_EXPORTS int64x2x1_t
{
    int64x2_t val[1];
};

struct AURA_EXPORTS float16x4x1_t
{
    float16x4_t val[1];
};

struct AURA_EXPORTS float16x8x1_t
{
    float16x8_t val[1];
};

struct AURA_EXPORTS float32x2x1_t
{
    float32x2_t val[1];
};

struct AURA_EXPORTS float32x4x1_t
{
    float32x4_t val[1];
};

#if defined(__aarch64__)
struct AURA_EXPORTS float64x1x1_t
{
    float64x1_t val[1];
};

struct AURA_EXPORTS float64x2x1_t
{
    float64x2_t val[1];
};
#endif // __aarch64__

namespace neon
{

template <typename Tp> struct Scalar;
template <> struct Scalar<uint8x8_t>   { using SType = MI_U8;  };
template <> struct Scalar<int8x8_t>    { using SType = MI_S8;  };
template <> struct Scalar<uint16x4_t>  { using SType = MI_U16; };
template <> struct Scalar<int16x4_t>   { using SType = MI_S16; };
template <> struct Scalar<uint32x2_t>  { using SType = MI_U32; };
template <> struct Scalar<int32x2_t>   { using SType = MI_S32; };
template <> struct Scalar<uint64x1_t>  { using SType = MI_U64; };
template <> struct Scalar<int64x1_t>   { using SType = MI_S64; };
template <> struct Scalar<float32x2_t> { using SType = MI_F32; };
#if defined(__aarch64__)
template <> struct Scalar<float64x1_t> { using SType = MI_F64; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct Scalar<float16x4_t> { using SType = MI_F16; };
#endif

template <> struct Scalar<uint8x16_t>  { using SType = MI_U8;  };
template <> struct Scalar<int8x16_t>   { using SType = MI_S8;  };
template <> struct Scalar<uint16x8_t>  { using SType = MI_U16; };
template <> struct Scalar<int16x8_t>   { using SType = MI_S16; };
template <> struct Scalar<uint32x4_t>  { using SType = MI_U32; };
template <> struct Scalar<int32x4_t>   { using SType = MI_S32; };
template <> struct Scalar<uint64x2_t>  { using SType = MI_U64; };
template <> struct Scalar<int64x2_t>   { using SType = MI_S64; };
template <> struct Scalar<float32x4_t> { using SType = MI_F32; };
#if defined(__aarch64__)
template <> struct Scalar<float64x2_t> { using SType = MI_F64; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct Scalar<float16x8_t> { using SType = MI_F16; };
#endif

template <> struct Scalar<uint8x8x1_t>   { using SType = MI_U8;  };
template <> struct Scalar<uint8x8x2_t>   { using SType = MI_U8;  };
template <> struct Scalar<uint8x8x3_t>   { using SType = MI_U8;  };
template <> struct Scalar<uint8x8x4_t>   { using SType = MI_U8;  };
template <> struct Scalar<int8x8x1_t>    { using SType = MI_S8;  };
template <> struct Scalar<int8x8x2_t>    { using SType = MI_S8;  };
template <> struct Scalar<int8x8x3_t>    { using SType = MI_S8;  };
template <> struct Scalar<int8x8x4_t>    { using SType = MI_S8;  };
template <> struct Scalar<uint16x4x1_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x4x2_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x4x3_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x4x4_t>  { using SType = MI_U16; };
template <> struct Scalar<int16x4x1_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x4x2_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x4x3_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x4x4_t>   { using SType = MI_S16; };
template <> struct Scalar<uint32x2x1_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x2x2_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x2x3_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x2x4_t>  { using SType = MI_U32; };
template <> struct Scalar<int32x2x1_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x2x2_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x2x3_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x2x4_t>   { using SType = MI_S32; };
template <> struct Scalar<uint64x1x1_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x1x2_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x1x3_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x1x4_t>  { using SType = MI_U64; };
template <> struct Scalar<int64x1x1_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x1x2_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x1x3_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x1x4_t>   { using SType = MI_S64; };
template <> struct Scalar<float32x2x1_t> { using SType = MI_F32; };
template <> struct Scalar<float32x2x2_t> { using SType = MI_F32; };
template <> struct Scalar<float32x2x3_t> { using SType = MI_F32; };
template <> struct Scalar<float32x2x4_t> { using SType = MI_F32; };
#if defined(__aarch64__)
template <> struct Scalar<float64x1x1_t> { using SType = MI_F64; };
template <> struct Scalar<float64x1x2_t> { using SType = MI_F64; };
template <> struct Scalar<float64x1x3_t> { using SType = MI_F64; };
template <> struct Scalar<float64x1x4_t> { using SType = MI_F64; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct Scalar<float16x4x1_t> { using SType = MI_F16; };
template <> struct Scalar<float16x4x2_t> { using SType = MI_F16; };
template <> struct Scalar<float16x4x3_t> { using SType = MI_F16; };
template <> struct Scalar<float16x4x4_t> { using SType = MI_F16; };
#endif

template <> struct Scalar<uint8x16x1_t>  { using SType = MI_U8;  };
template <> struct Scalar<uint8x16x2_t>  { using SType = MI_U8;  };
template <> struct Scalar<uint8x16x3_t>  { using SType = MI_U8;  };
template <> struct Scalar<uint8x16x4_t>  { using SType = MI_U8;  };
template <> struct Scalar<int8x16x1_t>   { using SType = MI_S8;  };
template <> struct Scalar<int8x16x2_t>   { using SType = MI_S8;  };
template <> struct Scalar<int8x16x3_t>   { using SType = MI_S8;  };
template <> struct Scalar<int8x16x4_t>   { using SType = MI_S8;  };
template <> struct Scalar<uint16x8x1_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x8x2_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x8x3_t>  { using SType = MI_U16; };
template <> struct Scalar<uint16x8x4_t>  { using SType = MI_U16; };
template <> struct Scalar<int16x8x1_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x8x2_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x8x3_t>   { using SType = MI_S16; };
template <> struct Scalar<int16x8x4_t>   { using SType = MI_S16; };
template <> struct Scalar<uint32x4x1_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x4x2_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x4x3_t>  { using SType = MI_U32; };
template <> struct Scalar<uint32x4x4_t>  { using SType = MI_U32; };
template <> struct Scalar<int32x4x1_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x4x2_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x4x3_t>   { using SType = MI_S32; };
template <> struct Scalar<int32x4x4_t>   { using SType = MI_S32; };
template <> struct Scalar<uint64x2x1_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x2x2_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x2x3_t>  { using SType = MI_U64; };
template <> struct Scalar<uint64x2x4_t>  { using SType = MI_U64; };
template <> struct Scalar<int64x2x1_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x2x2_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x2x3_t>   { using SType = MI_S64; };
template <> struct Scalar<int64x2x4_t>   { using SType = MI_S64; };
template <> struct Scalar<float32x4x1_t> { using SType = MI_F32; };
template <> struct Scalar<float32x4x2_t> { using SType = MI_F32; };
template <> struct Scalar<float32x4x3_t> { using SType = MI_F32; };
template <> struct Scalar<float32x4x4_t> { using SType = MI_F32; };
#if defined(__aarch64__)
template <> struct Scalar<float64x2x1_t> { using SType = MI_F64; };
template <> struct Scalar<float64x2x2_t> { using SType = MI_F64; };
template <> struct Scalar<float64x2x3_t> { using SType = MI_F64; };
template <> struct Scalar<float64x2x4_t> { using SType = MI_F64; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct Scalar<float16x8x1_t> { using SType = MI_F16; };
template <> struct Scalar<float16x8x2_t> { using SType = MI_F16; };
template <> struct Scalar<float16x8x3_t> { using SType = MI_F16; };
template <> struct Scalar<float16x8x4_t> { using SType = MI_F16; };
#endif

template <typename Tp> struct DVector;
template <> struct DVector<MI_U8>  { using VType = uint8x8_t;   };
template <> struct DVector<MI_S8>  { using VType = int8x8_t;    };
template <> struct DVector<MI_U16> { using VType = uint16x4_t;  };
template <> struct DVector<MI_S16> { using VType = int16x4_t;   };
template <> struct DVector<MI_U32> { using VType = uint32x2_t;  };
template <> struct DVector<MI_S32> { using VType = int32x2_t;   };
template <> struct DVector<MI_U64> { using VType = uint64x1_t;  };
template <> struct DVector<MI_S64> { using VType = int64x1_t;   };
template <> struct DVector<MI_F32> { using VType = float32x2_t; };
#if defined(__aarch64__)
template <> struct DVector<MI_F64> { using VType = float64x1_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct DVector<MI_F16> { using VType = float16x4_t; };
#endif

template <typename Tp> struct QVector;
template <> struct QVector<MI_U8>  { using VType = uint8x16_t;  };
template <> struct QVector<MI_S8>  { using VType = int8x16_t;   };
template <> struct QVector<MI_U16> { using VType = uint16x8_t;  };
template <> struct QVector<MI_S16> { using VType = int16x8_t;   };
template <> struct QVector<MI_U32> { using VType = uint32x4_t;  };
template <> struct QVector<MI_S32> { using VType = int32x4_t;   };
template <> struct QVector<MI_U64> { using VType = uint64x2_t;  };
template <> struct QVector<MI_S64> { using VType = int64x2_t;   };
template <> struct QVector<MI_F32> { using VType = float32x4_t; };
#if defined(__aarch64__)
template <> struct QVector<MI_F64> { using VType = float64x2_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct QVector<MI_F16> { using VType = float16x8_t; };
#endif

template <typename Tp, int C> struct MDVector;
template <> struct MDVector<MI_U8, 1>  { using MVType = uint8x8x1_t;   };
template <> struct MDVector<MI_U8, 2>  { using MVType = uint8x8x2_t;   };
template <> struct MDVector<MI_U8, 3>  { using MVType = uint8x8x3_t;   };
template <> struct MDVector<MI_U8, 4>  { using MVType = uint8x8x4_t;   };
template <> struct MDVector<MI_S8, 1>  { using MVType = int8x8x1_t;    };
template <> struct MDVector<MI_S8, 2>  { using MVType = int8x8x2_t;    };
template <> struct MDVector<MI_S8, 3>  { using MVType = int8x8x3_t;    };
template <> struct MDVector<MI_S8, 4>  { using MVType = int8x8x4_t;    };
template <> struct MDVector<MI_U16, 1> { using MVType = uint16x4x1_t;  };
template <> struct MDVector<MI_U16, 2> { using MVType = uint16x4x2_t;  };
template <> struct MDVector<MI_U16, 3> { using MVType = uint16x4x3_t;  };
template <> struct MDVector<MI_U16, 4> { using MVType = uint16x4x4_t;  };
template <> struct MDVector<MI_S16, 1> { using MVType = int16x4x1_t;   };
template <> struct MDVector<MI_S16, 2> { using MVType = int16x4x2_t;   };
template <> struct MDVector<MI_S16, 3> { using MVType = int16x4x3_t;   };
template <> struct MDVector<MI_S16, 4> { using MVType = int16x4x4_t;   };
template <> struct MDVector<MI_U32, 1> { using MVType = uint32x2x1_t;  };
template <> struct MDVector<MI_U32, 2> { using MVType = uint32x2x2_t;  };
template <> struct MDVector<MI_U32, 3> { using MVType = uint32x2x3_t;  };
template <> struct MDVector<MI_U32, 4> { using MVType = uint32x2x4_t;  };
template <> struct MDVector<MI_S32, 1> { using MVType = int32x2x1_t;   };
template <> struct MDVector<MI_S32, 2> { using MVType = int32x2x2_t;   };
template <> struct MDVector<MI_S32, 3> { using MVType = int32x2x3_t;   };
template <> struct MDVector<MI_S32, 4> { using MVType = int32x2x4_t;   };
template <> struct MDVector<MI_U64, 1> { using MVType = uint64x1x1_t;  };
template <> struct MDVector<MI_U64, 2> { using MVType = uint64x1x2_t;  };
template <> struct MDVector<MI_U64, 3> { using MVType = uint64x1x3_t;  };
template <> struct MDVector<MI_U64, 4> { using MVType = uint64x1x4_t;  };
template <> struct MDVector<MI_S64, 1> { using MVType = int64x1x1_t;   };
template <> struct MDVector<MI_S64, 2> { using MVType = int64x1x2_t;   };
template <> struct MDVector<MI_S64, 3> { using MVType = int64x1x3_t;   };
template <> struct MDVector<MI_S64, 4> { using MVType = int64x1x4_t;   };
template <> struct MDVector<MI_F32, 1> { using MVType = float32x2x1_t; };
template <> struct MDVector<MI_F32, 2> { using MVType = float32x2x2_t; };
template <> struct MDVector<MI_F32, 3> { using MVType = float32x2x3_t; };
template <> struct MDVector<MI_F32, 4> { using MVType = float32x2x4_t; };
#if defined(__aarch64__)
template <> struct MDVector<MI_F64, 1> { using MVType = float64x1x1_t; };
template <> struct MDVector<MI_F64, 2> { using MVType = float64x1x2_t; };
template <> struct MDVector<MI_F64, 3> { using MVType = float64x1x3_t; };
template <> struct MDVector<MI_F64, 4> { using MVType = float64x1x4_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct MDVector<MI_F16, 1> { using MVType = float16x4x1_t; };
template <> struct MDVector<MI_F16, 2> { using MVType = float16x4x2_t; };
template <> struct MDVector<MI_F16, 3> { using MVType = float16x4x3_t; };
template <> struct MDVector<MI_F16, 4> { using MVType = float16x4x4_t; };
#endif

template <typename Tp, int C> struct MQVector;
template <> struct MQVector<MI_U8, 1>  { using MVType = uint8x16x1_t;  };
template <> struct MQVector<MI_U8, 2>  { using MVType = uint8x16x2_t;  };
template <> struct MQVector<MI_U8, 3>  { using MVType = uint8x16x3_t;  };
template <> struct MQVector<MI_U8, 4>  { using MVType = uint8x16x4_t;  };
template <> struct MQVector<MI_S8, 1>  { using MVType = int8x16x1_t;   };
template <> struct MQVector<MI_S8, 2>  { using MVType = int8x16x2_t;   };
template <> struct MQVector<MI_S8, 3>  { using MVType = int8x16x3_t;   };
template <> struct MQVector<MI_S8, 4>  { using MVType = int8x16x4_t;   };
template <> struct MQVector<MI_U16, 1> { using MVType = uint16x8x1_t;  };
template <> struct MQVector<MI_U16, 2> { using MVType = uint16x8x2_t;  };
template <> struct MQVector<MI_U16, 3> { using MVType = uint16x8x3_t;  };
template <> struct MQVector<MI_U16, 4> { using MVType = uint16x8x4_t;  };
template <> struct MQVector<MI_S16, 1> { using MVType = int16x8x1_t;   };
template <> struct MQVector<MI_S16, 2> { using MVType = int16x8x2_t;   };
template <> struct MQVector<MI_S16, 3> { using MVType = int16x8x3_t;   };
template <> struct MQVector<MI_S16, 4> { using MVType = int16x8x4_t;   };
template <> struct MQVector<MI_U32, 1> { using MVType = uint32x4x1_t;  };
template <> struct MQVector<MI_U32, 2> { using MVType = uint32x4x2_t;  };
template <> struct MQVector<MI_U32, 3> { using MVType = uint32x4x3_t;  };
template <> struct MQVector<MI_U32, 4> { using MVType = uint32x4x4_t;  };
template <> struct MQVector<MI_S32, 1> { using MVType = int32x4x1_t;   };
template <> struct MQVector<MI_S32, 2> { using MVType = int32x4x2_t;   };
template <> struct MQVector<MI_S32, 3> { using MVType = int32x4x3_t;   };
template <> struct MQVector<MI_S32, 4> { using MVType = int32x4x4_t;   };
template <> struct MQVector<MI_U64, 1> { using MVType = uint64x2x1_t;  };
template <> struct MQVector<MI_U64, 2> { using MVType = uint64x2x2_t;  };
template <> struct MQVector<MI_U64, 3> { using MVType = uint64x2x3_t;  };
template <> struct MQVector<MI_U64, 4> { using MVType = uint64x2x4_t;  };
template <> struct MQVector<MI_S64, 1> { using MVType = int64x2x1_t;   };
template <> struct MQVector<MI_S64, 2> { using MVType = int64x2x2_t;   };
template <> struct MQVector<MI_S64, 3> { using MVType = int64x2x3_t;   };
template <> struct MQVector<MI_S64, 4> { using MVType = int64x2x4_t;   };
template <> struct MQVector<MI_F32, 1> { using MVType = float32x4x1_t; };
template <> struct MQVector<MI_F32, 2> { using MVType = float32x4x2_t; };
template <> struct MQVector<MI_F32, 3> { using MVType = float32x4x3_t; };
template <> struct MQVector<MI_F32, 4> { using MVType = float32x4x4_t; };
#if defined(__aarch64__)
template <> struct MQVector<MI_F64, 1> { using MVType = float64x2x1_t; };
template <> struct MQVector<MI_F64, 2> { using MVType = float64x2x2_t; };
template <> struct MQVector<MI_F64, 3> { using MVType = float64x2x3_t; };
template <> struct MQVector<MI_F64, 4> { using MVType = float64x2x4_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct MQVector<MI_F16, 1> { using MVType = float16x8x1_t; };
template <> struct MQVector<MI_F16, 2> { using MVType = float16x8x2_t; };
template <> struct MQVector<MI_F16, 3> { using MVType = float16x8x3_t; };
template <> struct MQVector<MI_F16, 4> { using MVType = float16x8x4_t; };
#endif

template <typename Tp> struct WVectorBits;
template <> struct WVectorBits<uint8x8_t>  { using VType = uint16x8_t; };
template <> struct WVectorBits<int8x8_t>   { using VType = int16x8_t;  };
template <> struct WVectorBits<uint16x4_t> { using VType = uint32x4_t; };
template <> struct WVectorBits<int16x4_t>  { using VType = int32x4_t;  };
template <> struct WVectorBits<uint32x2_t> { using VType = uint64x2_t; };
template <> struct WVectorBits<int32x2_t>  { using VType = int64x2_t;  };

template <typename Tp> struct WMVectorBits;
template <> struct WMVectorBits<uint8x8x1_t>  { using MVType = uint16x8x1_t; };
template <> struct WMVectorBits<uint8x8x2_t>  { using MVType = uint16x8x2_t; };
template <> struct WMVectorBits<uint8x8x3_t>  { using MVType = uint16x8x3_t; };
template <> struct WMVectorBits<uint8x8x4_t>  { using MVType = uint16x8x4_t; };
template <> struct WMVectorBits<int8x8x1_t>   { using MVType = int16x8x1_t;  };
template <> struct WMVectorBits<int8x8x2_t>   { using MVType = int16x8x2_t;  };
template <> struct WMVectorBits<int8x8x3_t>   { using MVType = int16x8x3_t;  };
template <> struct WMVectorBits<int8x8x4_t>   { using MVType = int16x8x4_t;  };
template <> struct WMVectorBits<uint16x4x1_t> { using MVType = uint32x4x1_t; };
template <> struct WMVectorBits<uint16x4x2_t> { using MVType = uint32x4x2_t; };
template <> struct WMVectorBits<uint16x4x3_t> { using MVType = uint32x4x3_t; };
template <> struct WMVectorBits<uint16x4x4_t> { using MVType = uint32x4x4_t; };
template <> struct WMVectorBits<int16x4x1_t>  { using MVType = int32x4x1_t;  };
template <> struct WMVectorBits<int16x4x2_t>  { using MVType = int32x4x2_t;  };
template <> struct WMVectorBits<int16x4x3_t>  { using MVType = int32x4x3_t;  };
template <> struct WMVectorBits<int16x4x4_t>  { using MVType = int32x4x4_t;  };
template <> struct WMVectorBits<uint32x2x1_t> { using MVType = uint64x2x1_t; };
template <> struct WMVectorBits<uint32x2x2_t> { using MVType = uint64x2x2_t; };
template <> struct WMVectorBits<uint32x2x3_t> { using MVType = uint64x2x3_t; };
template <> struct WMVectorBits<uint32x2x4_t> { using MVType = uint64x2x4_t; };
template <> struct WMVectorBits<int32x2x1_t>  { using MVType = int64x2x1_t;  };
template <> struct WMVectorBits<int32x2x2_t>  { using MVType = int64x2x2_t;  };
template <> struct WMVectorBits<int32x2x3_t>  { using MVType = int64x2x3_t;  };
template <> struct WMVectorBits<int32x2x4_t>  { using MVType = int64x2x4_t;  };
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <> struct WMVectorBits<float16x4x1_t> { using MVType = float32x4x1_t; };
template <> struct WMVectorBits<float16x4x2_t> { using MVType = float32x4x2_t; };
template <> struct WMVectorBits<float16x4x3_t> { using MVType = float32x4x3_t; };
template <> struct WMVectorBits<float16x4x4_t> { using MVType = float32x4x4_t; };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename Tp> struct WVectorNums;
template <> struct WVectorNums<uint8x8_t>   { using VType = uint8x16_t;  };
template <> struct WVectorNums<int8x8_t>    { using VType = int8x16_t;   };
template <> struct WVectorNums<uint16x4_t>  { using VType = uint16x8_t;  };
template <> struct WVectorNums<int16x4_t>   { using VType = int16x8_t;   };
template <> struct WVectorNums<uint32x2_t>  { using VType = uint32x4_t;  };
template <> struct WVectorNums<int32x2_t>   { using VType = int32x4_t;   };
template <> struct WVectorNums<uint64x1_t>  { using VType = uint64x2_t;  };
template <> struct WVectorNums<int64x1_t>   { using VType = int64x2_t;   };
template <> struct WVectorNums<float32x2_t> { using VType = float32x4_t; };
#if defined(__aarch64__)
template <> struct WVectorNums<float64x1_t> { using VType = float64x2_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct WVectorNums<float16x4_t> { using VType = float16x8_t; };
#endif

template <typename Tp> struct WMVectorNums;
template <> struct WMVectorNums<uint8x8x1_t>   { using MVType = uint8x16x1_t;  };
template <> struct WMVectorNums<uint8x8x2_t>   { using MVType = uint8x16x2_t;  };
template <> struct WMVectorNums<uint8x8x3_t>   { using MVType = uint8x16x3_t;  };
template <> struct WMVectorNums<uint8x8x4_t>   { using MVType = uint8x16x4_t;  };
template <> struct WMVectorNums<int8x8x1_t>    { using MVType = int8x16x1_t;   };
template <> struct WMVectorNums<int8x8x2_t>    { using MVType = int8x16x2_t;   };
template <> struct WMVectorNums<int8x8x3_t>    { using MVType = int8x16x3_t;   };
template <> struct WMVectorNums<int8x8x4_t>    { using MVType = int8x16x4_t;   };
template <> struct WMVectorNums<uint16x4x1_t>  { using MVType = uint16x8x1_t;  };
template <> struct WMVectorNums<uint16x4x2_t>  { using MVType = uint16x8x2_t;  };
template <> struct WMVectorNums<uint16x4x3_t>  { using MVType = uint16x8x3_t;  };
template <> struct WMVectorNums<uint16x4x4_t>  { using MVType = uint16x8x4_t;  };
template <> struct WMVectorNums<int16x4x1_t>   { using MVType = int16x8x1_t;   };
template <> struct WMVectorNums<int16x4x2_t>   { using MVType = int16x8x2_t;   };
template <> struct WMVectorNums<int16x4x3_t>   { using MVType = int16x8x3_t;   };
template <> struct WMVectorNums<int16x4x4_t>   { using MVType = int16x8x4_t;   };
template <> struct WMVectorNums<uint32x2x1_t>  { using MVType = uint32x4x1_t;  };
template <> struct WMVectorNums<uint32x2x2_t>  { using MVType = uint32x4x2_t;  };
template <> struct WMVectorNums<uint32x2x3_t>  { using MVType = uint32x4x3_t;  };
template <> struct WMVectorNums<uint32x2x4_t>  { using MVType = uint32x4x4_t;  };
template <> struct WMVectorNums<int32x2x1_t>   { using MVType = int32x4x1_t;   };
template <> struct WMVectorNums<int32x2x2_t>   { using MVType = int32x4x2_t;   };
template <> struct WMVectorNums<int32x2x3_t>   { using MVType = int32x4x3_t;   };
template <> struct WMVectorNums<int32x2x4_t>   { using MVType = int32x4x4_t;   };
template <> struct WMVectorNums<uint64x1x1_t>  { using MVType = uint64x2x1_t;  };
template <> struct WMVectorNums<uint64x1x2_t>  { using MVType = uint64x2x2_t;  };
template <> struct WMVectorNums<uint64x1x3_t>  { using MVType = uint64x2x3_t;  };
template <> struct WMVectorNums<uint64x1x4_t>  { using MVType = uint64x2x4_t;  };
template <> struct WMVectorNums<int64x1x1_t>   { using MVType = int64x2x1_t;   };
template <> struct WMVectorNums<int64x1x2_t>   { using MVType = int64x2x2_t;   };
template <> struct WMVectorNums<int64x1x3_t>   { using MVType = int64x2x3_t;   };
template <> struct WMVectorNums<int64x1x4_t>   { using MVType = int64x2x4_t;   };
template <> struct WMVectorNums<float32x2x1_t> { using MVType = float32x4x1_t; };
template <> struct WMVectorNums<float32x2x2_t> { using MVType = float32x4x2_t; };
template <> struct WMVectorNums<float32x2x3_t> { using MVType = float32x4x3_t; };
template <> struct WMVectorNums<float32x2x4_t> { using MVType = float32x4x4_t; };
#if defined(__aarch64__)
template <> struct WMVectorNums<float64x1x1_t> { using MVType = float64x2x1_t; };
template <> struct WMVectorNums<float64x1x2_t> { using MVType = float64x2x2_t; };
template <> struct WMVectorNums<float64x1x3_t> { using MVType = float64x2x3_t; };
template <> struct WMVectorNums<float64x1x4_t> { using MVType = float64x2x4_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct WMVectorNums<float16x4x1_t> { using MVType = float16x8x1_t; };
template <> struct WMVectorNums<float16x4x2_t> { using MVType = float16x8x2_t; };
template <> struct WMVectorNums<float16x4x3_t> { using MVType = float16x8x3_t; };
template <> struct WMVectorNums<float16x4x4_t> { using MVType = float16x8x4_t; };
#endif

template <typename Tp> struct NVectorBits;
template <> struct NVectorBits<uint16x8_t> { using VType = uint8x8_t;  };
template <> struct NVectorBits<int16x8_t>  { using VType = int8x8_t;   };
template <> struct NVectorBits<uint32x4_t> { using VType = uint16x4_t; };
template <> struct NVectorBits<int32x4_t>  { using VType = int16x4_t;  };
template <> struct NVectorBits<uint64x2_t> { using VType = uint32x2_t; };
template <> struct NVectorBits<int64x2_t>  { using VType = int32x2_t;  };

template <typename Tp> struct NMVectorBits;
template <> struct NMVectorBits<uint16x8x1_t> { using MVType = uint8x8x1_t;  };
template <> struct NMVectorBits<uint16x8x2_t> { using MVType = uint8x8x2_t;  };
template <> struct NMVectorBits<uint16x8x3_t> { using MVType = uint8x8x3_t;  };
template <> struct NMVectorBits<uint16x8x4_t> { using MVType = uint8x8x4_t;  };
template <> struct NMVectorBits<int16x8x1_t>  { using MVType = int8x8x1_t;   };
template <> struct NMVectorBits<int16x8x2_t>  { using MVType = int8x8x2_t;   };
template <> struct NMVectorBits<int16x8x3_t>  { using MVType = int8x8x3_t;   };
template <> struct NMVectorBits<int16x8x4_t>  { using MVType = int8x8x4_t;   };
template <> struct NMVectorBits<uint32x4x1_t> { using MVType = uint16x4x1_t; };
template <> struct NMVectorBits<uint32x4x2_t> { using MVType = uint16x4x2_t; };
template <> struct NMVectorBits<uint32x4x3_t> { using MVType = uint16x4x3_t; };
template <> struct NMVectorBits<uint32x4x4_t> { using MVType = uint16x4x4_t; };
template <> struct NMVectorBits<int32x4x1_t>  { using MVType = int16x4x1_t;  };
template <> struct NMVectorBits<int32x4x2_t>  { using MVType = int16x4x2_t;  };
template <> struct NMVectorBits<int32x4x3_t>  { using MVType = int16x4x3_t;  };
template <> struct NMVectorBits<int32x4x4_t>  { using MVType = int16x4x4_t;  };
template <> struct NMVectorBits<uint64x2x1_t> { using MVType = uint32x2x1_t; };
template <> struct NMVectorBits<uint64x2x2_t> { using MVType = uint32x2x2_t; };
template <> struct NMVectorBits<uint64x2x3_t> { using MVType = uint32x2x3_t; };
template <> struct NMVectorBits<uint64x2x4_t> { using MVType = uint32x2x4_t; };
template <> struct NMVectorBits<int64x2x1_t>  { using MVType = int32x2x1_t;  };
template <> struct NMVectorBits<int64x2x2_t>  { using MVType = int32x2x2_t;  };
template <> struct NMVectorBits<int64x2x3_t>  { using MVType = int32x2x3_t;  };
template <> struct NMVectorBits<int64x2x4_t>  { using MVType = int32x2x4_t;  };

template <typename Tp> struct NVectorNums;
template <> struct NVectorNums<uint8x16_t>  { using VType = uint8x8_t;   };
template <> struct NVectorNums<int8x16_t>   { using VType = int8x8_t;    };
template <> struct NVectorNums<uint16x8_t>  { using VType = uint16x4_t;  };
template <> struct NVectorNums<int16x8_t>   { using VType = int16x4_t;   };
template <> struct NVectorNums<uint32x4_t>  { using VType = uint32x2_t;  };
template <> struct NVectorNums<int32x4_t>   { using VType = int32x2_t;   };
template <> struct NVectorNums<uint64x2_t>  { using VType = uint64x1_t;  };
template <> struct NVectorNums<int64x2_t>   { using VType = int64x1_t;   };
template <> struct NVectorNums<float32x4_t> { using VType = float32x2_t; };
#if defined(__aarch64__)
template <> struct NVectorNums<float64x2_t> { using VType = float64x1_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct NVectorNums<float16x8_t> { using VType = float16x4_t; };
#endif

template <typename Tp> struct NMVectorNums;
template <> struct NMVectorNums<uint8x16x1_t>  { using MVType = uint8x8x1_t;   };
template <> struct NMVectorNums<uint8x16x2_t>  { using MVType = uint8x8x2_t;   };
template <> struct NMVectorNums<uint8x16x3_t>  { using MVType = uint8x8x3_t;   };
template <> struct NMVectorNums<uint8x16x4_t>  { using MVType = uint8x8x4_t;   };
template <> struct NMVectorNums<int8x16x1_t>   { using MVType = int8x8x1_t;    };
template <> struct NMVectorNums<int8x16x2_t>   { using MVType = int8x8x2_t;    };
template <> struct NMVectorNums<int8x16x3_t>   { using MVType = int8x8x3_t;    };
template <> struct NMVectorNums<int8x16x4_t>   { using MVType = int8x8x4_t;    };
template <> struct NMVectorNums<uint16x8x1_t>  { using MVType = uint16x4x1_t;  };
template <> struct NMVectorNums<uint16x8x2_t>  { using MVType = uint16x4x2_t;  };
template <> struct NMVectorNums<uint16x8x3_t>  { using MVType = uint16x4x3_t;  };
template <> struct NMVectorNums<uint16x8x4_t>  { using MVType = uint16x4x4_t;  };
template <> struct NMVectorNums<int16x8x1_t>   { using MVType = int16x4x1_t;   };
template <> struct NMVectorNums<int16x8x2_t>   { using MVType = int16x4x2_t;   };
template <> struct NMVectorNums<int16x8x3_t>   { using MVType = int16x4x3_t;   };
template <> struct NMVectorNums<int16x8x4_t>   { using MVType = int16x4x4_t;   };
template <> struct NMVectorNums<uint32x4x1_t>  { using MVType = uint32x2x1_t;  };
template <> struct NMVectorNums<uint32x4x2_t>  { using MVType = uint32x2x2_t;  };
template <> struct NMVectorNums<uint32x4x3_t>  { using MVType = uint32x2x3_t;  };
template <> struct NMVectorNums<uint32x4x4_t>  { using MVType = uint32x2x4_t;  };
template <> struct NMVectorNums<int32x4x1_t>   { using MVType = int32x2x1_t;   };
template <> struct NMVectorNums<int32x4x2_t>   { using MVType = int32x2x2_t;   };
template <> struct NMVectorNums<int32x4x3_t>   { using MVType = int32x2x3_t;   };
template <> struct NMVectorNums<int32x4x4_t>   { using MVType = int32x2x4_t;   };
template <> struct NMVectorNums<uint64x2x1_t>  { using MVType = uint64x1x1_t;  };
template <> struct NMVectorNums<uint64x2x2_t>  { using MVType = uint64x1x2_t;  };
template <> struct NMVectorNums<uint64x2x3_t>  { using MVType = uint64x1x3_t;  };
template <> struct NMVectorNums<uint64x2x4_t>  { using MVType = uint64x1x4_t;  };
template <> struct NMVectorNums<int64x2x1_t>   { using MVType = int64x1x1_t;   };
template <> struct NMVectorNums<int64x2x2_t>   { using MVType = int64x1x2_t;   };
template <> struct NMVectorNums<int64x2x3_t>   { using MVType = int64x1x3_t;   };
template <> struct NMVectorNums<int64x2x4_t>   { using MVType = int64x1x4_t;   };
template <> struct NMVectorNums<float32x4x1_t> { using MVType = float32x2x1_t; };
template <> struct NMVectorNums<float32x4x2_t> { using MVType = float32x2x2_t; };
template <> struct NMVectorNums<float32x4x3_t> { using MVType = float32x2x3_t; };
template <> struct NMVectorNums<float32x4x4_t> { using MVType = float32x2x4_t; };
#if defined(__aarch64__)
template <> struct NMVectorNums<float64x2x1_t> { using MVType = float64x1x1_t; };
template <> struct NMVectorNums<float64x2x2_t> { using MVType = float64x1x2_t; };
template <> struct NMVectorNums<float64x2x3_t> { using MVType = float64x1x3_t; };
template <> struct NMVectorNums<float64x2x4_t> { using MVType = float64x1x4_t; };
#endif // __aarch64__
#if defined(AURA_ENABLE_NEON_FP16)
template <> struct NMVectorNums<float16x8x1_t> { using MVType = float16x4x1_t; };
template <> struct NMVectorNums<float16x8x2_t> { using MVType = float16x4x2_t; };
template <> struct NMVectorNums<float16x8x3_t> { using MVType = float16x4x3_t; };
template <> struct NMVectorNums<float16x8x4_t> { using MVType = float16x4x4_t; };
#endif

template <typename Tp> struct VectorSign;
template <> struct VectorSign<uint8x8_t>  { using SVType = int8x8_t;  using UVType = uint8x8_t;  };
template <> struct VectorSign<int8x8_t>   { using SVType = int8x8_t;  using UVType = uint8x8_t;  };
template <> struct VectorSign<uint16x4_t> { using SVType = int16x4_t; using UVType = uint16x4_t; };
template <> struct VectorSign<int16x4_t>  { using SVType = int16x4_t; using UVType = uint16x4_t; };
template <> struct VectorSign<uint32x2_t> { using SVType = int32x2_t; using UVType = uint32x2_t; };
template <> struct VectorSign<int32x2_t>  { using SVType = int32x2_t; using UVType = uint32x2_t; };
template <> struct VectorSign<uint64x1_t> { using SVType = int64x1_t; using UVType = uint64x1_t; };
template <> struct VectorSign<int64x1_t>  { using SVType = int64x1_t; using UVType = uint64x1_t; };
template <> struct VectorSign<uint8x16_t> { using SVType = int8x16_t; using UVType = uint8x16_t; };
template <> struct VectorSign<int8x16_t>  { using SVType = int8x16_t; using UVType = uint8x16_t; };
template <> struct VectorSign<uint16x8_t> { using SVType = int16x8_t; using UVType = uint16x8_t; };
template <> struct VectorSign<int16x8_t>  { using SVType = int16x8_t; using UVType = uint16x8_t; };
template <> struct VectorSign<uint32x4_t> { using SVType = int32x4_t; using UVType = uint32x4_t; };
template <> struct VectorSign<int32x4_t>  { using SVType = int32x4_t; using UVType = uint32x4_t; };
template <> struct VectorSign<uint64x2_t> { using SVType = int64x2_t; using UVType = uint64x2_t; };
template <> struct VectorSign<int64x2_t>  { using SVType = int64x2_t; using UVType = uint64x2_t; };

template <typename Tp> struct MVectorSign;
template <> struct MVectorSign<uint8x8x1_t>  { using MSVType = int8x8x1_t;  using MUVType = uint8x8x1_t;  };
template <> struct MVectorSign<uint8x8x2_t>  { using MSVType = int8x8x2_t;  using MUVType = uint8x8x2_t;  };
template <> struct MVectorSign<uint8x8x3_t>  { using MSVType = int8x8x3_t;  using MUVType = uint8x8x3_t;  };
template <> struct MVectorSign<uint8x8x4_t>  { using MSVType = int8x8x4_t;  using MUVType = uint8x8x4_t;  };
template <> struct MVectorSign<int8x8x1_t>   { using MSVType = int8x8x1_t;  using MUVType = uint8x8x1_t;  };
template <> struct MVectorSign<int8x8x2_t>   { using MSVType = int8x8x2_t;  using MUVType = uint8x8x2_t;  };
template <> struct MVectorSign<int8x8x3_t>   { using MSVType = int8x8x3_t;  using MUVType = uint8x8x3_t;  };
template <> struct MVectorSign<int8x8x4_t>   { using MSVType = int8x8x4_t;  using MUVType = uint8x8x4_t;  };
template <> struct MVectorSign<uint16x4x1_t> { using MSVType = int16x4x1_t; using MUVType = uint16x4x1_t; };
template <> struct MVectorSign<uint16x4x2_t> { using MSVType = int16x4x2_t; using MUVType = uint16x4x2_t; };
template <> struct MVectorSign<uint16x4x3_t> { using MSVType = int16x4x3_t; using MUVType = uint16x4x3_t; };
template <> struct MVectorSign<uint16x4x4_t> { using MSVType = int16x4x4_t; using MUVType = uint16x4x4_t; };
template <> struct MVectorSign<int16x4x1_t>  { using MSVType = int16x4x1_t; using MUVType = uint16x4x1_t; };
template <> struct MVectorSign<int16x4x2_t>  { using MSVType = int16x4x2_t; using MUVType = uint16x4x2_t; };
template <> struct MVectorSign<int16x4x3_t>  { using MSVType = int16x4x3_t; using MUVType = uint16x4x3_t; };
template <> struct MVectorSign<int16x4x4_t>  { using MSVType = int16x4x4_t; using MUVType = uint16x4x4_t; };
template <> struct MVectorSign<uint32x2x1_t> { using MSVType = int32x2x1_t; using MUVType = uint32x2x1_t; };
template <> struct MVectorSign<uint32x2x2_t> { using MSVType = int32x2x2_t; using MUVType = uint32x2x2_t; };
template <> struct MVectorSign<uint32x2x3_t> { using MSVType = int32x2x3_t; using MUVType = uint32x2x3_t; };
template <> struct MVectorSign<uint32x2x4_t> { using MSVType = int32x2x4_t; using MUVType = uint32x2x4_t; };
template <> struct MVectorSign<int32x2x1_t>  { using MSVType = int32x2x1_t; using MUVType = uint32x2x1_t; };
template <> struct MVectorSign<int32x2x2_t>  { using MSVType = int32x2x2_t; using MUVType = uint32x2x2_t; };
template <> struct MVectorSign<int32x2x3_t>  { using MSVType = int32x2x3_t; using MUVType = uint32x2x3_t; };
template <> struct MVectorSign<int32x2x4_t>  { using MSVType = int32x2x4_t; using MUVType = uint32x2x4_t; };
template <> struct MVectorSign<uint64x1x1_t> { using MSVType = int64x1x1_t; using MUVType = uint64x1x1_t; };
template <> struct MVectorSign<uint64x1x2_t> { using MSVType = int64x1x2_t; using MUVType = uint64x1x2_t; };
template <> struct MVectorSign<uint64x1x3_t> { using MSVType = int64x1x3_t; using MUVType = uint64x1x3_t; };
template <> struct MVectorSign<uint64x1x4_t> { using MSVType = int64x1x4_t; using MUVType = uint64x1x4_t; };
template <> struct MVectorSign<int64x1x1_t>  { using MSVType = int64x1x1_t; using MUVType = uint64x1x1_t; };
template <> struct MVectorSign<int64x1x2_t>  { using MSVType = int64x1x2_t; using MUVType = uint64x1x2_t; };
template <> struct MVectorSign<int64x1x3_t>  { using MSVType = int64x1x3_t; using MUVType = uint64x1x3_t; };
template <> struct MVectorSign<int64x1x4_t>  { using MSVType = int64x1x4_t; using MUVType = uint64x1x4_t; };
template <> struct MVectorSign<uint8x16x1_t> { using MSVType = int8x16x1_t; using MUVType = uint8x16x1_t; };
template <> struct MVectorSign<uint8x16x2_t> { using MSVType = int8x16x2_t; using MUVType = uint8x16x2_t; };
template <> struct MVectorSign<uint8x16x3_t> { using MSVType = int8x16x3_t; using MUVType = uint8x16x3_t; };
template <> struct MVectorSign<uint8x16x4_t> { using MSVType = int8x16x4_t; using MUVType = uint8x16x4_t; };
template <> struct MVectorSign<int8x16x1_t>  { using MSVType = int8x16x1_t; using MUVType = uint8x16x1_t; };
template <> struct MVectorSign<int8x16x2_t>  { using MSVType = int8x16x2_t; using MUVType = uint8x16x2_t; };
template <> struct MVectorSign<int8x16x3_t>  { using MSVType = int8x16x3_t; using MUVType = uint8x16x3_t; };
template <> struct MVectorSign<int8x16x4_t>  { using MSVType = int8x16x4_t; using MUVType = uint8x16x4_t; };
template <> struct MVectorSign<uint16x8x1_t> { using MSVType = int16x8x1_t; using MUVType = uint16x8x1_t; };
template <> struct MVectorSign<uint16x8x2_t> { using MSVType = int16x8x2_t; using MUVType = uint16x8x2_t; };
template <> struct MVectorSign<uint16x8x3_t> { using MSVType = int16x8x3_t; using MUVType = uint16x8x3_t; };
template <> struct MVectorSign<uint16x8x4_t> { using MSVType = int16x8x4_t; using MUVType = uint16x8x4_t; };
template <> struct MVectorSign<int16x8x1_t>  { using MSVType = int16x8x1_t; using MUVType = uint16x8x1_t; };
template <> struct MVectorSign<int16x8x2_t>  { using MSVType = int16x8x2_t; using MUVType = uint16x8x2_t; };
template <> struct MVectorSign<int16x8x3_t>  { using MSVType = int16x8x3_t; using MUVType = uint16x8x3_t; };
template <> struct MVectorSign<int16x8x4_t>  { using MSVType = int16x8x4_t; using MUVType = uint16x8x4_t; };
template <> struct MVectorSign<uint32x4x1_t> { using MSVType = int32x4x1_t; using MUVType = uint32x4x1_t; };
template <> struct MVectorSign<uint32x4x2_t> { using MSVType = int32x4x2_t; using MUVType = uint32x4x2_t; };
template <> struct MVectorSign<uint32x4x3_t> { using MSVType = int32x4x3_t; using MUVType = uint32x4x3_t; };
template <> struct MVectorSign<uint32x4x4_t> { using MSVType = int32x4x4_t; using MUVType = uint32x4x4_t; };
template <> struct MVectorSign<int32x4x1_t>  { using MSVType = int32x4x1_t; using MUVType = uint32x4x1_t; };
template <> struct MVectorSign<int32x4x2_t>  { using MSVType = int32x4x2_t; using MUVType = uint32x4x2_t; };
template <> struct MVectorSign<int32x4x3_t>  { using MSVType = int32x4x3_t; using MUVType = uint32x4x3_t; };
template <> struct MVectorSign<int32x4x4_t>  { using MSVType = int32x4x4_t; using MUVType = uint32x4x4_t; };
template <> struct MVectorSign<uint64x2x1_t> { using MSVType = int64x2x1_t; using MUVType = uint64x2x1_t; };
template <> struct MVectorSign<uint64x2x2_t> { using MSVType = int64x2x2_t; using MUVType = uint64x2x2_t; };
template <> struct MVectorSign<uint64x2x3_t> { using MSVType = int64x2x3_t; using MUVType = uint64x2x3_t; };
template <> struct MVectorSign<uint64x2x4_t> { using MSVType = int64x2x4_t; using MUVType = uint64x2x4_t; };
template <> struct MVectorSign<int64x2x1_t>  { using MSVType = int64x2x1_t; using MUVType = uint64x2x1_t; };
template <> struct MVectorSign<int64x2x2_t>  { using MSVType = int64x2x2_t; using MUVType = uint64x2x2_t; };
template <> struct MVectorSign<int64x2x3_t>  { using MSVType = int64x2x3_t; using MUVType = uint64x2x3_t; };
template <> struct MVectorSign<int64x2x4_t>  { using MSVType = int64x2x4_t; using MUVType = uint64x2x4_t; };

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_TRAITS_HPP__