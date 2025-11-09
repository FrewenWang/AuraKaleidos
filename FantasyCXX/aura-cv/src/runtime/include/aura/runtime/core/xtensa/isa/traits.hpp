#ifndef AURA_RUNTIME_CORE_XTENSA_ISA_TRAITS_HPP__
#define AURA_RUNTIME_CORE_XTENSA_ISA_TRAITS_HPP__

#include "aura/runtime/core/xtensa/types/built-in.hpp"

#define AURA_VDSP_VECTOR_TYPE(name, c, type)    \
struct name##X##c                               \
{                                               \
    type val[c];                                \
};

namespace aura
{
namespace xtensa
{

AURA_VDSP_VECTOR_TYPE(VdspVectorU8, 1, xb_vec2Nx8U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU8, 2, xb_vec2Nx8U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU8, 3, xb_vec2Nx8U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU8, 4, xb_vec2Nx8U);

AURA_VDSP_VECTOR_TYPE(VdspVectorS8, 1, xb_vec2Nx8);
AURA_VDSP_VECTOR_TYPE(VdspVectorS8, 2, xb_vec2Nx8);
AURA_VDSP_VECTOR_TYPE(VdspVectorS8, 3, xb_vec2Nx8);
AURA_VDSP_VECTOR_TYPE(VdspVectorS8, 4, xb_vec2Nx8);

AURA_VDSP_VECTOR_TYPE(VdspVectorU16, 1, xb_vecNx16U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU16, 2, xb_vecNx16U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU16, 3, xb_vecNx16U);
AURA_VDSP_VECTOR_TYPE(VdspVectorU16, 4, xb_vecNx16U);

AURA_VDSP_VECTOR_TYPE(VdspVectorS16, 1, xb_vecNx16);
AURA_VDSP_VECTOR_TYPE(VdspVectorS16, 2, xb_vecNx16);
AURA_VDSP_VECTOR_TYPE(VdspVectorS16, 3, xb_vecNx16);
AURA_VDSP_VECTOR_TYPE(VdspVectorS16, 4, xb_vecNx16);

AURA_VDSP_VECTOR_TYPE(VdspVectorU32, 1, xb_vecN_2x32Uv);
AURA_VDSP_VECTOR_TYPE(VdspVectorU32, 2, xb_vecN_2x32Uv);
AURA_VDSP_VECTOR_TYPE(VdspVectorU32, 3, xb_vecN_2x32Uv);
AURA_VDSP_VECTOR_TYPE(VdspVectorU32, 4, xb_vecN_2x32Uv);

AURA_VDSP_VECTOR_TYPE(VdspVectorS32, 1, xb_vecN_2x32v);
AURA_VDSP_VECTOR_TYPE(VdspVectorS32, 2, xb_vecN_2x32v);
AURA_VDSP_VECTOR_TYPE(VdspVectorS32, 3, xb_vecN_2x32v);
AURA_VDSP_VECTOR_TYPE(VdspVectorS32, 4, xb_vecN_2x32v);

template <typename Tp, DT_S32 C> struct MDVector;
template <> struct MDVector<DT_U8, 1>    { using MVType = VdspVectorU16X1; };
template <> struct MDVector<DT_U8, 2>    { using MVType = VdspVectorU16X2; };
template <> struct MDVector<DT_U8, 3>    { using MVType = VdspVectorU16X3; };
template <> struct MDVector<DT_U8, 4>    { using MVType = VdspVectorU16X4; };

template <> struct MDVector<DT_S8, 1>    { using MVType = VdspVectorS16X1; };
template <> struct MDVector<DT_S8, 2>    { using MVType = VdspVectorS16X2; };
template <> struct MDVector<DT_S8, 3>    { using MVType = VdspVectorS16X3; };
template <> struct MDVector<DT_S8, 4>    { using MVType = VdspVectorS16X4; };

template <typename Tp, DT_S32 C> struct MQVector;
template <> struct MQVector<DT_U8, 1>    { using MVType = VdspVectorU8X1; };
template <> struct MQVector<DT_U8, 2>    { using MVType = VdspVectorU8X2; };
template <> struct MQVector<DT_U8, 3>    { using MVType = VdspVectorU8X3; };
template <> struct MQVector<DT_U8, 4>    { using MVType = VdspVectorU8X4; };

template <> struct MQVector<DT_S8, 1>    { using MVType = VdspVectorS8X1; };
template <> struct MQVector<DT_S8, 2>    { using MVType = VdspVectorS8X2; };
template <> struct MQVector<DT_S8, 3>    { using MVType = VdspVectorS8X3; };
template <> struct MQVector<DT_S8, 4>    { using MVType = VdspVectorS8X4; };

template <> struct MQVector<DT_U16, 1>   { using MVType = VdspVectorU16X1; };
template <> struct MQVector<DT_U16, 2>   { using MVType = VdspVectorU16X2; };
template <> struct MQVector<DT_U16, 3>   { using MVType = VdspVectorU16X3; };
template <> struct MQVector<DT_U16, 4>   { using MVType = VdspVectorU16X4; };

template <> struct MQVector<DT_S16, 1>   { using MVType = VdspVectorS16X1; };
template <> struct MQVector<DT_S16, 2>   { using MVType = VdspVectorS16X2; };
template <> struct MQVector<DT_S16, 3>   { using MVType = VdspVectorS16X3; };
template <> struct MQVector<DT_S16, 4>   { using MVType = VdspVectorS16X4; };

template <> struct MQVector<DT_U32, 1>   { using MVType = VdspVectorU32X1; };
template <> struct MQVector<DT_U32, 2>   { using MVType = VdspVectorU32X2; };
template <> struct MQVector<DT_U32, 3>   { using MVType = VdspVectorU32X3; };
template <> struct MQVector<DT_U32, 4>   { using MVType = VdspVectorU32X4; };

template <> struct MQVector<DT_S32, 1>   { using MVType = VdspVectorS32X1; };
template <> struct MQVector<DT_S32, 2>   { using MVType = VdspVectorS32X2; };
template <> struct MQVector<DT_S32, 3>   { using MVType = VdspVectorS32X3; };
template <> struct MQVector<DT_S32, 4>   { using MVType = VdspVectorS32X4; };

template <typename Tp> struct DVector;
template <> struct DVector<DT_U8>        { using VType = xb_vecNx8U;     };
template <> struct DVector<DT_S8>        { using VType = xb_vecNx8;      };

template <typename Tp> struct QVector;
template <> struct QVector<DT_U8>        { using VType = xb_vec2Nx8U;    };
template <> struct QVector<DT_S8>        { using VType = xb_vec2Nx8;     };
template <> struct QVector<DT_U16>       { using VType = xb_vecNx16U;    };
template <> struct QVector<DT_S16>       { using VType = xb_vecNx16;     };
template <> struct QVector<DT_U32>       { using VType = xb_vecN_2x32Uv; };
template <> struct QVector<DT_S32>       { using VType = xb_vecN_2x32v;  };

template <typename Tp> struct WDVector;
template <> struct WDVector<DT_U16>      { using WVType = xb_vec2Nx16Uw; };
template <> struct WDVector<DT_S16>      { using WVType = xb_vec2Nx16w;  };
template <> struct WDVector<DT_U32>      { using WVType = xb_vecNx32U;   };
template <> struct WDVector<DT_S32>      { using WVType = xb_vecNx32;    };
template <> struct WDVector<DT_S64>      { using WVType = xb_vecN_2x64w; };

template <typename Tp> struct WQVector;
template <> struct WQVector<DT_S32>      { using WVType = xb_vec2Nx32w;  };
template <> struct WQVector<DT_S64>      { using WVType = xb_vecNx64w;   };

} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_ISA_TRAITS_HPP__