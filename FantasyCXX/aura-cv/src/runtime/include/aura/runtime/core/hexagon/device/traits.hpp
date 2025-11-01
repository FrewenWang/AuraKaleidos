#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_TRAITS_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_TRAITS_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

#define HVX_VECTOR_TYPE(type, c)    \
struct AURA_EXPORTS type##X##c      \
{                                   \
    type val[c];                    \
};

HVX_VECTOR_TYPE(HVX_Vector, 1);
HVX_VECTOR_TYPE(HVX_Vector, 2);
HVX_VECTOR_TYPE(HVX_Vector, 3);
HVX_VECTOR_TYPE(HVX_Vector, 4);
HVX_VECTOR_TYPE(HVX_Vector, 8);
HVX_VECTOR_TYPE(HVX_Vector, 16);
HVX_VECTOR_TYPE(HVX_VectorPair, 1);
HVX_VECTOR_TYPE(HVX_VectorPair, 2);
HVX_VECTOR_TYPE(HVX_VectorPair, 3);
HVX_VECTOR_TYPE(HVX_VectorPair, 4);

template <MI_S32 C> struct MVHvxVector;
template <> struct MVHvxVector<1>   { using Type = HVX_VectorX1; };
template <> struct MVHvxVector<2>   { using Type = HVX_VectorX2; };
template <> struct MVHvxVector<3>   { using Type = HVX_VectorX3; };
template <> struct MVHvxVector<4>   { using Type = HVX_VectorX4; };

template <MI_S32 C> struct MWHvxVector;
template <> struct MWHvxVector<1>   { using Type = HVX_VectorPairX1; };
template <> struct MWHvxVector<2>   { using Type = HVX_VectorPairX2; };
template <> struct MWHvxVector<3>   { using Type = HVX_VectorPairX3; };
template <> struct MWHvxVector<4>   { using Type = HVX_VectorPairX4; };

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_TRAITS_HPP__