#ifndef AURA_RUNTIME_CORE_SATURATE_HPP__
#define AURA_RUNTIME_CORE_SATURATE_HPP__

#if defined(AURA_BUILD_HOST)
#  include "aura/runtime/core/limits.hpp"
#endif // AURA_BUILD_HOST
#include "aura/runtime/core/maths.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup saturate Runtime Core Saturate
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup saturate
 * @{
*/
/**
 * @brief Template function for accurate conversion from one primitive type to another.
 *
 * The function saturate_cast resembles the standard C++ cast operations, such as static_cast\<T\>()
 * and others. It perform an efficient and accurate conversion from one primitive type to another.
 * When the input value v is out of the range of the target type, the result is not formed just
 * by taking low bits of the input, but instead the value is clipped.
 * For example:
 * @code
 * uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
 * short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
 * @endcode
 * Such clipping is done when the target type is an integer.
 *
 * When the parameter is a floating-point value and the target type is an integer (8-, 16- or 32-bit),
 * the floating-point value is first rounded to the nearest integer and then clipped if needed (when
 * the target type is 8- or 16-bit).
 *
 * @tparam Tp The output data type.
 * 
 * @param v Function parameter.
 * 
 * @return The casted value of type Tp.
 */
template<typename Tp> static inline Tp SaturateCast(DT_U8  v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_S8  v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_U16 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_S16 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_U32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_S32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_F32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_F64 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_S64 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(DT_U64 v) { return Tp(v); }
#if defined(AURA_BUILD_HOST)
template<typename Tp> static inline Tp SaturateCast(MI_F16 v) { return SaturateCast<Tp>(static_cast<DT_F32>(v)); }
#endif // AURA_BUILD_HOST
template<> inline DT_U8 SaturateCast<DT_U8>(DT_S8  v)       { return (DT_U8)Max((DT_S32)v, (DT_S32)0); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_U16 v)       { return (DT_U8)Min((DT_U32)v, (DT_U32)UCHAR_MAX); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_S32 v)       { return (DT_U8)((DT_U32)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_S16 v)       { return SaturateCast<DT_U8>((DT_S32)v); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_U32 v)       { return (DT_U8)Min(v, (DT_U32)UCHAR_MAX); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_F32 v)       { DT_S32 iv = Round(v); return SaturateCast<DT_U8>(iv); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_F64 v)       { DT_S32 iv = Round(v); return SaturateCast<DT_U8>(iv); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_S64 v)       { return (DT_U8)((DT_U64)v <= (DT_U64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline DT_U8 SaturateCast<DT_U8>(DT_U64 v)       { return (DT_U8)Min(v, (DT_U64)UCHAR_MAX); }

template<> inline DT_S8 SaturateCast<DT_S8>(DT_U8  v)       { return (DT_S8)Min((DT_S32)v, (DT_S32)SCHAR_MAX); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_U16 v)       { return (DT_S8)Min((DT_U32)v, (DT_U32)SCHAR_MAX); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_S32 v)       { return (DT_S8)((DT_U32)(v-SCHAR_MIN) <= (DT_U32)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_S16 v)       { return SaturateCast<DT_S8>((DT_S32)v); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_U32 v)       { return (DT_S8)Min(v, (DT_U32)SCHAR_MAX); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_F32 v)       { DT_S32 iv = Round(v); return SaturateCast<DT_S8>(iv); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_F64 v)       { DT_S32 iv = Round(v); return SaturateCast<DT_S8>(iv); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_S64 v)       { return (DT_S8)((DT_U64)((DT_S64)v-SCHAR_MIN) <= (DT_U64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline DT_S8 SaturateCast<DT_S8>(DT_U64 v)       { return (DT_S8)Min(v, (DT_U64)SCHAR_MAX); }

template<> inline DT_U16 SaturateCast<DT_U16>(DT_S8 v)      { return (DT_U16)Max((DT_S32)v, (DT_S32)0); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_S16 v)     { return (DT_U16)Max((DT_S32)v, (DT_S32)0); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_S32 v)     { return (DT_U16)((DT_U32)v <= (DT_U32)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_U32 v)     { return (DT_U16)Min(v, (DT_U32)USHRT_MAX); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_F32 v)     { DT_S32 iv = Round(v); return SaturateCast<DT_U16>(iv); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_F64 v)     { DT_S32 iv = Round(v); return SaturateCast<DT_U16>(iv); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_S64 v)     { return (DT_U16)((DT_U64)v <= (DT_U64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline DT_U16 SaturateCast<DT_U16>(DT_U64 v)     { return (DT_U16)Min(v, (DT_U64)USHRT_MAX); }

template<> inline DT_S16 SaturateCast<DT_S16>(DT_U16 v)     { return (DT_S16)Min((DT_S32)v, (DT_S32)SHRT_MAX); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_S32 v)     { return (DT_S16)((DT_U32)(v - SHRT_MIN) <= (DT_U32)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_U32 v)     { return (DT_S16)Min(v, (DT_U32)SHRT_MAX); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_F32 v)     { DT_S32 iv = Round(v); return SaturateCast<DT_S16>(iv); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_F64 v)     { DT_S32 iv = Round(v); return SaturateCast<DT_S16>(iv); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_S64 v)     { return (DT_S16)((DT_U64)((DT_S64)v - SHRT_MIN) <= (DT_U64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline DT_S16 SaturateCast<DT_S16>(DT_U64 v)     { return (DT_S16)Min(v, (DT_U64)SHRT_MAX); }

template<> inline DT_S32 SaturateCast<DT_S32>(DT_U32 v)     { return (DT_S32)Min(v, (DT_U32)INT_MAX); }
template<> inline DT_S32 SaturateCast<DT_S32>(DT_S64 v)     { return (DT_S32)((DT_U64)(v - INT_MIN) <= (DT_U64)UINT_MAX ? v : v > 0 ? INT_MAX : INT_MIN); }
template<> inline DT_S32 SaturateCast<DT_S32>(DT_U64 v)     { return (DT_S32)Min(v, (DT_U64)INT_MAX); }
template<> inline DT_S32 SaturateCast<DT_S32>(DT_F32 v)     { return Round(v); }
template<> inline DT_S32 SaturateCast<DT_S32>(DT_F64 v)     { return Round(v); }

template<> inline DT_U32 SaturateCast<DT_U32>(DT_S8 v )     { return (DT_U32)Max(v, (DT_S8)0); }
template<> inline DT_U32 SaturateCast<DT_U32>(DT_S16 v)     { return (DT_U32)Max(v, (DT_S16)0); }
template<> inline DT_U32 SaturateCast<DT_U32>(DT_S32 v)     { return (DT_U32)Max(v, (DT_S32)0); }
template<> inline DT_U32 SaturateCast<DT_U32>(DT_S64 v)     { return (DT_U32)((DT_U64)v <= (DT_U64)UINT_MAX ? v : v > 0 ? UINT_MAX : 0); }
template<> inline DT_U32 SaturateCast<DT_U32>(DT_U64 v)     { return (DT_U32)Min(v, (DT_U64)UINT_MAX); }

template<> inline DT_U32 SaturateCast<DT_U32>(DT_F32 v)     { return SaturateCast<DT_U32>(Round(v)); }
template<> inline DT_U32 SaturateCast<DT_U32>(DT_F64 v)     { return SaturateCast<DT_U32>(Round(v)); }

template<> inline DT_U64 SaturateCast<DT_U64>(DT_S8 v )     { return (DT_U64)Max(v, (DT_S8)0); }
template<> inline DT_U64 SaturateCast<DT_U64>(DT_S16 v)     { return (DT_U64)Max(v, (DT_S16)0); }
template<> inline DT_U64 SaturateCast<DT_U64>(DT_S32 v)     { return (DT_U64)Max(v, (DT_S32)0); }
template<> inline DT_U64 SaturateCast<DT_U64>(DT_S64 v)     { return (DT_U64)Max(v, (DT_S64)0); }

template<> inline DT_S64 SaturateCast<DT_S64>(DT_U64 v)     { return (DT_S64)Min(v, (DT_U64)LLONG_MAX); }

#if defined(AURA_BUILD_HOST)
template<> inline MI_F16 SaturateCast<MI_F16>(DT_U16 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_U32 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_S32 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_U64 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_S64 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_F32 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(DT_F64 v)     { return static_cast<MI_F16>(static_cast<DT_F32>(v)); }
#endif // AURA_BUILD_HOST

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_SATURATE_HPP__