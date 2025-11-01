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
template<typename Tp> static inline Tp SaturateCast(MI_U8  v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_S8  v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_U16 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_S16 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_U32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_S32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_F32 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_F64 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_S64 v) { return Tp(v); }
template<typename Tp> static inline Tp SaturateCast(MI_U64 v) { return Tp(v); }
#if defined(AURA_BUILD_HOST)
template<typename Tp> static inline Tp SaturateCast(MI_F16 v) { return SaturateCast<Tp>(static_cast<MI_F32>(v)); }
#endif // AURA_BUILD_HOST
template<> inline MI_U8 SaturateCast<MI_U8>(MI_S8  v)       { return (MI_U8)Max((MI_S32)v, (MI_S32)0); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_U16 v)       { return (MI_U8)Min((MI_U32)v, (MI_U32)UCHAR_MAX); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_S32 v)       { return (MI_U8)((MI_U32)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_S16 v)       { return SaturateCast<MI_U8>((MI_S32)v); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_U32 v)       { return (MI_U8)Min(v, (MI_U32)UCHAR_MAX); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_F32 v)       { MI_S32 iv = Round(v); return SaturateCast<MI_U8>(iv); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_F64 v)       { MI_S32 iv = Round(v); return SaturateCast<MI_U8>(iv); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_S64 v)       { return (MI_U8)((MI_U64)v <= (MI_U64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline MI_U8 SaturateCast<MI_U8>(MI_U64 v)       { return (MI_U8)Min(v, (MI_U64)UCHAR_MAX); }

template<> inline MI_S8 SaturateCast<MI_S8>(MI_U8  v)       { return (MI_S8)Min((MI_S32)v, (MI_S32)SCHAR_MAX); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_U16 v)       { return (MI_S8)Min((MI_U32)v, (MI_U32)SCHAR_MAX); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_S32 v)       { return (MI_S8)((MI_U32)(v-SCHAR_MIN) <= (MI_U32)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_S16 v)       { return SaturateCast<MI_S8>((MI_S32)v); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_U32 v)       { return (MI_S8)Min(v, (MI_U32)SCHAR_MAX); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_F32 v)       { MI_S32 iv = Round(v); return SaturateCast<MI_S8>(iv); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_F64 v)       { MI_S32 iv = Round(v); return SaturateCast<MI_S8>(iv); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_S64 v)       { return (MI_S8)((MI_U64)((MI_S64)v-SCHAR_MIN) <= (MI_U64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline MI_S8 SaturateCast<MI_S8>(MI_U64 v)       { return (MI_S8)Min(v, (MI_U64)SCHAR_MAX); }

template<> inline MI_U16 SaturateCast<MI_U16>(MI_S8 v)      { return (MI_U16)Max((MI_S32)v, (MI_S32)0); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_S16 v)     { return (MI_U16)Max((MI_S32)v, (MI_S32)0); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_S32 v)     { return (MI_U16)((MI_U32)v <= (MI_U32)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_U32 v)     { return (MI_U16)Min(v, (MI_U32)USHRT_MAX); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_F32 v)     { MI_S32 iv = Round(v); return SaturateCast<MI_U16>(iv); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_F64 v)     { MI_S32 iv = Round(v); return SaturateCast<MI_U16>(iv); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_S64 v)     { return (MI_U16)((MI_U64)v <= (MI_U64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline MI_U16 SaturateCast<MI_U16>(MI_U64 v)     { return (MI_U16)Min(v, (MI_U64)USHRT_MAX); }

template<> inline MI_S16 SaturateCast<MI_S16>(MI_U16 v)     { return (MI_S16)Min((MI_S32)v, (MI_S32)SHRT_MAX); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_S32 v)     { return (MI_S16)((MI_U32)(v - SHRT_MIN) <= (MI_U32)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_U32 v)     { return (MI_S16)Min(v, (MI_U32)SHRT_MAX); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_F32 v)     { MI_S32 iv = Round(v); return SaturateCast<MI_S16>(iv); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_F64 v)     { MI_S32 iv = Round(v); return SaturateCast<MI_S16>(iv); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_S64 v)     { return (MI_S16)((MI_U64)((MI_S64)v - SHRT_MIN) <= (MI_U64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline MI_S16 SaturateCast<MI_S16>(MI_U64 v)     { return (MI_S16)Min(v, (MI_U64)SHRT_MAX); }

template<> inline MI_S32 SaturateCast<MI_S32>(MI_U32 v)     { return (MI_S32)Min(v, (MI_U32)INT_MAX); }
template<> inline MI_S32 SaturateCast<MI_S32>(MI_S64 v)     { return (MI_S32)((MI_U64)(v - INT_MIN) <= (MI_U64)UINT_MAX ? v : v > 0 ? INT_MAX : INT_MIN); }
template<> inline MI_S32 SaturateCast<MI_S32>(MI_U64 v)     { return (MI_S32)Min(v, (MI_U64)INT_MAX); }
template<> inline MI_S32 SaturateCast<MI_S32>(MI_F32 v)     { return Round(v); }
template<> inline MI_S32 SaturateCast<MI_S32>(MI_F64 v)     { return Round(v); }

template<> inline MI_U32 SaturateCast<MI_U32>(MI_S8 v )     { return (MI_U32)Max(v, (MI_S8)0); }
template<> inline MI_U32 SaturateCast<MI_U32>(MI_S16 v)     { return (MI_U32)Max(v, (MI_S16)0); }
template<> inline MI_U32 SaturateCast<MI_U32>(MI_S32 v)     { return (MI_U32)Max(v, (MI_S32)0); }
template<> inline MI_U32 SaturateCast<MI_U32>(MI_S64 v)     { return (MI_U32)((MI_U64)v <= (MI_U64)UINT_MAX ? v : v > 0 ? UINT_MAX : 0); }
template<> inline MI_U32 SaturateCast<MI_U32>(MI_U64 v)     { return (MI_U32)Min(v, (MI_U64)UINT_MAX); }

template<> inline MI_U32 SaturateCast<MI_U32>(MI_F32 v)     { return SaturateCast<MI_U32>(Round(v)); }
template<> inline MI_U32 SaturateCast<MI_U32>(MI_F64 v)     { return SaturateCast<MI_U32>(Round(v)); }

template<> inline MI_U64 SaturateCast<MI_U64>(MI_S8 v )     { return (MI_U64)Max(v, (MI_S8)0); }
template<> inline MI_U64 SaturateCast<MI_U64>(MI_S16 v)     { return (MI_U64)Max(v, (MI_S16)0); }
template<> inline MI_U64 SaturateCast<MI_U64>(MI_S32 v)     { return (MI_U64)Max(v, (MI_S32)0); }
template<> inline MI_U64 SaturateCast<MI_U64>(MI_S64 v)     { return (MI_U64)Max(v, (MI_S64)0); }

template<> inline MI_S64 SaturateCast<MI_S64>(MI_U64 v)     { return (MI_S64)Min(v, (MI_U64)LLONG_MAX); }

#if defined(AURA_BUILD_HOST)
template<> inline MI_F16 SaturateCast<MI_F16>(MI_U16 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_U32 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_S32 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_U64 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_S64 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_F32 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
template<> inline MI_F16 SaturateCast<MI_F16>(MI_F64 v)     { return static_cast<MI_F16>(static_cast<MI_F32>(v)); }
#endif // AURA_BUILD_HOST

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_SATURATE_HPP__