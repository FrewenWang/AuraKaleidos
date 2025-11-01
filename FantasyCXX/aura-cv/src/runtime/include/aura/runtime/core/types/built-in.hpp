#ifndef AURA_RUNTIME_CORE_TYPES_BUILTIN_HPP__
#define AURA_RUNTIME_CORE_TYPES_BUILTIN_HPP__

#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cstddef>
#include <cstdbool>
#include <cfloat>

#if !defined(AURA_VOID_DEFINED)
#define AURA_VOID_DEFINED
typedef void            AURA_VOID;
#endif // AURA_VOID_DEFINED

#if !defined(MI_CHAR_DEFINED)
#define MI_CHAR_DEFINED
typedef char            MI_CHAR;
#endif // MI_CHAR_DEFINED

#if !defined(MI_UCHAR_DEFINED)
#define MI_UCHAR_DEFINED
typedef unsigned char   MI_UCHAR;
#endif // MI_UCHAR_DEFINED

#if !defined(MI_NULL_DEFINED)
#define MI_NULL_DEFINED
#define MI_NULL         nullptr
#endif // MI_NULL_DEFINED

#if !defined(MI_BOOL_DEFINED)
#define MI_BOOL_DEFINED
#define MI_BOOL         bool
#endif // MI_BOOL_DEFINED

#if !defined(MI_TRUE_DEFINED)
#define MI_TRUE_DEFINED
#define MI_TRUE         true
#endif // MI_TRUE_DEFINED

#if !defined(MI_FALSE_DEFINED)
#define MI_FALSE_DEFINED
#define MI_FALSE        false
#endif // MI_FALSE_DEFINED

#if !defined(MI_S8_DEFINED)
#define MI_S8_DEFINED
typedef int8_t          MI_S8;
#endif // MI_S8_DEFINED

#if !defined(MI_U8_DEFINED)
#define MI_U8_DEFINED
typedef uint8_t         MI_U8;
#endif // MI_U8_DEFINED

#if !defined(MI_S16_DEFINED)
#define MI_S16_DEFINED
typedef int16_t         MI_S16;
#endif // MI_S16_DEFINED

#if !defined(MI_U16_DEFINED)
#define MI_U16_DEFINED
typedef uint16_t        MI_U16;
#endif // MI_U16_DEFINED

#if !defined(MI_S32_DEFINED)
#define MI_S32_DEFINED
typedef int32_t         MI_S32;
#endif // MI_S32_DEFINED

#if !defined(MI_U32_DEFINED)
#define MI_U32_DEFINED
typedef uint32_t        MI_U32;
#endif // MI_U32_DEFINED

#if !defined(MI_S64_DEFINED)
#define MI_S64_DEFINED
typedef int64_t         MI_S64;
#endif // MI_S64_DEFINED

#if !defined(MI_U64_DEFINED)
#define MI_U64_DEFINED
typedef uint64_t        MI_U64;
#endif // MI_U64_DEFINED

#if !defined(MI_F32_DEFINED)
#define MI_F32_DEFINED
typedef float           MI_F32;
#endif // MI_F32_DEFINED

#if !defined(MI_F64_DEFINED)
#define MI_F64_DEFINED
typedef double          MI_F64;
#endif // MI_F64_DEFINED

#if !defined(MI_SPTR_T_DEFINED)
#define MI_SPTR_T_DEFINED
typedef std::intptr_t    MI_SPTR_T;
#endif // MI_SPTR_T_DEFINED

#if !defined(MI_UPTR_T_DEFINED)
#define MI_UPTR_T_DEFINED
typedef std::uintptr_t   MI_UPTR_T;
#endif // MI_UPTR_T_DEFINED

#if !defined(MI_PTR_DIFF_DEFINED)
#define MI_PTR_DIFF_DEFINED
typedef std::ptrdiff_t   MI_PTR_DIFF;
#endif // MI_PTR_DIFF_DEFINED

#endif // AURA_RUNTIME_CORE_TYPES_BUILTIN_HPP__