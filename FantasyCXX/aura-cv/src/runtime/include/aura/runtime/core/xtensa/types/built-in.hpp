#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_BUILTIN_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_BUILTIN_HPP__

#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cstddef>
#include <cstdbool>
#include <cfloat>

#if !defined(DT_VOID_DEFINED)
#define DT_VOID_DEFINED
typedef void            DT_VOID;
#endif // DT_VOID_DEFINED

#if !defined(DT_CHAR_DEFINED)
#define DT_CHAR_DEFINED
typedef char            DT_CHAR;
#endif // DT_CHAR_DEFINED

#if !defined(DT_UCHAR_DEFINED)
#define DT_UCHAR_DEFINED
typedef unsigned char   DT_UCHAR;
#endif // DT_UCHAR_DEFINED

#if !defined(DT_NULL_DEFINED)
#define DT_NULL_DEFINED
#define DT_NULL         nullptr
#endif // DT_NULL_DEFINED

#if !defined(DT_BOOL_DEFINED)
#define DT_BOOL_DEFINED
#define DT_BOOL         bool
#endif // DT_BOOL_DEFINED

#if !defined(DT_TRUE_DEFINED)
#define DT_TRUE_DEFINED
#define DT_TRUE         true
#endif // DT_TRUE_DEFINED

#if !defined(DT_FALSE_DEFINED)
#define DT_FALSE_DEFINED
#define DT_FALSE        false
#endif // DT_FALSE_DEFINED

#if !defined(DT_S8_DEFINED)
#define DT_S8_DEFINED
typedef int8_t          DT_S8;
#endif // DT_S8_DEFINED

#if !defined(DT_U8_DEFINED)
#define DT_U8_DEFINED
typedef uint8_t         DT_U8;
#endif // DT_U8_DEFINED

#if !defined(DT_S16_DEFINED)
#define DT_S16_DEFINED
typedef int16_t         DT_S16;
#endif // DT_S16_DEFINED

#if !defined(DT_U16_DEFINED)
#define DT_U16_DEFINED
typedef uint16_t        DT_U16;
#endif // DT_U16_DEFINED

#if !defined(DT_S32_DEFINED)
#define DT_S32_DEFINED
typedef int32_t         DT_S32;
#endif // DT_S32_DEFINED

#if !defined(DT_U32_DEFINED)
#define DT_U32_DEFINED
typedef uint32_t        DT_U32;
#endif // DT_U32_DEFINED

#if !defined(DT_S64_DEFINED)
#define DT_S64_DEFINED
typedef int64_t         DT_S64;
#endif // DT_S64_DEFINED

#if !defined(DT_U64_DEFINED)
#define DT_U64_DEFINED
typedef uint64_t        DT_U64;
#endif // DT_U64_DEFINED

#if !defined(DT_F32_DEFINED)
#define DT_F32_DEFINED
typedef float           DT_F32;
#endif // DT_F32_DEFINED

#if !defined(DT_F64_DEFINED)
#define DT_F64_DEFINED
typedef double          DT_F64;
#endif // DT_F64_DEFINED

#if !defined(DT_SPTR_T_DEFINED)
#define DT_SPTR_T_DEFINED
typedef std::intptr_t    DT_SPTR_T;
#endif // DT_SPTR_T_DEFINED

#if !defined(DT_UPTR_T_DEFINED)
#define DT_UPTR_T_DEFINED
typedef std::uintptr_t   DT_UPTR_T;
#endif // DT_UPTR_T_DEFINED

#if !defined(DT_PTR_DIFF_DEFINED)
#define DT_PTR_DIFF_DEFINED
typedef std::ptrdiff_t   DT_PTR_DIFF;
#endif // DT_PTR_DIFF_DEFINED

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_BUILTIN_HPP__