#ifndef AURA_UTILS_CORE_TYPES_BUILTIN_HPP__
#define AURA_UTILS_CORE_TYPES_BUILTIN_HPP__

#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cstddef>
#include <cstdbool>
#include <cfloat>

#ifndef AURA_VOID_DEFINED
#define AURA_VOID_DEFINED
typedef void AURA_VOID;
#endif // AURA_VOID_DEFINED

#ifndef AURA_CHAR_DEFINED
#define AURA_CHAR_DEFINED
typedef char AURA_CHAR;
#endif // AURA_CHAR_DEFINED

#ifndef AURA_UCHAR_DEFINED
#define AURA_UCHAR_DEFINED
typedef unsigned char AURA_UCHAR;
#endif // AURA_UCHAR_DEFINED

#ifndef AURA_NULL_DEFINED
#define AURA_NULL_DEFINED
#define AURA_NULL         nullptr
#endif // AURA_NULL_DEFINED

#ifndef AURA_BOOL_DEFINED
#define AURA_BOOL_DEFINED
#define AURA_BOOL         bool
#endif // AURA_BOOL_DEFINED

#ifndef AURA_TRUE_DEFINED
#define AURA_TRUE_DEFINED
#define AURA_TRUE         true
#endif // AURA_TRUE_DEFINED

#ifndef AURA_FALSE_DEFINED
#define AURA_FALSE_DEFINED
#define AURA_FALSE        false
#endif // AURA_FALSE_DEFINED

#ifndef AURA_S8_DEFINED
#define AURA_S8_DEFINED
typedef int8_t AURA_S8;
#endif // AURA_S8_DEFINED

#ifndef AURA_U8_DEFINED
#define AURA_U8_DEFINED
typedef uint8_t AURA_U8;
#endif // AURA_U8_DEFINED

#ifndef AURA_S16_DEFINED
#define AURA_S16_DEFINED
typedef int16_t AURA_S16;
#endif // AURA_S16_DEFINED

#ifndef AURA_U16_DEFINED
#define AURA_U16_DEFINED
typedef uint16_t AURA_U16;
#endif // AURA_U16_DEFINED

#ifndef AURA_S32_DEFINED
#define AURA_S32_DEFINED
typedef int32_t AURA_S32;
#endif // AURA_S32_DEFINED

#ifndef AURA_U32_DEFINED
#define AURA_U32_DEFINED
typedef uint32_t AURA_U32;
#endif // AURA_U32_DEFINED

#ifndef AURA_S64_DEFINED
#define AURA_S64_DEFINED
typedef int64_t AURA_S64;
#endif // AURA_S64_DEFINED

#ifndef AURA_U64_DEFINED
#define AURA_U64_DEFINED
typedef uint64_t AURA_U64;
#endif // AURA_U64_DEFINED

#ifndef AURA_F32_DEFINED
#define AURA_F32_DEFINED
typedef float AURA_F32;
#endif // AURA_F32_DEFINED

#ifndef AURA_F64_DEFINED
#define AURA_F64_DEFINED
typedef double AURA_F64;
#endif // AURA_F64_DEFINED

#ifndef AURA_SPTR_T_DEFINED
#define AURA_SPTR_T_DEFINED
typedef std::intptr_t AURA_SPTR_T;
#endif // AURA_SPTR_T_DEFINED

#ifndef AURA_UPTR_T_DEFINED
#define AURA_UPTR_T_DEFINED
typedef std::uintptr_t AURA_UPTR_T;
#endif // AURA_UPTR_T_DEFINED

#ifndef AURA_PTR_DIFF_DEFINED
#define AURA_PTR_DIFF_DEFINED
typedef std::ptrdiff_t AURA_PTR_DIFF;
#endif // AURA_PTR_DIFF_DEFINED

#endif // AURA_UTILS_CORE_TYPES_BUILTIN_HPP__