#ifndef AURA_RUNTIME_CORE_NEON_PRINT_HPP__
#define AURA_RUNTIME_CORE_NEON_PRINT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

template <typename Tp>
inline AURA_VOID FormatPrintDec(const MI_CHAR *s, Tp *v, MI_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (MI_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %02d", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp>
inline AURA_VOID FormatPrintHex(const MI_CHAR *s, Tp *v, MI_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (MI_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %04x", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp>
inline AURA_VOID FormatPrintFp(const MI_CHAR *s, Tp *v, MI_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (MI_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %.4f", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp> inline AURA_VOID PrintDec(const MI_CHAR *s, const Tp &v)
{
    fprintf(stderr, "%s:", s);
    fprintf(stderr, "\n");
    (AURA_VOID)v;
}

template <> inline AURA_VOID PrintDec<uint8x8_t>(const MI_CHAR *s, const uint8x8_t &v)
{
    MI_U8 *ptr = (MI_U8 *)(&v);
    FormatPrintDec<MI_U8>(s, ptr, 8);
}

template <> inline AURA_VOID PrintDec<int8x8_t>(const MI_CHAR *s, const int8x8_t &v)
{
    MI_S8 *ptr = (MI_S8 *)(&v);
    FormatPrintDec<MI_S8>(s, ptr, 8);
}

template <> inline AURA_VOID PrintDec<uint8x16_t>(const MI_CHAR *s, const uint8x16_t &v)
{
    MI_U8 *ptr = (MI_U8 *)(&v);
    FormatPrintDec<MI_U8>(s, ptr, 16);
}

template <> inline AURA_VOID PrintDec<int8x16_t>(const MI_CHAR *s, const int8x16_t &v)
{
    MI_S8 *ptr = (MI_S8 *)(&v);
    FormatPrintDec<MI_S8>(s, ptr, 16);
}

template <> inline AURA_VOID PrintDec<uint16x4_t>(const MI_CHAR *s, const uint16x4_t &v)
{
    MI_U16 *ptr = (MI_U16 *)(&v);
    FormatPrintDec<MI_U16>(s, ptr, 4);
}

template <> inline AURA_VOID PrintDec<int16x4_t>(const MI_CHAR *s, const int16x4_t &v)
{
    MI_S16 *ptr = (MI_S16 *)(&v);
    FormatPrintDec<MI_S16>(s, ptr, 4);
}

template <> inline AURA_VOID PrintDec<uint16x8_t>(const MI_CHAR *s, const uint16x8_t &v)
{
    MI_U16 *ptr = (MI_U16 *)(&v);
    FormatPrintDec<MI_U16>(s, ptr, 8);
}

template <> inline AURA_VOID PrintDec<int16x8_t>(const MI_CHAR *s, const int16x8_t &v)
{
    MI_S16 *ptr = (MI_S16 *)(&v);
    FormatPrintDec<MI_S16>(s, ptr, 8);
}

template <> inline AURA_VOID PrintDec<uint32x2_t>(const MI_CHAR *s, const uint32x2_t &v)
{
    MI_U32 *ptr = (MI_U32 *)(&v);
    FormatPrintDec<MI_U32>(s, ptr, 2);
}

template <> inline AURA_VOID PrintDec<int32x2_t>(const MI_CHAR *s, const int32x2_t &v)
{
    MI_S32 *ptr = (MI_S32 *)(&v);
    FormatPrintDec<MI_S32>(s, ptr, 2);
}

template <> inline AURA_VOID PrintDec<uint32x4_t>(const MI_CHAR *s, const uint32x4_t &v)
{
    MI_U32 *ptr = (MI_U32 *)(&v);
    FormatPrintDec<MI_U32>(s, ptr, 4);
}

template <> inline AURA_VOID PrintDec<int32x4_t>(const MI_CHAR *s, const int32x4_t &v)
{
    MI_S32 *ptr = (MI_S32 *)(&v);
    FormatPrintDec<MI_S32>(s, ptr, 4);
}

template <> inline AURA_VOID PrintDec<float32x2_t>(const MI_CHAR *s, const float32x2_t &v)
{
    MI_F32 *ptr = (MI_F32 *)(&v);
    FormatPrintFp<MI_F32>(s, ptr, 2);
}

template <> inline AURA_VOID PrintDec<float32x4_t>(const MI_CHAR *s, const float32x4_t &v)
{
    MI_F32 *ptr = (MI_F32 *)(&v);
    FormatPrintFp<MI_F32>(s, ptr, 4);
}

#if defined(AURA_ENABLE_NEON_FP16)
template <> inline AURA_VOID PrintDec<float16x4_t>(const MI_CHAR *s, const float16x4_t &v)
{
    float16_t *ptr = (float16_t *)(&v);
    FormatPrintFp<float16_t>(s, ptr, 4);
}

template <> inline AURA_VOID PrintDec<float16x8_t>(const MI_CHAR *s, const float16x8_t &v)
{
    float16_t *ptr = (float16_t *)(&v);
    FormatPrintFp<float16_t>(s, ptr, 8);
}
#endif

template <typename Tp> inline AURA_VOID PrintHex(const MI_CHAR *s, const Tp &v)
{
    fprintf(stderr, "%s:", s);
    fprintf(stderr, "\n");
    (AURA_VOID)v;
}

template <> inline AURA_VOID PrintHex<uint8x8_t>(const MI_CHAR *s, const uint8x8_t &v)
{
    MI_U8 *ptr = (MI_U8 *)(&v);
    FormatPrintHex<MI_U8>(s, ptr, 8);
}

template <> inline AURA_VOID PrintHex<int8x8_t>(const MI_CHAR *s, const int8x8_t &v)
{
    MI_S8 *ptr = (MI_S8 *)(&v);
    FormatPrintHex<MI_S8>(s, ptr, 8);
}

template <> inline AURA_VOID PrintHex<uint8x16_t>(const MI_CHAR *s, const uint8x16_t &v)
{
    MI_U8 *ptr = (MI_U8 *)(&v);
    FormatPrintHex<MI_U8>(s, ptr, 16);
}

template <> inline AURA_VOID PrintHex<int8x16_t>(const MI_CHAR *s, const int8x16_t &v)
{
    MI_S8 *ptr = (MI_S8 *)(&v);
    FormatPrintHex<MI_S8>(s, ptr, 16);
}

template <> inline AURA_VOID PrintHex<uint16x4_t>(const MI_CHAR *s, const uint16x4_t &v)
{
    MI_U16 *ptr = (MI_U16 *)(&v);
    FormatPrintHex<MI_U16>(s, ptr, 4);
}

template <> inline AURA_VOID PrintHex<int16x4_t>(const MI_CHAR *s, const int16x4_t &v)
{
    MI_S16 *ptr = (MI_S16 *)(&v);
    FormatPrintHex<MI_S16>(s, ptr, 4);
}

template <> inline AURA_VOID PrintHex<uint16x8_t>(const MI_CHAR *s, const uint16x8_t &v)
{
    MI_U16 *ptr = (MI_U16 *)(&v);
    FormatPrintHex<MI_U16>(s, ptr, 8);
}

template <> inline AURA_VOID PrintHex<int16x8_t>(const MI_CHAR *s, const int16x8_t &v)
{
    MI_S16 *ptr = (MI_S16 *)(&v);
    FormatPrintHex<MI_S16>(s, ptr, 8);
}

template <> inline AURA_VOID PrintHex<uint32x2_t>(const MI_CHAR *s, const uint32x2_t &v)
{
    MI_U32 *ptr = (MI_U32 *)(&v);
    FormatPrintHex<MI_U32>(s, ptr, 2);
}

template <> inline AURA_VOID PrintHex<int32x2_t>(const MI_CHAR *s, const int32x2_t &v)
{
    MI_S32 *ptr = (MI_S32 *)(&v);
    FormatPrintHex<MI_S32>(s, ptr, 2);
}

template <> inline AURA_VOID PrintHex<uint32x4_t>(const MI_CHAR *s, const uint32x4_t &v)
{
    MI_U32 *ptr = (MI_U32 *)(&v);
    FormatPrintHex<MI_U32>(s, ptr, 4);
}

template <> inline AURA_VOID PrintHex<int32x4_t>(const MI_CHAR *s, const int32x4_t &v)
{
    MI_S32 *ptr = (MI_S32 *)(&v);
    FormatPrintHex<MI_S32>(s, ptr, 4);
}

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_PRINT_HPP__