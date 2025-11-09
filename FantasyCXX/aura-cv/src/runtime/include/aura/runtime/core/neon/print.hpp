#ifndef AURA_RUNTIME_CORE_NEON_PRINT_HPP__
#define AURA_RUNTIME_CORE_NEON_PRINT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

template <typename Tp>
inline DT_VOID FormatPrintDec(const DT_CHAR *s, Tp *v, DT_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (DT_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %02d", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp>
inline DT_VOID FormatPrintHex(const DT_CHAR *s, Tp *v, DT_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (DT_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %04x", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp>
inline DT_VOID FormatPrintFp(const DT_CHAR *s, Tp *v, DT_S32 n)
{
    fprintf(stderr, "%s:", s);
    for (DT_S32 i = 0; i < n; i++)
    {
        fprintf(stderr, " %.4f", v[i]);
    }
    fprintf(stderr, "\n");
}

template <typename Tp> inline DT_VOID PrintDec(const DT_CHAR *s, const Tp &v)
{
    fprintf(stderr, "%s:", s);
    fprintf(stderr, "\n");
    (DT_VOID)v;
}

template <> inline DT_VOID PrintDec<uint8x8_t>(const DT_CHAR *s, const uint8x8_t &v)
{
    DT_U8 *ptr = (DT_U8 *)(&v);
    FormatPrintDec<DT_U8>(s, ptr, 8);
}

template <> inline DT_VOID PrintDec<int8x8_t>(const DT_CHAR *s, const int8x8_t &v)
{
    DT_S8 *ptr = (DT_S8 *)(&v);
    FormatPrintDec<DT_S8>(s, ptr, 8);
}

template <> inline DT_VOID PrintDec<uint8x16_t>(const DT_CHAR *s, const uint8x16_t &v)
{
    DT_U8 *ptr = (DT_U8 *)(&v);
    FormatPrintDec<DT_U8>(s, ptr, 16);
}

template <> inline DT_VOID PrintDec<int8x16_t>(const DT_CHAR *s, const int8x16_t &v)
{
    DT_S8 *ptr = (DT_S8 *)(&v);
    FormatPrintDec<DT_S8>(s, ptr, 16);
}

template <> inline DT_VOID PrintDec<uint16x4_t>(const DT_CHAR *s, const uint16x4_t &v)
{
    DT_U16 *ptr = (DT_U16 *)(&v);
    FormatPrintDec<DT_U16>(s, ptr, 4);
}

template <> inline DT_VOID PrintDec<int16x4_t>(const DT_CHAR *s, const int16x4_t &v)
{
    DT_S16 *ptr = (DT_S16 *)(&v);
    FormatPrintDec<DT_S16>(s, ptr, 4);
}

template <> inline DT_VOID PrintDec<uint16x8_t>(const DT_CHAR *s, const uint16x8_t &v)
{
    DT_U16 *ptr = (DT_U16 *)(&v);
    FormatPrintDec<DT_U16>(s, ptr, 8);
}

template <> inline DT_VOID PrintDec<int16x8_t>(const DT_CHAR *s, const int16x8_t &v)
{
    DT_S16 *ptr = (DT_S16 *)(&v);
    FormatPrintDec<DT_S16>(s, ptr, 8);
}

template <> inline DT_VOID PrintDec<uint32x2_t>(const DT_CHAR *s, const uint32x2_t &v)
{
    DT_U32 *ptr = (DT_U32 *)(&v);
    FormatPrintDec<DT_U32>(s, ptr, 2);
}

template <> inline DT_VOID PrintDec<int32x2_t>(const DT_CHAR *s, const int32x2_t &v)
{
    DT_S32 *ptr = (DT_S32 *)(&v);
    FormatPrintDec<DT_S32>(s, ptr, 2);
}

template <> inline DT_VOID PrintDec<uint32x4_t>(const DT_CHAR *s, const uint32x4_t &v)
{
    DT_U32 *ptr = (DT_U32 *)(&v);
    FormatPrintDec<DT_U32>(s, ptr, 4);
}

template <> inline DT_VOID PrintDec<int32x4_t>(const DT_CHAR *s, const int32x4_t &v)
{
    DT_S32 *ptr = (DT_S32 *)(&v);
    FormatPrintDec<DT_S32>(s, ptr, 4);
}

template <> inline DT_VOID PrintDec<float32x2_t>(const DT_CHAR *s, const float32x2_t &v)
{
    DT_F32 *ptr = (DT_F32 *)(&v);
    FormatPrintFp<DT_F32>(s, ptr, 2);
}

template <> inline DT_VOID PrintDec<float32x4_t>(const DT_CHAR *s, const float32x4_t &v)
{
    DT_F32 *ptr = (DT_F32 *)(&v);
    FormatPrintFp<DT_F32>(s, ptr, 4);
}

#if defined(AURA_ENABLE_NEON_FP16)
template <> inline DT_VOID PrintDec<float16x4_t>(const DT_CHAR *s, const float16x4_t &v)
{
    float16_t *ptr = (float16_t *)(&v);
    FormatPrintFp<float16_t>(s, ptr, 4);
}

template <> inline DT_VOID PrintDec<float16x8_t>(const DT_CHAR *s, const float16x8_t &v)
{
    float16_t *ptr = (float16_t *)(&v);
    FormatPrintFp<float16_t>(s, ptr, 8);
}
#endif

template <typename Tp> inline DT_VOID PrintHex(const DT_CHAR *s, const Tp &v)
{
    fprintf(stderr, "%s:", s);
    fprintf(stderr, "\n");
    (DT_VOID)v;
}

template <> inline DT_VOID PrintHex<uint8x8_t>(const DT_CHAR *s, const uint8x8_t &v)
{
    DT_U8 *ptr = (DT_U8 *)(&v);
    FormatPrintHex<DT_U8>(s, ptr, 8);
}

template <> inline DT_VOID PrintHex<int8x8_t>(const DT_CHAR *s, const int8x8_t &v)
{
    DT_S8 *ptr = (DT_S8 *)(&v);
    FormatPrintHex<DT_S8>(s, ptr, 8);
}

template <> inline DT_VOID PrintHex<uint8x16_t>(const DT_CHAR *s, const uint8x16_t &v)
{
    DT_U8 *ptr = (DT_U8 *)(&v);
    FormatPrintHex<DT_U8>(s, ptr, 16);
}

template <> inline DT_VOID PrintHex<int8x16_t>(const DT_CHAR *s, const int8x16_t &v)
{
    DT_S8 *ptr = (DT_S8 *)(&v);
    FormatPrintHex<DT_S8>(s, ptr, 16);
}

template <> inline DT_VOID PrintHex<uint16x4_t>(const DT_CHAR *s, const uint16x4_t &v)
{
    DT_U16 *ptr = (DT_U16 *)(&v);
    FormatPrintHex<DT_U16>(s, ptr, 4);
}

template <> inline DT_VOID PrintHex<int16x4_t>(const DT_CHAR *s, const int16x4_t &v)
{
    DT_S16 *ptr = (DT_S16 *)(&v);
    FormatPrintHex<DT_S16>(s, ptr, 4);
}

template <> inline DT_VOID PrintHex<uint16x8_t>(const DT_CHAR *s, const uint16x8_t &v)
{
    DT_U16 *ptr = (DT_U16 *)(&v);
    FormatPrintHex<DT_U16>(s, ptr, 8);
}

template <> inline DT_VOID PrintHex<int16x8_t>(const DT_CHAR *s, const int16x8_t &v)
{
    DT_S16 *ptr = (DT_S16 *)(&v);
    FormatPrintHex<DT_S16>(s, ptr, 8);
}

template <> inline DT_VOID PrintHex<uint32x2_t>(const DT_CHAR *s, const uint32x2_t &v)
{
    DT_U32 *ptr = (DT_U32 *)(&v);
    FormatPrintHex<DT_U32>(s, ptr, 2);
}

template <> inline DT_VOID PrintHex<int32x2_t>(const DT_CHAR *s, const int32x2_t &v)
{
    DT_S32 *ptr = (DT_S32 *)(&v);
    FormatPrintHex<DT_S32>(s, ptr, 2);
}

template <> inline DT_VOID PrintHex<uint32x4_t>(const DT_CHAR *s, const uint32x4_t &v)
{
    DT_U32 *ptr = (DT_U32 *)(&v);
    FormatPrintHex<DT_U32>(s, ptr, 4);
}

template <> inline DT_VOID PrintHex<int32x4_t>(const DT_CHAR *s, const int32x4_t &v)
{
    DT_S32 *ptr = (DT_S32 *)(&v);
    FormatPrintHex<DT_S32>(s, ptr, 4);
}

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_PRINT_HPP__