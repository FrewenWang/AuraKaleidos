#include "aura/runtime/core.h"

#include "tileManager.h"
#include "tileManager_FIK_api.h"

extern application_symbol_tray *g_symbol_tray;

namespace aura
{
namespace xtensa
{

DT_S32 Print(const DT_CHAR *format, ...)
{
    va_list args;
    va_start(args, format);
    DT_S32 ret = g_symbol_tray->tray_vprintf(format, args);
    va_end(args);
    return ret;
}

DT_VOID DCacheInvalidate(const DT_VOID *addr, DT_U32 size)
{
    return g_symbol_tray->tray_xthal_dcache_region_invalidate(const_cast<void*>(addr), size);
}

DT_VOID DCacheWriteback(const DT_VOID *addr, DT_U32 size)
{
    return g_symbol_tray->tray_xthal_dcache_region_writeback_inv(const_cast<void*>(addr), size);
}

DT_VOID* Memcpy(DT_VOID *dst, const DT_VOID *src, size_t size)
{
    return g_symbol_tray->tray_memcpy(dst, src, size);
}

DT_VOID* Memset(DT_VOID *data, DT_S32 value, size_t size)
{
    return g_symbol_tray->tray_memset(data, value, size);
}

DT_S32 Strcmp(const DT_CHAR *str1, const DT_CHAR *str2)
{
    return g_symbol_tray->tray_strcmp(str1, str2);
}

size_t Strlen(const DT_CHAR *str)
{
    return g_symbol_tray->tray_strlen(str);
}

DT_CHAR* Strcpy(DT_CHAR *dst, const DT_CHAR *src)
{
    return g_symbol_tray->tray_strcpy(dst, src);
}

DT_VOID* Memmove(DT_VOID *dst, const DT_VOID *src, size_t size)
{
    return g_symbol_tray->tray_memmove(dst, src, size);
}

const DT_CHAR* Strstr(const DT_CHAR *str1, const DT_CHAR *str2)
{
    return g_symbol_tray->tray_strstr(str1, str2);
}

DT_F32 Modff(DT_F32 value, DT_F32* iptr)
{
    return g_symbol_tray->tray_modff(value, iptr);
}

DT_F64 Modf(DT_F64 value, DT_F64* iptr)
{
    return g_symbol_tray->tray_modf(value, iptr);
}

DT_F32 Fabsf(DT_F32 value)
{
    return g_symbol_tray->tray_fabsf(value);
}

DT_F64 Fabs(DT_F64 value)
{
    return g_symbol_tray->tray_fabs(value);
}

DT_F32 Sqrtf(DT_F32 value)
{
    return g_symbol_tray->tray_sqrtf(value);
}

DT_F64 Sqrt(DT_F64 value)
{
    return g_symbol_tray->tray_sqrt(value);
}

DT_F32 Expf(DT_F32 value)
{
    return g_symbol_tray->tray_expf(value);
}

DT_F64 Exp(DT_F64 value)
{
    return g_symbol_tray->tray_exp(value);
}

DT_F32 Exp2f(DT_F32 value)
{
    return g_symbol_tray->tray_exp2f(value);
}

DT_F64 Exp2(DT_F64 value)
{
    return g_symbol_tray->tray_exp2(value);
}

DT_F32 Logf(DT_F32 value)
{
    return g_symbol_tray->tray_logf(value);
}

DT_F64 Log(DT_F64 value)
{
    return g_symbol_tray->tray_log(value);
}

DT_F32 Log2f(DT_F32 value)
{
    return g_symbol_tray->tray_log2f(value);
}

DT_F64 Log2(DT_F64 value)
{
    return g_symbol_tray->tray_log2(value);
}

DT_F32 Log10f(DT_F32 value)
{
    return g_symbol_tray->tray_log10f(value);
}

DT_F64 Log10(DT_F64 value)
{
    return g_symbol_tray->tray_log10(value);
}

DT_F32 Powf(DT_F32 base, DT_F32 exponent)
{
    return g_symbol_tray->tray_powf(base, exponent);
}

DT_F64 Pow(DT_F64 base, DT_F64 exponent)
{
    return g_symbol_tray->tray_pow(base, exponent);
}

DT_F32 Sinf(DT_F32 value)
{
    return g_symbol_tray->tray_sinf(value);
}

DT_F64 Sin(DT_F64 value)
{
    return g_symbol_tray->tray_sin(value);
}

DT_F32 Cosf(DT_F32 value)
{
    return g_symbol_tray->tray_cosf(value);
}

DT_F64 Cos(DT_F64 value)
{
    return g_symbol_tray->tray_cos(value);
}

DT_F32 Tanf(DT_F32 value)
{
    return g_symbol_tray->tray_tanf(value);
}

DT_F64 Tan(DT_F64 value)
{
    return g_symbol_tray->tray_tan(value);
}

DT_F32 Asinf(DT_F32 value)
{
    return g_symbol_tray->tray_asinf(value);
}

DT_F64 Asin(DT_F64 value)
{
    return g_symbol_tray->tray_asin(value);
}

DT_F32 Acosf(DT_F32 value)
{
    return g_symbol_tray->tray_acosf(value);
}

DT_F64 Acos(DT_F64 value)
{
    return g_symbol_tray->tray_acos(value);
}

DT_F32 Atanf(DT_F32 value)
{
    return g_symbol_tray->tray_atanf(value);
}

DT_F64 Atan(DT_F64 value)
{
    return g_symbol_tray->tray_atan(value);
}

DT_F32 Atan2f(DT_F32 y, DT_F32 value)
{
    return g_symbol_tray->tray_atan2f(y, value);
}

DT_F64 Atan2(DT_F64 y, DT_F64 value)
{
    return g_symbol_tray->tray_atan2(y, value);
}

} // namespace xtensa
} // namepsace aura