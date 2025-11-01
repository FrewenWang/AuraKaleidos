#include "aura/runtime/core.h"

#include "tileManager.h"
#include "tileManager_FIK_api.h"

extern application_symbol_tray *g_symbol_tray;

namespace aura
{
namespace xtensa
{

MI_S32 Print(const MI_CHAR *format, ...)
{
    va_list args;
    va_start(args, format);
    MI_S32 ret = g_symbol_tray->tray_vprintf(format, args);
    va_end(args);
    return ret;
}

AURA_VOID DCacheInvalidate(const AURA_VOID *addr, MI_U32 size)
{
    return g_symbol_tray->tray_xthal_dcache_region_invalidate(const_cast<void*>(addr), size);
}

AURA_VOID DCacheWriteback(const AURA_VOID *addr, MI_U32 size)
{
    return g_symbol_tray->tray_xthal_dcache_region_writeback_inv(const_cast<void*>(addr), size);
}

AURA_VOID* Memcpy(AURA_VOID *dst, const AURA_VOID *src, size_t size)
{
    return g_symbol_tray->tray_memcpy(dst, src, size);
}

AURA_VOID* Memset(AURA_VOID *data, MI_S32 value, size_t size)
{
    return g_symbol_tray->tray_memset(data, value, size);
}

MI_S32 Strcmp(const MI_CHAR *str1, const MI_CHAR *str2)
{
    return g_symbol_tray->tray_strcmp(str1, str2);
}

size_t Strlen(const MI_CHAR *str)
{
    return g_symbol_tray->tray_strlen(str);
}

MI_CHAR* Strcpy(MI_CHAR *dst, const MI_CHAR *src)
{
    return g_symbol_tray->tray_strcpy(dst, src);
}

AURA_VOID* Memmove(AURA_VOID *dst, const AURA_VOID *src, size_t size)
{
    return g_symbol_tray->tray_memmove(dst, src, size);
}

const MI_CHAR* Strstr(const MI_CHAR *str1, const MI_CHAR *str2)
{
    return g_symbol_tray->tray_strstr(str1, str2);
}

MI_F32 Modff(MI_F32 value, MI_F32* iptr)
{
    return g_symbol_tray->tray_modff(value, iptr);
}

MI_F64 Modf(MI_F64 value, MI_F64* iptr)
{
    return g_symbol_tray->tray_modf(value, iptr);
}

MI_F32 Fabsf(MI_F32 value)
{
    return g_symbol_tray->tray_fabsf(value);
}

MI_F64 Fabs(MI_F64 value)
{
    return g_symbol_tray->tray_fabs(value);
}

MI_F32 Sqrtf(MI_F32 value)
{
    return g_symbol_tray->tray_sqrtf(value);
}

MI_F64 Sqrt(MI_F64 value)
{
    return g_symbol_tray->tray_sqrt(value);
}

MI_F32 Expf(MI_F32 value)
{
    return g_symbol_tray->tray_expf(value);
}

MI_F64 Exp(MI_F64 value)
{
    return g_symbol_tray->tray_exp(value);
}

MI_F32 Exp2f(MI_F32 value)
{
    return g_symbol_tray->tray_exp2f(value);
}

MI_F64 Exp2(MI_F64 value)
{
    return g_symbol_tray->tray_exp2(value);
}

MI_F32 Logf(MI_F32 value)
{
    return g_symbol_tray->tray_logf(value);
}

MI_F64 Log(MI_F64 value)
{
    return g_symbol_tray->tray_log(value);
}

MI_F32 Log2f(MI_F32 value)
{
    return g_symbol_tray->tray_log2f(value);
}

MI_F64 Log2(MI_F64 value)
{
    return g_symbol_tray->tray_log2(value);
}

MI_F32 Log10f(MI_F32 value)
{
    return g_symbol_tray->tray_log10f(value);
}

MI_F64 Log10(MI_F64 value)
{
    return g_symbol_tray->tray_log10(value);
}

MI_F32 Powf(MI_F32 base, MI_F32 exponent)
{
    return g_symbol_tray->tray_powf(base, exponent);
}

MI_F64 Pow(MI_F64 base, MI_F64 exponent)
{
    return g_symbol_tray->tray_pow(base, exponent);
}

MI_F32 Sinf(MI_F32 value)
{
    return g_symbol_tray->tray_sinf(value);
}

MI_F64 Sin(MI_F64 value)
{
    return g_symbol_tray->tray_sin(value);
}

MI_F32 Cosf(MI_F32 value)
{
    return g_symbol_tray->tray_cosf(value);
}

MI_F64 Cos(MI_F64 value)
{
    return g_symbol_tray->tray_cos(value);
}

MI_F32 Tanf(MI_F32 value)
{
    return g_symbol_tray->tray_tanf(value);
}

MI_F64 Tan(MI_F64 value)
{
    return g_symbol_tray->tray_tan(value);
}

MI_F32 Asinf(MI_F32 value)
{
    return g_symbol_tray->tray_asinf(value);
}

MI_F64 Asin(MI_F64 value)
{
    return g_symbol_tray->tray_asin(value);
}

MI_F32 Acosf(MI_F32 value)
{
    return g_symbol_tray->tray_acosf(value);
}

MI_F64 Acos(MI_F64 value)
{
    return g_symbol_tray->tray_acos(value);
}

MI_F32 Atanf(MI_F32 value)
{
    return g_symbol_tray->tray_atanf(value);
}

MI_F64 Atan(MI_F64 value)
{
    return g_symbol_tray->tray_atan(value);
}

MI_F32 Atan2f(MI_F32 y, MI_F32 value)
{
    return g_symbol_tray->tray_atan2f(y, value);
}

MI_F64 Atan2(MI_F64 y, MI_F64 value)
{
    return g_symbol_tray->tray_atan2(y, value);
}

} // namespace xtensa
} // namepsace aura