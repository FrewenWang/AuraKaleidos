#ifndef AURA_RUNTIME_CORE_XTENSA_ISA_LOAD_HPP__
#define AURA_RUNTIME_CORE_XTENSA_ISA_LOAD_HPP__

#include "aura/runtime/core/xtensa/isa/traits.hpp"

#include <xtensa/tie/xt_ivpn.h>

namespace aura
{
namespace xtensa
{

extern const DT_U8 sel_deinterleave_d16_c3_step_0[]     __attribute__((aligned(128)));
extern const DT_U8 sel_deinterleave_d16_c3_step_1[]     __attribute__((aligned(128)));
extern const DT_U8 sel_deinterleave_d16_c3_step_1_msk[] __attribute__((aligned(128)));
extern const DT_U8 dsel_deinterleave_d8_1[]             __attribute__((aligned(128)));
extern const DT_U8 dsel_deinterleave_d16_1[]            __attribute__((aligned(128)));
extern const DT_U8 sel_interleave_d16_c3_step_0[]       __attribute__((aligned(128)));
extern const DT_U8 sel_interleave_d16_c3_step_1[]       __attribute__((aligned(128)));
extern const DT_U8 sel_interleave_d16_c3_step_1_msk[]   __attribute__((aligned(128)));
extern const DT_U8 dsel_interleave_d8_1[]               __attribute__((aligned(128)));
extern const DT_U8 dsel_interleave_d16_1[]              __attribute__((aligned(128)));

#define DECLFUN(vtype, postfix)                                                                      \
    inline DT_VOID align(valign &align, vtype *__restrict addr)                                      \
    {                                                                                                \
        align = IVP_LA##postfix##_PP(addr);                                                          \
    }

DECLFUN(xb_vecNx8U,     NX8U)
DECLFUN(xb_vecNx8,      NX8S)
DECLFUN(xb_vec2Nx8U,    2NX8U)
DECLFUN(xb_vec2Nx8,     2NX8)
DECLFUN(xb_vecNx16U,    NX16U)
DECLFUN(xb_vecNx16,     NX16)
DECLFUN(xb_vecN_2x32Uv, N_2X32U)
DECLFUN(xb_vecN_2x32v,  N_2X32)
#undef DECLFUN

#define DECLFUN(vtype, postfix)                                                                      \
    inline DT_VOID vflush(valign &align, vtype *__restrict addr)                                     \
    {                                                                                                \
        IVP_SAPOS##postfix##_FP(align, addr);                                                        \
    }

DECLFUN(xb_vecNx8U,     NX8U)
DECLFUN(xb_vecNx8,      NX8S)
DECLFUN(xb_vec2Nx8U,    2NX8U)
DECLFUN(xb_vec2Nx8,     2NX8)
DECLFUN(xb_vecNx16U,    NX16U)
DECLFUN(xb_vecNx16,     NX16)
DECLFUN(xb_vecN_2x32Uv, N_2X32U)
DECLFUN(xb_vecN_2x32v,  N_2X32)
#undef DECLFUN

#define DECLFUN(vtype, postfix)                                                                       \
    inline DT_VOID vload(vtype *__restrict &addr, vtype &v_out, DT_S32 byte_num)                      \
    {                                                                                                 \
        valign v_load;                                                                                \
        align(v_load, addr);                                                                          \
        IVP_LAV##postfix##_XP(v_out, v_load, addr, byte_num);                                         \
    }

DECLFUN(xb_vec2Nx8U,    2NX8U)
DECLFUN(xb_vec2Nx8,     2NX8)
DECLFUN(xb_vecNx16U,    NX16U)
DECLFUN(xb_vecNx16,     NX16)
DECLFUN(xb_vecN_2x32Uv, N_2X32U)
DECLFUN(xb_vecN_2x32v,  N_2X32)
#undef DECLFUN

#define DECLFUN(vtype, postfix)                                                                       \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, vtype &v_in, DT_S32 byte_num)     \
    {                                                                                                 \
        IVP_SAV##postfix##_XP(v_in, v_store, addr, byte_num);                                         \
    }

DECLFUN(xb_vec2Nx8U,    2NX8U)
DECLFUN(xb_vec2Nx8,     2NX8)
DECLFUN(xb_vecNx16U,    NX16U)
DECLFUN(xb_vecNx16,     NX16)
DECLFUN(xb_vecN_2x32Uv, N_2X32U)
DECLFUN(xb_vecN_2x32v,  N_2X32)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                               \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                     \
    {                                                                                                 \
        valign v_load;                                                                                \
        align(v_load, addr);                                                                          \
        IVP_LAV##postfix##_XP(v_out.val[0], v_load, addr, byte_num);                                  \
    }

DECLFUN(VdspVectorU8X1,  xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X1,  xb_vec2Nx8,  2NX8)
DECLFUN(VdspVectorU16X1, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X1, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                              \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)   \
    {                                                                                                \
        IVP_SAV##postfix##_XP(v_in.val[0], v_store, addr, byte_num);                                 \
    }

DECLFUN(VdspVectorU8X1,  xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X1,  xb_vec2Nx8,  2NX8)
DECLFUN(VdspVectorU16X1, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X1, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                         \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1)   \
    {                                                                                                           \
        valign v_load;                                                                                          \
        align(v_load, addr);                                                                                    \
        IVP_LAV##postfix##_XP(mv_x0.val[0], v_load, addr, len_x0);                                              \
        IVP_LAV##postfix##_XP(mv_x1.val[0], v_load, addr, len_x1);                                              \
    }

DECLFUN(VdspVectorU8X1,  xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X1,  xb_vec2Nx8,  2NX8)
DECLFUN(VdspVectorU16X1, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X1, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix, byte)                                                                 \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1) \
    {                                                                                                         \
        valign v_load;                                                                                        \
        align(v_load, addr);                                                                                  \
        vtype vrg_lo, vrg_hi;                                                                                 \
        IVP_LAV##postfix##_XP(vrg_lo, v_load, addr, len_x0);                                                  \
        IVP_LAV##postfix##_XP(vrg_hi, v_load, addr, len_x0 - 128);                                            \
        IVP_DSEL##postfix##I(mv_x0.val[1], mv_x0.val[0], vrg_hi, vrg_lo, IVP_DSELI_##byte##B_DEINTERLEAVE_1); \
        IVP_LAV##postfix##_XP(vrg_lo, v_load, addr, len_x1);                                                  \
        IVP_LAV##postfix##_XP(vrg_hi, v_load, addr, len_x1 - 128);                                            \
        IVP_DSEL##postfix##I(mv_x1.val[1], mv_x1.val[0], vrg_hi, vrg_lo, IVP_DSELI_##byte##B_DEINTERLEAVE_1); \
    }

DECLFUN(VdspVectorU8X2,  xb_vec2Nx8U, 2NX8U, 8)
DECLFUN(VdspVectorS8X2,  xb_vec2Nx8,  2NX8,  8)
DECLFUN(VdspVectorU16X2, xb_vecNx16U, NX16U, 16)
DECLFUN(VdspVectorS16X2, xb_vecNx16,  NX16,  16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix, byte)                                                                \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                            \
    {                                                                                                        \
        valign v_load;                                                                                       \
        align(v_load, addr);                                                                                 \
        vtype vrg_lo, vrg_hi;                                                                                \
        IVP_LAV##postfix##_XP(vrg_lo, v_load, addr, byte_num);                                               \
        IVP_LAV##postfix##_XP(vrg_hi, v_load, addr, byte_num - 128);                                         \
        IVP_DSEL##postfix##I(v_out.val[1], v_out.val[0], vrg_hi, vrg_lo, IVP_DSELI_##byte##B_DEINTERLEAVE_1);\
    }

DECLFUN(VdspVectorU8X2,  xb_vec2Nx8U, 2NX8U, 8)
DECLFUN(VdspVectorS8X2,  xb_vec2Nx8,  2NX8,  8)
DECLFUN(VdspVectorU16X2, xb_vecNx16U, NX16U, 16)
DECLFUN(VdspVectorS16X2, xb_vecNx16,  NX16,  16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                             \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1)       \
    {                                                                                                               \
        valign v_load;                                                                                              \
        align(v_load, addr);                                                                                        \
        vtype vrg0, vrg1;                                                                                           \
        vtype vrgb0, vrgb1, vrgb2;                                                                                  \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, len_x0);                                                         \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, len_x0 - 128);                                                   \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, len_x0 - 256);                                                   \
        IVP_DSEL##postfix##I(mv_x0.val[2], vrg0, vrgb1, vrgb0, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_0);                \
        IVP_DSEL##postfix##I_H(mv_x0.val[2], vrg1, vrgb2, vrgb1, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_1);              \
        IVP_DSEL##postfix##I(mv_x0.val[1], mv_x0.val[0], vrg1, vrg0, IVP_DSELI_8B_DEINTERLEAVE_1);                  \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, len_x1);                                                         \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, len_x1 - 128);                                                   \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, len_x1 - 256);                                                   \
        IVP_DSEL##postfix##I(mv_x1.val[2], vrg0, vrgb1, vrgb0, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_0);                \
        IVP_DSEL##postfix##I_H(mv_x1.val[2], vrg1, vrgb2, vrgb1, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_1);              \
        IVP_DSEL##postfix##I(mv_x1.val[1], mv_x1.val[0], vrg1, vrg0, IVP_DSELI_8B_DEINTERLEAVE_1);                  \
    }

DECLFUN(VdspVectorU8X3, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X3, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                    \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                          \
    {                                                                                                      \
        valign v_load;                                                                                     \
        align(v_load, addr);                                                                               \
        vtype vrg0, vrg1;                                                                                  \
        vtype vrgb0, vrgb1, vrgb2;                                                                         \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, byte_num);                                              \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, byte_num - 128);                                        \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, byte_num - 256);                                        \
        IVP_DSEL##postfix##I(v_out.val[2], vrg0, vrgb1, vrgb0, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_0);       \
        IVP_DSEL##postfix##I_H(v_out.val[2], vrg1, vrgb2, vrgb1, IVP_DSELI_8B_DEINTERLEAVE_C3_STEP_1);     \
        IVP_DSEL##postfix##I(v_out.val[1], v_out.val[0], vrg1, vrg0, IVP_DSELI_8B_DEINTERLEAVE_1);         \
    }

DECLFUN(VdspVectorU8X3, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X3, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                          \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1)    \
    {                                                                                                            \
        valign v_load;                                                                                           \
        align(v_load, addr);                                                                                     \
        vtype vrg0, vrg1;                                                                                        \
        vtype vrgb0, vrgb1, vrgb2;                                                                               \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, len_x0);                                                      \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, len_x0 - 128);                                                \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, len_x0 - 256);                                                \
        IVP_DSEL##postfix(mv_x0.val[2], vrg0, vrgb1, vrgb0, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_0));     \
        IVP_DSEL##postfix##T(mv_x0.val[2], vrg1, vrgb1, vrgb2, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_1),   \
                             *((vbool2N*)&sel_deinterleave_d16_c3_step_1_msk));                                  \
        IVP_DSEL##postfix##I(mv_x0.val[1], mv_x0.val[0], vrg1, vrg0, IVP_DSELI_DEINTERLEAVE_1);                  \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, len_x1);                                                      \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, len_x1 - 128);                                                \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, len_x1 - 256);                                                \
        IVP_DSEL##postfix(mv_x1.val[2], vrg0, vrgb1, vrgb0, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_0));     \
        IVP_DSEL##postfix##T(mv_x1.val[2], vrg1, vrgb1, vrgb2, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_1),   \
                             *((vbool2N*)&sel_deinterleave_d16_c3_step_1_msk));                                  \
        IVP_DSEL##postfix##I(mv_x1.val[1], mv_x1.val[0], vrg1, vrg0, IVP_DSELI_DEINTERLEAVE_1);                  \
    }

DECLFUN(VdspVectorU16X3, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X3, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                            \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                                  \
    {                                                                                                              \
        valign v_load;                                                                                             \
        align(v_load, addr);                                                                                       \
        vtype vrg0, vrg1;                                                                                          \
        vtype vrgb0, vrgb1, vrgb2;                                                                                 \
        IVP_LAV##postfix##_XP(vrgb0, v_load, addr, byte_num);                                                      \
        IVP_LAV##postfix##_XP(vrgb1, v_load, addr, byte_num - 128);                                                \
        IVP_LAV##postfix##_XP(vrgb2, v_load, addr, byte_num - 256);                                                \
        IVP_DSEL##postfix(v_out.val[2], vrg0, vrgb1, vrgb0, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_0));       \
        IVP_DSEL##postfix##T(v_out.val[2], vrg1, vrgb1, vrgb2, *((xb_vec2Nx8*)sel_deinterleave_d16_c3_step_1),     \
                             *((vbool2N*)&sel_deinterleave_d16_c3_step_1_msk));                                    \
        IVP_DSEL##postfix##I(v_out.val[1], v_out.val[0], vrg1, vrg0, IVP_DSELI_DEINTERLEAVE_1);                    \
    }

DECLFUN(VdspVectorU16X3, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X3, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                          \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1)    \
    {                                                                                                            \
        valign v_load;                                                                                           \
        align(v_load, addr);                                                                                     \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                                    \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                                    \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, len_x0);                                                     \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, len_x0 - 128);                                               \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, len_x0 - 256);                                               \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, len_x0 - 384);                                               \
        IVP_DSEL##postfix##I(vba_lo, vrg_lo, vrgba1, vrgba0, IVP_DSELI_DEINTERLEAVE_1);                          \
        IVP_DSEL##postfix##I(vba_hi, vrg_hi, vrgba3, vrgba2, IVP_DSELI_DEINTERLEAVE_1);                          \
        IVP_DSEL##postfix##I(mv_x0.val[1], mv_x0.val[0], vrg_hi, vrg_lo, IVP_DSELI_8B_DEINTERLEAVE_1);           \
        IVP_DSEL##postfix##I(mv_x0.val[3], mv_x0.val[2], vba_hi, vba_lo, IVP_DSELI_8B_DEINTERLEAVE_1);           \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, len_x1);                                                     \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, len_x1 - 128);                                               \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, len_x1 - 256);                                               \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, len_x1 - 384);                                               \
        IVP_DSEL##postfix##I(vba_lo, vrg_lo, vrgba1, vrgba0, IVP_DSELI_DEINTERLEAVE_1);                          \
        IVP_DSEL##postfix##I(vba_hi, vrg_hi, vrgba3, vrgba2, IVP_DSELI_DEINTERLEAVE_1);                          \
        IVP_DSEL##postfix##I(mv_x1.val[1], mv_x1.val[0], vrg_hi, vrg_lo, IVP_DSELI_8B_DEINTERLEAVE_1);           \
        IVP_DSEL##postfix##I(mv_x1.val[3], mv_x1.val[2], vba_hi, vba_lo, IVP_DSELI_8B_DEINTERLEAVE_1);           \
    }

DECLFUN(VdspVectorU8X4, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X4, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                    \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                          \
    {                                                                                                      \
        valign v_load;                                                                                     \
        align(v_load, addr);                                                                               \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                              \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                              \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, byte_num);                                             \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, byte_num - 128);                                       \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, byte_num - 256);                                       \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, byte_num - 384);                                       \
        IVP_DSEL##postfix##I(vba_lo, vrg_lo, vrgba1, vrgba0, IVP_DSELI_DEINTERLEAVE_1);                    \
        IVP_DSEL##postfix##I(vba_hi, vrg_hi, vrgba3, vrgba2, IVP_DSELI_DEINTERLEAVE_1);                    \
        IVP_DSEL##postfix##I(v_out.val[1], v_out.val[0], vrg_hi, vrg_lo, IVP_DSELI_8B_DEINTERLEAVE_1);     \
        IVP_DSEL##postfix##I(v_out.val[3], v_out.val[2], vba_hi, vba_lo, IVP_DSELI_8B_DEINTERLEAVE_1);     \
    }

DECLFUN(VdspVectorU8X4, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X4, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                            \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &mv_x0, mvtype &mv_x1, DT_S32 len_x0, DT_S32 len_x1)      \
    {                                                                                                              \
        valign v_load;                                                                                             \
        align(v_load, addr);                                                                                       \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                                      \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                                      \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, len_x0);                                                       \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, len_x0 - 128);                                                 \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, len_x0 - 256);                                                 \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, len_x0 - 384);                                                 \
        IVP_DSEL##postfix(vba_lo, vrg_lo, vrgba1, vrgba0, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(vba_hi, vrg_hi, vrgba3, vrgba2, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(mv_x0.val[1], mv_x0.val[0], vrg_hi, vrg_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
        IVP_DSEL##postfix(mv_x0.val[3], mv_x0.val[2], vba_hi, vba_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, len_x1);                                                       \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, len_x1 - 128);                                                 \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, len_x1 - 256);                                                 \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, len_x1 - 384);                                                 \
        IVP_DSEL##postfix(vba_lo, vrg_lo, vrgba1, vrgba0, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(vba_hi, vrg_hi, vrgba3, vrgba2, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(mv_x1.val[1], mv_x1.val[0], vrg_hi, vrg_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
        IVP_DSEL##postfix(mv_x1.val[3], mv_x1.val[2], vba_hi, vba_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
    }

DECLFUN(VdspVectorU16X4, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X4, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                            \
    inline DT_VOID vload(vtype *__restrict &addr, mvtype &v_out, DT_S32 byte_num)                                  \
    {                                                                                                              \
        valign v_load;                                                                                             \
        align(v_load, addr);                                                                                       \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                                      \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                                      \
        IVP_LAV##postfix##_XP(vrgba0, v_load, addr, byte_num);                                                     \
        IVP_LAV##postfix##_XP(vrgba1, v_load, addr, byte_num - 128);                                               \
        IVP_LAV##postfix##_XP(vrgba2, v_load, addr, byte_num - 256);                                               \
        IVP_LAV##postfix##_XP(vrgba3, v_load, addr, byte_num - 384);                                               \
        IVP_DSEL##postfix(vba_lo, vrg_lo, vrgba1, vrgba0, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(vba_hi, vrg_hi, vrgba3, vrgba2, *((xb_vec2Nx8*)dsel_deinterleave_d8_1));                 \
        IVP_DSEL##postfix(v_out.val[1], v_out.val[0], vrg_hi, vrg_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
        IVP_DSEL##postfix(v_out.val[3], v_out.val[2], vba_hi, vba_lo, *((xb_vec2Nx8*)dsel_deinterleave_d16_1));    \
    }

DECLFUN(VdspVectorU16X4, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X4, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix, bit)                                                             \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)       \
    {                                                                                                    \
        vtype vrg_lo, vrg_hi;                                                                            \
        IVP_DSEL##postfix##I(vrg_hi, vrg_lo, v_in.val[1], v_in.val[0], IVP_DSELI_##bit##B_INTERLEAVE_1); \
        IVP_SAV##postfix##_XP(vrg_lo, v_store, addr, byte_num);                                          \
        IVP_SAV##postfix##_XP(vrg_hi, v_store, addr, byte_num - 128);                                    \
    }

DECLFUN(VdspVectorU8X2,  xb_vec2Nx8U, 2NX8U, 8)
DECLFUN(VdspVectorS8X2,  xb_vec2Nx8,  2NX8,  8)
DECLFUN(VdspVectorU16X2, xb_vecNx16U, NX16U, 16)
DECLFUN(VdspVectorS16X2, xb_vecNx16,  NX16,  16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                 \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)      \
    {                                                                                                   \
        vtype vrg0, vrg1;                                                                               \
        vtype vrgb0, vrgb1, vrgb2;                                                                      \
        IVP_DSEL##postfix##I(vrg1, vrg0, v_in.val[1], v_in.val[0], IVP_DSELI_8B_INTERLEAVE_1);          \
        IVP_DSEL##postfix##I(vrgb1, vrgb0, v_in.val[2], vrg0, IVP_DSELI_8B_INTERLEAVE_C3_STEP_0);       \
        IVP_DSEL##postfix##I_H(vrgb1, vrgb2, v_in.val[2], vrg1, IVP_DSELI_8B_INTERLEAVE_C3_STEP_1);     \
        IVP_SAV##postfix##_XP(vrgb0, v_store, addr, byte_num);                                          \
        IVP_SAV##postfix##_XP(vrgb1, v_store, addr, byte_num - 128);                                    \
        IVP_SAV##postfix##_XP(vrgb2, v_store, addr, byte_num - 256);                                    \
    }

DECLFUN(VdspVectorU8X3, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X3, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                       \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)            \
    {                                                                                                         \
        vtype vrg0, vrg1;                                                                                     \
        vtype vrgb0, vrgb1, vrgb2;                                                                            \
        IVP_DSEL##postfix##I(vrg1, vrg0, v_in.val[1], v_in.val[0], IVP_DSELI_INTERLEAVE_1);                   \
        IVP_DSEL##postfix(vrgb1, vrgb0, v_in.val[2], vrg0, *((xb_vec2Nx8*)sel_interleave_d16_c3_step_0));     \
        IVP_DSEL##postfix##T(vrgb2, vrgb1, v_in.val[2], vrg1, *((xb_vec2Nx8*)sel_interleave_d16_c3_step_1),   \
                             *((vbool2N*)&sel_interleave_d16_c3_step_1_msk));                                 \
        IVP_SAV##postfix##_XP(vrgb0, v_store, addr, byte_num);                                                \
        IVP_SAV##postfix##_XP(vrgb1, v_store, addr, byte_num - 128);                                          \
        IVP_SAV##postfix##_XP(vrgb2, v_store, addr, byte_num - 256);                                          \
    }

DECLFUN(VdspVectorU16X3, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X3, xb_vecNx16,  NX16)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                             \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)  \
    {                                                                                               \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                       \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                       \
        IVP_DSEL##postfix##I(vrg_hi, vrg_lo, v_in.val[1], v_in.val[0], IVP_DSELI_8B_INTERLEAVE_1);  \
        IVP_DSEL##postfix##I(vba_hi, vba_lo, v_in.val[3], v_in.val[2], IVP_DSELI_8B_INTERLEAVE_1);  \
        IVP_DSEL##postfix##I(vrgba1, vrgba0, vba_lo, vrg_lo, IVP_DSELI_INTERLEAVE_1);               \
        IVP_DSEL##postfix##I(vrgba3, vrgba2, vba_hi, vrg_hi, IVP_DSELI_INTERLEAVE_1);               \
        IVP_SAV##postfix##_XP(vrgba0, v_store, addr, byte_num);                                     \
        IVP_SAV##postfix##_XP(vrgba1, v_store, addr, byte_num - 128);                               \
        IVP_SAV##postfix##_XP(vrgba2, v_store, addr, byte_num - 256);                               \
        IVP_SAV##postfix##_XP(vrgba3, v_store, addr, byte_num - 384);                               \
    }

DECLFUN(VdspVectorU8X4, xb_vec2Nx8U, 2NX8U)
DECLFUN(VdspVectorS8X4, xb_vec2Nx8,  2NX8)
#undef DECLFUN

#define DECLFUN(mvtype, vtype, postfix)                                                                      \
    inline DT_VOID vstore(vtype *__restrict &addr, valign &v_store, mvtype &v_in, DT_S32 byte_num)           \
    {                                                                                                        \
        vtype vrg_lo, vba_lo, vrg_hi, vba_hi;                                                                \
        vtype vrgba0, vrgba1, vrgba2, vrgba3;                                                                \
        IVP_DSEL##postfix(vrg_hi, vrg_lo, v_in.val[1], v_in.val[0], *((xb_vec2Nx8*)dsel_interleave_d8_1));   \
        IVP_DSEL##postfix(vba_hi, vba_lo, v_in.val[3], v_in.val[2], *((xb_vec2Nx8*)dsel_interleave_d8_1));   \
        IVP_DSEL##postfix(vrgba1, vrgba0, vba_lo, vrg_lo, *((xb_vec2Nx8*)dsel_interleave_d16_1));            \
        IVP_DSEL##postfix(vrgba3, vrgba2, vba_hi, vrg_hi, *((xb_vec2Nx8*)dsel_interleave_d16_1));            \
        IVP_SAV##postfix##_XP(vrgba0, v_store, addr, byte_num);                                              \
        IVP_SAV##postfix##_XP(vrgba1, v_store, addr, byte_num - 128);                                        \
        IVP_SAV##postfix##_XP(vrgba2, v_store, addr, byte_num - 256);                                        \
        IVP_SAV##postfix##_XP(vrgba3, v_store, addr, byte_num - 384);                                        \
    }

DECLFUN(VdspVectorU16X4, xb_vecNx16U, NX16U)
DECLFUN(VdspVectorS16X4, xb_vecNx16,  NX16)
#undef DECLFUN

} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_ISA_LOAD_HPP__