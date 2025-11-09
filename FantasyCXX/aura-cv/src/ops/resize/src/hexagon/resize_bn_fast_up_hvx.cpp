#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
namespace aura
{

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, DT_VOID>::type
ResizeBnUpX2BorderRow(const Tp *src_c, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = (width + elem_counts - 1) & (~(elem_counts - 1));
    DT_S32 multiplier_3  = 3 + (3 << 8) + (3 << 16) + (3 << 24);
    HVX_Vector v_const_2 = Q6_Vb_vsplat_R(2);

    MVType mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);
    Tp src_l[C] = {};
    memcpy(src_l, src_c, sizeof(Tp) * C);

    auto resizex2_border_func = [&]()
    {
        MVType mv_c_result, mv_r0_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vb_vsplat_R(src_l[ch]), 1);
            HVX_VectorPair w_tmp    = Q6_Wuh_vmpy_VubRub(mv_c_src.val[ch], multiplier_3);
            HVX_VectorPair w_result = Q6_Wh_vadd_VubVub(v_tmp, v_const_2);

            w_result            = Q6_Wh_vadd_WhWh(w_result, w_tmp);
            mv_c_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            v_tmp                = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 1);
            w_result             = Q6_Wh_vadd_VubVub(v_tmp, v_const_2);
            w_result             = Q6_Wh_vadd_WhWh(w_result, w_tmp);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            w_result             = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
            src_l[ch]        = src_c[i - C + ch];
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_r0_src);

        resizex2_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);
        memcpy(src_l, src_c + i - C, sizeof(Tp) * C);
        i += elem_counts;

        resizex2_border_func();

        j += 2 * elem_counts;
    }

    resizex2_border_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width * 2 - C + ch] = src_c[width - C + ch];
    }
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, DT_VOID>::type
ResizeBnUpX2BorderRow(const Tp *src_c, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = (width + elem_counts - 1) & (~(elem_counts - 1));
    HVX_Vector v_const_3 = Q6_Vb_vsplat_R(3);

    MVType mv_c_result, mv_r0_result, mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);
    Tp src_l[C] = {};
    memcpy(src_l, src_c, sizeof(Tp) * C);

    auto resizex2_border_func = [&]()
    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_tmp    = Q6_Wh_vmpy_VbVb(mv_c_src.val[ch], v_const_3);
            HVX_Vector v_tmp        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vb_vsplat_R(src_l[ch]), 1);
            HVX_VectorPair w_result = Q6_Wh_vsxt_Vb(v_tmp);

            w_result            = Q6_Wh_vadd_WhWh(w_result, w_tmp);
            mv_c_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            v_tmp                = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 1);
            w_result             = Q6_Wh_vsxt_Vb(v_tmp);
            w_result             = Q6_Wh_vadd_WhWh(w_result, w_tmp);
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            w_result             = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
            src_l[ch]        = src_c[i - C + ch];
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_r0_src);

        resizex2_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);
        memcpy(src_l, src_c + i - C, sizeof(Tp) * C);
        i += elem_counts;

        resizex2_border_func();

        j += 2 * elem_counts;
    }

    resizex2_border_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width * 2 - C + ch] = src_c[width - C + ch];
    }
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, DT_VOID>::type
ResizeBnUpX2BorderRow(const Tp *src_c, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = (width + elem_counts - 1) & (~(elem_counts - 1));
    DT_S32 contant_3     = 3 + (3 << 16);
    HVX_Vector v_const_0 = Q6_V_vzero();

    MVType mv_c_result, mv_r0_result, mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);
    Tp src_l[C] = {0};
    memcpy(src_l, src_c, sizeof(Tp) * C);

    auto resizex2_border_func = [&]()
    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vh_vsplat_R(src_l[ch]), 2);
            HVX_VectorPair w_temp   = Q6_Wuw_vmpy_VuhRuh(mv_c_src.val[ch], contant_3);
            HVX_VectorPair w_result = Q6_Ww_vaddacc_WwVuhVuh(w_temp, v_const_0, v_tmp);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            v_tmp                = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 2);
            w_result             = Q6_Ww_vaddacc_WwVuhVuh(w_temp, v_const_0, v_tmp);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            w_result             = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
            src_l[ch]        = src_c[i - C + ch];
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_r0_src);

        resizex2_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);
        memcpy(src_l, src_c + i - C, sizeof(Tp) * C);
        i += elem_counts;

        resizex2_border_func();

        j += 2 * elem_counts;
    }

    resizex2_border_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width * 2 - C + ch] = src_c[width - C + ch];
    }
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, DT_VOID>::type
ResizeBnUpX2BorderRow(const Tp *src_c, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = (width + elem_counts - 1) & (~(elem_counts - 1));
    DT_S32 contant_3     = 3 + (3 << 16);
    HVX_Vector v_const_0 = Q6_Vh_vsplat_R(0);

    MVType mv_c_result, mv_r0_result, mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);
    Tp src_l[C] = {0};
    memcpy(src_l, src_c, sizeof(Tp) * C);

    auto resizex2_border_func = [&]()
    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vh_vsplat_R(src_l[ch]), 2);
            HVX_VectorPair w_temp   = Q6_Ww_vmpy_VhRh(mv_c_src.val[ch], contant_3);
            HVX_VectorPair w_result = Q6_Ww_vaddacc_WwVhVh(w_temp, v_const_0, v_tmp);
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            v_tmp                = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 2);
            w_result             = Q6_Ww_vaddacc_WwVhVh(w_temp, v_const_0, v_tmp);
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 2);

            w_result             = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
            src_l[ch]        = src_c[i - C + ch];
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_r0_src);

        resizex2_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);
        memcpy(src_l, src_c + i - C, sizeof(Tp) * C);
        i += elem_counts;

        resizex2_border_func();

        j += 2 * elem_counts;
    }

    resizex2_border_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width * 2 - C + ch] = src_c[width - C + ch];
    }
}

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, DT_VOID>::type
ResizeBnUpX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = width & (~(elem_counts - 1));
    DT_S32 multiplier_3  = 3 + (3 << 8) + (3 << 16) + (3 << 24);
    HVX_Vector v_const_0 = Q6_V_vzero();

    MVType mv_c_result, mv_r0_result, mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    HVX_VectorPair w_c_c_result, w_c_r0_result, w_n0_c_result, w_n0_r0_result;

#define mv_n0_c_result mv_c_r0_src
#define mv_n0_r0_result mv_n0_r0_src

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);
    Tp src_c_l[C]  = {0};
    Tp src_n0_l[C] = {0};
    memcpy(src_c_l, src_c, sizeof(Tp) * C);
    memcpy(src_n0_l, src_n0, sizeof(Tp) * C);

    auto resizex2_func = [&](int ch, HVX_Vector &v_tmp0, HVX_Vector &v_tmp1)
    {
        HVX_VectorPair w_tmp = Q6_Wuh_vmpy_VubRub(mv_c_c_src.val[ch], multiplier_3);
        w_c_c_result         = Q6_Wh_vadd_VubVub(v_tmp0, v_const_0);
        w_c_c_result         = Q6_Wh_vadd_WhWh(w_c_c_result, w_tmp);

        v_tmp0        = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
        w_c_r0_result = Q6_Wh_vadd_VubVub(v_tmp0, v_const_0);
        w_c_r0_result = Q6_Wh_vadd_WhWh(w_c_r0_result, w_tmp);

        w_tmp         = Q6_Wuh_vmpy_VubRub(mv_n0_c_src.val[ch], multiplier_3);
        w_n0_c_result = Q6_Wh_vadd_VubVub(v_tmp1, v_const_0);
        w_n0_c_result = Q6_Wh_vadd_WhWh(w_n0_c_result, w_tmp);

        v_tmp0         = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
        w_n0_r0_result = Q6_Wh_vadd_VubVub(v_tmp0, v_const_0);
        w_n0_r0_result = Q6_Wh_vadd_WhWh(w_n0_r0_result, w_tmp);

        v_tmp0              = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_n0_c_result), Q6_V_lo_W(w_c_c_result), multiplier_3);
        v_tmp1              = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_n0_c_result), Q6_V_hi_W(w_c_c_result), multiplier_3);
        mv_c_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0               = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_n0_r0_result), Q6_V_lo_W(w_c_r0_result), multiplier_3);
        v_tmp1               = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_n0_r0_result), Q6_V_hi_W(w_c_r0_result), multiplier_3);
        mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_tmp);

        mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
        mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        src_c_l[ch]         = src_c[i - C + ch];
        src_n0_l[ch]        = src_n0[i - C + ch];

        // dst_n0
        v_tmp0                 = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_c_c_result), Q6_V_lo_W(w_n0_c_result), multiplier_3);
        v_tmp1                 = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_c_c_result), Q6_V_hi_W(w_n0_c_result), multiplier_3);
        mv_n0_c_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0                  = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_c_r0_result), Q6_V_lo_W(w_n0_r0_result), multiplier_3);
        v_tmp1                  = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_c_r0_result), Q6_V_hi_W(w_n0_r0_result), multiplier_3);
        mv_n0_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                   = Q6_W_vshuff_VVR(mv_n0_r0_result.val[ch], mv_n0_c_result.val[ch], -1);
        mv_n0_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_n0_r0_result.val[ch] = Q6_V_hi_W(w_tmp);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c_l[ch]),  1);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0_l[ch]), 1);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);
        i += elem_counts;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c[i - elem_counts - C + ch]),  1);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0[i - elem_counts - C + ch]), 1);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);

        j += 2 * elem_counts;
    }

    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c_l[ch]),  1);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0_l[ch]), 1);

            mv_c_r0_src.val[ch]  = Q6_Vb_vsplat_R(src_c[width - C + ch]);
            mv_n0_r0_src.val[ch] = Q6_Vb_vsplat_R(src_n0[width - C + ch]);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, DT_VOID>::type
ResizeBnUpX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts   = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align   = width & (~(elem_counts - 1));
    DT_S32 multiplier_3  = 3 + (3 << 8) + (3 << 16) + (3 << 24);
    HVX_Vector v_const_3 = Q6_Vb_vsplat_R(3);

    MVType mv_c_result, mv_r0_result, mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    HVX_VectorPair w_c_c_result, w_c_r0_result, w_n0_c_result, w_n0_r0_result;

#define mv_n0_c_result mv_c_r0_src
#define mv_n0_r0_result mv_n0_r0_src

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);
    Tp src_c_l[C]  = {0};
    Tp src_n0_l[C] = {0};
    memcpy(src_c_l,  src_c,  sizeof(Tp) * C);
    memcpy(src_n0_l, src_n0, sizeof(Tp) * C);

    auto resizex2_func = [&](int ch, HVX_Vector &v_tmp0, HVX_Vector &v_tmp1)
    {
        HVX_VectorPair w_tmp = Q6_Wh_vsxt_Vb(v_tmp0);
        w_c_c_result         = Q6_Wh_vmpyacc_WhVbVb(w_tmp, mv_c_c_src.val[ch], v_const_3);

        v_tmp0        = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
        w_tmp         = Q6_Wh_vsxt_Vb(v_tmp0);
        w_c_r0_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp, mv_c_c_src.val[ch], v_const_3);

        w_tmp         = Q6_Wh_vsxt_Vb(v_tmp1);
        w_n0_c_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp, mv_n0_c_src.val[ch], v_const_3);

        v_tmp0         = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
        w_tmp          = Q6_Wh_vsxt_Vb(v_tmp0);
        w_n0_r0_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp, mv_n0_c_src.val[ch], v_const_3);

        v_tmp0              = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_n0_c_result), Q6_V_lo_W(w_c_c_result), multiplier_3);
        v_tmp1              = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_n0_c_result), Q6_V_hi_W(w_c_c_result), multiplier_3);
        mv_c_result.val[ch] = Q6_Vb_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0               = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_n0_r0_result), Q6_V_lo_W(w_c_r0_result), multiplier_3);
        v_tmp1               = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_n0_r0_result), Q6_V_hi_W(w_c_r0_result), multiplier_3);
        mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_tmp);

        mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
        mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        src_c_l[ch]         = src_c[i - C + ch];
        src_n0_l[ch]        = src_n0[i - C + ch];

        // dst_n0
        v_tmp0                 = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_c_c_result), Q6_V_lo_W(w_n0_c_result), multiplier_3);
        v_tmp1                 = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_c_c_result), Q6_V_hi_W(w_n0_c_result), multiplier_3);
        mv_n0_c_result.val[ch] = Q6_Vb_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0                  = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_lo_W(w_c_r0_result), Q6_V_lo_W(w_n0_r0_result), multiplier_3);
        v_tmp1                  = Q6_Vh_vmpyiacc_VhVhRb(Q6_V_hi_W(w_c_r0_result), Q6_V_hi_W(w_n0_r0_result), multiplier_3);
        mv_n0_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                   = Q6_W_vshuff_VVR(mv_n0_r0_result.val[ch], mv_n0_c_result.val[ch], -1);
        mv_n0_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_n0_r0_result.val[ch] = Q6_V_hi_W(w_tmp);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c_l[ch]),  1);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0_l[ch]), 1);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);
        i += elem_counts;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c[i - elem_counts - C + ch]),  1);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0[i - elem_counts - C + ch]), 1);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);

        j += 2 * elem_counts;
    }

    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0    = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c_l[ch]),  1);
            HVX_Vector v_tmp1    = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0_l[ch]), 1);
            mv_c_r0_src.val[ch]  = Q6_Vb_vsplat_R(src_c[width - C + ch]);
            mv_n0_r0_src.val[ch] = Q6_Vb_vsplat_R(src_n0[width - C + ch]);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, DT_VOID>::type
ResizeBnUpX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts  = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align  = width & (~(elem_counts - 1));
    DT_S32 multiplier_3 = 3 + (3 << 16);

    MVType mv_c_result, mv_r0_result, mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    HVX_VectorPair w_c_c_result, w_c_r0_result, w_n0_c_result, w_n0_r0_result;

#define mv_n0_c_result mv_c_r0_src
#define mv_n0_r0_result mv_n0_r0_src

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);
    Tp src_c_l[C]  = {0};
    Tp src_n0_l[C] = {0};
    memcpy(src_c_l,  src_c,  sizeof(Tp) * C);
    memcpy(src_n0_l, src_n0, sizeof(Tp) * C);

    auto resizex2_func = [&](int ch, HVX_Vector &v_tmp0, HVX_Vector &v_tmp1)
    {
        HVX_VectorPair w_tmp = Q6_Wuw_vzxt_Vuh(v_tmp0);
        w_c_c_result         = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp, mv_c_c_src.val[ch], multiplier_3);

        v_tmp0        = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
        w_tmp         = Q6_Wuw_vzxt_Vuh(v_tmp0);
        w_c_r0_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp, mv_c_c_src.val[ch], multiplier_3);

        w_tmp         = Q6_Wuw_vzxt_Vuh(v_tmp1);
        w_n0_c_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp, mv_n0_c_src.val[ch], multiplier_3);

        v_tmp0         = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
        w_tmp          = Q6_Wuw_vzxt_Vuh(v_tmp0);
        w_n0_r0_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp, mv_n0_c_src.val[ch], multiplier_3);

        v_tmp0              = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_n0_c_result), Q6_V_lo_W(w_c_c_result), multiplier_3);
        v_tmp1              = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_n0_c_result), Q6_V_hi_W(w_c_c_result), multiplier_3);
        mv_c_result.val[ch] = Q6_Vuh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0               = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_n0_r0_result), Q6_V_lo_W(w_c_r0_result), multiplier_3);
        v_tmp1               = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_n0_r0_result), Q6_V_hi_W(w_c_r0_result), multiplier_3);
        mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_tmp);

        mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
        mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        src_c_l[ch]         = src_c[i - C + ch];
        src_n0_l[ch]        = src_n0[i - C + ch];

        // dst_n0
        v_tmp0                 = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_c_c_result), Q6_V_lo_W(w_n0_c_result), multiplier_3);
        v_tmp1                 = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_c_c_result), Q6_V_hi_W(w_n0_c_result), multiplier_3);
        mv_n0_c_result.val[ch] = Q6_Vuh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0                  = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_c_r0_result), Q6_V_lo_W(w_n0_r0_result), multiplier_3);
        v_tmp1                  = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_c_r0_result), Q6_V_hi_W(w_n0_r0_result), multiplier_3);
        mv_n0_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                   = Q6_W_vshuff_VVR(mv_n0_r0_result.val[ch], mv_n0_c_result.val[ch], -2);
        mv_n0_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_n0_r0_result.val[ch] = Q6_V_hi_W(w_tmp);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c + i,  mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c_l[ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0_l[ch]), 2);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);
        i += elem_counts;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c[i - elem_counts - C + ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0[i - elem_counts - C + ch]), 2);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);

        j += 2 * elem_counts;
    }

    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c_l[ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0_l[ch]), 2);

            mv_c_r0_src.val[ch]  = Q6_Vh_vsplat_R(src_c[width - C + ch]);
            mv_n0_r0_src.val[ch] = Q6_Vh_vsplat_R(src_n0[width - C + ch]);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, DT_VOID>::type
ResizeBnUpX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts  = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align  = width & (~(elem_counts - 1));
    DT_S32 multiplier_3 = 3 + (3 << 16);

    MVType mv_c_result, mv_r0_result, mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    HVX_VectorPair w_c_c_result, w_c_r0_result, w_n0_c_result, w_n0_r0_result;

#define mv_n0_c_result mv_c_r0_src
#define mv_n0_r0_result mv_n0_r0_src

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);
    Tp src_c_l[C]  = {0};
    Tp src_n0_l[C] = {0};
    memcpy(src_c_l,  src_c,  sizeof(Tp) * C);
    memcpy(src_n0_l, src_n0, sizeof(Tp) * C);

    auto resizex2_func = [&](int ch, HVX_Vector &v_tmp0, HVX_Vector &v_tmp1)
    {
        HVX_VectorPair w_tmp = Q6_Ww_vsxt_Vh(v_tmp0);
        w_c_c_result         = Q6_Ww_vmpyacc_WwVhRh(w_tmp, mv_c_c_src.val[ch], multiplier_3);

        v_tmp0        = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
        w_tmp         = Q6_Ww_vsxt_Vh(v_tmp0);
        w_c_r0_result = Q6_Ww_vmpyacc_WwVhRh(w_tmp, mv_c_c_src.val[ch], multiplier_3);

        w_tmp         = Q6_Ww_vsxt_Vh(v_tmp1);
        w_n0_c_result = Q6_Ww_vmpyacc_WwVhRh(w_tmp, mv_n0_c_src.val[ch], multiplier_3);

        v_tmp0         = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
        w_tmp          = Q6_Ww_vsxt_Vh(v_tmp0);
        w_n0_r0_result = Q6_Ww_vmpyacc_WwVhRh(w_tmp, mv_n0_c_src.val[ch], multiplier_3);

        v_tmp0              = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_n0_c_result), Q6_V_lo_W(w_c_c_result), multiplier_3);
        v_tmp1              = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_n0_c_result), Q6_V_hi_W(w_c_c_result), multiplier_3);
        mv_c_result.val[ch] = Q6_Vh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0               = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_n0_r0_result), Q6_V_lo_W(w_c_r0_result), multiplier_3);
        v_tmp1               = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_n0_r0_result), Q6_V_hi_W(w_c_r0_result), multiplier_3);
        mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_tmp);

        mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
        mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        src_c_l[ch]         = src_c[i - C + ch];
        src_n0_l[ch]        = src_n0[i - C + ch];

        // dst_n0
        v_tmp0                 = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_c_c_result), Q6_V_lo_W(w_n0_c_result), multiplier_3);
        v_tmp1                 = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_c_c_result), Q6_V_hi_W(w_n0_c_result), multiplier_3);
        mv_n0_c_result.val[ch] = Q6_Vh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        v_tmp0                  = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_lo_W(w_c_r0_result), Q6_V_lo_W(w_n0_r0_result), multiplier_3);
        v_tmp1                  = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_hi_W(w_c_r0_result), Q6_V_hi_W(w_n0_r0_result), multiplier_3);
        mv_n0_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_sat(v_tmp1, v_tmp0, 4);

        w_tmp                   = Q6_W_vshuff_VVR(mv_n0_r0_result.val[ch], mv_n0_c_result.val[ch], -2);
        mv_n0_c_result.val[ch]  = Q6_V_lo_W(w_tmp);
        mv_n0_r0_result.val[ch] = Q6_V_hi_W(w_tmp);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 2)
    {
        vload(src_c  + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c_l[ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0_l[ch]), 2);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 2;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);
        i += elem_counts;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c[i - elem_counts - C + ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0[i - elem_counts - C + ch]), 2);
            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);

        j += 2 * elem_counts;
    }

    {
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp0 = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vh_vsplat_R(src_c_l[ch]),  2);
            HVX_Vector v_tmp1 = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0_l[ch]), 2);

            mv_c_r0_src.val[ch]  = Q6_Vh_vsplat_R(src_c[width - C + ch]);
            mv_n0_r0_src.val[ch] = Q6_Vh_vsplat_R(src_n0[width - C + ch]);

            resizex2_func(ch, v_tmp0, v_tmp1);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j, mv_n0_c_result);
        vstore(dst_n0 + j + elem_counts, mv_n0_r0_result);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
}

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeBnUpX4BorderRow(const Tp *src_c, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts  = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align  = width & (~(elem_counts - 1));
    DT_S32 multiplier_1 = 1 + (1 << 8) + (1 << 16) + (1 << 24);
    DT_S32 multiplier_3 = 3 + (3 << 8) + (3 << 16) + (3 << 24);
    DT_S32 multiplier_5 = 5 + (5 << 8) + (5 << 16) + (5 << 24);
    DT_S32 multiplier_7 = 7 + (7 << 8) + (7 << 16) + (7 << 24);

    MVType mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);

    auto resizex4_border_func = [&]()
    {
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_3);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_5);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(mv_c_src.val[ch], multiplier_1);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, v_tmp, multiplier_7);
            mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c + i, mv_r0_src);

        resizex4_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);

        resizex4_border_func();
    }

    MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
    j = (width - elem_counts - C) * 4;

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector v_tmp        = mv_c_src.val[ch];
        mv_c_src.val[ch]        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vb_vsplat_R(src_c[width - elem_counts - C + ch]), 1);
        HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_1);
        HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_7);
        mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_3);
        w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_5);
        mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuh_vmpy_VubRub(v_tmp, multiplier_5);
        w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_src.val[ch], multiplier_3);
        mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuh_vmpy_VubRub(mv_c_src.val[ch], multiplier_1);
        w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, v_tmp, multiplier_7);
        mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
        HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

        w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

        w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
        mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
        mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
    }

    vstore(dst_c  + j, mv_c_result);
    vstore(dst_c  + j + elem_counts, mv_r0_result);
    vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
    vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
    vstore(dst_n0 + j, mv_c_result);
    vstore(dst_n0 + j + elem_counts, mv_r0_result);
    vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
    vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        dst_n0[width * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        *(dst_c  - 2 * C + ch)                 = src_c[ch];
        *(dst_n0 - 2 * C + ch)                 = src_c[ch];
        *(dst_c  - C + ch)                     = src_c[ch];
        *(dst_n0 - C + ch)                     = src_c[ch];
    }

    return Status::OK;
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnUpX4BorderRow(const Tp *src_c, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));

    MVType mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);

    auto resizex4_border_func = [&]()
    {
        HVX_Vector v_const_1 = Q6_Vb_vsplat_R(1);
        HVX_Vector v_const_3 = Q6_Vb_vsplat_R(3);
        HVX_Vector v_const_5 = Q6_Vb_vsplat_R(5);
        HVX_Vector v_const_7 = Q6_Vb_vsplat_R(7);
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_tmp, v_const_1);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_7);
            mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_tmp, v_const_3);
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_5);
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_tmp, v_const_5);
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_3);
            mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wh_vmpy_VbVb(mv_c_src.val[ch], v_const_1);
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, v_tmp, v_const_7);
            mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c + i, mv_r0_src);

        resizex4_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);

        resizex4_border_func();
    }

    HVX_Vector v_const_1 = Q6_Vb_vsplat_R(1);
    HVX_Vector v_const_3 = Q6_Vb_vsplat_R(3);
    HVX_Vector v_const_5 = Q6_Vb_vsplat_R(5);
    HVX_Vector v_const_7 = Q6_Vb_vsplat_R(7);
    MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;

    j = (width - elem_counts - C) * 4;
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector v_tmp        = mv_c_src.val[ch];
        mv_c_src.val[ch]        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vb_vsplat_R(src_c[width - elem_counts - C + ch]), 1);
        HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_tmp, v_const_1);
        HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_7);
        mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wh_vmpy_VbVb(v_tmp, v_const_3);
        w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_5);
        mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wh_vmpy_VbVb(v_tmp, v_const_5);
        w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_src.val[ch], v_const_3);
        mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wh_vmpy_VbVb(mv_c_src.val[ch], v_const_1);
        w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, v_tmp, v_const_7);
        mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
        HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

        w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

        w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
        mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
        mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
    }

    vstore(dst_c  + j, mv_c_result);
    vstore(dst_c  + j + elem_counts, mv_r0_result);
    vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
    vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
    vstore(dst_n0 + j, mv_c_result);
    vstore(dst_n0 + j + elem_counts, mv_r0_result);
    vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
    vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        dst_n0[width * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        *(dst_c  - 2 * C + ch)                 = src_c[ch];
        *(dst_n0 - 2 * C + ch)                 = src_c[ch];
        *(dst_c  - C + ch)                     = src_c[ch];
        *(dst_n0 - C + ch)                     = src_c[ch];
    }

    return Status::OK;
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeBnUpX4BorderRow(const Tp *src_c, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts  = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align  = width & (~(elem_counts - 1));
    DT_S32 multiplier_1 = 1 + (1 << 16);
    DT_S32 multiplier_3 = 3 + (3 << 16);
    DT_S32 multiplier_5 = 5 + (5 << 16);
    DT_S32 multiplier_7 = 7 + (7 << 16);

    MVType mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);

    auto resizex4_border_func = [&]()
    {
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_3);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_5);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(mv_c_src.val[ch], multiplier_1);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, v_tmp, multiplier_7);
            mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c + i, mv_r0_src);

        resizex4_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);

        resizex4_border_func();
    }

    MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
    j = (width - elem_counts - C) * 4;
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector v_tmp        = mv_c_src.val[ch];
        mv_c_src.val[ch]        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vh_vsplat_R(src_c[width - elem_counts - C + ch]), 2);
        HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_1);
        HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_7);
        mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_3);
        w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_5);
        mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_tmp, multiplier_5);
        w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_src.val[ch], multiplier_3);
        mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0               = Q6_Wuw_vmpy_VuhRuh(mv_c_src.val[ch], multiplier_1);
        w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, v_tmp, multiplier_7);
        mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

        w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
        HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

        w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
        mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
        mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

        w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
        mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
        mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
    }

    vstore(dst_c  + j, mv_c_result);
    vstore(dst_c  + j + elem_counts, mv_r0_result);
    vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
    vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
    vstore(dst_n0 + j, mv_c_result);
    vstore(dst_n0 + j + elem_counts, mv_r0_result);
    vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
    vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        dst_n0[width * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        *(dst_c  - 2 * C + ch)                 = src_c[ch];
        *(dst_n0 - 2 * C + ch)                 = src_c[ch];
        *(dst_c  - C + ch)                     = src_c[ch];
        *(dst_n0 - C + ch)                     = src_c[ch];
    }

    return Status::OK;
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnUpX4BorderRow(const Tp *src_c, Tp *dst_c, Tp *dst_n0, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));

    MVType mv_c_src, mv_r0_src;

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    vload(src_c, mv_c_src);

    auto resizex4_border_func = [&]()
    {
        DT_S32 multiplier_1 = 1 + (1 << 16);
        DT_S32 multiplier_3 = 3 + (3 << 16);
        DT_S32 multiplier_5 = 5 + (5 << 16);
        DT_S32 multiplier_7 = 7 + (7 << 16);
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = Q6_V_valign_VVR(mv_r0_src.val[ch], mv_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_3);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_5);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(mv_c_src.val[ch], multiplier_1);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, v_tmp, multiplier_7);
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_src.val[ch] = mv_r0_src.val[ch];
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c + i, mv_r0_src);

        resizex4_border_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c + i, mv_c_src);
        vload(src_c + i + elem_counts, mv_r0_src);

        resizex4_border_func();
    }

    {
        DT_S32 multiplier_1 = 1 + (1 << 16);
        DT_S32 multiplier_3 = 3 + (3 << 16);
        DT_S32 multiplier_5 = 5 + (5 << 16);
        DT_S32 multiplier_7 = 7 + (7 << 16);
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;

        j = (width - elem_counts - C) * 4;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_tmp        = mv_c_src.val[ch];
            mv_c_src.val[ch]        = Q6_V_vlalign_VVR(mv_c_src.val[ch], Q6_Vh_vsplat_R(src_c[width - elem_counts - C + ch]), 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_3);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(v_tmp, multiplier_5);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, mv_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0               = Q6_Ww_vmpy_VhRh(mv_c_src.val[ch], multiplier_1);
            w_result             = Q6_Ww_vmpyacc_WwVhRh(w_tmp0, v_tmp, multiplier_7);
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 3);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_c  + j, mv_c_result);
        vstore(dst_c  + j + elem_counts, mv_r0_result);
        vstore(dst_c  + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c  + j + elem_counts * 3, mv_r2_result);
        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);
    }

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        dst_n0[width * 4 - C + ch - 2 * C]     = src_c[width - C + ch];
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = src_c[width - C + ch];
        *(dst_c  - 2 * C + ch)                 = src_c[ch];
        *(dst_n0 - 2 * C + ch)                 = src_c[ch];
        *(dst_c  - C + ch)                     = src_c[ch];
        *(dst_n0 - C + ch)                     = src_c[ch];
    }

    return Status::OK;
}

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeBnUpX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0,
                Tp *dst_n1, Tp *dst_n2, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    MVType mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    auto resizex4_func = [&]()
    {
        DT_S32 multiplier_1  = 1  + (1  << 8) + (1  << 16) + (1  << 24);
        DT_S32 multiplier_3  = 3  + (3  << 8) + (3  << 16) + (3  << 24);
        DT_S32 multiplier_5  = 5  + (5  << 8) + (5  << 16) + (5  << 24);
        DT_S32 multiplier_7  = 7  + (7  << 8) + (7  << 16) + (7  << 24);
        DT_S32 multiplier_9  = 9  + (9  << 8) + (9  << 16) + (9  << 24);
        DT_S32 multiplier_15 = 15 + (15 << 8) + (15 << 16) + (15 << 24);
        DT_S32 multiplier_21 = 21 + (21 << 8) + (21 << 16) + (21 << 24);
        DT_S32 multiplier_25 = 25 + (25 << 8) + (25 << 16) + (25 << 24);
        DT_S32 multiplier_35 = 35 + (35 << 8) + (35 << 16) + (35 << 24);
        DT_S32 multiplier_49 = 49 + (49 << 8) + (49 << 16) + (49 << 24);

        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_7);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_49);
            w_tmp0                  = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_1);
            w_result                = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_21);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_35);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_3);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_35);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_21);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_5);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_49);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_7);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_7);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_1);
            mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
        vstore(dst_c + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_5);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_35);
            w_tmp0                  = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_3);
            w_result                = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_21);
            mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_15);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_25);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_9);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_15);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_25);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_15);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_15);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_9);
            mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_35);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_5);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_21);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_3);
            mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_3);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_21);
            w_tmp0                  = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_5);
            w_result                = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_35);
            mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_9);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_15);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_15);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_25);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_15);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_9);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_25);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_15);
            mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_21);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_3);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_35);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_5);
            mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n1 + j, mv_c_result);
        vstore(dst_n1 + j + elem_counts, mv_r0_result);
        vstore(dst_n1 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n1 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_7);
            w_tmp0                  = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_7);
            w_result                = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_49);
            mv_c_result.val[ch]     = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_3);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_5);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_21);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_35);
            mv_r0_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_5);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_3);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_35);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_21);
            mv_r1_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuh_vmpy_VubRub(v_c_tmp, multiplier_7);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_c_c_src.val[ch], multiplier_1);
            w_tmp0               = Q6_Wh_vmpyacc_WhVubRb(w_result, v_n0_tmp, multiplier_49);
            w_result             = Q6_Wh_vmpyacc_WhVubRb(w_tmp0, mv_n0_c_src.val[ch], multiplier_7);
            mv_r2_result.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
            mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        }

        vstore(dst_n2 + j, mv_c_result);
        vstore(dst_n2 + j + elem_counts, mv_r0_result);
        vstore(dst_n2 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n2 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c  + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        resizex4_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);

        resizex4_func();
    }

    i = width - elem_counts - C;
    j = i * 4;
    Tp src_c_r[C] = {};
    Tp src_n0_r[C] = {};
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        mv_c_c_src.val[ch]   = Q6_V_vlalign_VVR(mv_c_c_src.val[ch],  Q6_Vb_vsplat_R(src_c[i + ch]),  1);
        mv_n0_c_src.val[ch]  = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0[i + ch]), 1);
        src_c_r[ch]          = src_c[width - C + ch];
        src_n0_r[ch]         = src_n0[width - C + ch];
        mv_c_r0_src.val[ch]  = Q6_Vb_vsplat_R(src_c_r[ch]);
        mv_n0_r0_src.val[ch] = Q6_Vb_vsplat_R(src_n0_r[ch]);
    }

    resizex4_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 7 + src_n0_r[ch] + 4) >> 3;
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = dst_c[width * 4 - C + ch - 2 * C];
        dst_n0[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 5 + src_n0_r[ch] * 3 + 4) >> 3;
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = dst_n0[width * 4 - C + ch - 2 * C];
        dst_n1[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 3 + src_n0_r[ch] * 5 + 4) >> 3;
        dst_n1[width * 4 - 2 * C + ch - 2 * C] = dst_n1[width * 4 - C + ch - 2 * C];
        dst_n2[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] + src_n0_r[ch] * 7 + 4) >> 3;
        dst_n2[width * 4 - 2 * C + ch - 2 * C] = dst_n2[width * 4 - C + ch - 2 * C];

        *(dst_c  - C + ch)     = (src_c[ch] * 7 + src_n0[ch] + 4) >> 3;
        *(dst_c  - 2 * C + ch) = *(dst_c - C + ch);
        *(dst_n0 - C + ch)     = (src_c[ch] * 5 + src_n0[ch] * 3 + 4) >> 3;
        *(dst_n0 - 2 * C + ch) = *(dst_n0 - C + ch);
        *(dst_n1 - C + ch)     = (src_c[ch] * 3 + src_n0[ch] * 5 + 4) >> 3;
        *(dst_n1 - 2 * C + ch) = *(dst_n1 - C + ch);
        *(dst_n2 - C + ch)     = (src_c[ch] + src_n0[ch] * 7 + 4) >> 3;
        *(dst_n2 - 2 * C + ch) = *(dst_n2 - C + ch);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
    return Status::OK;
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnUpX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0,
                Tp *dst_n1, Tp *dst_n2, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));

    MVType mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    vload(src_c,  mv_c_c_src);
    vload(src_n0, mv_n0_c_src);

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    auto resizex4_func = [&]()
    {
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(7));
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(49));
            w_tmp0                  = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(1));
            w_result                = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(7));
            mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(21));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(35));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(3));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(5));
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(35));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(21));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(5));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(3));
            mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(49));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(7));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(7));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(1));
            mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
        vstore(dst_c + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(5));
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(35));
            w_tmp0                  = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(3));
            w_result                = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(21));
            mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(15));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(25));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(9));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(15));
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(25));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(15));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(15));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(9));
            mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(35));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(5));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(21));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(3));
            mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(3));
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(21));
            w_tmp0                  = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(5));
            w_result                = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(35));
            mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(9));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(15));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(15));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(25));
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(15));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(9));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(25));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(15));
            mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(21));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(3));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(35));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(5));
            mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n1 + j, mv_c_result);
        vstore(dst_n1 + j + elem_counts, mv_r0_result);
        vstore(dst_n1 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n1 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 1);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 1);
            HVX_VectorPair w_tmp0   = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(1));
            HVX_VectorPair w_result = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(7));
            w_tmp0                  = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(7));
            w_result                = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(49));
            mv_c_result.val[ch]     = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(3));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(5));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(21));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(35));
            mv_r0_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(5));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(3));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(35));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(21));
            mv_r1_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wh_vmpy_VbVb(v_c_tmp, Q6_Vb_vsplat_R(7));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_c_c_src.val[ch], Q6_Vb_vsplat_R(1));
            w_tmp0               = Q6_Wh_vmpyacc_WhVbVb(w_result, v_n0_tmp, Q6_Vb_vsplat_R(49));
            w_result             = Q6_Wh_vmpyacc_WhVbVb(w_tmp0, mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(7));
            mv_r2_result.val[ch] = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -1);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -1);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -2);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -2);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
            mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        }

        vstore(dst_n2 + j, mv_c_result);
        vstore(dst_n2 + j + elem_counts, mv_r0_result);
        vstore(dst_n2 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n2 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c  + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        resizex4_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);

        resizex4_func();
    }

    i = width - elem_counts - C;
    j = i * 4;
    Tp src_c_r[C] = {};
    Tp src_n0_r[C] = {};
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        mv_c_c_src.val[ch]   = Q6_V_vlalign_VVR(mv_c_c_src.val[ch], Q6_Vb_vsplat_R(src_c[i + ch]), 1);
        mv_n0_c_src.val[ch]  = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vb_vsplat_R(src_n0[i + ch]), 1);
        src_c_r[ch]          = src_c[width - C + ch];
        src_n0_r[ch]         = src_n0[width - C + ch];
        mv_c_r0_src.val[ch]  = Q6_Vb_vsplat_R(src_c_r[ch]);
        mv_n0_r0_src.val[ch] = Q6_Vb_vsplat_R(src_n0_r[ch]);
    }

    resizex4_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 7 + src_n0_r[ch] + 4) >> 3;
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = dst_c[width * 4 - C + ch - 2 * C];
        dst_n0[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 5 + src_n0_r[ch] * 3 + 4) >> 3;
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = dst_n0[width * 4 - C + ch - 2 * C];
        dst_n1[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 3 + src_n0_r[ch] * 5 + 4) >> 3;
        dst_n1[width * 4 - 2 * C + ch - 2 * C] = dst_n1[width * 4 - C + ch - 2 * C];
        dst_n2[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] + src_n0_r[ch] * 7 + 4) >> 3;
        dst_n2[width * 4 - 2 * C + ch - 2 * C] = dst_n2[width * 4 - C + ch - 2 * C];

        *(dst_c  - C + ch)     = (src_c[ch] * 7 + src_n0[ch] + 4) >> 3;
        *(dst_c  - 2 * C + ch) = *(dst_c - C + ch);
        *(dst_n0 - C + ch)     = (src_c[ch] * 5 + src_n0[ch] * 3 + 4) >> 3;
        *(dst_n0 - 2 * C + ch) = *(dst_n0 - C + ch);
        *(dst_n1 - C + ch)     = (src_c[ch] * 3 + src_n0[ch] * 5 + 4) >> 3;
        *(dst_n1 - 2 * C + ch) = *(dst_n1 - C + ch);
        *(dst_n2 - C + ch)     = (src_c[ch] + src_n0[ch] * 7 + 4) >> 3;
        *(dst_n2 - 2 * C + ch) = *(dst_n2 - C + ch);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
    return Status::OK;
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeBnUpX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0,
                Tp *dst_n1, Tp *dst_n2, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));

    MVType mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    auto resizex4_func = [&]()
    {
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
        DT_S32 multiplier_1  = 1 + (1 << 16);
        DT_S32 multiplier_3  = 3 + (3 << 16);
        DT_S32 multiplier_5  = 5 + (5 << 16);
        DT_S32 multiplier_7  = 7 + (7 << 16);
        DT_S32 multiplier_9  = 9 + (9 << 16);
        DT_S32 multiplier_15 = 15 + (15 << 16);
        DT_S32 multiplier_21 = 21 + (21 << 16);
        DT_S32 multiplier_25 = 25 + (25 << 16);
        DT_S32 multiplier_35 = 35 + (35 << 16);
        DT_S32 multiplier_49 = 49 + (49 << 16);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_7);
            HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_49);
            w_tmp0                  = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_1);
            w_result                = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_7);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_21);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_35);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_3);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_5);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_35);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_21);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_5);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_3);
            mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_49);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_7);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_7);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_1);
            mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
        vstore(dst_c + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_5);
            HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_35);
            w_tmp0                  = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_3);
            w_result                = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_21);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_15);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_25);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_9);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_15);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_25);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_15);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_15);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_9);
            mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_35);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_5);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_21);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_3);
            mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_3);
            HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_21);
            w_tmp0                  = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_5);
            w_result                = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_35);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_9);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_15);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_15);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_25);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_15);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_9);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_25);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_15);
            mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_21);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_3);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_35);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_5);
            mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n1 + j, mv_c_result);
        vstore(dst_n1 + j + elem_counts, mv_r0_result);
        vstore(dst_n1 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n1 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_1);
            HVX_VectorPair w_result = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_7);
            w_tmp0                  = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_7);
            w_result                = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_49);
            mv_c_result.val[ch]     = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_3);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_5);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_21);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_35);
            mv_r0_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_5);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_3);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_35);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_21);
            mv_r1_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Wuw_vmpy_VuhRuh(v_c_tmp, multiplier_7);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_c_c_src.val[ch], multiplier_1);
            w_tmp0               = Q6_Wuw_vmpyacc_WuwVuhRuh(w_result, v_n0_tmp, multiplier_49);
            w_result             = Q6_Wuw_vmpyacc_WuwVuhRuh(w_tmp0, mv_n0_c_src.val[ch], multiplier_7);
            mv_r2_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
            mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        }

        vstore(dst_n2 + j, mv_c_result);
        vstore(dst_n2 + j + elem_counts, mv_r0_result);
        vstore(dst_n2 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n2 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c  + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        resizex4_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);

        resizex4_func();
    }

    i = width - elem_counts - C;
    j = i * 4;
    Tp src_c_r[C] = {};
    Tp src_n0_r[C] = {};
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        mv_c_c_src.val[ch]   = Q6_V_vlalign_VVR(mv_c_c_src.val[ch], Q6_Vh_vsplat_R(src_c[i + ch]), 2);
        mv_n0_c_src.val[ch]  = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0[i + ch]), 2);
        src_c_r[ch]          = src_c[width - C + ch];
        src_n0_r[ch]         = src_n0[width - C + ch];
        mv_c_r0_src.val[ch]  = Q6_Vh_vsplat_R(src_c_r[ch]);
        mv_n0_r0_src.val[ch] = Q6_Vh_vsplat_R(src_n0_r[ch]);
    }

    resizex4_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 7 + src_n0_r[ch] + 4) >> 3;
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = dst_c[width * 4 - C + ch - 2 * C];
        dst_n0[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 5 + src_n0_r[ch] * 3 + 4) >> 3;
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = dst_n0[width * 4 - C + ch - 2 * C];
        dst_n1[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 3 + src_n0_r[ch] * 5 + 4) >> 3;
        dst_n1[width * 4 - 2 * C + ch - 2 * C] = dst_n1[width * 4 - C + ch - 2 * C];
        dst_n2[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] + src_n0_r[ch] * 7 + 4) >> 3;
        dst_n2[width * 4 - 2 * C + ch - 2 * C] = dst_n2[width * 4 - C + ch - 2 * C];

        *(dst_c  - C + ch)     = (src_c[ch] * 7 + src_n0[ch] + 4) >> 3;
        *(dst_c  - 2 * C + ch) = *(dst_c - C + ch);
        *(dst_n0 - C + ch)     = (src_c[ch] * 5 + src_n0[ch] * 3 + 4) >> 3;
        *(dst_n0 - 2 * C + ch) = *(dst_n0 - C + ch);
        *(dst_n1 - C + ch)     = (src_c[ch] * 3 + src_n0[ch] * 5 + 4) >> 3;
        *(dst_n1 - 2 * C + ch) = *(dst_n1 - C + ch);
        *(dst_n2 - C + ch)     = (src_c[ch] + src_n0[ch] * 7 + 4) >> 3;
        *(dst_n2 - 2 * C + ch) = *(dst_n2 - C + ch);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
    return Status::OK;
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnUpX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, Tp *dst_n0,
                Tp *dst_n1, Tp *dst_n2, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));

    MVType mv_c_c_src, mv_c_r0_src, mv_n0_c_src, mv_n0_r0_src;
    vload(src_c, mv_c_c_src);
    vload(src_n0, mv_n0_c_src);

    DT_S32 i = elem_counts;
    DT_S32 j = 0;

    auto resizex4_func = [&]()
    {
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(7));
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(49));
            w_tmp0                  = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(1));
            w_result                = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(7));
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(21));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(35));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(3));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(5));
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(35));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(21));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(5));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(3));
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(49));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(7));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(7));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(1));
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_c + j, mv_c_result);
        vstore(dst_c + j + elem_counts, mv_r0_result);
        vstore(dst_c + j + elem_counts * 2, mv_r1_result);
        vstore(dst_c + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(5));
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(35));
            w_tmp0                  = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(3));
            w_result                = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(21));
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(15));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(25));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(9));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(15));
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(25));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(15));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(15));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(9));
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(35));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(5));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(21));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(3));
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n0 + j, mv_c_result);
        vstore(dst_n0 + j + elem_counts, mv_r0_result);
        vstore(dst_n0 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n0 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(3));
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(21));
            w_tmp0                  = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(5));
            w_result                = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(35));
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(9));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(15));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(15));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(25));
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(15));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(9));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(25));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(15));
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(21));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(3));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(35));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(5));
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);
        }

        vstore(dst_n1 + j, mv_c_result);
        vstore(dst_n1 + j + elem_counts, mv_r0_result);
        vstore(dst_n1 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n1 + j + elem_counts * 3, mv_r2_result);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_tmp      = Q6_V_valign_VVR(mv_c_r0_src.val[ch], mv_c_c_src.val[ch], 2);
            HVX_Vector v_n0_tmp     = Q6_V_valign_VVR(mv_n0_r0_src.val[ch], mv_n0_c_src.val[ch], 2);
            HVX_VectorPair w_tmp0   = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(1));
            HVX_VectorPair w_result = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(7));
            w_tmp0                  = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(7));
            w_result                = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(49));
            mv_c_result.val[ch]     = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(3));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(5));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(21));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(35));
            mv_r0_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(5));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(3));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(35));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(21));
            mv_r1_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0               = Q6_Ww_vmpy_VhVh(v_c_tmp, Q6_Vh_vsplat_R(7));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_c_c_src.val[ch], Q6_Vh_vsplat_R(1));
            w_tmp0               = Q6_Ww_vmpyacc_WwVhVh(w_result, v_n0_tmp, Q6_Vh_vsplat_R(49));
            w_result             = Q6_Ww_vmpyacc_WwVhVh(w_tmp0, mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(7));
            mv_r2_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), 6);

            w_tmp0                = Q6_W_vshuff_VVR(mv_r0_result.val[ch], mv_c_result.val[ch], -2);
            HVX_VectorPair w_tmp1 = Q6_W_vshuff_VVR(mv_r2_result.val[ch], mv_r1_result.val[ch], -2);

            w_result             = Q6_W_vshuff_VVR(Q6_V_lo_W(w_tmp1), Q6_V_lo_W(w_tmp0), -4);
            mv_c_result.val[ch]  = Q6_V_lo_W(w_result);
            mv_r0_result.val[ch] = Q6_V_hi_W(w_result);

            w_result             = Q6_W_vshuff_VVR(Q6_V_hi_W(w_tmp1), Q6_V_hi_W(w_tmp0), -4);
            mv_r1_result.val[ch] = Q6_V_lo_W(w_result);
            mv_r2_result.val[ch] = Q6_V_hi_W(w_result);

            mv_c_c_src.val[ch]  = mv_c_r0_src.val[ch];
            mv_n0_c_src.val[ch] = mv_n0_r0_src.val[ch];
        }

        vstore(dst_n2 + j, mv_c_result);
        vstore(dst_n2 + j + elem_counts, mv_r0_result);
        vstore(dst_n2 + j + elem_counts * 2, mv_r1_result);
        vstore(dst_n2 + j + elem_counts * 3, mv_r2_result);
    };

    for (; i < width_align; i += elem_counts, j += elem_counts * 4)
    {
        vload(src_c  + i, mv_c_r0_src);
        vload(src_n0 + i, mv_n0_r0_src);

        resizex4_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts * 2;
        j = i * 4;

        vload(src_c  + i, mv_c_c_src);
        vload(src_n0 + i, mv_n0_c_src);
        vload(src_c  + i + elem_counts, mv_c_r0_src);
        vload(src_n0 + i + elem_counts, mv_n0_r0_src);

        resizex4_func();
    }

    i = width - elem_counts - C;
    j = i * 4;
    Tp src_c_r[C] = {};
    Tp src_n0_r[C] = {};
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        mv_c_c_src.val[ch]   = Q6_V_vlalign_VVR(mv_c_c_src.val[ch], Q6_Vh_vsplat_R(src_c[i + ch]), 2);
        mv_n0_c_src.val[ch]  = Q6_V_vlalign_VVR(mv_n0_c_src.val[ch], Q6_Vh_vsplat_R(src_n0[i + ch]), 2);
        src_c_r[ch]          = src_c[width - C + ch];
        src_n0_r[ch]         = src_n0[width - C + ch];
        mv_c_r0_src.val[ch]  = Q6_Vh_vsplat_R(src_c_r[ch]);
        mv_n0_r0_src.val[ch] = Q6_Vh_vsplat_R(src_n0_r[ch]);
    }

    resizex4_func();

    for (DT_S32 ch = 0; ch < C; ch++)
    {
        dst_c[width  * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 7 + src_n0_r[ch] + 4) >> 3;
        dst_c[width  * 4 - 2 * C + ch - 2 * C] = dst_c[width * 4 - C + ch - 2 * C];
        dst_n0[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 5 + src_n0_r[ch] * 3 + 4) >> 3;
        dst_n0[width * 4 - 2 * C + ch - 2 * C] = dst_n0[width * 4 - C + ch - 2 * C];
        dst_n1[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] * 3 + src_n0_r[ch] * 5 + 4) >> 3;
        dst_n1[width * 4 - 2 * C + ch - 2 * C] = dst_n1[width * 4 - C + ch - 2 * C];
        dst_n2[width * 4 - C + ch - 2 * C]     = (src_c_r[ch] + src_n0_r[ch] * 7 + 4) >> 3;
        dst_n2[width * 4 - 2 * C + ch - 2 * C] = dst_n2[width * 4 - C + ch - 2 * C];

        *(dst_c  - C + ch)     = (src_c[ch] * 7 + src_n0[ch] + 4) >> 3;
        *(dst_c  - 2 * C + ch) = *(dst_c - C + ch);
        *(dst_n0 - C + ch)     = (src_c[ch] * 5 + src_n0[ch] * 3 + 4) >> 3;
        *(dst_n0 - 2 * C + ch) = *(dst_n0 - C + ch);
        *(dst_n1 - C + ch)     = (src_c[ch] * 3 + src_n0[ch] * 5 + 4) >> 3;
        *(dst_n1 - 2 * C + ch) = *(dst_n1 - C + ch);
        *(dst_n2 - C + ch)     = (src_c[ch] + src_n0[ch] * 7 + 4) >> 3;
        *(dst_n2 - 2 * C + ch) = *(dst_n2 - C + ch);
    }

#undef mv_n0_c_result
#undef mv_n0_r0_result
    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnUpX2HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_S32 y        = start_row;
    DT_S32 loop_row = Min(iheight - 1, end_row);
    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);

    if (0 == y)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        Tp *dst_c       = dst.Ptr<Tp>(0);
        L2Fetch(reinterpret_cast<DT_U64>(src_c), l2fetch_param);
        ResizeBnUpX2BorderRow<Tp, C>(src_c, dst_c, iwidth * C);
    }

    for (; y < loop_row; y++)
    {
        if (y + 4 < iheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y);
        const Tp *src_n0 = src.Ptr<Tp>(y + 1);
        Tp *dst_c        = dst.Ptr<Tp>(y * 2 + 1);
        Tp *dst_n0       = dst.Ptr<Tp>(y * 2 + 2);
        ResizeBnUpX2Row<Tp, C>(src_c, src_n0, dst_c, dst_n0, iwidth * C);
    }

    if (iheight == end_row)
    {
        const Tp *src_c = src.Ptr<Tp>(iheight - 1);
        Tp *dst_c       = dst.Ptr<Tp>(oheight - 1);
        ResizeBnUpX2BorderRow<Tp, C>(src_c, dst_c, iwidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnUpX4HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;
    DT_S32 ostride = dst.GetStrides().m_width;

    DT_S32 y         = start_row;
    DT_S32 loop_row  = Min(iheight - 1, end_row);
    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);

    if (0 == y)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        Tp *dst_c       = dst.Ptr<Tp>(0) + 2 * C;
        Tp *dst_n0      = dst_c + ostride / sizeof(Tp);
        L2Fetch(reinterpret_cast<DT_U64>(src_c), l2fetch_param);
        ResizeBnUpX4BorderRow<Tp, C>(src_c, dst_c, dst_n0, iwidth * C);
    }

    for (; y < loop_row; y++)
    {
        if (y + 4 < iheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y);
        const Tp *src_n0 = src.Ptr<Tp>(y + 1);
        Tp *dst_c        = dst.Ptr<Tp>(y * 4 + 2) + 2 * C;
        Tp *dst_n0       = dst_c  + ostride / sizeof(Tp);
        Tp *dst_n1       = dst_n0 + ostride / sizeof(Tp);
        Tp *dst_n2       = dst_n1 + ostride / sizeof(Tp);
        ResizeBnUpX4Row<Tp, C>(src_c, src_n0, dst_c, dst_n0, dst_n1, dst_n2, iwidth * C);
    }

    if (iheight == end_row)
    {
        const Tp *src_c  = src.Ptr<Tp>(iheight - 1);
        Tp *dst_c        = dst.Ptr<Tp>(oheight - 2) + 2 * C;
        Tp *dst_n0       = dst_c + ostride;
        ResizeBnUpX4BorderRow<Tp, C>(src_c, dst_c, dst_n0, iwidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnFastUpHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 oheight = dst.GetSizes().m_height;

    if ((owidth == 2 * iwidth) && (oheight == 2 * iheight))
    {
        ret = wp->ParallelFor((DT_S32)0, src.GetSizes().m_height, ResizeBnUpX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else if ((owidth == 4 * iwidth) && (oheight == 4 * iheight))
    {
        ret = wp->ParallelFor((DT_S32)0, src.GetSizes().m_height, ResizeBnUpX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "only support scale_x 0.5, 0.25, 2, 4");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ResizeBnFastUpHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = ResizeBnFastUpHvxHelper<Tp, 1>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper failed for c1");
            }
            break;
        }

        case 2:
        {
            ret = ResizeBnFastUpHvxHelper<Tp, 2>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper failed for c2");
            }
            break;
        }

        case 3:
        {
            ret = ResizeBnFastUpHvxHelper<Tp, 3>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper failed for c3");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3");
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeBnFastUpHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnFastUpHvxHelper<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnFastUpHvxHelper<DT_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnFastUpHvxHelper<DT_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnFastUpHvxHelper<DT_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastUpHvxHelper run failed, type: DT_S16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type is not supported.");
            ret = Status::ERROR;
        }
    }

    return Status::OK;
}

} // namespace aura