#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

enum class ResizeFastMethod
{
    DOWN_X2 = 0,
    DOWN_X4,
    UP_X2,
    UP_X4
};

template <typename Tp, MI_S32 C, ResizeFastMethod METHOD, typename Tp1 = AURA_VOID> struct ResizeNnFastRow;

// using Tp = MI_U8
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::UP_X2, typename std::enable_if<(sizeof(Tp) == 1)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_c_result, mv_r_result, mv_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu16_x_result = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
                wu16_x_result = Q6_Wh_vunpackoor_WhVb(wu16_x_result, mv_src.val[ch]);

                mv_c_result.val[ch] = Q6_V_lo_W(wu16_x_result);
                mv_r_result.val[ch] = Q6_V_hi_W(wu16_x_result);
            }
            vstore(dst_row + x * 2 * C, mv_c_result);
            vstore(dst_row + (x * 2 + elem_counts) * C, mv_r_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu16_x_result = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
                wu16_x_result = Q6_Wh_vunpackoor_WhVb(wu16_x_result, mv_src.val[ch]);

                mv_c_result.val[ch] = Q6_V_lo_W(wu16_x_result);
                mv_r_result.val[ch] = Q6_V_hi_W(wu16_x_result);
            }
            vstore(dst_row + x * 2 * C, mv_c_result);
            vstore(dst_row + (x * 2 + elem_counts) * C, mv_r_result);
        }
        return Status::OK;
    }
};

// using Tp = MI_U16
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::UP_X2, typename std::enable_if<(sizeof(Tp) == 2)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_c_result, mv_r_result, mv_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu32_x_result = Q6_Wuw_vunpack_Vuh(mv_src.val[ch]);
                wu32_x_result = Q6_Ww_vunpackoor_WwVh(wu32_x_result, mv_src.val[ch]);

                mv_c_result.val[ch] = Q6_V_lo_W(wu32_x_result);
                mv_r_result.val[ch] = Q6_V_hi_W(wu32_x_result);
            }
            vstore(dst_row + x * 2 * C, mv_c_result);
            vstore(dst_row + (x * 2 + elem_counts) * C, mv_r_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu32_x_result = Q6_Wuw_vunpack_Vuh(mv_src.val[ch]);
                wu32_x_result = Q6_Ww_vunpackoor_WwVh(wu32_x_result, mv_src.val[ch]);

                mv_c_result.val[ch] = Q6_V_lo_W(wu32_x_result);
                mv_r_result.val[ch] = Q6_V_hi_W(wu32_x_result);
            }
            vstore(dst_row + x * 2 * C, mv_c_result);
            vstore(dst_row + (x * 2 + elem_counts) * C, mv_r_result);
        }

        return Status::OK;
    }
};

// using Tp = MI_U8
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::UP_X4, typename std::enable_if<(sizeof(Tp) == 1)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result, mv_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu16_x_result = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
                wu16_x_result = Q6_Wh_vunpackoor_WhVb(wu16_x_result, mv_src.val[ch]);

                HVX_VectorPair wu16_result_l = Q6_Wuh_vunpack_Vub(Q6_V_lo_W(wu16_x_result));
                HVX_VectorPair wu16_result_h = Q6_Wuh_vunpack_Vub(Q6_V_hi_W(wu16_x_result));

                wu16_result_l = Q6_Wh_vunpackoor_WhVb(wu16_result_l, Q6_V_lo_W(wu16_x_result));
                wu16_result_h = Q6_Wh_vunpackoor_WhVb(wu16_result_h, Q6_V_hi_W(wu16_x_result));

                mv_c_result.val[ch]  = Q6_V_lo_W(wu16_result_l);
                mv_r0_result.val[ch] = Q6_V_hi_W(wu16_result_l);
                mv_r1_result.val[ch] = Q6_V_lo_W(wu16_result_h);
                mv_r2_result.val[ch] = Q6_V_hi_W(wu16_result_h);
            }
            vstore(dst_row + x * 4 * C, mv_c_result);
            vstore(dst_row + (x * 4 + elem_counts)        * C, mv_r0_result);
            vstore(dst_row + (x * 4 + (elem_counts << 1)) * C, mv_r1_result);
            vstore(dst_row + (x * 4 + (elem_counts *  3)) * C, mv_r2_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu16_x_result = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
                wu16_x_result = Q6_Wh_vunpackoor_WhVb(wu16_x_result, mv_src.val[ch]);

                HVX_VectorPair wu16_result_l = Q6_Wuh_vunpack_Vub(Q6_V_lo_W(wu16_x_result));
                HVX_VectorPair wu16_result_h = Q6_Wuh_vunpack_Vub(Q6_V_hi_W(wu16_x_result));

                wu16_result_l = Q6_Wh_vunpackoor_WhVb(wu16_result_l, Q6_V_lo_W(wu16_x_result));
                wu16_result_h = Q6_Wh_vunpackoor_WhVb(wu16_result_h, Q6_V_hi_W(wu16_x_result));

                mv_c_result.val[ch]  = Q6_V_lo_W(wu16_result_l);
                mv_r0_result.val[ch] = Q6_V_hi_W(wu16_result_l);
                mv_r1_result.val[ch] = Q6_V_lo_W(wu16_result_h);
                mv_r2_result.val[ch] = Q6_V_hi_W(wu16_result_h);
            }
            vstore(dst_row + x * 4 * C, mv_c_result);
            vstore(dst_row + (x * 4 + elem_counts)        * C, mv_r0_result);
            vstore(dst_row + (x * 4 + (elem_counts << 1)) * C, mv_r1_result);
            vstore(dst_row + (x * 4 + (elem_counts *  3)) * C, mv_r2_result);
        }

        return Status::OK;
    }
};

// using Tp = MI_U16
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::UP_X4, typename std::enable_if<(sizeof(Tp) == 2)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_c_result, mv_r0_result, mv_r1_result, mv_r2_result, mv_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu32_x_result = Q6_Wuw_vunpack_Vuh(mv_src.val[ch]);
                wu32_x_result = Q6_Ww_vunpackoor_WwVh(wu32_x_result, mv_src.val[ch]);

                HVX_VectorPair wu32_result_l = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(wu32_x_result));
                HVX_VectorPair wu32_result_h = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(wu32_x_result));

                wu32_result_l = Q6_Ww_vunpackoor_WwVh(wu32_result_l, Q6_V_lo_W(wu32_x_result));
                wu32_result_h = Q6_Ww_vunpackoor_WwVh(wu32_result_h, Q6_V_hi_W(wu32_x_result));

                mv_c_result.val[ch]  = Q6_V_lo_W(wu32_result_l);
                mv_r0_result.val[ch] = Q6_V_hi_W(wu32_result_l);
                mv_r1_result.val[ch] = Q6_V_lo_W(wu32_result_h);
                mv_r2_result.val[ch] = Q6_V_hi_W(wu32_result_h);
            }
            vstore(dst_row + x * 4 * C, mv_c_result);
            vstore(dst_row + (x * 4 + elem_counts)        * C, mv_r0_result);
            vstore(dst_row + (x * 4 + (elem_counts << 1)) * C, mv_r1_result);
            vstore(dst_row + (x * 4 + (elem_counts *  3)) * C, mv_r2_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * C, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair wu32_x_result = Q6_Wuw_vunpack_Vuh(mv_src.val[ch]);
                wu32_x_result = Q6_Ww_vunpackoor_WwVh(wu32_x_result, mv_src.val[ch]);

                HVX_VectorPair wu32_result_l = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(wu32_x_result));
                HVX_VectorPair wu32_result_h = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(wu32_x_result));

                wu32_result_l = Q6_Ww_vunpackoor_WwVh(wu32_result_l, Q6_V_lo_W(wu32_x_result));
                wu32_result_h = Q6_Ww_vunpackoor_WwVh(wu32_result_h, Q6_V_hi_W(wu32_x_result));

                mv_c_result.val[ch]  = Q6_V_lo_W(wu32_result_l);
                mv_r0_result.val[ch] = Q6_V_hi_W(wu32_result_l);
                mv_r1_result.val[ch] = Q6_V_lo_W(wu32_result_h);
                mv_r2_result.val[ch] = Q6_V_hi_W(wu32_result_h);
            }
            vstore(dst_row + x * 4 * C, mv_c_result);
            vstore(dst_row + (x * 4 + elem_counts)        * C, mv_r0_result);
            vstore(dst_row + (x * 4 + (elem_counts << 1)) * C, mv_r1_result);
            vstore(dst_row + (x * 4 + (elem_counts *  3)) * C, mv_r2_result);
        }

        return Status::OK;
    }
};

// using Tp = MI_U8
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::DOWN_X2, typename std::enable_if<(sizeof(Tp) == 1)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;
        width >>= 1;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_result, mv_c_src, mv_r_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * 2 * C, mv_c_src);
            vload(src_row + (x * 2 + elem_counts) * C, mv_r_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Q6_Vb_vpacke_VhVh(mv_r_src.val[ch], mv_c_src.val[ch]);
            }
            vstore(dst_row + x * C, mv_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * 2 * C, mv_c_src);
            vload(src_row + (x * 2 + elem_counts) * C, mv_r_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Q6_Vb_vpacke_VhVh(mv_r_src.val[ch], mv_c_src.val[ch]);
            }
            vstore(dst_row + x * C, mv_result);
        }

        return Status::OK;
    }
};

// using Tp = MI_U16
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::DOWN_X2, typename std::enable_if<(sizeof(Tp) == 2)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;
        width >>= 1;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_result, mv_c_src, mv_r_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * 2 * C, mv_c_src);
            vload(src_row + (x * 2 + elem_counts) * C, mv_r_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Q6_Vh_vpacke_VwVw(mv_r_src.val[ch], mv_c_src.val[ch]);
            }
            vstore(dst_row + x * C, mv_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * 2 * C, mv_c_src);
            vload(src_row + (x * 2 + elem_counts) * C, mv_r_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Q6_Vh_vpacke_VwVw(mv_r_src.val[ch], mv_c_src.val[ch]);
            }
            vstore(dst_row + x * C, mv_result);
        }

        return Status::OK;
    }
};

// using Tp = MI_U8
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::DOWN_X4, typename std::enable_if<(sizeof(Tp) == 1)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;
        width >>= 2;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_result, mv_c_src, mv_r0_src, mv_r1_src, mv_r2_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * 4 * C, mv_c_src);
            vload(src_row + (x * 4 + 1 * elem_counts) * C, mv_r0_src);
            vload(src_row + (x * 4 + 2 * elem_counts) * C, mv_r1_src);
            vload(src_row + (x * 4 + 3 * elem_counts) * C, mv_r2_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector vu8_x0_result = Q6_Vb_vpacke_VhVh(mv_r0_src.val[ch], mv_c_src.val[ch]);
                HVX_Vector vu8_x1_result = Q6_Vb_vpacke_VhVh(mv_r2_src.val[ch], mv_r1_src.val[ch]);
                mv_result.val[ch]      = Q6_Vb_vpacke_VhVh(vu8_x1_result, vu8_x0_result);
            }
            vstore(dst_row + x * C, mv_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * 4 * C, mv_c_src);
            vload(src_row + (x * 4 + 1 * elem_counts) * C, mv_r0_src);
            vload(src_row + (x * 4 + 2 * elem_counts) * C, mv_r1_src);
            vload(src_row + (x * 4 + 3 * elem_counts) * C, mv_r2_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector vu8_x0_result = Q6_Vb_vpacke_VhVh(mv_r0_src.val[ch], mv_c_src.val[ch]);
                HVX_Vector vu8_x1_result = Q6_Vb_vpacke_VhVh(mv_r2_src.val[ch], mv_r1_src.val[ch]);
                mv_result.val[ch]      = Q6_Vb_vpacke_VhVh(vu8_x1_result, vu8_x0_result);
            }
            vstore(dst_row + x * C, mv_result);
        }
        return Status::OK;
    }
};

// using Tp = MI_U16
template <typename Tp, MI_S32 C>
struct ResizeNnFastRow<Tp, C, ResizeFastMethod::DOWN_X4, typename std::enable_if<(sizeof(Tp) == 2)>::type>
{
    Status operator()(const Tp *src_row, Tp *dst_row, MI_S32 width)
    {
        using MVType = typename MVHvxVector<C>::Type;
        width >>= 2;

        MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
        MI_S32 width_align = width & (-elem_counts);
        MVType mv_result, mv_c_src, mv_r0_src, mv_r1_src, mv_r2_src;

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src_row + x * 4 * C, mv_c_src);
            vload(src_row + (x * 4 + 1 * elem_counts) * C, mv_r0_src);
            vload(src_row + (x * 4 + 2 * elem_counts) * C, mv_r1_src);
            vload(src_row + (x * 4 + 3 * elem_counts) * C, mv_r2_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector vu16_x0_result = Q6_Vh_vpacke_VwVw(mv_r0_src.val[ch], mv_c_src.val[ch]);
                HVX_Vector vu16_x1_result = Q6_Vh_vpacke_VwVw(mv_r2_src.val[ch], mv_r1_src.val[ch]);
                mv_result.val[ch]         = Q6_Vh_vpacke_VwVw(vu16_x1_result, vu16_x0_result);
            }
            vstore(dst_row + x * C, mv_result);
        }

        if (width_align < width)
        {
            MI_S32 x = width - elem_counts;
            vload(src_row + x * 4 * C, mv_c_src);
            vload(src_row + (x * 4 + 1 * elem_counts) * C, mv_r0_src);
            vload(src_row + (x * 4 + 2 * elem_counts) * C, mv_r1_src);
            vload(src_row + (x * 4 + 3 * elem_counts) * C, mv_r2_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector vu16_x0_result = Q6_Vh_vpacke_VwVw(mv_r0_src.val[ch], mv_c_src.val[ch]);
                HVX_Vector vu16_x1_result = Q6_Vh_vpacke_VwVw(mv_r2_src.val[ch], mv_r1_src.val[ch]);
                mv_result.val[ch]         = Q6_Vh_vpacke_VwVw(vu16_x1_result, vu16_x0_result);
            }
            vstore(dst_row + x * C, mv_result);
        }
        return Status::OK;
    }
};

template<typename Tp, MI_S32 C, ResizeFastMethod METHOD>
static Status ResizeNnFastHvxImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    auto functor = ResizeNnFastRow<Tp, C, METHOD>();

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;
    MI_S32 ostride = dst.GetStrides().m_width;
    MI_F32 scale_y = static_cast<MI_F32>(iheight) / oheight;

    MI_U32 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 1, 0);
    MI_S32 idx_src_p     = Min(static_cast<MI_S32>(start_row * scale_y), iheight - 1);
    MI_S32 idx_src_c     = idx_src_p;
    MI_S32 idx_src_n     = idx_src_p;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        idx_src_n = Min(static_cast<MI_S32>((y + 1) * scale_y), iheight - 1);

        if ((y + 1 < oheight) && (idx_src_n != idx_src_c))
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(idx_src_n)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(idx_src_c);
        Tp *dst_c = dst.Ptr<Tp>(y);

        if ((y == start_row) || (idx_src_p != idx_src_c))
        {
            functor(src_c, dst_c, iwidth);
        }
        else
        {
            Tp *dst_p = dst.Ptr<Tp>(y - 1);
            AuraMemCopy(dst_c, dst_p, ostride);
        }

        idx_src_p = idx_src_c;
        idx_src_c = idx_src_n;
    }
    return Status::OK;
}

template<typename Tp, MI_S32 C>
static Status ResizeNnFastHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;

    if (owidth == 2 * iwidth)
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeNnFastHvxImpl<Tp, C, ResizeFastMethod::UP_X2>, std::cref(src), std::ref(dst));
    }
    else if (owidth == 4 * iwidth)
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeNnFastHvxImpl<Tp, C, ResizeFastMethod::UP_X4>, std::cref(src), std::ref(dst));
    }
    else if (iwidth == 2 * owidth)
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeNnFastHvxImpl<Tp, C, ResizeFastMethod::DOWN_X2>, std::cref(src), std::ref(dst));
    }
    else if (iwidth == 4 * owidth)
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeNnFastHvxImpl<Tp, C, ResizeFastMethod::DOWN_X4>, std::cref(src), std::ref(dst));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "only support scale_x 0.5, 0.25, 2, 4");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

template<typename Tp>
static Status ResizeNnFastHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    MI_S32 channel = src.GetSizes().m_channel;
    Status ret     = Status::ERROR;
    switch (channel)
    {
        case 1:
        {
            ret = ResizeNnFastHvxHelper<Tp, 1>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnFastHvxHelper failed for c1");
            }
            break;
        }

        case 2:
        {
            ret = ResizeNnFastHvxHelper<Tp, 2>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnFastHvxHelper failed for c2");
            }
            break;
        }

        case 3:
        {
            ret = ResizeNnFastHvxHelper<Tp, 3>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnFastHvxHelper failed for c3");
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

Status ResizeNnFastHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = ResizeNnFastHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnFastHvxHelper run failed, type: MI_U8/MI_S8");
            }
            break;
        }

        case ElemType::U16:
        case ElemType::S16:
        {
            ret = ResizeNnFastHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnFastHvxHelper run failed, type: MI_U16/MI_S16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type is not supported.");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura