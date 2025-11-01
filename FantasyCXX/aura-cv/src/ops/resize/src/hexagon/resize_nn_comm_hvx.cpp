#include "resize_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

struct ResizeNnVtcmBuffer
{
    MI_U8 *xofs;
    MI_U8 *yofs;
    MI_U8 *src_buffer;
    MI_S32 src_buffer_pitch;
    MI_U8 *gather_buffer;
};

static Status GetResizeNnCommOffset(MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight, ResizeNnVtcmBuffer *vtcm_buffer)
{
    if (MI_NULL == vtcm_buffer)
    {
        return Status::ERROR;
    }

    MI_F64 inv_scale_x = static_cast<MI_F64>(owidth) / iwidth;
    MI_F64 inv_scale_y = static_cast<MI_F64>(oheight) / iheight;
    MI_F64 scale_x = 1. / inv_scale_x;
    MI_F64 scale_y = 1. / inv_scale_y;

    MI_U16 *xofs = reinterpret_cast<MI_U16*>(vtcm_buffer->xofs);
    MI_U16 *yofs = reinterpret_cast<MI_U16*>(vtcm_buffer->yofs);

    {
        MI_S32 x = 0;
        MI_S32 t_max = static_cast<MI_S32>(Ceil((iwidth - 1) / scale_x));
        MI_S32 xmax = Min(owidth, t_max);

        for (; x < xmax; x++)
        {
            xofs[x] = static_cast<MI_U16>(Floor(x * scale_x)) << 1;
        }

        for (; x < owidth; x++)
        {
            xofs[x] = (iwidth - 1) << 1;
        }
    }

    {
        MI_S32 y = 0;
        MI_S32 t_max =static_cast<MI_S32>(Ceil((iheight - 1) * inv_scale_y));
        MI_S32 ymax = Min(oheight, t_max);

        for (; y < ymax; y++)
        {
            yofs[y] = static_cast<MI_U16>(Floor(y * scale_y));
        }

        for (; y < oheight; y++)
        {
            yofs[y] = iheight - 1;
        }
    }

    return Status::OK;
}

// using Tp = MI_U8
template<typename Tp, MI_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U8, Tp>::value, Status>::type
ResizeNnCommRowVCore(Tp *src_row, MI_U16 *src_buffer, MI_S32 iwidth, MI_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = iwidth & (-elem_counts);
    MVType mv_src;

    for (MI_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *vu16_src_buffer_ptr = (HVX_Vector *)(src_buffer + x + ch * (src_buffer_pitch >> 1));
            HVX_VectorPair wu16_result      = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
            *vu16_src_buffer_ptr++          = Q6_V_lo_W(wu16_result);
            *vu16_src_buffer_ptr            = Q6_V_hi_W(wu16_result);
        }
    }

    if (width_align < iwidth)
    {
        MI_S32 x = iwidth - elem_counts;
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr      = src_buffer + x + ch * (src_buffer_pitch >> 1);
            HVX_VectorPair wu16_result  = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);

            HVX_Vector vu16_result_lo, vu16_result_hi;
            vu16_result_lo = Q6_V_lo_W(wu16_result);
            vu16_result_hi = Q6_V_hi_W(wu16_result);
            vstore(src_buffer_ptr, vu16_result_lo);
            vstore(src_buffer_ptr + AURA_HALF_HVLEN, vu16_result_hi);
        }
    }

    return Status::OK;
}

// using Tp = MI_U8
template<typename Tp, MI_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U8, Tp>::value, Status>::type
ResizeNnCommRowHCore(MI_U16 *src_buffer, MI_U16 *xofs, MI_U16 *gather_buffer, Tp *dst_row, MI_S32 iwidth, MI_S32 owidth, MI_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = owidth & (-elem_counts);
    MVType mv_result;
    MI_S32 x = 0;

    HVX_Vector *vu16_x_ofs        = (HVX_Vector *)xofs;
    HVX_Vector *vu16_x_swap[2]    = {MI_NULL};
    HVX_Vector *vu16_x0_gather[C] = {MI_NULL};
    HVX_Vector *vu16_x1_gather[C] = {MI_NULL};
    HVX_Vector *vu16_x2_gather[C] = {MI_NULL};
    HVX_Vector *vu16_x3_gather[C] = {MI_NULL};

    HVX_Vector vu16_x0_idx = *vu16_x_ofs++;
    HVX_Vector vu16_x1_idx = *vu16_x_ofs++;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector *vu16_gather_buffer_ptr = (HVX_Vector *)(gather_buffer + ch * (AURA_HALF_HVLEN << 2));
        vu16_x0_gather[ch] = vu16_gather_buffer_ptr + 0;
        vu16_x1_gather[ch] = vu16_gather_buffer_ptr + 1;
        vu16_x2_gather[ch] = vu16_gather_buffer_ptr + 2;
        vu16_x3_gather[ch] = vu16_gather_buffer_ptr + 3;

        MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);

        Q6_vgather_ARMVh(vu16_x0_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
        Q6_vgather_ARMVh(vu16_x1_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);
    }

    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vu16_x0_idx = *vu16_x_ofs++;
        vu16_x1_idx = *vu16_x_ofs++;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x2_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
            Q6_vgather_ARMVh(vu16_x3_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);

            mv_result.val[ch] = Q6_Vb_vshuffe_VbVb(*vu16_x1_gather[ch], *vu16_x0_gather[ch]);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(mv_result.val[ch]);

            vu16_x_swap[0]     = vu16_x0_gather[ch];
            vu16_x0_gather[ch] = vu16_x2_gather[ch];
            vu16_x2_gather[ch] = vu16_x_swap[0];

            vu16_x_swap[1]     = vu16_x1_gather[ch];
            vu16_x1_gather[ch] = vu16_x3_gather[ch];
            vu16_x3_gather[ch] = vu16_x_swap[1];
        }
        vstore(dst_row + x * C, mv_result);
    }

    {
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_result.val[ch] = Q6_Vb_vshuffe_VbVb(*vu16_x1_gather[ch], *vu16_x0_gather[ch]);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(mv_result.val[ch]);
        }
        vstore(dst_row + x * C, mv_result);
    }

    if (width_align < owidth)
    {
        x = owidth - elem_counts;
        vload(xofs + x, vu16_x0_idx);
        vload(xofs + x + AURA_HALF_HVLEN, vu16_x1_idx);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x0_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
            Q6_vgather_ARMVh(vu16_x1_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);
            mv_result.val[ch] = Q6_Vb_vshuffe_VbVb(*vu16_x1_gather[ch], *vu16_x0_gather[ch]);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(mv_result.val[ch]);
        }
        vstore(dst_row + x * C, mv_result);
    }
    return Status::OK;
}

// using Tp = MI_U16
template<typename Tp, MI_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U16, Tp>::value, Status>::type
ResizeNnCommRowVCore(Tp *src_row, MI_U16 *src_buffer, MI_S32 iwidth, MI_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = iwidth & (-elem_counts);
    MVType mv_src;

    for (MI_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *vu16_src_buffer_ptr = (HVX_Vector *)(src_buffer + x + ch * (src_buffer_pitch >> 1));
            *vu16_src_buffer_ptr            = mv_src.val[ch];
        }
    }

    if (width_align < iwidth)
    {
        MI_S32 x = iwidth - elem_counts;
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr = src_buffer + x + ch * (src_buffer_pitch >> 1);
            vstore(src_buffer_ptr, mv_src.val[ch]);
        }
    }

    return Status::OK;
}

// using Tp = MI_U16
template<typename Tp, MI_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U16, Tp>::value, Status>::type
ResizeNnCommRowHCore(MI_U16 *src_buffer, MI_U16 *xofs, MI_U16 *gather_buffer, Tp *dst_row, MI_S32 iwidth, MI_S32 owidth, MI_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = owidth & (-elem_counts);
    MI_S32 x = 0;
    MVType mv_result;

    HVX_Vector *vu16_x_ofs        = (HVX_Vector *)xofs;
    HVX_Vector *vu16_x_swap       = MI_NULL;
    HVX_Vector *vu16_x0_gather[C] = {MI_NULL};
    HVX_Vector *vu16_x1_gather[C] = {MI_NULL};

    HVX_Vector vu16_x_idx = *vu16_x_ofs++;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector *vu16_gather_buffer_ptr = (HVX_Vector *)(gather_buffer + ch * (AURA_HALF_HVLEN << 1));
        vu16_x0_gather[ch] = vu16_gather_buffer_ptr + 0;
        vu16_x1_gather[ch] = vu16_gather_buffer_ptr + 1;

        MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);

        Q6_vgather_ARMVh(vu16_x0_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);
    }

    for(; x < width_align - elem_counts; x += elem_counts)
    {
        vu16_x_idx = *vu16_x_ofs++;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x1_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);

            mv_result.val[ch]  = *vu16_x0_gather[ch];
            vu16_x_swap        = vu16_x1_gather[ch];
            vu16_x1_gather[ch] = vu16_x0_gather[ch];
            vu16_x0_gather[ch] = vu16_x_swap;
        }
        vstore(dst_row + x * C, mv_result);
    }

    {
        vu16_x_idx = *vu16_x_ofs++;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_result.val[ch] = *vu16_x0_gather[ch];
        }
        vstore(dst_row + x * C, mv_result);
    }

    if (width_align < owidth)
    {
        x          = owidth - elem_counts;
        vload(xofs + x, vu16_x_idx);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            MI_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x0_gather[ch], (MI_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);
            mv_result.val[ch] = *vu16_x0_gather[ch];
        }
        vstore(dst_row + x * C, mv_result);
    }
    return Status::OK;
}

template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value | std::is_same<MI_U16, Tp>::value,  Status>::type
ResizeNnCommHvxImpl(const Mat &src, Mat &dst, ResizeNnVtcmBuffer *vtcm_buffer, MI_S32 thread_num, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 ostride   = dst.GetStrides().m_width;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_row) * thread_num / oheight);

    MI_U16 *xofs          = reinterpret_cast<MI_U16*>(vtcm_buffer->xofs);
    MI_U16 *yofs          = reinterpret_cast<MI_U16*>(vtcm_buffer->yofs);
    MI_U16 *src_buffer    = reinterpret_cast<MI_U16*>(vtcm_buffer->src_buffer + C * vtcm_buffer->src_buffer_pitch * thread_id);
    MI_U16 *gather_buffer = reinterpret_cast<MI_U16*>(vtcm_buffer->gather_buffer + C * (AURA_HVLEN << 2) * thread_id);
    MI_U32 l2fetch_param  = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if ((y + 1 < end_row) && (yofs[y + 1] != yofs[y]))
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
        }

        Tp *src_c = (Tp *)src.Ptr<Tp>(yofs[y]);
        Tp *dst_c = (Tp *)dst.Ptr<Tp>(y);

        if ((start_row == y) || (yofs[y] != yofs[y - 1]))
        {
            ResizeNnCommRowVCore<Tp, C>(src_c, src_buffer, iwidth, vtcm_buffer->src_buffer_pitch);
            ResizeNnCommRowHCore<Tp, C>(src_buffer, xofs, gather_buffer, dst_c, iwidth, owidth, vtcm_buffer->src_buffer_pitch);
        }
        else
        {
            Tp *dst_p = (Tp *)dst.Ptr<Tp>(y - 1);
            AuraMemCopy(dst_c, dst_p, ostride);
        }
    }
    return Status::OK;
}

template<typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value | std::is_same<MI_U16, Tp>::value,  Status>::type
ResizeNnCommHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheihgt = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;
    MI_S32 thread_num = wp->GetComputeThreadNum();

    MI_S32 xofs_size          = AURA_ALIGN(owidth  * sizeof(MI_U16), AURA_HVLEN);
    MI_S32 yofs_size          = AURA_ALIGN(oheight * sizeof(MI_U16), AURA_HVLEN);
    MI_S32 src_buffer_size    = AURA_ALIGN(iwidth  * sizeof(MI_U16), AURA_HVLEN) * thread_num * channel;
    MI_S32 gather_buffer_size = (AURA_HVLEN << 2) * channel * thread_num;
    MI_S32 total_buffer_size  = xofs_size + yofs_size + src_buffer_size + gather_buffer_size;

    MI_U8 *vtcm_mem = static_cast<MI_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
    if (MI_NULL == vtcm_mem)
    {
        AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
        AURA_FREE(ctx, vtcm_mem);
        return Status::ABORT;
    }

    struct ResizeNnVtcmBuffer vtcm_buffer;
    vtcm_buffer.xofs             = vtcm_mem;
    vtcm_buffer.yofs             = vtcm_buffer.xofs + xofs_size;
    vtcm_buffer.src_buffer       = vtcm_buffer.yofs + yofs_size;
    vtcm_buffer.src_buffer_pitch = src_buffer_size / (thread_num * channel);
    vtcm_buffer.gather_buffer    = vtcm_buffer.src_buffer  + src_buffer_size;

    ret = GetResizeNnCommOffset(iwidth, iheihgt, owidth, oheight, &vtcm_buffer);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetResizeNnCommOffset failed");
        AURA_FREE(ctx, vtcm_mem);
        return ret;
    }

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 1>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeNnCommHvxImpl of c1 failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 2>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeNnCommHvxImpl of c2 failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 3>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeNnCommHvxImpl of c3 failed");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3");
        }
    }

    AURA_FREE(ctx, vtcm_mem);
    AURA_RETURN(ctx, ret);
}

Status ResizeNnCommHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        case ElemType::S8:
        {
            ret = ResizeNnCommHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnCommHvxHelper run failed, type: MI_U8/MI_S8");
            }
            break;
        }

        case ElemType::S16:
        case ElemType::U16:
        {
            ret = ResizeNnCommHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnCommHvxHelper run failed, type: U16/S16");
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