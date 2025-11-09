#include "resize_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

struct ResizeNnVtcmBuffer
{
    DT_U8 *xofs;
    DT_U8 *yofs;
    DT_U8 *src_buffer;
    DT_S32 src_buffer_pitch;
    DT_U8 *gather_buffer;
};

static Status GetResizeNnCommOffset(DT_S32 iwidth, DT_S32 iheight, DT_S32 owidth, DT_S32 oheight, ResizeNnVtcmBuffer *vtcm_buffer)
{
    if (DT_NULL == vtcm_buffer)
    {
        return Status::ERROR;
    }

    DT_F64 inv_scale_x = static_cast<DT_F64>(owidth) / iwidth;
    DT_F64 inv_scale_y = static_cast<DT_F64>(oheight) / iheight;
    DT_F64 scale_x = 1. / inv_scale_x;
    DT_F64 scale_y = 1. / inv_scale_y;

    DT_U16 *xofs = reinterpret_cast<DT_U16*>(vtcm_buffer->xofs);
    DT_U16 *yofs = reinterpret_cast<DT_U16*>(vtcm_buffer->yofs);

    {
        DT_S32 x = 0;
        DT_S32 t_max = static_cast<DT_S32>(Ceil((iwidth - 1) / scale_x));
        DT_S32 xmax = Min(owidth, t_max);

        for (; x < xmax; x++)
        {
            xofs[x] = static_cast<DT_U16>(Floor(x * scale_x)) << 1;
        }

        for (; x < owidth; x++)
        {
            xofs[x] = (iwidth - 1) << 1;
        }
    }

    {
        DT_S32 y = 0;
        DT_S32 t_max =static_cast<DT_S32>(Ceil((iheight - 1) * inv_scale_y));
        DT_S32 ymax = Min(oheight, t_max);

        for (; y < ymax; y++)
        {
            yofs[y] = static_cast<DT_U16>(Floor(y * scale_y));
        }

        for (; y < oheight; y++)
        {
            yofs[y] = iheight - 1;
        }
    }

    return Status::OK;
}

// using Tp = DT_U8
template<typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeNnCommRowVCore(Tp *src_row, DT_U16 *src_buffer, DT_S32 iwidth, DT_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align = iwidth & (-elem_counts);
    MVType mv_src;

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *vu16_src_buffer_ptr = (HVX_Vector *)(src_buffer + x + ch * (src_buffer_pitch >> 1));
            HVX_VectorPair wu16_result      = Q6_Wuh_vunpack_Vub(mv_src.val[ch]);
            *vu16_src_buffer_ptr++          = Q6_V_lo_W(wu16_result);
            *vu16_src_buffer_ptr            = Q6_V_hi_W(wu16_result);
        }
    }

    if (width_align < iwidth)
    {
        DT_S32 x = iwidth - elem_counts;
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr      = src_buffer + x + ch * (src_buffer_pitch >> 1);
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

// using Tp = DT_U8
template<typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeNnCommRowHCore(DT_U16 *src_buffer, DT_U16 *xofs, DT_U16 *gather_buffer, Tp *dst_row, DT_S32 iwidth, DT_S32 owidth, DT_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align = owidth & (-elem_counts);
    MVType mv_result;
    DT_S32 x = 0;

    HVX_Vector *vu16_x_ofs        = (HVX_Vector *)xofs;
    HVX_Vector *vu16_x_swap[2]    = {DT_NULL};
    HVX_Vector *vu16_x0_gather[C] = {DT_NULL};
    HVX_Vector *vu16_x1_gather[C] = {DT_NULL};
    HVX_Vector *vu16_x2_gather[C] = {DT_NULL};
    HVX_Vector *vu16_x3_gather[C] = {DT_NULL};

    HVX_Vector vu16_x0_idx = *vu16_x_ofs++;
    HVX_Vector vu16_x1_idx = *vu16_x_ofs++;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector *vu16_gather_buffer_ptr = (HVX_Vector *)(gather_buffer + ch * (AURA_HALF_HVLEN << 2));
        vu16_x0_gather[ch] = vu16_gather_buffer_ptr + 0;
        vu16_x1_gather[ch] = vu16_gather_buffer_ptr + 1;
        vu16_x2_gather[ch] = vu16_gather_buffer_ptr + 2;
        vu16_x3_gather[ch] = vu16_gather_buffer_ptr + 3;

        DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);

        Q6_vgather_ARMVh(vu16_x0_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
        Q6_vgather_ARMVh(vu16_x1_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);
    }

    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vu16_x0_idx = *vu16_x_ofs++;
        vu16_x1_idx = *vu16_x_ofs++;

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x2_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
            Q6_vgather_ARMVh(vu16_x3_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);

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
        for (DT_S32 ch = 0; ch < C; ch++)
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
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x0_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x0_idx);
            Q6_vgather_ARMVh(vu16_x1_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x1_idx);
            mv_result.val[ch] = Q6_Vb_vshuffe_VbVb(*vu16_x1_gather[ch], *vu16_x0_gather[ch]);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(mv_result.val[ch]);
        }
        vstore(dst_row + x * C, mv_result);
    }
    return Status::OK;
}

// using Tp = DT_U16
template<typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeNnCommRowVCore(Tp *src_row, DT_U16 *src_buffer, DT_S32 iwidth, DT_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align = iwidth & (-elem_counts);
    MVType mv_src;

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *vu16_src_buffer_ptr = (HVX_Vector *)(src_buffer + x + ch * (src_buffer_pitch >> 1));
            *vu16_src_buffer_ptr            = mv_src.val[ch];
        }
    }

    if (width_align < iwidth)
    {
        DT_S32 x = iwidth - elem_counts;
        vload(src_row + x * C, mv_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr = src_buffer + x + ch * (src_buffer_pitch >> 1);
            vstore(src_buffer_ptr, mv_src.val[ch]);
        }
    }

    return Status::OK;
}

// using Tp = DT_U16
template<typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeNnCommRowHCore(DT_U16 *src_buffer, DT_U16 *xofs, DT_U16 *gather_buffer, Tp *dst_row, DT_S32 iwidth, DT_S32 owidth, DT_S32 src_buffer_pitch)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align = owidth & (-elem_counts);
    DT_S32 x = 0;
    MVType mv_result;

    HVX_Vector *vu16_x_ofs        = (HVX_Vector *)xofs;
    HVX_Vector *vu16_x_swap       = DT_NULL;
    HVX_Vector *vu16_x0_gather[C] = {DT_NULL};
    HVX_Vector *vu16_x1_gather[C] = {DT_NULL};

    HVX_Vector vu16_x_idx = *vu16_x_ofs++;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector *vu16_gather_buffer_ptr = (HVX_Vector *)(gather_buffer + ch * (AURA_HALF_HVLEN << 1));
        vu16_x0_gather[ch] = vu16_gather_buffer_ptr + 0;
        vu16_x1_gather[ch] = vu16_gather_buffer_ptr + 1;

        DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);

        Q6_vgather_ARMVh(vu16_x0_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);
    }

    for(; x < width_align - elem_counts; x += elem_counts)
    {
        vu16_x_idx = *vu16_x_ofs++;

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x1_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);

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
        for (DT_S32 ch = 0; ch < C; ch++)
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
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *src_buffer_ptr = src_buffer + ch * (src_buffer_pitch >> 1);
            Q6_vgather_ARMVh(vu16_x0_gather[ch], (DT_U32)src_buffer_ptr, (iwidth << 1) - 1, vu16_x_idx);
            mv_result.val[ch] = *vu16_x0_gather[ch];
        }
        vstore(dst_row + x * C, mv_result);
    }
    return Status::OK;
}

template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value | std::is_same<DT_U16, Tp>::value,  Status>::type
ResizeNnCommHvxImpl(const Mat &src, Mat &dst, ResizeNnVtcmBuffer *vtcm_buffer, DT_S32 thread_num, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth    = src.GetSizes().m_width;
    DT_S32 istride   = src.GetStrides().m_width;
    DT_S32 ostride   = dst.GetStrides().m_width;
    DT_S32 owidth    = dst.GetSizes().m_width;
    DT_S32 oheight   = dst.GetSizes().m_height;
    DT_S32 thread_id = SaturateCast<DT_S32>(static_cast<DT_F32>(start_row) * thread_num / oheight);

    DT_U16 *xofs          = reinterpret_cast<DT_U16*>(vtcm_buffer->xofs);
    DT_U16 *yofs          = reinterpret_cast<DT_U16*>(vtcm_buffer->yofs);
    DT_U16 *src_buffer    = reinterpret_cast<DT_U16*>(vtcm_buffer->src_buffer + C * vtcm_buffer->src_buffer_pitch * thread_id);
    DT_U16 *gather_buffer = reinterpret_cast<DT_U16*>(vtcm_buffer->gather_buffer + C * (AURA_HVLEN << 2) * thread_id);
    DT_U32 l2fetch_param  = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if ((y + 1 < end_row) && (yofs[y + 1] != yofs[y]))
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
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
static typename std::enable_if<std::is_same<DT_U8, Tp>::value | std::is_same<DT_U16, Tp>::value,  Status>::type
ResizeNnCommHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheihgt = src.GetSizes().m_height;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 thread_num = wp->GetComputeThreadNum();

    DT_S32 xofs_size          = AURA_ALIGN(owidth  * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 yofs_size          = AURA_ALIGN(oheight * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 src_buffer_size    = AURA_ALIGN(iwidth  * sizeof(DT_U16), AURA_HVLEN) * thread_num * channel;
    DT_S32 gather_buffer_size = (AURA_HVLEN << 2) * channel * thread_num;
    DT_S32 total_buffer_size  = xofs_size + yofs_size + src_buffer_size + gather_buffer_size;

    DT_U8 *vtcm_mem = static_cast<DT_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
    if (DT_NULL == vtcm_mem)
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
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 1>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeNnCommHvxImpl of c1 failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 2>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeNnCommHvxImpl of c2 failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeNnCommHvxImpl<Tp, 3>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
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
            ret = ResizeNnCommHvxHelper<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnCommHvxHelper run failed, type: DT_U8/DT_S8");
            }
            break;
        }

        case ElemType::S16:
        case ElemType::U16:
        {
            ret = ResizeNnCommHvxHelper<DT_U16>(ctx, src, dst);
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