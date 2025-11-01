#include "bilateral_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
struct BilateralTraits
{
    // Tp = MI_U8 MI_F16 MI_F32, MI_U8 MI_F32 MI_F32
    using InterType = typename std::conditional<sizeof(Tp) == 1, MI_U8, MI_F32>::type;
    static constexpr MI_U32 Q = sizeof(Tp) == 1 ? 8 : 0;
};

template <typename Tp,
          typename std::enable_if<std::is_same<Tp, MI_U8>::value, Tp>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BilateralNoneCore(MI_F32 *color_weight_data, MI_F32 &cur_color_weight, MI_F32 alpha, MI_S32 max_idx)
{
    AURA_UNUSED(max_idx);

    MI_S32 idx       = Round(alpha);
    cur_color_weight = color_weight_data[idx];
}

template <typename Tp, 
          typename std::enable_if<!std::is_same<Tp, MI_U8>::value, Tp>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BilateralNoneCore(MI_F32 *color_weight_data, MI_F32 &cur_color_weight, MI_F32 alpha, MI_S32 max_idx)
{
    MI_S32 idx       = Floor(alpha);
    alpha           -= idx;
    idx              = Clamp(idx, static_cast<MI_S32>(0), max_idx);
    cur_color_weight = color_weight_data[idx] + alpha * (color_weight_data[idx + 1] - color_weight_data[idx]);
}

template <typename Tp, MI_S32 C,
          typename std::enable_if<(C == 1), Tp>::type* = MI_NULL>
static Status BilateralNoneImpl(const Mat &src, Mat &dst, const Mat &space_weight, const Mat &space_ofs, const Mat &color_weight,
                                MI_S32 ksize, MI_F32 scale_index, MI_S32 valid_num, MI_S32 start_row, MI_S32 end_row)
{
    using InterType = typename BilateralTraits<Tp>::InterType;

    MI_F32 *space_weight_data = (MI_F32*)space_weight.Ptr<MI_F32>(0);
    MI_S32 *space_ofs_data    = (MI_S32*)space_ofs.Ptr<MI_S32>(0);
    MI_F32 *color_weight_data = (MI_F32*)color_weight.Ptr<MI_F32>(0);

    const MI_S32 width        = dst.GetSizes().m_width;
    const MI_S32 ksh          = ksize >> 1;
    const MI_S32 max_idx      = color_weight.GetSizes().m_width - 2;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y + ksh);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            MI_F32 weight_sum = 0.f;
            MI_F32 sum        = 0.f;
            InterType val0    = static_cast<InterType>(src_row[x + ksh]);

            for (MI_S32 k = 0; k < valid_num; k++)
            {
                InterType val1 = static_cast<InterType>(src_row[x + ksh + space_ofs_data[k]]);

                MI_F32 cur_color_weight = 0.f;

                MI_F32 alpha = Abs(val1 - val0) * scale_index;
                BilateralNoneCore<Tp>(color_weight_data, cur_color_weight, alpha, max_idx);

                weight_sum += space_weight_data[k] * cur_color_weight;
                sum        += val1 * (space_weight_data[k] * cur_color_weight);
            }

            dst_row[x] = ShiftSatCast<MI_F32, Tp, BilateralTraits<Tp>::Q << 1>(sum / weight_sum);
        }
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C,
          typename std::enable_if<(C == 3), Tp>::type* = MI_NULL>
static Status BilateralNoneImpl(const Mat &src, Mat &dst, const Mat &space_weight, const Mat &space_ofs, const Mat &color_weight,
                                MI_S32 ksize, MI_F32 scale_index, MI_S32 valid_num, MI_S32 start_row, MI_S32 end_row)
{
    using InterType = typename BilateralTraits<Tp>::InterType;

    MI_F32 *space_weight_data = (MI_F32*)space_weight.Ptr<MI_F32>(0);
    MI_S32 *space_ofs_data    = (MI_S32*)space_ofs.Ptr<MI_S32>(0);
    MI_F32 *color_weight_data = (MI_F32*)color_weight.Ptr<MI_F32>(0);

    const MI_S32 width        = dst.GetSizes().m_width;
    const MI_S32 ksh          = ksize >> 1;
    const MI_S32 max_idx      = color_weight.GetSizes().m_width - 2;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y + ksh);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            MI_F32 weight_sum = 0.f;
            MI_F32 sum_r      = 0.f;
            MI_F32 sum_g      = 0.f;
            MI_F32 sum_b      = 0.f;
            InterType r0      = SaturateCast<InterType>(src_row[3 * (x + ksh)]);
            InterType g0      = SaturateCast<InterType>(src_row[3 * (x + ksh) + 1]);
            InterType b0      = SaturateCast<InterType>(src_row[3 * (x + ksh) + 2]);

            for (MI_S32 k = 0; k < valid_num; k++)
            {
                InterType r1 = SaturateCast<InterType>(src_row[3 * (x + ksh) + space_ofs_data[k]]);
                InterType g1 = SaturateCast<InterType>(src_row[3 * (x + ksh) + space_ofs_data[k] + 1]);
                InterType b1 = SaturateCast<InterType>(src_row[3 * (x + ksh) + space_ofs_data[k] + 2]);

                MI_F32 cur_color_weight = 0.f;
                MI_F32 alpha = (Abs(r1 - r0) + Abs(g1 - g0) + Abs(b1 - b0)) * scale_index;
                BilateralNoneCore<Tp>(color_weight_data, cur_color_weight, alpha, max_idx);

                MI_F32 weight = space_weight_data[k] * cur_color_weight;
                weight_sum   += weight;
                sum_r        += r1 * weight;
                sum_g        += g1 * weight;
                sum_b        += b1 * weight;
            }

            MI_F32 weight_inv  = 1.f / weight_sum;
            dst_row[3 * x    ] = SaturateCast<Tp>(sum_r * weight_inv);
            dst_row[3 * x + 1] = SaturateCast<Tp>(sum_g * weight_inv);
            dst_row[3 * x + 2] = SaturateCast<Tp>(sum_b * weight_inv);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status BilateralNoneHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &space_weight, const Mat &space_ofs, const Mat &color_weight,
                                  MI_S32 ksize, MI_F32 scale_index, MI_S32 valid_num, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::ERROR;

    switch(src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = BilateralNoneImpl<Tp, 1>(std::cref(src), std::ref(dst), std::cref(space_weight),
                                           std::cref(space_ofs), std::cref(color_weight),
                                           ksize, scale_index, valid_num, start_row, end_row);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BilateralNoneImpl<Tp, 1> failed");
            }
            break;
        }

        case 3:
        {
            ret = BilateralNoneImpl<Tp, 3>(std::cref(src), std::ref(dst), std::cref(space_weight),
                                           std::cref(space_ofs), std::cref(color_weight),
                                           ksize, scale_index, valid_num, start_row, end_row);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BilateralNoneImpl<Tp, 3> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "channel number error");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

BilateralNone::BilateralNone(Context *ctx, const OpTarget &target) : BilateralImpl(ctx, target)
{}

Status BilateralNone::SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space,
                                 MI_S32 ksize, BorderType border_type, const Scalar &border_value)
{
    if (BilateralImpl::SetArgs(src, dst, sigma_color, sigma_space, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BilateralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::F16 && elem_type != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/F16/F32");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BilateralNone::Run()
{
    Status ret = Status::ERROR;
    //
    MI_S32 ksh          = m_ksize >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(ksh << 1, ksh << 1, 0);

    const Mat *src_mat = dynamic_cast<const Mat*>(m_src);
    Mat *dst_mat = dynamic_cast<Mat*>(m_dst);
    if ((MI_NULL == src_mat) || (MI_NULL == dst_mat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is not mat");
        return Status::ERROR;
    }

    Mat src_border = Mat(m_ctx, m_src->GetElemType(), border_sizes, AURA_MEM_DEFAULT, m_src->GetStrides());
    if (!src_border.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src_border inValid..");
        AURA_RETURN(m_ctx, ret);
    }

    ret = IMakeBorder(m_ctx, *src_mat, src_border, ksh, ksh, ksh, ksh, m_border_type, m_border_value, OpTarget::None());
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
        return ret;
    }

    MI_S32 oheight = dst_mat->GetSizes().m_height;

#define BILATERAL_NONE_IMPL(type)                                                                                                 \
    if (m_target.m_data.none.enable_mt)                                                                                           \
    {                                                                                                                             \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                                  \
        if (MI_NULL == wp)                                                                                                        \
        {                                                                                                                         \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                                 \
            return Status::ERROR;                                                                                                 \
        }                                                                                                                         \
                                                                                                                                  \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, BilateralNoneHelper<type>, m_ctx, std::cref(src_border),           \
                              std::ref(*dst_mat), std::cref(m_space_weight), std::cref(m_space_ofs), std::cref(m_color_weight), \
                              m_ksize, m_scale_index, m_valid_num);                                                               \
    }                                                                                                                             \
    else                                                                                                                          \
    {                                                                                                                             \
        ret = BilateralNoneHelper<type>(m_ctx, src_border, *dst_mat, m_space_weight, m_space_ofs,                                \
                                        m_color_weight, m_ksize, m_scale_index, m_valid_num, 0, oheight);                        \
    }                                                                                                                             \
    if (ret != Status::OK)                                                                                                        \
    {                                                                                                                             \
        MI_CHAR error_msg[128];                                                                                                   \
        std::snprintf(error_msg, sizeof(error_msg), "BilateralNoneHelper<%s> failed", #type);                                     \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                                  \
    }

    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            BILATERAL_NONE_IMPL(MI_U8)
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            BILATERAL_NONE_IMPL(MI_F16)
            break;
        }
#endif // AURA_BUILD_HOST

        case ElemType::F32:
        {
            BILATERAL_NONE_IMPL(MI_F32)
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            break;
        }
    }

#undef BILATERAL_NONE_IMPL

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura