#include "make_border_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
static DT_VOID MakeBorderConstantNoneImpl(const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom,
                                          DT_S32 left, DT_S32 right, const Scalar &border_value)
{
    const Sizes3 &src_sz = src.GetSizes();
    const Sizes3 &dst_sz = dst.GetSizes();

    std::vector<Tp> border_val(Max(static_cast<DT_S32>(4), src_sz.m_channel), SaturateCast<Tp>(border_value.m_val[0]));
    if (src_sz.m_channel <= 4)
    {
        border_val[1] = SaturateCast<Tp>(border_value.m_val[1]);
        border_val[2] = SaturateCast<Tp>(border_value.m_val[2]);
        border_val[3] = SaturateCast<Tp>(border_value.m_val[3]);
    }

    // Process the top rows
    for (DT_S32 y = 0; y < top; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        for (DT_S32 x = 0; x < dst_sz.m_width; ++x)
        {
            for (DT_S32 ch = 0; ch < dst_sz.m_channel; ++ch)
            {
                *dst_row++ = border_val[ch];
            }
        }
    }

    // Process the middle rows
    for (DT_S32 y = 0; y < src_sz.m_height; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_row       = dst.Ptr<Tp>(y + top);

        // Process left border
        for (DT_S32 x = 0; x < left; ++x)
        {
            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = border_val[ch];
            }
        }

        // Process middle cols
        memcpy(dst_row, src_row, src_sz.m_width * src_sz.m_channel * sizeof(Tp));

        // Process right border
        dst_row += src_sz.m_width * src_sz.m_channel;
        for (DT_S32 x = 0; x < right; ++x)
        {
            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = border_val[ch];
            }
        }
    }

    // Process the bottom rows
    for (DT_S32 y = dst_sz.m_height - bottom; y < dst_sz.m_height; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        for (DT_S32 x = 0; x < dst_sz.m_width; ++x)
        {
            for (DT_S32 ch = 0; ch < dst_sz.m_channel; ++ch)
            {
                *dst_row++ = border_val[ch];
            }
        }
    }
}

template <typename Tp>
static DT_VOID MakeBorderReplicateNoneImpl(const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom,
                                           DT_S32 left, DT_S32 right)
{
    const Sizes3 &src_sz = src.GetSizes();

    // Process the middle rows
    for (DT_S32 y = 0; y < src_sz.m_height; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp *dst_row       = dst.Ptr<Tp>(y + top);

        // Process left border
        for (DT_S32 x = 0; x < left; ++x)
        {
            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = src_row[ch];
            }
        }

        // Process middle cols
        memcpy(dst_row, src_row, src_sz.m_width * src_sz.m_channel * sizeof(Tp));

        // Process right border
        dst_row += src_sz.m_width * src_sz.m_channel;
        src_row += (src_sz.m_width - 1) * src_sz.m_channel;
        for (DT_S32 x = 0; x < right; ++x)
        {
            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = src_row[ch];
            }
        }
    }

    // Process the top rows
    Tp *src_row = dst.Ptr<Tp>(top);
    for (DT_S32 y = 0; y < top; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        memcpy(dst_row, src_row, dst.GetRowPitch());
    }

    // Process the bottom rows
    src_row = dst.Ptr<Tp>(src_sz.m_height + top - 1);
    for (DT_S32 y = src_sz.m_height + top; y < src_sz.m_height + top + bottom; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        memcpy(dst_row, src_row, dst.GetRowPitch());
    }
}

template <typename Tp>
static DT_VOID MakeBorderReflect101NoneImpl(const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom,
                                            DT_S32 left, DT_S32 right)
{
    const Sizes3 &src_sz = src.GetSizes();

    // build idx table
    DT_S32 buffer_size = Max(top + bottom, left + right);
    std::vector<DT_S32> idx_tab(buffer_size, 0);
    for (DT_S32 i = 0; i < left; ++i)
    {
        idx_tab[i] = GetBorderIdx<BorderType::REFLECT_101>(i - left, src_sz.m_width) * src_sz.m_channel;
    }

    for (DT_S32 i = 0; i < right; i++)
    {
        idx_tab[i + left] = GetBorderIdx<BorderType::REFLECT_101>(i + src_sz.m_width, src_sz.m_width) * src_sz.m_channel;
    }

    // Process the middle rows
    for (DT_S32 y = top; y < src_sz.m_height + top; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y - top);
        Tp *dst_row = dst.Ptr<Tp>(y);

        // Process left border
        for (DT_S32 x_dst = 0; x_dst < left; ++x_dst)
        {
            DT_S32 x_src = idx_tab[x_dst];

            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = src_row[x_src + ch];
            }
        }

        // Process middle cols
        memcpy(dst_row, src_row, src_sz.m_width * src_sz.m_channel * sizeof(Tp));

        // Process right border
        dst_row = dst.Ptr<Tp>(y) + (left + src_sz.m_width) * src_sz.m_channel;
        for (DT_S32 x_dst = 0; x_dst < right; ++x_dst)
        {
            DT_S32 x_src = idx_tab[x_dst + left];

            for (DT_S32 ch = 0; ch < src_sz.m_channel; ++ch)
            {
                *dst_row++ = src_row[x_src + ch];
            }
        }
    }

    for (DT_S32 i = 0; i < top; ++i)
    {
        idx_tab[i] = GetBorderIdx<BorderType::REFLECT_101>(i - top, src_sz.m_height);
    }

    for (DT_S32 i = 0; i < bottom; i++)
    {
        idx_tab[i + top] = GetBorderIdx<BorderType::REFLECT_101>(i + src_sz.m_height, src_sz.m_height);
    }

    // Process the top rows
    for (DT_S32 y = 0; y < top; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        Tp *src_row = dst.Ptr<Tp>(idx_tab[y] + top);
        memcpy(dst_row, src_row, dst.GetRowPitch());
    }

    // Process the bottom rows
    for (DT_S32 y = 0; y < bottom; ++y)
    {
        Tp *dst_row = dst.Ptr<Tp>(y + top + src_sz.m_height);
        Tp *src_row = dst.Ptr<Tp>(idx_tab[y + top] + top);
        memcpy(dst_row, src_row, dst.GetRowPitch());
    }
}

template <typename Tp>
static Status MakeBorderNoneHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom, DT_S32 left, DT_S32 right,
                                   BorderType type, const Scalar &border_value)
{
    switch (type)
    {
        case BorderType::CONSTANT:
        {
            MakeBorderConstantNoneImpl<Tp>(src, dst, top, bottom, left, right, border_value);
            break;
        }
        case BorderType::REPLICATE:
        {
            MakeBorderReplicateNoneImpl<Tp>(src, dst, top, bottom, left, right);
            break;
        }
        case BorderType::REFLECT_101:
        {
            MakeBorderReflect101NoneImpl<Tp>(src, dst, top, bottom, left, right);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported border type.");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

MakeBorderNone::MakeBorderNone(Context *ctx, const OpTarget &target) : MakeBorderImpl(ctx, target)
{}

Status MakeBorderNone::SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom, DT_S32 left, DT_S32 right,
                                  BorderType type, const Scalar &border_value)
{
    if (MakeBorderImpl::SetArgs(src, dst, top, bottom, left, right, type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MakeBorderNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MakeBorderNoneHelper<DT_U8>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = MakeBorderNoneHelper<DT_S8>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = MakeBorderNoneHelper<DT_U16>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = MakeBorderNoneHelper<DT_S16>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = MakeBorderNoneHelper<DT_U32>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = MakeBorderNoneHelper<DT_S32>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = MakeBorderNoneHelper<MI_F16>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = MakeBorderNoneHelper<DT_F32>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_F32> failed.");
            }
            break;
        }
        case ElemType::F64:
        {
            ret = MakeBorderNoneHelper<DT_F64>(m_ctx, *src, *dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNoneHelper<DT_F64> failed.");
            }
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNone with invalid ElemType.");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura