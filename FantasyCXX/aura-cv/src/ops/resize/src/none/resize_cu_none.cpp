#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
static Status ResizeCuNoneImpl(Context *ctx, const Mat &src, Mat &dst)
{
    using BufType           = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType         = typename ResizeBnCuTraits<Tp>::AlphaType;
    auto  ResizeCastFunctor = typename ResizeBnCuTraits<Tp>::ResizeCastFunctor();

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_S32 buffer_size = (owidth + oheight) * sizeof(MI_S32) + (owidth * 4 + oheight * 4) * sizeof(AlphaType);
    MI_S32 *buffer = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, buffer_size, 0));
    if (MI_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    GetCuOffset<Tp>(buffer, iwidth, owidth, iheight, oheight);

    MI_S32 *xofs     = buffer;
    MI_S32 *yofs     = xofs + owidth;
    AlphaType *alpha = reinterpret_cast<AlphaType*>(yofs + oheight);
    AlphaType *beta  = reinterpret_cast<AlphaType*>(alpha + (owidth * 4));

    MI_S32 rows_size = owidth * channel * 4 * sizeof(BufType);
    BufType *rows = static_cast<BufType*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, rows_size, 0));
    if (MI_NULL == rows)
    {
        AURA_FREE(ctx, buffer);
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows0 + owidth * channel;
    BufType *rows2 = rows1 + owidth * channel;
    BufType *rows3 = rows2 + owidth * channel;

    MI_S32 prev_sy1 = -5;
    for (MI_S32 dy = 0; dy < oheight; dy++)
    {
        MI_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            BufType *rows0_tmp = rows0;
            rows0              = rows1;
            rows1              = rows2;
            rows2              = rows3;
            rows3              = rows0_tmp;

            const Tp *src_c3 = src.Ptr<Tp>(sy + 3);
            const AlphaType *alpha_ptr = alpha;

            for (MI_S32 dx = 0; dx < owidth; dx++)
            {
                MI_S32 sx = xofs[dx] * channel;
                MI_S32 x_id = dx * channel;
                const Tp *src_c3_ptr = src_c3 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                AlphaType a2 = alpha_ptr[2];
                AlphaType a3 = alpha_ptr[3];

                for (MI_S32 c = 0; c < channel; c++)
                {
                    rows3[x_id + c] = src_c3_ptr[c] * a0 + src_c3_ptr[c + channel] * a1 + src_c3_ptr[c + channel * 2] * a2 + src_c3_ptr[c + channel * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize two row
            BufType *rows0_tmp = rows0;
            BufType *rows1_tmp = rows1;
            rows0              = rows2;
            rows1              = rows3;
            rows2              = rows0_tmp;
            rows3              = rows1_tmp;

            const Tp *src_c2 = src.Ptr<Tp>(sy + 2);
            const Tp *src_c3 = src.Ptr<Tp>(sy + 3);
            const AlphaType *alpha_ptr = alpha;

            for (MI_S32 dx = 0; dx < owidth; dx++)
            {
                MI_S32 sx = xofs[dx] * channel;
                MI_S32 x_id = dx * channel;
                const Tp *src_c2_ptr = src_c2 + sx;
                const Tp *src_c3_ptr = src_c3 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                AlphaType a2 = alpha_ptr[2];
                AlphaType a3 = alpha_ptr[3];

                for (MI_S32 c = 0; c < channel; c++)
                {
                    rows2[x_id + c] = src_c2_ptr[c] * a0 + src_c2_ptr[c + channel] * a1 + src_c2_ptr[c + channel * 2] * a2 + src_c2_ptr[c + channel * 3] * a3;
                    rows3[x_id + c] = src_c3_ptr[c] * a0 + src_c3_ptr[c + channel] * a1 + src_c3_ptr[c + channel * 2] * a2 + src_c3_ptr[c + channel * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize three row
            BufType *rows0_tmp = rows0;
            rows0              = rows3;
            rows3              = rows2;
            rows2              = rows1;
            rows1              = rows0_tmp;

            const Tp *src_c1 = src.Ptr<Tp>(sy + 1);
            const Tp *src_c2 = src.Ptr<Tp>(sy + 2);
            const Tp *src_c3 = src.Ptr<Tp>(sy + 3);
            const AlphaType *alpha_ptr = alpha;

            for (MI_S32 dx = 0; dx < owidth; dx++)
            {
                MI_S32 sx = xofs[dx] * channel;
                MI_S32 x_id = dx * channel;
                const Tp *src_c1_ptr = src_c1 + sx;
                const Tp *src_c2_ptr = src_c2 + sx;
                const Tp *src_c3_ptr = src_c3 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                AlphaType a2 = alpha_ptr[2];
                AlphaType a3 = alpha_ptr[3];

                for (MI_S32 c = 0; c < channel; c++)
                {
                    rows1[x_id + c] = src_c1_ptr[c] * a0 + src_c1_ptr[c + channel] * a1 + src_c1_ptr[c + channel * 2] * a2 + src_c1_ptr[c + channel * 3] * a3;
                    rows2[x_id + c] = src_c2_ptr[c] * a0 + src_c2_ptr[c + channel] * a1 + src_c2_ptr[c + channel * 2] * a2 + src_c2_ptr[c + channel * 3] * a3;
                    rows3[x_id + c] = src_c3_ptr[c] * a0 + src_c3_ptr[c + channel] * a1 + src_c3_ptr[c + channel * 2] * a2 + src_c3_ptr[c + channel * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }
        else if (sy > prev_sy1 + 2)
        {
            // hresize four rows
            const Tp *src_c0 = src.Ptr<Tp>(sy);
            const Tp *src_c1 = src.Ptr<Tp>(sy + 1);
            const Tp *src_c2 = src.Ptr<Tp>(sy + 2);
            const Tp *src_c3 = src.Ptr<Tp>(sy + 3);
            const AlphaType *alpha_ptr = alpha;

            for (MI_S32 dx = 0; dx < owidth; dx++)
            {
                MI_S32 x_id = dx * channel;
                MI_S32 sx = xofs[dx] * channel;
                const Tp *src_c0_ptr = src_c0 + sx;
                const Tp *src_c1_ptr = src_c1 + sx;
                const Tp *src_c2_ptr = src_c2 + sx;
                const Tp *src_c3_ptr = src_c3 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                AlphaType a2 = alpha_ptr[2];
                AlphaType a3 = alpha_ptr[3];

                for (MI_S32 c = 0; c < channel; c++)
                {
                    rows0[x_id + c] = src_c0_ptr[c] * a0 + src_c0_ptr[c + channel] * a1 + src_c0_ptr[c + channel * 2] * a2 + src_c0_ptr[c + channel * 3] * a3;
                    rows1[x_id + c] = src_c1_ptr[c] * a0 + src_c1_ptr[c + channel] * a1 + src_c1_ptr[c + channel * 2] * a2 + src_c1_ptr[c + channel * 3] * a3;
                    rows2[x_id + c] = src_c2_ptr[c] * a0 + src_c2_ptr[c + channel] * a1 + src_c2_ptr[c + channel * 2] * a2 + src_c2_ptr[c + channel * 3] * a3;
                    rows3[x_id + c] = src_c3_ptr[c] * a0 + src_c3_ptr[c + channel] * a1 + src_c3_ptr[c + channel * 2] * a2 + src_c3_ptr[c + channel * 3] * a3;
                }
                alpha_ptr += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];
        AlphaType b2 = beta[2];
        AlphaType b3 = beta[3];

        Tp *dst_c_ptr = dst.Ptr<Tp>(dy);

        for (MI_S32 dx = 0; dx < owidth * channel; dx++)
        {
            dst_c_ptr[dx] = ResizeCastFunctor(static_cast<BufType>((rows0[dx] * b0 + rows1[dx] * b1 + rows2[dx] * b2 + rows3[dx] * b3)));
        }
        beta += 4;
    }

    AURA_FREE(ctx, buffer);
    AURA_FREE(ctx, rows);

    return Status::OK;
}

Status ResizeCuNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuNoneImpl<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuNoneImpl<MI_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuNoneImpl<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuNoneImpl<MI_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_S16");
            }
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ResizeCuNoneImpl<MI_F16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_F16");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = ResizeCuNoneImpl<MI_F32>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuNoneImpl failed, type: MI_F32");
            }
            break;
        }
#endif // AURA_BUILD_HOST

        default :
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura