#include "morph_impl.hpp"

namespace aura
{

MorphNeon::MorphNeon(Context *ctx, MorphType type, const OpTarget &target) : MorphImpl(ctx, type, target)
{}

Status MorphNeon::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
{
    if (MorphImpl::SetArgs(src, dst, ksize, shape, iterations) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MorphNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Mat iter_mat;
    const Mat *iter_src = src;
    Mat *iter_dst = ((m_iterations & 1) == 1) ? dst : &iter_mat;

    Status ret = Status::ERROR;

    if (m_iterations > 1)
    {
        iter_mat = Mat(m_ctx, dst->GetElemType(), dst->GetSizes());
        if (!iter_mat.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid iter_mat");
            return Status::ERROR;
        }
    }

    for (MI_S32 i = 0; i < m_iterations; i++)
    {
        switch (m_ksize)
        {
            case 3:
            {
                ret = Morph3x3Neon(m_ctx, *iter_src, *iter_dst, m_type, m_shape, m_target);
                break;
            }

            case 5:
            {
                ret = Morph5x5Neon(m_ctx, *iter_src, *iter_dst, m_type, m_shape, m_target);
                break;
            }

            case 7:
            {
                ret = Morph7x7Neon(m_ctx, *iter_src, *iter_dst, m_type, m_shape, m_target);
                break;
            }

            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported kernel size");
                return Status::ERROR;
            }
        }

        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Morph" + std::to_string(m_ksize) + "x" + std::to_string(m_ksize) + "Neon failed").c_str());
            AURA_RETURN(m_ctx, ret);
        }

        iter_src = (((i + m_iterations) & 1) == 1) ? dst : &iter_mat;
        iter_dst = (((i + m_iterations) & 1) == 1) ? &iter_mat : dst;
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
