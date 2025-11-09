#include "laplacian_impl.hpp"
#include "aura/ops/filter/sobel.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"

namespace aura
{

LaplacianNone::LaplacianNone(Context *ctx, const OpTarget &target) : LaplacianImpl(ctx, target)
{}

Status LaplacianNone::SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                              BorderType border_type, const Scalar &border_value)
{
    if (LaplacianImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::Iinitialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status LaplacianNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input null ptr");
        return Status::ERROR;
    }

    Mat fx(m_ctx, ElemType::F32, dst->GetSizes());
    if (!fx.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Mat create failed");
        return Status::ERROR;
    }

    if (ISobel(m_ctx, *src, fx, 0, 2, m_ksize, 1.f, m_border_type, m_border_value, m_target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ISobel failed");
        return Status::ERROR;
    }

    Mat fy(m_ctx, ElemType::F32, dst->GetSizes());
    if (!fy.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Mat create failed");
        return Status::ERROR;
    }

    if (ISobel(m_ctx, *src, fy, 2, 0, m_ksize, 1.f, m_border_type, m_border_value, m_target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ISobel failed");
        return Status::ERROR;
    }

    if (IAdd(m_ctx, fx, fy, *dst, m_target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IAdd failed");
        return Status:: ERROR ;
    }

    return Status::OK;
}

} // namespace aura
