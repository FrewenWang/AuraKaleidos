#include "aura/ops/feature2d/tomasi.hpp"
#include "tomasi_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<TomasiImpl> CreateTomasiImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<TomasiImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new TomasiNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new TomasiNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Tomasi::Tomasi(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Tomasi::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                       DT_F64 quality_level, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                       DT_BOOL use_harris, DT_F64 harris_k)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateTomasiImpl(m_ctx, impl_target);
    }

    // run SetArgs
    TomasiImpl *tomasi_impl = dynamic_cast<TomasiImpl *>(m_impl.get());
    if (DT_NULL == tomasi_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "gaussian_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = tomasi_impl->SetArgs(src, key_points, max_num_corners, quality_level, min_distance,
                                      block_size, gradient_size, use_harris, harris_k);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status GoodFeaturesToTrack(Context *ctx, const Mat &src, std::vector<KeyPoint> &corners, DT_S32 max_corners, DT_F64 quality_level,
                                         DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size, DT_BOOL use_harris, DT_F64 harris_k,
                                         const OpTarget &target)
{
    Tomasi tomasi(ctx, target);

    return OpCall(ctx, tomasi, &src, corners, max_corners, quality_level, min_distance, block_size, gradient_size, use_harris, harris_k);
}

TomasiImpl::TomasiImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Tomasi", target),
                                                               m_max_corners(0), m_quality_level(0.0),
                                                               m_min_distance(0.0), m_block_size(0),
                                                               m_gradient_size(0), m_use_harris(DT_FALSE),
                                                               m_harris_k(0.0), m_src(DT_NULL)
{}

Status TomasiImpl::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                           DT_F64 quality_level, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                           DT_BOOL use_harris, DT_F64 harris_k)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src");
        return Status::ERROR;
    }

    if(0 != key_points.size())
    {
        key_points.clear();
    }

    if(key_points.capacity() < (DT_U32)max_num_corners)
    {
        key_points.reserve(max_num_corners);
    }

    if (src->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8");
        return Status::ERROR;
    }

    m_src            = src;
    m_key_points     = &key_points;
    m_max_corners    = max_num_corners;
    m_quality_level  = quality_level;
    m_min_distance   = min_distance;
    m_block_size     = block_size;
    m_gradient_size  = gradient_size;
    m_use_harris     = use_harris;
    m_harris_k       = harris_k;

    return Status::OK;
}

std::vector<const Array*> TomasiImpl::GetInputArrays() const
{
    return {m_src};
}

std::string TomasiImpl::ToString() const
{
    std::string str;

    DT_CHAR quality_level_str[20], min_distance_str[20], harris_k_str[20];
    snprintf(quality_level_str, sizeof(quality_level_str), "%.2f", m_quality_level);
    snprintf(min_distance_str, sizeof(min_distance_str), "%.2f", m_min_distance);
    snprintf(harris_k_str, sizeof(harris_k_str), "%.2f", m_harris_k);

    str = "op(Tomasi)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param( max_corners:" + std::to_string(m_max_corners) + " | " + "min_distance:" + min_distance_str + " | " +
            "quality_level:" + quality_level_str + " | " + "block_size:" + std::to_string(m_block_size) + " | " +
            "gradient_size:" + std::to_string(m_gradient_size) + " | " + "use_harris:" + std::to_string(m_use_harris) + " | " +
            "harris_k:" + harris_k_str + ")\n";

    return str;
}

DT_VOID TomasiImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_key_points, m_max_corners, m_quality_level,
                        m_min_distance, m_block_size, m_gradient_size, m_use_harris, m_harris_k);
}

} // namespace aura