#include "tomasi_impl.hpp"
#include "harris_impl.hpp"
#include "aura/ops/morph.h"
#include "aura/runtime/logger.h"
#include <algorithm>

namespace aura
{

Status GoodFeaturesToTrackU8Neon(Context *ctx, const Mat &src, std::vector<KeyPoint> &key_points,
                                 MI_S32 max_corners, MI_F64 quality_level, MI_F64 min_distance, MI_S32 block_size,
                                 MI_S32 gradient_size, MI_BOOL use_harris, MI_F64 harris_k, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Mat eigen(ctx, ElemType::F32, src.GetSizes());
    Scalar border_value;

    ret = CornerEigenValsVecsNeon(ctx, src, eigen, block_size, gradient_size, use_harris, harris_k, BorderType::REFLECT_101, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "CornerEigenValsVecsNeon excute failed");
        return Status::ERROR;
    }

    MI_S32 height = src.GetSizes().m_height;
    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 width_align4 = width & (-4);

    MI_F32 max_val = 0.f;
    float32x4_t vqf32_max;
    neon::vdup(vqf32_max, max_val);
    for (MI_S32 y = 0; y < height; y++)
    {
        const MI_F32 *eigen_data = eigen.Ptr<MI_F32>(y);
        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4_t vqf32_src = neon::vload1q(eigen_data);
            vqf32_max             = neon::vmax(vqf32_src, vqf32_max);

            eigen_data += 4;
        }
        for (; x < width; x++)
        {
            max_val = Max(*eigen_data++, max_val);
        }
    }
    max_val = Max(Max(Max(Max(neon::vgetlane<0>(vqf32_max),  neon::vgetlane<1>(vqf32_max)),
                              neon::vgetlane<2>(vqf32_max)), neon::vgetlane<3>(vqf32_max)), max_val);

    MI_F32 thresh = max_val * quality_level;
    float32x4_t vqf32_thresh, vqf32_zero;
    neon::vdup(vqf32_thresh, thresh);
    neon::vdup(vqf32_zero, 0.f);

    for (MI_S32 y = 0; y < height; y++)
    {
        MI_F32 *eigen_data = eigen.Ptr<MI_F32>(y);
        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4_t vqf32_src  = neon::vload1q(eigen_data);
            uint32x4_t  vqu32_mask = neon::vcgt(vqf32_src, vqf32_thresh);
            float32x4_t vqf32_dst  = neon::vbsl(vqu32_mask, vqf32_src, vqf32_zero);
            neon::vstore(eigen_data, vqf32_dst);

            eigen_data += 4;
        }
        for (; x < width; x++)
        {
            MI_F32 val = *eigen_data > thresh ? *eigen_data : 0.f;
            *eigen_data++ = val;
        }
    }

    Mat dilate_mat(ctx, ElemType::F32, src.GetSizes());
    ret = IDilate(ctx, eigen, dilate_mat, 3, MorphShape::RECT, 1, OpTarget::Neon());
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dilate run failed");
        return Status::ERROR;
    }

    std::vector<const MI_F32*> tmp_corners;

    for (MI_S32 y = 1; y < height - 1; y++)
    {
        const MI_F32 *eigen_data  = eigen.Ptr<MI_F32>(y);
        const MI_F32 *dilate_data = dilate_mat.Ptr<MI_F32>(y);
        for (MI_S32 x = 1; x < width - 1; x++)
        {
            MI_F32 val = eigen_data[x];
            if (val != 0 && val == dilate_data[x])
            {
                tmp_corners.push_back(eigen_data + x);
            }
        }
    }

    if (tmp_corners.empty())
    {
        return Status::OK;
    }

    auto func_greater = [](const MI_F32 *a, const MI_F32 *b) -> MI_BOOL
    {
        return (*a > *b) ? MI_TRUE : (*a < *b) ? MI_FALSE : (a > b);
    };
    std::sort(tmp_corners.begin(), tmp_corners.end(), func_greater);

    size_t total_num        = tmp_corners.size(), ncorners = 0;
    MI_S32 eigen_step       = eigen.GetStrides().m_width;
    const MI_U8 *eigen_data = eigen.Ptr<MI_U8>(0);

    if (min_distance >= 1)
    {
        const MI_S32 cell_size   = Round(min_distance);
        const MI_S32 grid_width  = (width  + cell_size - 1) / cell_size;
        const MI_S32 grid_height = (height + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f>> grid(grid_width * grid_height);
        min_distance *= min_distance;

        for (size_t i = 0; i < total_num; i++)
        {
            MI_S32 offset = reinterpret_cast<const MI_U8*>(tmp_corners[i]) - eigen_data;
            MI_S32 y      = offset / eigen_step;
            MI_S32 x      = (offset - y * eigen_step) / sizeof(MI_F32);

            MI_BOOL good = MI_TRUE;

            MI_S32 x_cell = x / cell_size;
            MI_S32 y_cell = y / cell_size;

            MI_S32 x1 = x_cell - 1;
            MI_S32 y1 = y_cell - 1;
            MI_S32 x2 = x_cell + 1;
            MI_S32 y2 = y_cell + 1;

            x1 = Max(0, x1);
            y1 = Max(0, y1);
            x2 = Min(grid_width - 1, x2);
            y2 = Min(grid_height - 1, y2);

            for (MI_S32 dy = y1; dy <= y2; dy++)
            {
                for (MI_S32 dx = x1; dx <= x2; dx++)
                {
                    std::vector<Point2f> &m = grid[dy * grid_width + dx];

                    if (m.size())
                    {
                        for (size_t j = 0; j < m.size(); j++)
                        {
                            MI_F32 dx = x - m[j].m_x;
                            MI_F32 dy = y - m[j].m_y;
                            if ((dx * dx + dy * dy) < min_distance)
                            {
                                good = MI_FALSE;
                                goto EXIT;
                            }
                        }
                    }
                }
            }

EXIT:
            if (good)
            {
                grid[y_cell * grid_width + x_cell].push_back(Point2f(x, y));
                key_points.push_back(KeyPoint(Point2f(x, y), 0));
                ++ncorners;

                if (max_corners > 0 && static_cast<MI_S32>(ncorners) == max_corners)
                {
                    break;
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < total_num; i++)
        {
            MI_S32 offset = reinterpret_cast<const MI_U8*>(tmp_corners[i]) - eigen_data;
            MI_S32 y      = offset / eigen_step;
            MI_S32 x      = (offset - y * eigen_step) / sizeof(MI_F32);

            key_points.push_back(KeyPoint(Point2f(x, y), 0));
            ++ncorners;

            if (max_corners > 0 && static_cast<MI_S32>(ncorners) == max_corners)
            {
                break;
            }
        }
    }

    return Status::OK;
}

TomasiNeon::TomasiNeon(Context *ctx, const OpTarget &target) : TomasiImpl(ctx, target)
{}

Status TomasiNeon::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, MI_S32 max_num_corners,
                           MI_F64 quality_leve, MI_F64 min_distance, MI_S32 block_size, MI_S32 gradient_size,
                           MI_BOOL use_harris, MI_F64 harris_k)
{
    if (TomasiImpl::SetArgs(src, key_points, max_num_corners, quality_leve, min_distance, block_size, gradient_size, use_harris, harris_k) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "TomasiImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status TomasiNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mat is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GoodFeaturesToTrackU8Neon(m_ctx, *src, *m_key_points, m_max_corners, m_quality_level,
                                            m_min_distance, m_block_size, m_gradient_size, m_use_harris,
                                            m_harris_k, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "GoodFeaturesToTrackU8Neon failed, ElemType: U8");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura