#include "tomasi_impl.hpp"
#include "harris_impl.hpp"
#include "aura/ops/morph.h"
#include "aura/runtime/logger.h"
#include <algorithm>

namespace aura
{

static Status GoodFeaturesToTrackU8None(Context *ctx, const Mat &mat, std::vector<KeyPoint> &key_points,
                                        DT_S32 max_corners, DT_F64 quality_level, DT_F64 min_distance, DT_S32 block_size,
                                        DT_S32 gradient_size, DT_BOOL use_harris, DT_F64 harris_k, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Mat eigen(ctx, ElemType::F32, mat.GetSizes());
    Scalar border_value;

    ret = CornerEigenValsVecsNone(ctx, mat, eigen, block_size, gradient_size, use_harris, harris_k, BorderType::REFLECT_101, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "CornerEigenValsVecsNone excute failed");
        return Status::ERROR;
    }

    DT_S32 height = mat.GetSizes().m_height;
    DT_S32 width  = mat.GetSizes().m_width;

    DT_F32 max_val = 0.f;
    for (DT_S32 y = 0; y < height; y++)
    {
        const DT_F32 *eig_data = eigen.Ptr<DT_F32>(y);
        for (DT_S32 x = 0; x < width; x++)
        {
            max_val = Max(eig_data[x], max_val);
        }
    }

    DT_F32 thresh = max_val * quality_level;
    for (DT_S32 y = 0; y < height; y++)
    {
        DT_F32 *eig_data = eigen.Ptr<DT_F32>(y);
        for (DT_S32 x = 0; x < width; x++)
        {
            eig_data[x] = eig_data[x] > thresh ? eig_data[x] : 0.f;
        }
    }

    Mat dilate_mat(ctx, ElemType::F32, mat.GetSizes());
    ret = IDilate(ctx, eigen, dilate_mat, 3);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dilate excute failed");
        return Status::ERROR;
    }

    std::vector<const DT_F32*> vec_corners;

    for (DT_S32 y = 1; y < height - 1; y++)
    {
        const DT_F32 *eig_data    = eigen.Ptr<DT_F32>(y);
        const DT_F32 *dilate_data = dilate_mat.Ptr<DT_F32>(y);
        for (DT_S32 x = 1; x < width - 1; x++)
        {
            DT_F32 val = eig_data[x];
            if (val != 0 && val == dilate_data[x])
            {
                vec_corners.push_back(eig_data + x);
            }
        }
    }

    if (vec_corners.empty())
    {
        return Status::OK;
    }

    auto func_greater = [](const DT_F32 *a, const DT_F32 *b) -> DT_BOOL
    {
        return (*a > *b) ? DT_TRUE : (*a < *b) ? DT_FALSE : (a > b);
    };
    std::sort(vec_corners.begin(), vec_corners.end(), func_greater);

    size_t total_num = vec_corners.size(), ncorners = 0;

    DT_S32 eig_step = eigen.GetStrides().m_width;
    const DT_U8 *eig_data = eigen.Ptr<DT_U8>(0);

    if (min_distance >= 1)
    {
        const DT_S32 cell_size   = Round(min_distance);
        const DT_S32 grid_width  = (width + cell_size - 1) / cell_size;
        const DT_S32 grid_height = (height + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f>> grid(grid_width * grid_height);
        min_distance *= min_distance;

        for (size_t i = 0; i < total_num; i++)
        {
            DT_S32 offset = reinterpret_cast<const DT_U8*>(vec_corners[i]) - eig_data;
            DT_S32 y      = offset / eig_step;
            DT_S32 x      = (offset - y * eig_step) / sizeof(DT_F32);

            DT_BOOL good = DT_TRUE;

            DT_S32 x_cell = x / cell_size;
            DT_S32 y_cell = y / cell_size;

            DT_S32 x1 = x_cell - 1;
            DT_S32 y1 = y_cell - 1;
            DT_S32 x2 = x_cell + 1;
            DT_S32 y2 = y_cell + 1;

            x1 = Max<DT_S32>(0, x1);
            y1 = Max<DT_S32>(0, y1);
            x2 = Min<DT_S32>(grid_width - 1, x2);
            y2 = Min<DT_S32>(grid_height - 1, y2);

            for (DT_S32 dy = y1; dy <= y2; dy++)
            {
                for (DT_S32 dx = x1; dx <= x2; dx++)
                {
                    std::vector<Point2f> &m = grid[dy * grid_width + dx];

                    if (m.size())
                    {
                        for (size_t j = 0; j < m.size(); j++)
                        {
                            DT_F32 dx = x - m[j].m_x;
                            DT_F32 dy = y - m[j].m_y;
                            if ((dx * dx + dy * dy) < min_distance)
                            {
                                good = DT_FALSE;
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

                if (max_corners > 0 && static_cast<DT_S32>(ncorners) == max_corners)
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
            DT_S32 offset = reinterpret_cast<const DT_U8*>(vec_corners[i]) - eig_data;
            DT_S32 y      = offset / eig_step;
            DT_S32 x      = (offset - y * eig_step) / sizeof(DT_F32);

            key_points.push_back(KeyPoint(Point2f(x, y), 0));
            ++ncorners;

            if (max_corners > 0 && static_cast<DT_S32>(ncorners) == max_corners)
            {
                break;
            }
        }
    }

    return Status::OK;
}

TomasiNone::TomasiNone(Context *ctx, const OpTarget &target) : TomasiImpl(ctx, target)
{}

Status TomasiNone::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                           DT_F64 quality_leve, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                           DT_BOOL use_harris, DT_F64 harris_k)
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

Status TomasiNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mat is null");
        return Status::ERROR;
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GoodFeaturesToTrackU8None(m_ctx, *src, *m_key_points, m_max_corners, m_quality_level,
                                            m_min_distance, m_block_size, m_gradient_size, m_use_harris,
                                            m_harris_k, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "GoodFeaturesToTrackU8None failed, ElemType: U8");
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