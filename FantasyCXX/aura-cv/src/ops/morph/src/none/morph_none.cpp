#include "morph_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, MorphType>
struct MorphFunctor;

template <typename Tp>
struct MorphFunctor<Tp, MorphType::ERODE>
{
    Tp operator()(const Tp a, const Tp b) const { return Min(a, b); }
};

template <typename Tp>
struct MorphFunctor<Tp, MorphType::DILATE>
{
    Tp operator()(const Tp a, const Tp b) const { return Max(a, b); }
};

static DT_S32 CountNonZero(const Mat &mat)
{
    const DT_S32 height = mat.GetSizes().m_height;
    const DT_S32 width  = mat.GetSizes().m_width;
    DT_S32 nonzero      = 0;

    for (DT_S32 y = 0; y < height; y++)
    {
        const DT_U8 *mat_row = mat.Ptr<DT_U8>(y);
        for (DT_S32 x = 0; x < width; x++)
        {
            nonzero += (mat_row[x] > 0) ? 1 : 0;
        }
    }

    return nonzero;
}

static Mat MorphKernelMat(Context *ctx, MorphShape shape, DT_S32 ksize)
{
    Mat kmat(ctx, ElemType::U8, Sizes3(ksize, ksize));
    if (!kmat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid kmat");
        return Mat();
    }

    const DT_S32 height = ksize;
    const DT_S32 width  = ksize;
    const DT_S32 ksh    = ksize / 2;

    switch (shape)
    {
        case MorphShape::RECT:
        {
            for (DT_S32 y = 0; y < height; y++)
            {
                DT_U8 *ker_row = kmat.Ptr<DT_U8>(y);
                for (DT_S32 x = 0; x < width; x++)
                {
                    ker_row[x] = 1;
                }
            }

            return kmat;
        }

        case MorphShape::CROSS:
        {
            for (DT_S32 y = 0; y < height; y++)
            {
                DT_U8 *ker_row = kmat.Ptr<DT_U8>(y);
                if (y == ksh)
                {
                    for (DT_S32 x = 0; x < width; x++)
                    {
                        ker_row[x] = 1;
                    }
                }
                else
                {
                    ker_row[ksh] = 1;
                }
            }

            return kmat;
        }

        case MorphShape::ELLIPSE:
        {
            DT_S32 r      = height >> 1;
            DT_S32 c      = width >> 1;
            DT_F64 inv_r2 = r ? (1. / (r * r)) : 0;
            for (DT_S32 y = 0; y < height; y++)
            {
                DT_U8 *ker_row = kmat.Ptr<DT_U8>(y);
                DT_S32 dy      = y - r;
                DT_S32 x = 0, x0 = 0, x1 = 0;
                if (Abs(dy) <= r)
                {
                    DT_S32 dx = SaturateCast<int>(c * Sqrt((r * r - dy * dy) * inv_r2));
                    x0        = Max<DT_S32>(c - dx, 0);
                    x1        = Min<DT_S32>(c + dx + 1, width);
                }

                for (; x < x0; x++)
                {
                    ker_row[x] = 0;
                }
                for (; x < x1; x++)
                {
                    ker_row[x] = 1;
                }
                for (; x < width; x++)
                {
                    ker_row[x] = 0;
                }
            }

            return kmat;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported morph shape");
            return Mat();
        }
    }
}

template <typename Tp, MorphType MORPH_TYPE>
static Status MorphNoneCore(const Mat &src, Mat &dst, const std::vector<Point2i> &coords, DT_S32 ksize, DT_S32 start_row, DT_S32 end_row)
{
    auto Functor = MorphFunctor<Tp, MORPH_TYPE>();

    const DT_S32 nz = coords.size();
    std::vector<const Tp*> src_row(ksize, DT_NULL);
    std::vector<const Tp*> src_ptr(nz, DT_NULL);

    const DT_S32 channel = dst.GetSizes().m_channel;
    const DT_S32 width   = dst.GetSizes().m_width * channel;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        for (DT_S32 dy = 0; dy < ksize; dy++)
        {
            src_row[dy] = src.Ptr<Tp>(y + dy);
        }
        Tp *dst_row = dst.Ptr<Tp>(y);

        for (DT_S32 k = 0; k < nz; k++)
        {
            src_ptr[k] = src_row[coords[k].m_y] + coords[k].m_x * channel;
        }

        for (DT_S32 x = 0; x < width; x++)
        {
            Tp val = src_ptr[0][x];
            for (DT_S32 k = 0; k < nz; k++)
            {
                val = Functor(val, src_ptr[k][x]);
            }
            dst_row[x] = val;
        }
    }

    return Status::OK;
}

template <typename Tp, MorphType MORPH_TYPE>
static Status MorphNoneImpl(Context *ctx, const Mat &src, Mat &dst, Mat &src_border, const Mat &kmat, DT_S32 iterations, const OpTarget &target)
{
    Status ret = Status::ERROR;

    // preprocess kernel
    const DT_S32 ksize = kmat.GetSizes().m_width;
    DT_S32 ksh         = ksize >> 1;
    DT_S32 nz          = CountNonZero(kmat);
    std::vector<Point2i> coords(nz);

    if (0 == nz)
    {
        AURA_ADD_ERROR_STRING(ctx, "morph kmat nz = 0");
        return Status::ERROR;
    }

    DT_S32 idx = 0;
    for (DT_S32 y = 0; y < ksize; y++)
    {
        const DT_U8 *ker_row = kmat.Ptr<DT_U8>(y);
        for (DT_S32 x = 0; x < ksize; x++)
        {
            DT_U8 val = ker_row[x];
            if (0 == val)
            {
                continue;
            }
            coords[idx++] = Point2i(x, y);
        }
    }

    DT_S32 oheight = dst.GetSizes().m_height;

    // basic implementation
    {
        if (IMakeBorder(ctx, src, src_border, ksh, ksh, ksh, ksh, BorderType::REPLICATE, Scalar(), target) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MakeBorder failed");
            return Status::ERROR;
        }

        if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (DT_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<DT_S32>(0), oheight, MorphNoneCore<Tp, MORPH_TYPE>, std::cref(src_border),
                                  std::ref(dst), std::cref(coords), ksize);
        }
        else
        {
            ret = MorphNoneCore<Tp, MORPH_TYPE>(src_border, dst, coords, ksize, 0, oheight);
        }
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MorphNoneCore failed");
            return Status::ERROR;
        }
    }

    // iterative processing
    for (DT_S32 iter = 1; iter < iterations; iter++)
    {
        if (IMakeBorder(ctx, dst, src_border, ksh, ksh, ksh, ksh, BorderType::REPLICATE, Scalar(), target) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MakeBorder failed");
            return Status::ERROR;
        }

        if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (DT_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<DT_S32>(0), oheight, MorphNoneCore<Tp, MORPH_TYPE>, std::cref(src_border),
                                  std::ref(dst), std::cref(coords), ksize);
        }
        else
        {
            ret = MorphNoneCore<Tp, MORPH_TYPE>(src_border, dst, coords, ksize, 0, oheight);
        }
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "MorphNoneCore failed");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
Status MorphNoneHelper(Context *ctx, const Mat &src, Mat &dst, Mat &src_border, const Mat &kmat, MorphType type, DT_S32 iterations, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case MorphType::ERODE:
        {
            ret = MorphNoneImpl<Tp, MorphType::ERODE>(ctx, src, dst, src_border, kmat, iterations, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "MorphNoneImpl<Tp, MorphType::ERODE> run failed!");
            }
            break;
        }

        case MorphType::DILATE:
        {
            ret = MorphNoneImpl<Tp, MorphType::DILATE>(ctx, src, dst, src_border, kmat, iterations, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "MorphNoneImpl<Tp, MorphType::DILATE> run failed!");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported morph type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

MorphNone::MorphNone(Context *ctx, MorphType type, const OpTarget &target) : MorphImpl(ctx, type, target)
{}

Status MorphNone::SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape, DT_S32 iterations)
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

    if ((MorphShape::RECT == m_shape) && (m_iterations > 1))
    {
        m_ksize += (m_ksize - 1) * (m_iterations - 1);
        m_iterations = 1;
    }

    return Status::OK;
}

Status MorphNone::Initialize()
{
    if (MorphImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::Initialize() failed");
        return Status::ERROR;
    }

    m_kmat = MorphKernelMat(m_ctx, m_shape, m_ksize);
    if (!m_kmat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphKernelMat failed");
        return Status::ERROR;
    }

    DT_S32 ksh          = m_ksize >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(ksh << 1, ksh << 1, 0);
    m_src_border        = Mat(m_ctx, m_src->GetElemType(), border_sizes);

    if (!m_src_border.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid m_src_border");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MorphNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MorphNoneHelper<DT_U8>(m_ctx, *src, *dst, m_src_border, m_kmat, m_type, m_iterations, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MorphNoneHelper<DT_U8> run failed!");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = MorphNoneHelper<DT_U16>(m_ctx, *src, *dst, m_src_border, m_kmat, m_type, m_iterations, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MorphNoneHelper<DT_U16> run failed!");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = MorphNoneHelper<DT_S16>(m_ctx, *src, *dst, m_src_border, m_kmat, m_type, m_iterations, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MorphNoneHelper<DT_S16> run failed!");
            }
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = MorphNoneHelper<MI_F16>(m_ctx, *src, *dst, m_src_border, m_kmat, m_type, m_iterations, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MorphNoneHelper<MI_F16> run failed!");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = MorphNoneHelper<DT_F32>(m_ctx, *src, *dst, m_src_border, m_kmat, m_type, m_iterations, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MorphNoneHelper<DT_F32> run failed!");
            }
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status MorphNone::DeInitialize()
{
    m_src_border.Release();
    m_kmat.Release();

    if (MorphImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura
