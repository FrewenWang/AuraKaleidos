#include "dct_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

#define M_PI        3.14159265358979323846

namespace aura
{

template <typename Tp>
static Status DctCommNoneImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Sizes3 sz     = src.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 height = sz.m_height;

    // The subsequent offset uses a pointer of its own type, so here use a stride
    DT_S32 row_stride = dst.GetRowPitch() / sizeof(DT_F32);

    DT_S32 buf_sz  = height * sizeof(DT_F32);
    Mat param_mat(ctx, ElemType::U8, {1, buf_sz, 1}, AURA_MEM_DEFAULT);
    Mat coeff_row_mat(ctx, ElemType::F32, {width,  width,  1}, AURA_MEM_DEFAULT);
    Mat coeff_col_mat(ctx, ElemType::F32, {height, height, 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid() || !coeff_row_mat.IsValid() || !coeff_col_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "DctCommNoneImpl failed to get param_mat / coeff_row_mat / coeff_col_mat");
        return Status::ERROR;
    }

    DT_F32 coef_x0  = Sqrt(0.5f);
    DT_F32 coef_col = Sqrt(2.f / height);
    DT_F32 coef_row = Sqrt(2.f / width);
    DT_F32 div_row  = 2.f * width;
    DT_F32 div_col  = 2.f * height;
    DT_F32 *buffer  = param_mat.Ptr<DT_F32>(0);

    for (DT_S32 idx_m = 0; idx_m < width; idx_m++)
    {
        DT_F32 *coeff_row = coeff_row_mat.Ptr<DT_F32>(idx_m);
        for (DT_S32 idx_k = 0; idx_k < width; idx_k++)
        {
            // Attempted to optimize parameter calculations, but errors will be amplified by other methods
            coeff_row[idx_k] = Cos(((M_PI * idx_m) * ((idx_k * 2.f) + 1.f)) / div_row);
        }
    }

    for (DT_S32 idx_m = 0; idx_m < height; idx_m++)
    {
        DT_F32 *coeff_col = coeff_col_mat.Ptr<DT_F32>(idx_m);
        for (DT_S32 idx_k = 0; idx_k < height; idx_k++)
        {
            coeff_col[idx_k] = Cos(((M_PI * idx_m) * ((idx_k * 2.f) + 1.f)) / div_col);
        }
    }

    auto row_coeff_func = [&](DT_S32 start_row, DT_S32 end_row)->Status
    {
        for (DT_S32 idx_row = start_row; idx_row < end_row; idx_row++)
        {
            const Tp *src_row = src.Ptr<Tp>(idx_row);
            DT_F32 *dst_row   = dst.Ptr<DT_F32>(idx_row);

            for (DT_S32 idx_m = 0; idx_m < width; idx_m++)
            {
                const DT_F32 *coeff_row = coeff_row_mat.Ptr<DT_F32>(idx_m);
                DT_F32 result = 0;

                for (DT_S32 idx_k = 0; idx_k < width; idx_k++)
                {
                    result += static_cast<DT_F32>(src_row[idx_k]) * coeff_row[idx_k];
                }

                dst_row[idx_m] = result * coef_row;
            }

            dst_row[0] *= coef_x0;
        }

        return Status::OK;
    };

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, row_coeff_func);
    }
    else
    {
        ret = row_coeff_func(0, height);
    }
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "DctCommNoneImpl row_coeff_func failed.");
        return Status::ERROR;
    }

    auto col_coeff_func = [&](DT_S32 start_col, DT_S32 end_col)->Status
    {
        for (DT_S32 idx_col = start_col; idx_col < end_col; idx_col++)
        {
            DT_F32 *transp_src = reinterpret_cast<DT_F32 *>(dst.GetData());
            DT_F32 *dst_row    = transp_src;

            for (DT_S32 idx_row = 0; idx_row < height; ++idx_row)
            {
                buffer[idx_row]  = transp_src[idx_col];
                transp_src      += row_stride;
            }

            for (DT_S32 idx_m = 0; idx_m < height; idx_m++)
            {
                DT_F32 result = 0.f;
                const DT_F32 *coeff_col = coeff_col_mat.Ptr<DT_F32>(idx_m);

                for (DT_S32 idx_k = 0; idx_k < height; idx_k++)
                {
                    result += buffer[idx_k] * coeff_col[idx_k];
                }

                dst_row[idx_col]  = result * coef_col;
                dst_row          += row_stride;
            }
        }

        return Status::OK;
    };

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, width, col_coeff_func);
    }
    else
    {
        ret = col_coeff_func(0, width);
    }
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "DctCommNoneImpl col_coeff_func failed.");
        return Status::ERROR;
    }

    DT_F32 *dst_row = dst.Ptr<DT_F32>(0);
    for (DT_S32 x = 0; x < width; ++x)
    {
        dst_row[x] *= coef_x0;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status DctRadix2NoneImpl(Context *ctx, const Mat &src, Mat &dst)
{
    Sizes3 sz            = src.GetSizes();
    DT_S32 width         = sz.m_width;
    DT_S32 height        = sz.m_height;
    DT_S32 half_w        = width / 2;
    DT_S32 half_h        = height / 2;
    DT_S32 dst_row_pitch = dst.GetRowPitch();
    DT_U32 max_len       = Max(width, height);
    DT_U64 buf_sz        = max_len * sizeof(DT_U16) + (max_len + half_w + half_h + width + height) * sizeof(DT_F32) * 2;

    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(buf_sz), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "DctRadix2NoneImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + max_len);
    std::complex<DT_F32> *exp_table         = buffer + max_len;
    std::complex<DT_F32> *dft_row_exp_table = exp_table;
    std::complex<DT_F32> *dft_col_exp_table = dft_row_exp_table + half_w;
    std::complex<DT_F32> *dct_row_exp_table = dft_col_exp_table + half_h;
    std::complex<DT_F32> *dct_col_exp_table = dct_row_exp_table + width;

    GetDftExpTable<0>(dft_row_exp_table, width);
    GetDftExpTable<0>(dft_col_exp_table, height);
    GetDctExpTable<0>(dct_row_exp_table, width);
    GetDctExpTable<0>(dct_col_exp_table, height);

    DT_F32 coef_x0  = Sqrt(0.5f);
    DT_F32 coef_row = Sqrt(2.0 / width);
    DT_F32 coef_col = Sqrt(2.0 / height);
    // Row Dct Process
    GetReverseIndex(idx_table, width);
    for (DT_S32 y = 0; y < height; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        DT_F32   *dst_row = dst.Ptr<DT_F32>(y);

        for (DT_S32 x = 0; x < half_w; ++x)
        {
            DT_S32 idx0 = idx_table[x];
            DT_S32 idx1 = idx_table[width - x - 1];

            buffer[idx0].real(static_cast<DT_F32>(src_row[2 * x]));
            buffer[idx0].imag(0.0f);
            buffer[idx1].real(static_cast<DT_F32>(src_row[2 * x + 1]));
            buffer[idx1].imag(0.0f);
        }

        ButterflyTransformNone(buffer, 2, width, DT_FALSE, dft_row_exp_table);

        for (DT_S32 x = 0; x < width; ++x)
        {
            DT_F32 cos_val = dct_row_exp_table[x].real();
            DT_F32 sin_val = dct_row_exp_table[x].imag();
            dst_row[x]     = (buffer[x].real() * cos_val + buffer[x].imag() * sin_val) * coef_row;
        }

        dst_row[0] *= coef_x0;
    }

    // Col Dct Process
    GetReverseIndex(idx_table, height);
    for (DT_S32 x = 0; x < width; ++x)
    {
        DT_U8 *dst_data = reinterpret_cast<DT_U8 *>(dst.GetData());

        for (DT_S32 y = 0; y < half_h; ++y)
        {
            DT_F32 even_value = reinterpret_cast<DT_F32 *>(dst_data)[x];
            DT_F32 odd_value  = reinterpret_cast<DT_F32 *>(dst_data + dst_row_pitch)[x];

            buffer[y].real(even_value);
            buffer[y].imag(0.0f);
            buffer[height - y - 1].real(odd_value);
            buffer[height - y - 1].imag(0.0f);

            dst_data += 2 * dst_row_pitch;
        }

        for (DT_S32 y = 0; y < height; ++y)
        {
            DT_S32 idx = idx_table[y];
            if (idx > y)
            {
                Swap(buffer[y], buffer[idx]);
            }
        }

        ButterflyTransformNone(buffer, 2, height, DT_FALSE, dft_col_exp_table);

        dst_data = reinterpret_cast<DT_U8 *>(dst.GetData());

        for (DT_S32 y = 0; y < height; ++y)
        {
            DT_F32 *dst_row = reinterpret_cast<DT_F32 *>(dst_data);
            DT_F32 cos_val  = dct_col_exp_table[y].real();
            DT_F32 sin_val  = dct_col_exp_table[y].imag();

            dst_row[x]  = (buffer[y].real() * cos_val + buffer[y].imag() * sin_val) * coef_col;
            dst_data   += dst_row_pitch;
        }
    }

    DT_F32 *dst_row = dst.Ptr<DT_F32>(0);

    for (DT_S32 x = 0; x < width; ++x)
    {
        dst_row[x] *= coef_x0;
    }

    return Status::OK;
}

static Status DctRadix2NoneHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    ElemType elem_type = src.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = DctRadix2NoneImpl<DT_U8>(ctx, src, dst);
            break;
        }
        case ElemType::S8:
        {
            ret = DctRadix2NoneImpl<DT_S8>(ctx, src, dst);
            break;
        }
        case ElemType::U16:
        {
            ret = DctRadix2NoneImpl<DT_U16>(ctx, src, dst);
            break;
        }
        case ElemType::S16:
        {
            ret = DctRadix2NoneImpl<DT_S16>(ctx, src, dst);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = DctRadix2NoneImpl<MI_F16>(ctx, src, dst);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = DctRadix2NoneImpl<DT_F32>(ctx, src, dst);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status DctCommNoneHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    ElemType elem_type = src.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = DctCommNoneImpl<DT_U8>(ctx, src, dst, target);
            break;
        }
        case ElemType::S8:
        {
            ret = DctCommNoneImpl<DT_S8>(ctx, src, dst, target);
            break;
        }
        case ElemType::U16:
        {
            ret = DctCommNoneImpl<DT_U16>(ctx, src, dst, target);
            break;
        }
        case ElemType::S16:
        {
            ret = DctCommNoneImpl<DT_S16>(ctx, src, dst, target);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = DctCommNoneImpl<MI_F16>(ctx, src, dst, target);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = DctCommNoneImpl<DT_F32>(ctx, src, dst, target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

DctNone::DctNone(Context *ctx, const OpTarget &target) : DctImpl(ctx, target)
{}

Status DctNone::SetArgs(const Array *src, Array *dst)
{
    if (DctImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DctImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) ||(dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType src_type = src->GetElemType();
    if (ElemType::F64 == src_type)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support DT_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DctNone::Run()
{
    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat *dst = dynamic_cast<Mat *>(m_dst);
    if ((DT_NULL == src) ||(DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input src or dst is null");
        return Status::ERROR;
    }

    Status ret    = Status::ERROR;
    Sizes3 src_sz = src->GetSizes();
    if (!IsPowOf2(src_sz.m_width) || !IsPowOf2(src_sz.m_height) || (1 == src_sz.m_width) || (1 == src_sz.m_height))
    {
        ret = DctCommNoneHelper(m_ctx, *src, *dst, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DctCommNoneHelper failed.");
        }
    }
    else
    {
        ret = DctRadix2NoneHelper(m_ctx, *src, *dst);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DctRadix2NoneHelper failed.");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp>
static Status IDctCommNoneImpl(Context *ctx, const Mat &src, Mat &mid, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    Sizes3 sz         = src.GetSizes();
    DT_S32 width      = sz.m_width;
    DT_S32 height     = sz.m_height;
    DT_S32 mid_stride = mid.GetRowPitch() / sizeof(DT_F32);
    DT_S32 dst_stride = dst.GetRowPitch() / sizeof(Tp);

    DT_U64 buffer_sz = height * sizeof(DT_F32);
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(buffer_sz), 1}, AURA_MEM_DEFAULT);
    Mat coeff_row_mat(ctx, ElemType::F32, {width,  width,  1}, AURA_MEM_DEFAULT);
    Mat coeff_col_mat(ctx, ElemType::F32, {height, height, 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid() || !coeff_row_mat.IsValid() || !coeff_col_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctCommNoneImpl failed to get param_mat / coeff_row_mat / coeff_col_mat");
        return Status::ERROR;
    }

    DT_F32 coef_x0  = Sqrt(0.5f);
    DT_F32 coef_col = Sqrt(2.f / height);
    DT_F32 coef_row = Sqrt(2.f / width);
    DT_F32 div_row  = 2.f * width;
    DT_F32 div_col  = 2.f * height;
    DT_F32 *buffer  = param_mat.Ptr<DT_F32>(0);

    for (DT_S32 idx_m = 0; idx_m < width; idx_m++)
    {
        DT_F32 *coeff_row = coeff_row_mat.Ptr<DT_F32>(idx_m);
        for (DT_S32 idx_k = 0; idx_k < width; idx_k++)
        {
            coeff_row[idx_k] = Cos(((M_PI * idx_k) * ((idx_m * 2.f) + 1.f)) / div_row);
        }

        coeff_row[0] *= coef_x0;
    }

    for (DT_S32 idx_m = 0; idx_m < height; idx_m++)
    {
        DT_F32 *coeff_col = coeff_col_mat.Ptr<DT_F32>(idx_m);
        for (DT_S32 idx_k = 0; idx_k < height; idx_k++)
        {
            coeff_col[idx_k] = Cos(((M_PI * idx_k) * ((idx_m * 2.f) + 1.f)) / div_col);
        }

        coeff_col[0] *= coef_x0;
    }

    auto row_coeff_func = [&](DT_S32 start_row, DT_S32 end_row)->Status
    {
        for (DT_S32 idx_row = start_row; idx_row < end_row; idx_row++)
        {
            const  DT_F32 *src_row = src.Ptr<DT_F32>(idx_row);
            DT_F32 *mid_row        = mid.Ptr<DT_F32>(idx_row);

            for (DT_S32 idx_m = 0; idx_m < width; idx_m++)
            {
                const DT_F32 *coeff_row = coeff_row_mat.Ptr<DT_F32>(idx_m);
                DT_F32 result = 0;

                for (DT_S32 idx_k = 0; idx_k < width; idx_k++)
                {
                    result += src_row[idx_k] * coeff_row[idx_k];
                }

                mid_row[idx_m] = result * coef_row;
            }
        }

        return Status::OK;
    };

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, height, row_coeff_func);
    }
    else
    {
        ret = row_coeff_func(0, height);
    }
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctCommNoneImpl row_coeff_func failed.");
        return Status::ERROR;
    }

    auto col_coeff_func = [&](DT_S32 start_col, DT_S32 end_col)->Status
    {
        for (DT_S32 idx_col = start_col; idx_col < end_col; idx_col++)
        {
            DT_F32 *transp_src = reinterpret_cast<DT_F32 *>(mid.GetData());
            Tp     *dst_row    = reinterpret_cast<Tp *>(dst.GetData());

            for (DT_S32 idx_row = 0; idx_row < height; ++idx_row)
            {
                buffer[idx_row]  = transp_src[idx_col];
                transp_src      += mid_stride;
            }

            for (DT_S32 idx_m = 0; idx_m < height; idx_m++)
            {
                DT_F32 result = 0.f;
                const DT_F32 *coeff_col = coeff_col_mat.Ptr<DT_F32>(idx_m);

                for (DT_S32 idx_k = 0; idx_k < height; idx_k++)
                {
                    result += buffer[idx_k] * coeff_col[idx_k];
                }

                dst_row[idx_col]  = SaturateCast<Tp>(result * coef_col);
                dst_row          += dst_stride;
            }
        }

        return Status::OK;
    };

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
            return Status::ERROR;
        }

        ret = wp->ParallelFor((DT_S32)0, width, col_coeff_func);
    }
    else
    {
        ret = col_coeff_func(0, width);
    }
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctCommNoneImpl col_coeff_func failed.");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status IDctRadix2NoneImpl(Context *ctx, const Mat &src, Mat &mid, Mat &dst)
{
    Sizes3 sz            = src.GetSizes();
    DT_S32 width         = sz.m_width;
    DT_S32 height        = sz.m_height;
    DT_S32 half_w        = width / 2;
    DT_S32 half_h        = height / 2;
    DT_S32 mid_row_pitch = mid.GetRowPitch();
    DT_S32 dst_row_pitch = dst.GetRowPitch();

    DT_U32 max_len   = Max(width, height);
    DT_U64 buffer_sz = max_len * sizeof(DT_U16) + (max_len + half_w + half_h + width + height) * sizeof(DT_F32) * 2;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(buffer_sz), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctRadix2NoneImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + max_len);
    std::complex<DT_F32> *exp_table         = buffer + max_len;
    std::complex<DT_F32> *dft_row_exp_table = exp_table;
    std::complex<DT_F32> *dft_col_exp_table = dft_row_exp_table + half_w;
    std::complex<DT_F32> *dct_row_exp_table = dft_col_exp_table + half_h;
    std::complex<DT_F32> *dct_col_exp_table = dct_row_exp_table + width;

    GetDftExpTable<0>(dft_row_exp_table, width);
    GetDftExpTable<0>(dft_col_exp_table, height);
    GetDctExpTable<1>(dct_row_exp_table, width);
    GetDctExpTable<1>(dct_col_exp_table, height);

    DT_F32 coef_row_x0 = Sqrt(1.0f / width);
    DT_F32 coef_col_x0 = Sqrt(1.0f / height);
    DT_F32 coef_row    = Sqrt(2.0f / width);
    DT_F32 coef_col    = Sqrt(2.0f / height);

    // Row IDct Process
    GetReverseIndex(idx_table, width);
    for (DT_S32 y = 0; y < height; ++y)
    {
        const  DT_F32 *src_row = src.Ptr<DT_F32>(y);
        DT_F32 *mid_row        = mid.Ptr<DT_F32>(y);

        buffer[0].real(src_row[0] * coef_row_x0);
        buffer[0].imag(0);

        for (DT_S32 x = 1; x < width; ++x)
        {
            DT_F32 cos_val = dct_row_exp_table[x].real();
            DT_F32 sin_val = dct_row_exp_table[x].imag();
            buffer[x].real(src_row[x] * coef_row * cos_val);
            buffer[x].imag(src_row[x] * coef_row * sin_val);
        }

        for (DT_S32 i = 0; i < width; ++i)
        {
            DT_S32 idx = idx_table[i];
            if (idx > i)
            {
                Swap(buffer[i], buffer[idx]);
            }
        }

        ButterflyTransformNone(buffer, 2, width, DT_FALSE, dft_row_exp_table);

        for (DT_S32 x = 0; x < half_w; ++x)
        {
            mid_row[2 * x]     = buffer[x].real();
            mid_row[2 * x + 1] = buffer[width - x - 1].real();
        }
    }

    // Col IDct Process
    GetReverseIndex(idx_table, height);

    for (DT_S32 x = 0; x < width; ++x)
    {
        DT_U8  *mid_data = reinterpret_cast<DT_U8 *>(mid.GetData());
        DT_F32 *mid_row  = reinterpret_cast<DT_F32 *>(mid_data);

        buffer[0].real(mid_row[x] * coef_col_x0);
        buffer[0].imag(0.0f);
        mid_data += mid_row_pitch;

        for (DT_S32 y = 1; y < height; ++y)
        {
            mid_row = reinterpret_cast<DT_F32 *>(mid_data);

            DT_F32 cos_val = dct_col_exp_table[y].real();
            DT_F32 sin_val = dct_col_exp_table[y].imag();

            buffer[y].real(mid_row[x] * coef_col * cos_val);
            buffer[y].imag(mid_row[x] * coef_col * sin_val);
            mid_data += mid_row_pitch;
        }

        for (DT_S32 y = 0; y < height; ++y)
        {
            DT_S32 idx = idx_table[y];
            if (idx > y)
            {
                Swap(buffer[y], buffer[idx]);
            }
        }

        ButterflyTransformNone(buffer, 2, height, DT_FALSE, dft_col_exp_table);

        DT_U8 *dst_data = reinterpret_cast<DT_U8 *>(dst.GetData());
        for (DT_S32 y = 0; y < half_h; ++y)
        {
            Tp *dst_even = reinterpret_cast<Tp *>(dst_data);
            Tp *dst_odd  = reinterpret_cast<Tp *>(dst_data + dst_row_pitch);

            dst_even[x]  = SaturateCast<Tp>(buffer[y].real());
            dst_odd[x]   = SaturateCast<Tp>(buffer[height - y - 1].real());
            dst_data    += 2 * dst_row_pitch;
        }
    }

    return Status::OK;
}

static Status IDctRadix2NoneHelper(Context *ctx, const Mat &src, Mat &mid, Mat &dst)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();

    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDctRadix2NoneImpl<DT_U8>(ctx, src, mid, dst);
            break;
        }
        case ElemType::S8:
        {
            ret = IDctRadix2NoneImpl<DT_S8>(ctx, src, mid, dst);
            break;
        }
        case ElemType::U16:
        {
            ret = IDctRadix2NoneImpl<DT_U16>(ctx, src, mid, dst);
            break;
        }
        case ElemType::S16:
        {
            ret = IDctRadix2NoneImpl<DT_S16>(ctx, src, mid, dst);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = IDctRadix2NoneImpl<MI_F16>(ctx, src, mid, dst);
            break;
        }
#endif //AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = IDctRadix2NoneImpl<DT_F32>(ctx, src, dst, dst);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status IDctCommNoneHelper(Context *ctx, const Mat &src, Mat &mid, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();

    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDctCommNoneImpl<DT_U8>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::S8:
        {
            ret = IDctCommNoneImpl<DT_S8>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::U16:
        {
            ret = IDctCommNoneImpl<DT_U16>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::S16:
        {
            ret = IDctCommNoneImpl<DT_S16>(ctx, src, mid, dst, target);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = IDctCommNoneImpl<MI_F16>(ctx, src, mid, dst, target);
            break;
        }
#endif //AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = IDctCommNoneImpl<DT_F32>(ctx, src, dst, dst, target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

IDctNone::IDctNone(Context *ctx, const OpTarget &target) : IDctImpl(ctx, target)
{}

Status IDctNone::SetArgs(const Array *src, Array *dst)
{
    if (IDctImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IDctImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) ||(dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType dst_type = dst->GetElemType();
    if (ElemType::F64 == dst_type)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current dst does not support DT_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IDctNone::Run()
{
    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat   *dst     = dynamic_cast<Mat *>(m_dst);
    Mat   *mid     = &m_mid;
    if ((DT_NULL == src) ||(DT_NULL == dst) || (DT_NULL == mid))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input src or dst or mid is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    Sizes3 src_sz = src->GetSizes();
    if (!IsPowOf2(src_sz.m_width) || !IsPowOf2(src_sz.m_height) || (1 == src_sz.m_width) || (1 == src_sz.m_height))
    {
        ret = IDctCommNoneHelper(m_ctx, *src, *mid, *dst, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDctCommNoneHelper failed.");
        }
    }
    else
    {
        ret = IDctRadix2NoneHelper(m_ctx, *src, *mid, *dst);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDctRadix2NoneHelper failed.");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura