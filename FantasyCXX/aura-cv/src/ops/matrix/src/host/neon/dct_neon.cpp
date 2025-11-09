#include "dct_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

template <typename Tp>
static Status DctRadix2SingleRowNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 half_w = width / 2;

    DT_U32 idx_bytes   = width * sizeof(DT_U16);
    DT_U64 total_bytes = 2 * width * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (DT_FALSE == param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "DctRadix2SingleRowNeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + width);
    std::complex<DT_F32> *exp_table         = buffer + width;
    std::complex<DT_F32> *dft_row_exp_table = exp_table;
    std::complex<DT_F32> *dct_row_exp_table = dft_row_exp_table + half_w;

    GetDftExpTable<0>(dft_row_exp_table, width);
    GetDctExpTable<0>(dct_row_exp_table, width);

    DT_F32 coef_x0  = Sqrt(0.5f);
    DT_F32 coef_row = Sqrt(2.0 / width);

    GetReverseIndex(idx_table, width);

    const Tp *src_c = src.Ptr<Tp>(0);
    DT_F32   *dst_c = dst.Ptr<DT_F32>(0);

    for (DT_S32 x = 0; x < half_w; ++x)
    {
        DT_S32 idx0 = idx_table[x];
        DT_S32 idx1 = idx_table[width - x - 1];

        buffer[idx0].real(static_cast<DT_F32>(src_c[2 * x]));
        buffer[idx0].imag(0.0f);
        buffer[idx1].real(static_cast<DT_F32>(src_c[2 * x + 1]));
        buffer[idx1].imag(0.0f);
    }

    ButterflyTransformNeon(buffer, 2, width, DT_FALSE, dft_row_exp_table);

    float32x4_t vqf32_coef_row;
    neon::vdup(vqf32_coef_row, coef_row);

    for (DT_S32 x = 0; x < width; x += 4)
    {
        DT_F32 *buf_ptr = reinterpret_cast<DT_F32 *>(&buffer[x]);
        DT_F32 *w_ptr   = reinterpret_cast<DT_F32 *>(&dct_row_exp_table[x]);

        float32x4x2_t v2qf32_buf = neon::vload2q(buf_ptr);
        float32x4x2_t v2qf32_w   = neon::vload2q(w_ptr);
        float32x4_t   v2qf32_res = neon::vmul(v2qf32_buf.val[0], v2qf32_w.val[0]);

        v2qf32_res = neon::vmla(v2qf32_res, v2qf32_buf.val[1], v2qf32_w.val[1]);
        v2qf32_res = neon::vmul(v2qf32_res, vqf32_coef_row);

        neon::vstore(dst_c + x, v2qf32_res);
    }

    dst_c[0] *= coef_x0;

    return Status::OK;
}

template <typename Tp>
static Status DctRadix2SingleColNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    DT_S32 height = sz.m_height;
    DT_S32 half_h = height / 2;

    DT_U32 idx_bytes   = height * sizeof(DT_U16);
    DT_U64 total_bytes = 2 * height * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (DT_FALSE == param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "DctRadix2SingleColNeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + height);
    std::complex<DT_F32> *exp_table         = buffer + height;
    std::complex<DT_F32> *dft_col_exp_table = exp_table;
    std::complex<DT_F32> *dct_col_exp_table = dft_col_exp_table + half_h;

    GetDftExpTable<0>(dft_col_exp_table, height);
    GetDctExpTable<0>(dct_col_exp_table, height);

    DT_F32 coef_x0  = Sqrt(0.5f);
    DT_F32 coef_col = Sqrt(2.0 / height);

    GetReverseIndex(idx_table, height);

    const Tp *src_col0 = src.Ptr<Tp>(0);
    const Tp *src_col1 = src.Ptr<Tp>(0);

    for (DT_S32 x = 0; x < half_h; ++x)
    {
        DT_S32 idx0    = idx_table[x];
        DT_S32 idx1    = idx_table[height - x - 1];
        DT_S32 col_idx = x << 1;

        src_col0 = src.Ptr<Tp>(col_idx);
        src_col1 = src.Ptr<Tp>(col_idx + 1);

        buffer[idx0].real(static_cast<DT_F32>(src_col0[col_idx]));
        buffer[idx0].imag(0.0f);
        buffer[idx1].real(static_cast<DT_F32>(src_col1[col_idx + 1]));
        buffer[idx1].imag(0.0f);
    }

    ButterflyTransformNeon(buffer, 2, height, DT_FALSE, dft_col_exp_table);

    DT_F32 *dst_col = NULL;
    for (DT_S32 x = 0; x < height; ++x)
    {
        DT_F32 cos_val = dct_col_exp_table[x].real();
        DT_F32 sin_val = dct_col_exp_table[x].imag();

        dst_col    = dst.Ptr<DT_F32>(x);
        dst_col[0] = (buffer[x].real() * cos_val + buffer[x].imag() * sin_val) * coef_col;
    }

    dst_col     = dst.Ptr<DT_F32>(0);
    dst_col[0] *= coef_x0;

    return Status::OK;
}

template <typename Tp>
static Status DctRadix2NeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();

    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 height = sz.m_height;

    if (1 == height)
    {
        return DctRadix2SingleRowNeonImpl<Tp>(ctx, src, dst);
    }

    if (1 == width)
    {
        // CheckNeonWidth runing in Dct::Initialize
        return DctRadix2SingleColNeonImpl<Tp>(ctx, src, dst);
    }

    DT_S32 half_w    = width / 2;
    DT_S32 half_h    = height / 2;
    DT_S32 row_pitch = dst.GetRowPitch();

    DT_U32 max_len     = Max(width, height);
    DT_U32 idx_bytes   = max_len * sizeof(DT_U16);
    DT_U64 total_bytes = 3 * max_len * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "DctRadix2NeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + max_len);
    std::complex<DT_F32> *dft_row_exp_table = buffer;
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

    DT_S32 buffer_sz = max_len * sizeof(std::complex<DT_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    auto row_process_func = [&](DT_S32 start_row, DT_S32 end_row)->Status
    {
        std::complex<DT_F32> *p_buf = thread_buffer.GetThreadData<std::complex<DT_F32>>();

        if (!p_buf)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (DT_S32 y = start_row; y < end_row; ++y)
        {
            const Tp *src_row = src.Ptr<Tp>(y);
            DT_F32   *dst_row = dst.Ptr<DT_F32>(y);

            for (DT_S32 x = 0; x < half_w; ++x)
            {
                DT_S32 idx0 = idx_table[x];
                DT_S32 idx1 = idx_table[width - x - 1];

                p_buf[idx0].real(static_cast<DT_F32>(src_row[2 * x]));
                p_buf[idx0].imag(0.0f);
                p_buf[idx1].real(static_cast<DT_F32>(src_row[2 * x + 1]));
                p_buf[idx1].imag(0.0f);
            }

            ButterflyTransformNeon(p_buf, 2, width, DT_FALSE, dft_row_exp_table);

            float32x4_t vqf32_coef_row;
            neon::vdup(vqf32_coef_row, coef_row);

            for (DT_S32 x = 0; x < width; x += 4)
            {
                DT_F32 *buf_ptr = reinterpret_cast<DT_F32 *>(&p_buf[x]);
                DT_F32 *w_ptr   = reinterpret_cast<DT_F32 *>(&dct_row_exp_table[x]);

                float32x4x2_t v2qf32_buf = neon::vload2q(buf_ptr);
                float32x4x2_t v2qf32_w   = neon::vload2q(w_ptr);
                float32x4_t   v2qf32_res = neon::vmul(v2qf32_buf.val[0], v2qf32_w.val[0]);

                v2qf32_res = neon::vmla(v2qf32_res, v2qf32_buf.val[1], v2qf32_w.val[1]);
                v2qf32_res = neon::vmul(v2qf32_res, vqf32_coef_row);

                neon::vstore(dst_row + x, v2qf32_res);
            }

            dst_row[0] *= coef_x0;
        }

        return Status::OK;
    };

    auto col_process_func = [&](DT_S32 start, DT_S32 end)->Status
    {

        std::complex<DT_F32> *p_buf = thread_buffer.GetThreadData<std::complex<DT_F32>>();

        if (!p_buf)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (DT_S32 x = start; x < end; ++x)
        {
            DT_U8 *dst_data = reinterpret_cast<DT_U8*>(dst.GetData());
            for (DT_S32 y = 0; y < half_h; ++y)
            {
                DT_F32 even_value = reinterpret_cast<DT_F32 *>(dst_data)[x];
                DT_F32 odd_value  = reinterpret_cast<DT_F32 *>(dst_data + row_pitch)[x];

                p_buf[y].real(even_value);
                p_buf[y].imag(0.0f);
                p_buf[height - y - 1].real(odd_value);
                p_buf[height - y - 1].imag(0.0f);
                dst_data += 2 * row_pitch;
            }

            for (DT_S32 y = 0; y < height; ++y)
            {
                DT_S32 idx = idx_table[y];
                if (idx > y)
                {
                    Swap(p_buf[y], p_buf[idx]);
                }
            }

            ButterflyTransformNeon(p_buf, 2, height, DT_FALSE, dft_col_exp_table);

            dst_data = reinterpret_cast<DT_U8 *>(dst.GetData());
            for (DT_S32 y = 0; y < height; ++y)
            {
                DT_F32 *dst_c  = reinterpret_cast<DT_F32 *>(dst_data);
                DT_F32 cos_val = dct_col_exp_table[y].real();
                DT_F32 sin_val = dct_col_exp_table[y].imag();

                dst_c[x] = (p_buf[y].real() * cos_val + p_buf[y].imag() * sin_val) * coef_col;
                dst_data += row_pitch;
            }
        }

        return Status::OK;
    };


    GetReverseIndex(idx_table, width);
    if (wp->ParallelFor(0, height, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dct DctRadix2NeonImpl parallel for row process failed.");
        return Status::ERROR;
    }

    GetReverseIndex(idx_table, height);
    if (wp->ParallelFor(0, width, col_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dct DctRadix2NeonImpl parallel for col process failed.");
        return Status::ERROR;
    }

    DT_F32 *dst_c = dst.Ptr<DT_F32>(0);
    for (DT_S32 x = 0; x < width; ++x)
    {
        dst_c[x] *= coef_x0;
    }

    return Status::OK;
}

DctNeon::DctNeon(Context *ctx, const OpTarget &target) : DctImpl(ctx, target)
{}

Status DctNeon::SetArgs(const Array *src, Array *dst)
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
    if ((ElemType::S32 == src_type) || (ElemType::U32 == src_type) || (ElemType::F64 == src_type))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support DT_S32/DT_U32/DT_F64 type.");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();
    if (!IsPowOf2(sz.m_width) || (!IsPowOf2(sz.m_height)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DctNeon current only support 2^n size.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DctNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat *dst = dynamic_cast<Mat *>(m_dst);
    if ((DT_NULL == src) ||(DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input src or dst is null");
    }

    Status ret = Status::ERROR;

    ElemType elem_type = src->GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = DctRadix2NeonImpl<DT_U8>(m_ctx, *src, *dst, m_target);
            break;
        }
        case ElemType::S8:
        {
            ret = DctRadix2NeonImpl<DT_S8>(m_ctx, *src, *dst, m_target);
            break;
        }
        case ElemType::U16:
        {
            ret = DctRadix2NeonImpl<DT_U16>(m_ctx, *src, *dst, m_target);
            break;
        }
        case ElemType::S16:
        {
            ret = DctRadix2NeonImpl<DT_S16>(m_ctx, *src, *dst, m_target);
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = DctRadix2NeonImpl<MI_F16>(m_ctx, *src, *dst, m_target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = DctRadix2NeonImpl<DT_F32>(m_ctx, *src, *dst, m_target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

template<typename Tp>
static Status IDctRadix2SingleRowNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 half_w = width / 2;

    DT_U32 idx_bytes   = width * sizeof(DT_U16);
    DT_U64 total_bytes = 2 * width * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctRadix2SingleRowNeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + width);
    std::complex<DT_F32> *exp_table         = buffer + width;
    std::complex<DT_F32> *dft_row_exp_table = exp_table;
    std::complex<DT_F32> *dct_row_exp_table = dft_row_exp_table + half_w;

    GetDftExpTable<0>(dft_row_exp_table, width);
    GetDctExpTable<1>(dct_row_exp_table, width);

    DT_F32 coef_row_x0 = Sqrt(1.0f / width);
    DT_F32 coef_row    = Sqrt(2.0f / width);

    GetReverseIndex(idx_table, width);

    const DT_F32 *src_c = src.Ptr<DT_F32>(0);

    buffer[0].real(src_c[0] * coef_row_x0);
    buffer[0].imag(0);

    for (DT_S32 x = 1; x < width; ++x)
    {
        DT_S32 idx = idx_table[x];
        DT_F32 cos_val = dct_row_exp_table[x].real();
        DT_F32 sin_val = dct_row_exp_table[x].imag();

        buffer[idx].real(src_c[x] * coef_row * cos_val);
        buffer[idx].imag(src_c[x] * coef_row * sin_val);
    }

    ButterflyTransformNeon(buffer, 2, width, DT_FALSE, dft_row_exp_table);

    Tp *dst_row = dst.Ptr<Tp>(0);
    for (DT_S32 x = 0; x < half_w; ++x)
    {
        DT_S32 row_idx       = x << 1;
        dst_row[row_idx]     = SaturateCast<Tp>(buffer[x].real());
        dst_row[row_idx + 1] = SaturateCast<Tp>(buffer[width - x - 1].real());
    }

    return Status::OK;
}

template<typename Tp>
static Status IDctRadix2SingleColNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    DT_S32 height = sz.m_height;
    DT_S32 half_h = height / 2;

    DT_U32 idx_bytes   = height * sizeof(DT_U16);
    DT_U64 total_bytes = 2 * height * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctRadix2SingleColNeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + height);
    std::complex<DT_F32> *exp_table         = buffer + height;
    std::complex<DT_F32> *dft_col_exp_table = exp_table;
    std::complex<DT_F32> *dct_col_exp_table = dft_col_exp_table + half_h;

    GetDftExpTable<0>(dft_col_exp_table, height);
    GetDctExpTable<1>(dct_col_exp_table, height);

    DT_F32 coef_col_x0 = Sqrt(1.0f / height);
    DT_F32 coef_col    = Sqrt(2.0 / height);

    GetReverseIndex(idx_table, height);

    const DT_F32 *src_c = src.Ptr<DT_F32>(0);

    buffer[0].real(src_c[0] * coef_col_x0);
    buffer[0].imag(0);

    for (DT_S32 x = 1; x < height; ++x)
    {
        DT_S32 idx     = idx_table[x];
        DT_F32 cos_val = dct_col_exp_table[x].real();
        DT_F32 sin_val = dct_col_exp_table[x].imag();

        src_c = src.Ptr<DT_F32>(x);
        buffer[idx].real(src_c[0] * coef_col * cos_val);
        buffer[idx].imag(src_c[0] * coef_col * sin_val);
    }

    ButterflyTransformNeon(buffer, 2, height, DT_FALSE, dft_col_exp_table);

    Tp *dst_col0 = dst.Ptr<Tp>(0);
    Tp *dst_col1 = dst.Ptr<Tp>(0);

    for (DT_S32 x = 0; x < half_h; ++x)
    {
        DT_S32 col_idx = x << 1;

        dst_col0    = dst.Ptr<Tp>(col_idx);
        dst_col1    = dst.Ptr<Tp>(col_idx + 1);
        dst_col0[0] = SaturateCast<Tp>(buffer[x].real());
        dst_col1[0] = SaturateCast<Tp>(buffer[height - x - 1].real());
    }

    return Status::OK;
}

template<typename Tp>
static Status IDctRadix2NeonImpl(Context *ctx, const Mat &src, Mat &mid, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();

    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 height = sz.m_height;

    if (1 == height)
    {
        return IDctRadix2SingleRowNeonImpl<Tp>(ctx, src, dst);
    }

    if (1 == width)
    {
        return IDctRadix2SingleColNeonImpl<Tp>(ctx, src, dst);
    }

    DT_S32 half_w        = width / 2;
    DT_S32 half_h        = height / 2;
    DT_S32 mid_pitch     = mid.GetRowPitch();
    DT_S32 dst_row_pitch = dst.GetRowPitch();

    DT_U32 max_len     = Max(width, height);
    DT_U32 idx_bytes   = max_len * sizeof(DT_U16);
    DT_U64 total_bytes = 3 * max_len * sizeof(DT_F32) * 2 + idx_bytes;
    Mat param_mat(ctx, ElemType::U8, {1, SaturateCast<DT_S32>(total_bytes), 1}, AURA_MEM_DEFAULT);
    if (!param_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "IDctRadix2NeonImpl failed to get param_mat");
        return Status::ERROR;
    }

    DT_U16 *idx_table = param_mat.Ptr<DT_U16>(0);

    std::complex<DT_F32> *buffer            = reinterpret_cast<std::complex<DT_F32> *>(idx_table + max_len);
    std::complex<DT_F32> *dft_row_exp_table = buffer;
    std::complex<DT_F32> *dft_col_exp_table = dft_row_exp_table + half_w;
    std::complex<DT_F32> *dct_row_exp_table = dft_col_exp_table + half_h;
    std::complex<DT_F32> *dct_col_exp_table = dct_row_exp_table + width;

    GetDftExpTable<0>(dft_row_exp_table, width);
    GetDftExpTable<0>(dft_col_exp_table, height);
    GetDctExpTable<1>(dct_row_exp_table, width);
    GetDctExpTable<1>(dct_col_exp_table, height);

    DT_F32 coef_row_x0 = Sqrt(1.0f / width);
    DT_F32 coef_col_x0 = Sqrt(1.0f / height);
    DT_F32 coef_row    = Sqrt(2.0 / width);
    DT_F32 coef_col    = Sqrt(2.0 / height);

    DT_S32 buffer_sz = max_len * sizeof(std::complex<DT_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    auto row_process_func = [&](DT_S32 start, DT_S32 end)->Status
    {

        std::complex<DT_F32> *p_buf = thread_buffer.GetThreadData<std::complex<DT_F32>>();

        if (!p_buf)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (DT_S32 y = start; y < end; ++y)
        {
            const  DT_F32 *src_row = src.Ptr<DT_F32>(y);
            DT_F32 *mid_row        = mid.Ptr<DT_F32>(y);

            p_buf[0].real(src_row[0] * coef_row_x0);
            p_buf[0].imag(0);

            for (DT_S32 x = 1; x < width; ++x)
            {
                DT_S32 idx     = idx_table[x];
                DT_F32 cos_val = dct_row_exp_table[x].real();
                DT_F32 sin_val = dct_row_exp_table[x].imag();

                p_buf[idx].real(src_row[x] * coef_row * cos_val);
                p_buf[idx].imag(src_row[x] * coef_row * sin_val);
            }

            ButterflyTransformNeon(p_buf, 2, width, DT_FALSE, dft_row_exp_table);

            DT_F32 *buf_head = reinterpret_cast<DT_F32 *>(p_buf);

            // width size is based radix2, so do specialized process for litter size
            if (2 == width)
            {
                mid_row[0] = p_buf[0].real();
                mid_row[1] = p_buf[1].real();
            }
            else if (4 == width)
            {
                DT_F32 *buf_tail = reinterpret_cast<DT_F32 *>(p_buf + width - 2);
                float32x2x2_t v2f32_res;
                float32x2x2_t v2f32_head = neon::vload2(buf_head);
                float32x2x2_t v2f32_tail = neon::vload2(buf_tail);
                float32x2_t   v2f32_rev  = neon::vrev64(v2f32_tail.val[0]);

                v2f32_res.val[0] = v2f32_head.val[0];
                v2f32_res.val[1] = v2f32_rev;
                neon::vstore(mid_row, v2f32_res);
            }
            else
            {
                DT_F32 *buf_tail = reinterpret_cast<DT_F32 *>(p_buf + width - 4);
                for (DT_S32 x = 0; x < half_w; x += 4)
                {
                    float32x4x2_t v2qf32_res;
                    float32x4x2_t v2qf32_head = neon::vload2q(buf_head);
                    float32x4x2_t v2qf32_tail = neon::vload2q(buf_tail);
                    float32x4_t   v2qf32_rev  = neon::vrev64(v2qf32_tail.val[0]);

                    v2qf32_res.val[0] = v2qf32_head.val[0];
                    v2qf32_res.val[1] = neon::vcombine(neon::vgethigh(v2qf32_rev), neon::vgetlow(v2qf32_rev));
                    neon::vstore(mid_row + 2 * x, v2qf32_res);

                    buf_head += 8;
                    buf_tail -= 8;
                }
            }
        }

        return Status::OK;
    };

    auto col_process_func = [&](DT_S32 start, DT_S32 end)->Status
    {
        std::complex<DT_F32> *p_buf = thread_buffer.GetThreadData<std::complex<DT_F32>>();

        if (!p_buf)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (DT_S32 x = start; x < end; ++x)
        {
            DT_U8  *mid_data = reinterpret_cast<DT_U8*>(mid.GetData());
            DT_F32 *mid_row  = reinterpret_cast<DT_F32*>(mid_data);

            p_buf[0].real(mid_row[x] * coef_col_x0);
            p_buf[0].imag(0.0f);
            mid_data += mid_pitch;

            for (DT_S32 y = 1; y < height; ++y)
            {
                mid_row  = reinterpret_cast<DT_F32*>(mid_data);

                DT_F32 cos_val = dct_col_exp_table[y].real();
                DT_F32 sin_val = dct_col_exp_table[y].imag();

                p_buf[y].real(mid_row[x] * coef_col * cos_val);
                p_buf[y].imag(mid_row[x] * coef_col * sin_val);
                mid_data += mid_pitch;
            }

            for (DT_S32 y = 0; y < height; ++y)
            {
                DT_S32 idx = idx_table[y];
                if (idx > y)
                {
                    Swap(p_buf[y], p_buf[idx]);
                }
            }

            ButterflyTransformNeon(p_buf, 2, height, DT_FALSE, dft_col_exp_table);

            DT_U8 *dst_data = reinterpret_cast<DT_U8 *>(dst.GetData());
            for (DT_S32 y = 0; y < half_h; ++y)
            {
                Tp *dst_even = reinterpret_cast<Tp *>(dst_data);
                Tp *dst_odd  = reinterpret_cast<Tp *>(dst_data + dst_row_pitch);

                dst_even[x]  = SaturateCast<Tp>(p_buf[y].real());
                dst_odd[x]   = SaturateCast<Tp>(p_buf[height - y - 1].real());
                dst_data    += 2 * dst_row_pitch;
            }
        }

        return Status::OK;
    };

    GetReverseIndex(idx_table, width);
    if (wp->ParallelFor(0, height, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IDct IDctRadix2NeonImpl parallel for row process failed.");
        return Status::ERROR;
    }

    GetReverseIndex(idx_table, height);
    if (wp->ParallelFor(0, width, col_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IDct IDctRadix2NeonImpl parallel for col process failed.");
        return Status::ERROR;
    }

    return Status::OK;
}

static Status IDctRadix2NeonHelper(Context *ctx, const Mat &src, Mat &mid, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();

    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDctRadix2NeonImpl<DT_U8>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::S8:
        {
            ret = IDctRadix2NeonImpl<DT_S8>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::U16:
        {
            ret = IDctRadix2NeonImpl<DT_U16>(ctx, src, mid, dst, target);
            break;
        }
        case ElemType::S16:
        {
            ret = IDctRadix2NeonImpl<DT_S16>(ctx, src, mid, dst, target);
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = IDctRadix2NeonImpl<MI_F16>(ctx, src, mid, dst, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = IDctRadix2NeonImpl<DT_F32>(ctx, src, dst, dst, target);
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

IDctNeon::IDctNeon(Context *ctx, const OpTarget &target) : IDctImpl(ctx, target)
{}

Status IDctNeon::SetArgs(const Array *src, Array *dst)
{
    // DctImpl::Initialize also verify shape and channel
    if (IDctImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DctImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) ||(dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType dst_type = dst->GetElemType();
    if ((ElemType::S32 == dst_type) || (ElemType::U32 == dst_type) || (ElemType::F64 == dst_type))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current dst does not support DT_S32/DT_U32/DT_F64 type.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    if (!IsPowOf2(src_sz.m_width) || !IsPowOf2(src_sz.m_height))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IDctNeon current only support 2^n size.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IDctNeon::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat *>(m_src);
    Mat   *dst     = dynamic_cast<Mat *>(m_dst);
    Mat   *mid     = &m_mid;
    if ((DT_NULL == src) ||(DT_NULL == dst) || (DT_NULL == mid))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "input src or dst or mid is null");
        return Status::ERROR;
    }

    ret = IDctRadix2NeonHelper(m_ctx, *src, *mid, *dst, m_target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IDctRadix2NeonHelper failed.");
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura