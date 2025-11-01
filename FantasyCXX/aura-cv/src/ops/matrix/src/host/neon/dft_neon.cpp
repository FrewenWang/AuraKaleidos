#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

// eg: VType = float32x2_t, V2Type = float32x2x2_t
template <typename VType, typename V2Type>
AURA_ALWAYS_INLINE AURA_VOID ButterflyBlockNeon(MI_F32 *x0_ptr, MI_F32 *x1_ptr, const MI_F32 *w_ptr)
{
    V2Type v2f32_x0, v2f32_x1, v2f32_w;

    neon::vload(x0_ptr, v2f32_x0);
    neon::vload(x1_ptr, v2f32_x1);
    neon::vload(w_ptr, v2f32_w);

    VType vf32_wrx1r = neon::vmul(v2f32_w.val[0], v2f32_x1.val[0]);
    VType vf32_wrx1i = neon::vmul(v2f32_w.val[0], v2f32_x1.val[1]);

    V2Type v2f32_wx1;
    v2f32_wx1.val[0] = neon::vmls(vf32_wrx1r, v2f32_w.val[1], v2f32_x1.val[1]);
    v2f32_wx1.val[1] = neon::vmla(vf32_wrx1i, v2f32_w.val[1], v2f32_x1.val[0]);

    V2Type v2f32_result0, v2f32_result1;
    v2f32_result0.val[0] = v2f32_x0.val[0] + v2f32_wx1.val[0];
    v2f32_result0.val[1] = v2f32_x0.val[1] + v2f32_wx1.val[1];
    v2f32_result1.val[0] = v2f32_x0.val[0] - v2f32_wx1.val[0];
    v2f32_result1.val[1] = v2f32_x0.val[1] - v2f32_wx1.val[1];

    neon::vstore(x0_ptr, v2f32_result0);
    neon::vstore(x1_ptr, v2f32_result1);
}

AURA_VOID ButterflyTransformNeon(std::complex<MI_F32> *src, MI_S32 start_level, MI_S32 n, MI_BOOL with_scale,
                               const std::complex<MI_F32> *dft_exp_table)
{
    if (start_level > n)
    {
        return;
    }

    if (n < 16)
    {
        ButterflyTransformNone(src, start_level, n, with_scale, dft_exp_table);
        return;
    }

    MI_S32 size = start_level;

    if (2 == size)
    {
        for (MI_S32 i = 0; i < n; i += 2)
        {
            MI_F32 *x0_ptr = reinterpret_cast<MI_F32*>(src + i);
            MI_F32 *x1_ptr = reinterpret_cast<MI_F32*>(src + i + 1);
            float32x2_t vdf32_x0 = neon::vload1(x0_ptr);
            float32x2_t vdf32_x1 = neon::vload1(x1_ptr);
            float32x2_t vdf32_v0 = neon::vadd(vdf32_x0, vdf32_x1);
            float32x2_t vdf32_v1 = neon::vsub(vdf32_x0, vdf32_x1);
            neon::vstore(x0_ptr, vdf32_v0);
            neon::vstore(x1_ptr, vdf32_v1);
        }
        size *= 2;
    }

    if (4 == size)
    {
        MI_S32 table_step = n / 4;
        MI_F32 w[4] = {dft_exp_table[0].real(), dft_exp_table[0].imag(),
                       dft_exp_table[table_step].real(), dft_exp_table[table_step].imag()};

        for (MI_S32 i = 0; i < n; i += 4) // i for group start index
        {
            MI_F32 *x0_ptr = reinterpret_cast<MI_F32*>(src + i);
            MI_F32 *x1_ptr = reinterpret_cast<MI_F32*>(src + i + 2);
            ButterflyBlockNeon<float32x2_t, float32x2x2_t>(x0_ptr, x1_ptr, w);
        }
        size *= 2;
    }

    std::complex<MI_F32> w[4] = {{0.0f, 0.0f}};
    for (; size < n; size *= 2)
    {
        MI_S32 half_size  = size / 2;
        MI_S32 table_step = n / size;

        for (MI_S32 i = 0; i < n; i += size) // i for group start index
        {
            for (MI_S32 j = 0; j < half_size; j += 4)
            {
                MI_F32 *x0_ptr = reinterpret_cast<MI_F32*>(src + i + j);
                MI_F32 *x1_ptr = reinterpret_cast<MI_F32*>(src + i + half_size + j);
                MI_F32 *w_ptr  = reinterpret_cast<MI_F32*>(w);
                MI_S32 exp_idx = j * table_step;

                for (MI_S32 k = 0; k < 4; ++k)
                {
                    w[k] = dft_exp_table[exp_idx + k * table_step];
                }

                ButterflyBlockNeon<float32x4_t, float32x4x2_t>(x0_ptr, x1_ptr, w_ptr);
            }
        }
    }

    // layer n
    {
        MI_S32 half_size  = n / 2;
        for (MI_S32 i = 0; i < half_size; i += 4)
        {
            MI_F32 *x0_ptr = reinterpret_cast<MI_F32*>(src + i);
            MI_F32 *x1_ptr = reinterpret_cast<MI_F32*>(src + half_size + i);
            const MI_F32 *w_ptr = reinterpret_cast<const MI_F32*>(&dft_exp_table[i]);

            ButterflyBlockNeon<float32x4_t, float32x4x2_t>(x0_ptr, x1_ptr, w_ptr);
        }
    }

    if (with_scale)
    {
        for (MI_S32 i = 0; i < n; ++i)
        {
            src[i] /= n;
        }
    }
}

static AURA_VOID FFTRadix2Neon(std::complex<MI_F32> *src, MI_S32 n,
                             const std::complex<MI_F32> *exp_table,
                             const MI_U16 *idx_table, MI_BOOL with_scale)
{
    if (!IsPowOf2(n))
    {
        return;
    }

    for (MI_S32 i = 0; i < n; ++i)
    {
        MI_S32 idx = idx_table[i];
        if (idx > i)
        {
            Swap(src[i], src[idx]);
        }
    }

    ButterflyTransformNeon(src, 2, n, with_scale, exp_table);
}

template <typename Tp, MI_U8 IS_INVERSE>
Status DftRadix2RowNeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    static_assert(0 == IS_INVERSE, "this branch is not for inverse dft");
    AURA_UNUSED(with_scale);
    AURA_UNUSED(target);
    Sizes3 sz = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;
    MI_S32 half_w = width / 2;

    if (1 == width)
    {
        DftRadix2RowProcCol1None<Tp, 0>(src, dst);
        return Status::OK;
    }
    if (2 == width)
    {
        DftRadix2RowProcCol2None<Tp, 0>(src, dst);
        return Status::OK;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    MI_S32 buffer_sz = half_w * sizeof(std::complex<MI_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    MI_S32 table_sz = half_w * sizeof(MI_U16) + ((half_w + width) / 2) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *dft_row_exp_table  = reinterpret_cast<std::complex<MI_F32>*>(idx_table + half_w);
    std::complex<MI_F32> *row_real_exp_table = dft_row_exp_table + half_w / 2;

    GetReverseIndex(idx_table, half_w);
    GetDftExpTable<0>(dft_row_exp_table, half_w);
    GetDftExpTable<0>(row_real_exp_table, width);

    // Row process use real values fft, ref: http://dsp-book.narod.ru/FFTBB/0270_PDF_C14.pdf
    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {

        std::complex<MI_F32> *buffer_ptr = thread_buffer.GetThreadData<std::complex<MI_F32>>();

        if (!buffer_ptr)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (MI_S32 y = start; y < end; ++y)
        {
            const Tp *src_row = src.Ptr<Tp>(y);
            std::complex<MI_F32> *dst_complex = dst.Ptr<std::complex<MI_F32>>(y);

            for (MI_S32 x = 0; x < half_w; x++)
            {
                MI_S32 idx = idx_table[x];
                buffer_ptr[x].real(SaturateCast<MI_F32>(src_row[2 * idx]));
                buffer_ptr[x].imag(SaturateCast<MI_F32>(src_row[2 * idx + 1]));
            }

            ButterflyTransformNeon(buffer_ptr, 2, half_w, MI_FALSE, dft_row_exp_table);

            for (MI_S32 x = 1; x < half_w; x++)
            {
                std::complex<MI_F32> yk = buffer_ptr[x];
                std::complex<MI_F32> yk_conj = std::conj(buffer_ptr[half_w - x]);

                std::complex<MI_F32> fk = std::complex<MI_F32>(0.5f, 0.0f) * (yk + yk_conj);
                std::complex<MI_F32> gk = std::complex<MI_F32>(0.0f, 0.5f) * (yk_conj - yk);

                std::complex<MI_F32> result = fk + row_real_exp_table[x] * gk;
                dst_complex[x] = result;
                dst_complex[width - x] = std::conj(result);
            }
            {
                std::complex<MI_F32> y0 = buffer_ptr[0];
                std::complex<MI_F32> y0_conj = std::conj(y0);
                std::complex<MI_F32> f0 = std::complex<MI_F32>(0.5f, 0.0f) * (y0 + y0_conj);
                std::complex<MI_F32> g0 = std::complex<MI_F32>(0.0f, 0.5f) * (y0_conj - y0);
                dst_complex[0] = f0 + g0;
                dst_complex[half_w] = f0 - g0;
            }
            // clear
            dst_complex[0].imag(0.0f);
            dst_complex[width / 2].imag(0.0f);
        }

        return Status::OK;
    };

    if (wp->ParallelFor(0, height, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "DftRadix2RowNeonImpl ParallelFor failed.");
        return Status::ERROR;
    }

    AURA_FREE(ctx, temp_buffer);
    return Status::OK;
}

// dft inverse row process method
template <>
Status DftRadix2RowNeonImpl<MI_F32, 1>(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    Sizes3 sz = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    if (1 == width)
    {
        DftRadix2RowProcCol1None<MI_F32, 1>(src, dst);
        return Status::OK;
    }
    if (2 == width)
    {
        DftRadix2RowProcCol2None<MI_F32, 1>(src, dst);
        return Status::OK;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    MI_S32 table_sz = width * sizeof(MI_U16) + width / 2 * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *dft_row_exp_table = reinterpret_cast<std::complex<MI_F32>*>(idx_table + width);

    GetDftExpTable<1>(dft_row_exp_table, width);
    GetReverseIndex(idx_table, width);

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        for (MI_S32 y = start; y < end; ++y)
        {
            const std::complex<MI_F32> *src_complex = src.Ptr<std::complex<MI_F32>>(y);
            std::complex<MI_F32> *dst_complex = dst.Ptr<std::complex<MI_F32>>(y);

            for (MI_S32 x = 0; x < width; x += 2)
            {
                MI_U16 idx0 = idx_table[x];
                MI_U16 idx1 = idx_table[x + 1];
                dst_complex[x]     = src_complex[idx0] + src_complex[idx1];
                dst_complex[x + 1] = src_complex[idx0] - src_complex[idx1];
            }

            ButterflyTransformNeon(dst_complex, 4, width, with_scale, dft_row_exp_table);
        }

        return Status::OK;
    };

    Status ret = wp->ParallelFor(0, height, row_process_func);

    AURA_FREE(ctx, temp_buffer);
    AURA_RETURN(ctx, ret);
}

static Status DftRadix2RowNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::OK;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = DftRadix2RowNeonImpl<MI_U8, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::S8:
        {
            ret = DftRadix2RowNeonImpl<MI_S8, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::U16:
        {
            ret = DftRadix2RowNeonImpl<MI_U16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::S16:
        {
            ret = DftRadix2RowNeonImpl<MI_S16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = DftRadix2RowNeonImpl<MI_F16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = DftRadix2RowNeonImpl<MI_F32, 0>(ctx, src, dst, MI_FALSE, target);
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

template <typename Tp, MI_BOOL IS_DST_C1>
AURA_INLINE AURA_VOID DftRadix2Col8Neon(Mat &src, Mat &dst, MI_U16 *idx_table, std::complex<MI_F32> *exp_table,
                                      std::complex<MI_F32> *buffer_data, MI_BOOL with_scale, MI_S32 x)
{
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 row_pitch = src.GetRowPitch();

    struct ComplexPacket
    {
        std::complex<MI_F32> data[8];
    };

    // Init buffer
    std::complex<MI_F32> *buffer_row0 = buffer_data + 0 * height;
    std::complex<MI_F32> *buffer_row1 = buffer_data + 1 * height;
    std::complex<MI_F32> *buffer_row2 = buffer_data + 2 * height;
    std::complex<MI_F32> *buffer_row3 = buffer_data + 3 * height;
    std::complex<MI_F32> *buffer_row4 = buffer_data + 4 * height;
    std::complex<MI_F32> *buffer_row5 = buffer_data + 5 * height;
    std::complex<MI_F32> *buffer_row6 = buffer_data + 6 * height;
    std::complex<MI_F32> *buffer_row7 = buffer_data + 7 * height;

    // move data into temp buffer
    MI_U8 *src_row = reinterpret_cast<MI_U8*>(src.GetData()) + x * sizeof(std::complex<MI_F32>);
    for (MI_S32 i = 0; i < height; ++i)
    {
        ComplexPacket *packet = reinterpret_cast<ComplexPacket*>(src_row);

        MI_S32 idx = idx_table[i];
        buffer_row0[idx] = packet->data[0];
        buffer_row1[idx] = packet->data[1];
        buffer_row2[idx] = packet->data[2];
        buffer_row3[idx] = packet->data[3];
        buffer_row4[idx] = packet->data[4];
        buffer_row5[idx] = packet->data[5];
        buffer_row6[idx] = packet->data[6];
        buffer_row7[idx] = packet->data[7];

        src_row += row_pitch;
    }

    // DFT process
    ButterflyTransformNeon(buffer_row0, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row1, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row2, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row3, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row4, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row5, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row6, 2, height, with_scale, exp_table);
    ButterflyTransformNeon(buffer_row7, 2, height, with_scale, exp_table);

    // Store data
    if (IS_DST_C1)
    {
        for (MI_S32 i = 0; i < height; ++i)
        {
            Tp *dst_row = dst.Ptr<Tp>(i) + x;

            dst_row[0] = SaturateCast<Tp>(buffer_row0[i].real());
            dst_row[1] = SaturateCast<Tp>(buffer_row1[i].real());
            dst_row[2] = SaturateCast<Tp>(buffer_row2[i].real());
            dst_row[3] = SaturateCast<Tp>(buffer_row3[i].real());
            dst_row[4] = SaturateCast<Tp>(buffer_row4[i].real());
            dst_row[5] = SaturateCast<Tp>(buffer_row5[i].real());
            dst_row[6] = SaturateCast<Tp>(buffer_row6[i].real());
            dst_row[7] = SaturateCast<Tp>(buffer_row7[i].real());
        }
    }
    else
    {
        for (MI_S32 i = 0; i < height; ++i)
        {
            std::complex<Tp> *dst_row = dst.Ptr<std::complex<Tp>>(i) + x;

            dst_row[0].real(SaturateCast<Tp>(buffer_row0[i].real()));
            dst_row[1].real(SaturateCast<Tp>(buffer_row1[i].real()));
            dst_row[2].real(SaturateCast<Tp>(buffer_row2[i].real()));
            dst_row[3].real(SaturateCast<Tp>(buffer_row3[i].real()));
            dst_row[4].real(SaturateCast<Tp>(buffer_row4[i].real()));
            dst_row[5].real(SaturateCast<Tp>(buffer_row5[i].real()));
            dst_row[6].real(SaturateCast<Tp>(buffer_row6[i].real()));
            dst_row[7].real(SaturateCast<Tp>(buffer_row7[i].real()));

            dst_row[0].imag(SaturateCast<Tp>(buffer_row0[i].imag()));
            dst_row[1].imag(SaturateCast<Tp>(buffer_row1[i].imag()));
            dst_row[2].imag(SaturateCast<Tp>(buffer_row2[i].imag()));
            dst_row[3].imag(SaturateCast<Tp>(buffer_row3[i].imag()));
            dst_row[4].imag(SaturateCast<Tp>(buffer_row4[i].imag()));
            dst_row[5].imag(SaturateCast<Tp>(buffer_row5[i].imag()));
            dst_row[6].imag(SaturateCast<Tp>(buffer_row6[i].imag()));
            dst_row[7].imag(SaturateCast<Tp>(buffer_row7[i].imag()));
        }
    }
}

template <typename Tp, MI_BOOL IS_DST_C1>
AURA_INLINE AURA_VOID DftRadix2Col1Neon(Mat &src, Mat &dst, MI_U16 *idx_table, std::complex<MI_F32> *exp_table,
                                      std::complex<MI_F32> *buffer_data, MI_BOOL with_scale, MI_S32 x)
{
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 row_pitch = src.GetRowPitch();

    MI_U8 *src_row = reinterpret_cast<MI_U8*>(src.GetData());
    for (MI_S32 i = 0; i < height; ++i)
    {
        std::complex<MI_F32> *dst_complex = reinterpret_cast<std::complex<MI_F32>*>(src_row);
        MI_S32 idx  = idx_table[i];
        buffer_data[idx] = dst_complex[x];
        src_row += row_pitch;
    }

    ButterflyTransformNeon(buffer_data, 2, height, with_scale, exp_table);

    if (IS_DST_C1)
    {
        for (MI_S32 i = 0; i < height; ++i)
        {
            Tp *dst_row = dst.Ptr<Tp>(i);
            dst_row[x] = SaturateCast<Tp>(buffer_data[i].real());
        }
    }
    else
    {
        for (MI_S32 i = 0; i < height; ++i)
        {
            std::complex<Tp> *dst_row = dst.Ptr<std::complex<Tp>>(i);
            dst_row[x].real(SaturateCast<Tp>(buffer_data[i].real()));
            dst_row[x].imag(SaturateCast<Tp>(buffer_data[i].imag()));
        }
    }
}

template <typename Tp, MI_BOOL IS_DST_C1>
Status DftRadix2ColCommNeon(Context *ctx, Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_U16 *idx_table,
                            std::complex<MI_F32> *exp_table, MI_BOOL with_scale, MI_S32 start, MI_S32 end)
{
    std::complex<MI_F32> *buffer_ptr = thread_buffer.GetThreadData<std::complex<MI_F32>>();

    if (!buffer_ptr)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    for (; start + 8 <= end; start += 8)
    {
        DftRadix2Col8Neon<Tp, IS_DST_C1>(src, dst, idx_table, exp_table, buffer_ptr, with_scale, start);
    }

    for (; start < end; start++)
    {
        DftRadix2Col1Neon<Tp, IS_DST_C1>(src, dst, idx_table, exp_table, buffer_ptr, with_scale, start);
    }

    return Status::OK;
}

template <typename Tp, MI_U8 IS_INVERSE, MI_BOOL IS_DST_C1>
static AURA_VOID DftRadix2ColProcRow2Neon(Mat &src, Mat &dst)
{
    std::complex<MI_F32> *src_c  = src.Ptr<std::complex<MI_F32>>(0);
    std::complex<MI_F32> *src_n0 = src.Ptr<std::complex<MI_F32>>(1);
    Tp                   *dst_c  = dst.Ptr<Tp>(0);
    Tp                   *dst_n0 = dst.Ptr<Tp>(1);

    MI_S32 width        = src.GetSizes().m_width;
    MI_S32 width_align4 = width & (-4);
    MI_S32 width_align8 = width & (-8);
    const  MI_F32 coef  = IS_INVERSE ? 0.5f : 1.0f;

    if (IS_DST_C1)
    {
        MI_S32 x = 0;
        for (; x < width_align8; x += 8)
        {
            float32x4x2_t v2qf32_c_l  = neon::vload2q((MI_F32 *)src_c + x * 2);
            float32x4x2_t v2qf32_c_r  = neon::vload2q((MI_F32 *)src_c + x * 2 + 8);
            float32x4x2_t v2qf32_n0_l = neon::vload2q((MI_F32 *)src_n0 + x * 2);
            float32x4x2_t v2qf32_n0_r = neon::vload2q((MI_F32 *)src_n0 + x * 2 + 8);

            float32x4_t vqf32_real_c_l  = neon::vadd(v2qf32_c_l.val[0], v2qf32_n0_l.val[0]);
            float32x4_t vqf32_real_c_r  = neon::vadd(v2qf32_c_r.val[0], v2qf32_n0_r.val[0]);
            float32x4_t vqf32_real_n0_l = neon::vsub(v2qf32_c_l.val[0], v2qf32_n0_l.val[0]);
            float32x4_t vqf32_real_n0_r = neon::vsub(v2qf32_c_r.val[0], v2qf32_n0_r.val[0]);

            float32x4x2_t v2qf32_real_c, v2qf32_real_n0;
            v2qf32_real_c.val[0]  = neon::vmul(vqf32_real_c_l, coef);
            v2qf32_real_c.val[1]  = neon::vmul(vqf32_real_c_r, coef);
            v2qf32_real_n0.val[0] = neon::vmul(vqf32_real_n0_l, coef);
            v2qf32_real_n0.val[1] = neon::vmul(vqf32_real_n0_r, coef);

            StoreF32PairAs<Tp>((dst_c + x), v2qf32_real_c);
            StoreF32PairAs<Tp>((dst_n0 + x), v2qf32_real_n0);
        }

        for (; x < width; x++)
        {
            std::complex<MI_F32> val_c  = src_c[x];
            std::complex<MI_F32> val_n0 = src_n0[x];

            dst_c[x]  = SaturateCast<Tp>(((val_c + val_n0) * coef).real());
            dst_n0[x] = SaturateCast<Tp>(((val_c - val_n0) * coef).real());
        }
    }
    else
    {
        MI_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4_t vqf32_row_c_l  = neon::vload1q((MI_F32 *)src_c + x * 2);
            float32x4_t vqf32_row_c_r  = neon::vload1q((MI_F32 *)src_c + x * 2 + 4);
            float32x4_t vqf32_row_n0_l = neon::vload1q((MI_F32 *)src_n0 + x * 2);
            float32x4_t vqf32_row_n0_r = neon::vload1q((MI_F32 *)src_n0 + x * 2 + 4);

            float32x4_t vqf32_result_c_l  = neon::vadd(vqf32_row_c_l, vqf32_row_n0_l);
            float32x4_t vqf32_result_c_r  = neon::vadd(vqf32_row_c_r, vqf32_row_n0_r);
            float32x4_t vqf32_result_n0_l = neon::vsub(vqf32_row_c_l, vqf32_row_n0_l);
            float32x4_t vqf32_result_n0_r = neon::vsub(vqf32_row_c_r, vqf32_row_n0_r);

            float32x4x2_t v2qf32_result_c, v2qf32_result_n0;
            v2qf32_result_c.val[0]  = neon::vmul(vqf32_result_c_l, coef);
            v2qf32_result_c.val[1]  = neon::vmul(vqf32_result_c_r, coef);
            v2qf32_result_n0.val[0] = neon::vmul(vqf32_result_n0_l, coef);
            v2qf32_result_n0.val[1] = neon::vmul(vqf32_result_n0_r, coef);

            StoreF32PairAs<Tp>((dst_c + x * 2), v2qf32_result_c);
            StoreF32PairAs<Tp>((dst_n0 + x * 2), v2qf32_result_n0);
        }

        for (; x < width; x++)
        {
            std::complex<MI_F32> val_c  = src_c[x];
            std::complex<MI_F32> val_n0 = src_n0[x];

            std::complex<MI_F32> result_c  = (val_c + val_n0) * coef;
            std::complex<MI_F32> result_n0 = (val_c - val_n0) * coef;

            dst_c[x * 2]      = SaturateCast<Tp>(result_c.real());
            dst_c[x * 2 + 1]  = SaturateCast<Tp>(result_c.imag());
            dst_n0[x * 2]     = SaturateCast<Tp>(result_n0.real());
            dst_n0[x * 2 + 1] = SaturateCast<Tp>(result_n0.imag());
        }
    }
}

template <typename Tp, MI_U8 IS_INVERSE, MI_BOOL IS_DST_C1>
static Status DftRadix2ColNeonImpl(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale)
{
    Sizes3 sz = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    if (1 == height)
    {
        if (IS_DST_C1)
        {
            std::complex<MI_F32> *src_c = src.Ptr<std::complex<MI_F32>>(0);
            Tp                   *dst_c = dst.Ptr<Tp>(0);
            for (MI_S32 x = 0; x < width; x++)
            {
                dst_c[x] = SaturateCast<Tp>(src_c[x].real());
            }
        }
        else
        {
            std::complex<MI_F32> *src_c = src.Ptr<std::complex<MI_F32>>(0);
            std::complex<Tp>     *dst_c = dst.Ptr<std::complex<Tp>>(0);
            for (MI_S32 x = 0; x < width; x++)
            {
                dst_c[x].real(SaturateCast<Tp>(src_c[x].real()));
                dst_c[x].imag(SaturateCast<Tp>(src_c[x].imag()));
            }
        }

        return Status::OK;
    }

    if (2 == height)
    {
        DftRadix2ColProcRow2Neon<Tp, IS_INVERSE, IS_DST_C1>(src, dst);
        return Status::OK;
    }

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    MI_S32 w_eff = (width >= 8) ? 8 : 1;
    MI_S32 buffer_sz = height * w_eff * sizeof(std::complex<MI_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    MI_S32 table_sz = height * sizeof(MI_U16) + height / 2 * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *exp_table = reinterpret_cast<std::complex<MI_F32>*>(idx_table + height);

    GetReverseIndex(idx_table, height);
    GetDftExpTable<IS_INVERSE>(exp_table, height);

    Status ret = wp->ParallelFor(0, width, DftRadix2ColCommNeon<Tp, IS_DST_C1>, ctx, std::ref(src), std::ref(dst), std::ref(thread_buffer),
                                 idx_table, exp_table, with_scale);

    if (!IS_INVERSE)
    {
        // clear special coordinate zero
        if (0 == width % 2)
        {
            if (0 == height % 2)
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(0, width / 2, 1) = 0.0f;
                src.At<MI_F32>(height / 2, 0, 1) = 0.0f;
                src.At<MI_F32>(height / 2, width / 2, 1) = 0.0f;
            }
            else
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(0, width / 2, 1) = 0.0f;
            }
        }
        else
        {
            if (0 == height % 2)
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(height / 2, 0, 1) = 0.0f;
            }
            else
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
            }
        }
    }

    AURA_FREE(ctx, temp_buffer);
    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status IDftRadix2ColNeonHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Status ret = Status::ERROR;

    MI_BOOL is_dst_c1 = (1 == dst.GetSizes().m_channel);
    if (is_dst_c1)
    {
        ret = DftRadix2ColNeonImpl<Tp, 1, MI_TRUE>(ctx, src, dst, with_scale);
    }
    else
    {
        ret = DftRadix2ColNeonImpl<Tp, 1, MI_FALSE>(ctx, src, dst, with_scale);
    }

    AURA_RETURN(ctx, ret);
}

static Status IDftRadix2ColNeonHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDftRadix2ColNeonHelper<MI_U8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S8:
        {
            ret = IDftRadix2ColNeonHelper<MI_S8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U16:
        {
            ret = IDftRadix2ColNeonHelper<MI_U16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S16:
        {
            ret = IDftRadix2ColNeonHelper<MI_S16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U32:
        {
            ret = IDftRadix2ColNeonHelper<MI_U32>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S32:
        {
            ret = IDftRadix2ColNeonHelper<MI_S32>(ctx, src, dst, with_scale);
            break;
        }
# if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = IDftRadix2ColNeonHelper<MI_F16>(ctx, src, dst, with_scale);
            break;
        }
# endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = IDftRadix2ColNeonHelper<MI_F32>(ctx, src, dst, with_scale);
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

template <typename Tp, MI_U8 IS_INVERSE>
static Status DftBluesteinRowNeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    MI_S32 width_padding = 1;
    while (width_padding / 2 <= width)
    {
        width_padding *= 2;
    }

    MI_S32 buffer_sz = width_padding * sizeof(std::complex<MI_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    MI_S32 table_sz = width_padding * sizeof(MI_U16) + (2 * width_padding + width) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table                    = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *b_hor          = reinterpret_cast<std::complex<MI_F32>*>(idx_table + width_padding);
    std::complex<MI_F32> *table_hor      = b_hor + width_padding;
    std::complex<MI_F32> *dft_exp_table  = table_hor + width;
    std::complex<MI_F32> *idft_exp_table = dft_exp_table + width_padding / 2;

    GetReverseIndex(idx_table, width_padding);
    GetBlueSteinExpTable<IS_INVERSE>(table_hor, width);
    GetDftExpTable<0>(dft_exp_table,  width_padding);
    GetDftExpTable<1>(idft_exp_table, width_padding);

    // Get serial b
    b_hor[0] = std::conj(table_hor[0]);
    for (MI_S32 x = 1; x < width; ++x)
    {
        b_hor[x] = b_hor[width_padding - x] = std::conj(table_hor[x]);
    }

    FFTRadix2Neon(b_hor, width_padding, dft_exp_table, idx_table, MI_FALSE);

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        std::complex<MI_F32> *a_hor = thread_buffer.GetThreadData<std::complex<MI_F32>>();

        if (!a_hor)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (MI_S32 y = start; y < end; y++)
        {
            // init a_hor
            if (IS_INVERSE)
            {
                // data of channel 2
                const Tp *src_row = src.Ptr<Tp>(y);
                for (MI_S32 x = 0; x < width; x++)
                {
                    a_hor[x].real(SaturateCast<MI_F32>(src_row[2 * x + 0]));
                    a_hor[x].imag(SaturateCast<MI_F32>(src_row[2 * x + 1]));
                    a_hor[x] *= table_hor[x];
                }
            }
            else
            {
                // data of channel 1
                const Tp *src_row = src.Ptr<Tp>(y);
                for (MI_S32 x = 0; x < width; x++)
                {
                    a_hor[x].real(SaturateCast<MI_F32>(src_row[x]));
                    a_hor[x].imag(0.0f);
                    a_hor[x] *= table_hor[x];
                }
            }

            for (MI_S32 x = width; x < width_padding; ++x)
            {
                a_hor[x] = std::complex<MI_F32>(0.0f, 0.0f);
            }

            FFTRadix2Neon(a_hor, width_padding, dft_exp_table, idx_table, MI_FALSE);
            for (MI_S32 x = 0; x < width_padding; ++x)
            {
                a_hor[x] *= b_hor[x];
            }
            FFTRadix2Neon(a_hor, width_padding, idft_exp_table, idx_table, MI_TRUE);

            std::complex<MI_F32> *dst_row = dst.Ptr<std::complex<MI_F32>>(y);
            MI_F32 norm_coef = with_scale ? (1.0f / width) : 1.0f;
            for (MI_S32 x = 0; x < width; x++)
            {
                dst_row[x] = a_hor[x] * table_hor[x] * norm_coef;
            }

            if (!IS_INVERSE)
            {
                if (0 == width % 2)
                {
                    dst_row[0].imag(0.0f);
                    dst_row[width / 2].imag(0.0f);
                }
                else
                {
                    dst_row[0].imag(0.0f);
                }
            }
        }

        return Status::OK;
    };

    Status ret = wp->ParallelFor(0, height, row_process_func);

    AURA_FREE(ctx, temp_buffer);
    AURA_RETURN(ctx, ret);
}

static Status DftBluesteinRowNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::OK;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = DftBluesteinRowNeonImpl<MI_U8, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::S8:
        {
            ret = DftBluesteinRowNeonImpl<MI_S8, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::U16:
        {
            ret = DftBluesteinRowNeonImpl<MI_U16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
        case ElemType::S16:
        {
            ret = DftBluesteinRowNeonImpl<MI_S16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = DftBluesteinRowNeonImpl<MI_F16, 0>(ctx, src, dst, MI_FALSE, target);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = DftBluesteinRowNeonImpl<MI_F32, 0>(ctx, src, dst, MI_FALSE, target);
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

template <typename Tp, MI_BOOL IS_DST_C1>
AURA_INLINE AURA_VOID FftConvolveColNeon(Mat &src, Mat &dst, MI_S32 h_padding, MI_U16 *idx_table,
                                       std::complex<MI_F32> *a_ver, std::complex<MI_F32> *b_ver,
                                       std::complex<MI_F32> *exp_table, std::complex<MI_F32> *dft_exp_table,
                                       std::complex<MI_F32> *idft_exp_table, MI_BOOL with_scale, MI_S32 x)
{
    MI_S32 height = src.GetSizes().m_height;

    for (MI_S32 y = 0; y < height; y++)
    {
        std::complex<MI_F32> *cur_data = src.Ptr<std::complex<MI_F32>>(y);
        a_ver[y] = cur_data[x] * exp_table[y];
    }

    for (MI_S32 y = height; y < h_padding; y++)
    {
        a_ver[y] = std::complex<MI_F32>(0.0f, 0.0f);
    }

    FFTRadix2Neon(a_ver, h_padding, dft_exp_table, idx_table, MI_FALSE);
    for (MI_S32 i = 0; i < h_padding; ++i)
    {
        a_ver[i] *= b_ver[i];
    }
    FFTRadix2Neon(a_ver, h_padding, idft_exp_table, idx_table, MI_TRUE);

    MI_F32 norm_coef = with_scale ? (1.0f / height) : 1.0f;

    if (IS_DST_C1)
    {
        for (MI_S32 y = 0; y < height; y++)
        {
            Tp *dst_row = dst.Ptr<Tp>(y);
            dst_row[x] = SaturateCast<Tp>((a_ver[y] * exp_table[y] * norm_coef).real());
        }
    }
    else
    {
        for (MI_S32 y = 0; y < height; y++)
        {
            std::complex<Tp>     *dst_row = dst.Ptr<std::complex<Tp>>(y);
            std::complex<MI_F32> result   = a_ver[y] * exp_table[y] * norm_coef;

            dst_row[x].real(SaturateCast<Tp>(result.real()));
            dst_row[x].imag(SaturateCast<Tp>(result.imag()));
        }
    }
}

template <typename Tp, MI_U8 IS_INVERSE, MI_BOOL IS_DST_C1>
static Status DftBluesteinColNeonImpl(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    MI_S32 height_padding = 1;
    while (height_padding / 2 <= height)
    {
        height_padding *= 2;
    }

    MI_S32 buffer_sz = height_padding * sizeof(std::complex<MI_F32>);
    ThreadBuffer thread_buffer(ctx, buffer_sz);

    MI_S32 table_sz = height_padding * sizeof(MI_U16) + (2 * height_padding + height) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table                    = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *b_ver          = reinterpret_cast<std::complex<MI_F32>*>(idx_table + height_padding);
    std::complex<MI_F32> *exp_table      = b_ver + height_padding;
    std::complex<MI_F32> *dft_exp_table  = exp_table + height;
    std::complex<MI_F32> *idft_exp_table = dft_exp_table + height_padding / 2;

    GetReverseIndex(idx_table, height_padding);
    GetBlueSteinExpTable<IS_INVERSE>(exp_table, height);
    GetDftExpTable<0>(dft_exp_table,  height_padding);
    GetDftExpTable<1>(idft_exp_table, height_padding);

    // Get serial b
    b_ver[0] = exp_table[0];
    for (MI_S32 i = 1; i < height; ++i)
    {
        b_ver[i] = b_ver[height_padding - i] = std::conj(exp_table[i]);
    }
    FFTRadix2Neon(b_ver, height_padding, dft_exp_table, idx_table, MI_FALSE);

    auto col_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        std::complex<MI_F32> *a_ver = thread_buffer.GetThreadData<std::complex<MI_F32>>();

        if (!a_ver)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        for (MI_S32 x = start; x < end; x++)
        {
            FftConvolveColNeon<Tp, IS_DST_C1>(src, dst, height_padding, idx_table, a_ver, b_ver, exp_table,
                                              dft_exp_table, idft_exp_table, with_scale, x);
        }

        return Status::OK;
    };

    Status ret = wp->ParallelFor(0, width, col_process_func);

    if (!IS_INVERSE)
    {
        // clear specific point
        if (0 == width % 2)
        {
            if (0 == height % 2)
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(0, width / 2, 1) = 0.0f;
                src.At<MI_F32>(height / 2, 0, 1) = 0.0f;
                src.At<MI_F32>(height / 2, width / 2, 1) = 0.0f;
            }
            else
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(0, width / 2, 1) = 0.0f;
            }
        }
        else
        {
            if (0 == height % 2)
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
                src.At<MI_F32>(height / 2, 0, 1) = 0.0f;
            }
            else
            {
                src.At<MI_F32>(0, 0, 1) = 0.0f;
            }
        }
    }

    AURA_FREE(ctx, temp_buffer);
    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status IDftBluesteinColNeonHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    Status ret = Status::ERROR;

    MI_BOOL is_dst_c1 = (1 == dst.GetSizes().m_channel);
    if (is_dst_c1)
    {
        ret = DftBluesteinColNeonImpl<Tp, 1, MI_TRUE>(ctx, src, dst, with_scale, target);
    }
    else
    {
        ret = DftBluesteinColNeonImpl<Tp, 1, MI_FALSE>(ctx, src, dst, with_scale, target);
    }

    AURA_RETURN(ctx, ret);
}

static Status IDftBluesteinColNeonHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDftBluesteinColNeonHelper<MI_U8>(ctx, src, dst, with_scale, target);
            break;
        }
        case ElemType::S8:
        {
            ret = IDftBluesteinColNeonHelper<MI_S8>(ctx, src, dst, with_scale, target);
            break;
        }
        case ElemType::U16:
        {
            ret = IDftBluesteinColNeonHelper<MI_U16>(ctx, src, dst, with_scale, target);
            break;
        }
        case ElemType::S16:
        {
            ret = IDftBluesteinColNeonHelper<MI_S16>(ctx, src, dst, with_scale, target);
            break;
        }
        case ElemType::U32:
        {
            ret = IDftBluesteinColNeonHelper<MI_U32>(ctx, src, dst, with_scale, target);
            break;
        }
        case ElemType::S32:
        {
            ret = IDftBluesteinColNeonHelper<MI_S32>(ctx, src, dst, with_scale, target);
            break;
        }
# if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = IDftBluesteinColNeonHelper<MI_F16>(ctx, src, dst, with_scale, target);
            break;
        }
# endif // AURA_ENABLE_NEON_FP16
        case ElemType::F32:
        {
            ret = IDftBluesteinColNeonHelper<MI_F32>(ctx, src, dst, with_scale, target);
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

DftNeon::DftNeon(Context *ctx, const OpTarget &target) : DftImpl(ctx, target)
{}

Status DftNeon::SetArgs(const Array *src, Array *dst)
{
    if (DftImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if ((ElemType::S32 == src->GetElemType()) || (ElemType::U32 == src->GetElemType()) || (ElemType::F64 == src->GetElemType()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support MI_S32/MI_U32/MI_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DftNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();
    MI_S32 width = sz.m_width;
    MI_S32 height = sz.m_height;
    Status ret = Status::ERROR;

    if (IsPowOf2(width))
    {
        ret = DftRadix2RowNeon(m_ctx, *src, *dst, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2RowNeon failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }
    else
    {
        ret = DftBluesteinRowNeon(m_ctx, *src, *dst, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinRowNeon failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }

    if (IsPowOf2(height))
    {
        ret = DftRadix2ColNeonImpl<MI_F32, 0, MI_FALSE>(m_ctx, *dst, *dst, MI_FALSE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2ColNeonImpl failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }
    else
    {
        ret = DftBluesteinColNeonImpl<MI_F32, 0, MI_FALSE>(m_ctx, *dst, *dst, MI_FALSE, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinColNeonImpl failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }

    AURA_RETURN(m_ctx, ret);
}

InverseDftNeon::InverseDftNeon(Context *ctx, const OpTarget &target) : InverseDftImpl(ctx, target)
{}

Status InverseDftNeon::SetArgs(const Array *src, Array *dst, MI_BOOL with_scale)
{
    if (InverseDftImpl::SetArgs(src, dst, with_scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InverseDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status InverseDftNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Sizes3 src_size = src->GetSizes();
    Sizes3 dst_size = dst->GetSizes();

    Mat *mid = MI_NULL;
    if ((1 == dst_size.m_channel) || (ElemType::F32 != dst->GetElemType()))
    {
        mid = &m_mid;
    }
    else
    {
        mid = dst;
    }

    if (IsPowOf2(src_size.m_width))
    {
        if (DftRadix2RowNeonImpl<MI_F32, 1>(m_ctx, *src, *mid, m_with_scale, m_target) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2RowNeonImpl failed.");
            return Status::ERROR;
        }
    }
    else
    {
        if (DftBluesteinRowNeonImpl<MI_F32, 1>(m_ctx, *src, *mid, m_with_scale, m_target) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinRowNeonImpl failed.");
            return Status::ERROR;
        }
    }

    if (IsPowOf2(src_size.m_height))
    {
        if (IDftRadix2ColNeonHelper(m_ctx, *mid, *dst, m_with_scale) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDftRadix2ColNeonHelper failed");
            return Status::ERROR;
        }
    }
    else
    {
        if (IDftBluesteinColNeonHelper(m_ctx, *mid, *dst, m_with_scale, m_target) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDftBluesteinColNeonHelper failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

} // aura