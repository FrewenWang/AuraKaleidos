#include "dft_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

AURA_INLINE AURA_VOID FFTRadix2None(std::complex<MI_F32> *src, MI_S32 n,
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

    ButterflyTransformNone(src, 2, n, with_scale, exp_table);
}

template <typename Tp, MI_U8 IS_INVERSE>
static Status DftRadix2RowNoneImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale)
{
    static_assert(0 == IS_INVERSE, "this branch is not for inverse dft");
    AURA_UNUSED(with_scale);

    Sizes3 sz     = src.GetSizes();
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

    MI_S32 table_sz = half_w * sizeof(MI_U16) + (half_w / 2 + 2 * half_w) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *dft_row_exp_table  = reinterpret_cast<std::complex<MI_F32>*>(idx_table + half_w);
    std::complex<MI_F32> *row_real_exp_table = dft_row_exp_table  + half_w / 2;
    std::complex<MI_F32> *buffer             = row_real_exp_table + half_w;

    GetReverseIndex(idx_table, half_w);
    GetDftExpTable<0>(dft_row_exp_table, half_w);
    GetDftExpTable<0>(row_real_exp_table, width);

    // Row process use real values fft, ref: http://dsp-book.narod.ru/FFTBB/0270_PDF_C14.pdf
    for (MI_S32 y = 0; y < height; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        std::complex<MI_F32> *dst_complex = dst.Ptr<std::complex<MI_F32>>(y);

        for (MI_S32 x = 0; x < half_w; x++)
        {
            MI_S32 idx = idx_table[x];
            buffer[x].real(SaturateCast<MI_F32>(src_row[2 * idx]));
            buffer[x].imag(SaturateCast<MI_F32>(src_row[2 * idx + 1]));
        }

        ButterflyTransformNone(buffer, 2, half_w, MI_FALSE, dft_row_exp_table);

        for (MI_S32 x = 1; x < half_w; x++)
        {
            std::complex<MI_F32> yk = buffer[x];
            std::complex<MI_F32> yk_conj = std::conj(buffer[half_w - x]);

            std::complex<MI_F32> fk = std::complex<MI_F32>(0.5f, 0.0f) * (yk + yk_conj);
            std::complex<MI_F32> gk = std::complex<MI_F32>(0.0f, 0.5f) * (yk_conj - yk);

            std::complex<MI_F32> result = fk + row_real_exp_table[x] * gk;
            dst_complex[x] = result;
            dst_complex[width - x] = std::conj(result);
        }
        {
            std::complex<MI_F32> y0 = buffer[0];
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

    AURA_FREE(ctx, temp_buffer);
    return Status::OK;
}

// inverse dft row process method
template <>
Status DftRadix2RowNoneImpl<MI_F32, 1>(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale)
{
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

    MI_S32 table_sz = width * sizeof(MI_U16) + width / 2 * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *dft_row_exp_table = reinterpret_cast<std::complex<MI_F32>*>(idx_table + width);

    GetDftExpTable<1>(dft_row_exp_table, width);
    GetReverseIndex(idx_table, width);

    for (MI_S32 y = 0; y < height; ++y)
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

        ButterflyTransformNone(dst_complex, 4, width, with_scale, dft_row_exp_table);
    }

    AURA_FREE(ctx, temp_buffer);
    return Status::OK;
}

static Status DftRadix2RowNone(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = DftRadix2RowNoneImpl<MI_U8, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S8:
        {
            ret = DftRadix2RowNoneImpl<MI_S8, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::U16:
        {
            ret = DftRadix2RowNoneImpl<MI_U16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S16:
        {
            ret = DftRadix2RowNoneImpl<MI_S16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::U32:
        {
            ret = DftRadix2RowNoneImpl<MI_U32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S32:
        {
            ret = DftRadix2RowNoneImpl<MI_S32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = DftRadix2RowNoneImpl<MI_F16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::F32:
        {
            ret = DftRadix2RowNoneImpl<MI_F32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
#endif
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
static AURA_VOID DftRadix2Col8None(Mat &src, Mat &dst, MI_U16 *idx_table, std::complex<MI_F32> *exp_table,
                                 std::complex<MI_F32> *buffer_data, MI_BOOL with_scale, MI_S32 x)
{
    MI_S32 height    = src.GetSizes().m_height;
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
    ButterflyTransformNone(buffer_row0, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row1, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row2, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row3, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row4, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row5, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row6, 2, height, with_scale, exp_table);
    ButterflyTransformNone(buffer_row7, 2, height, with_scale, exp_table);

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
static AURA_VOID DftRadix2Col1None(Mat &src, Mat &dst, MI_U16 *idx_table, std::complex<MI_F32> *exp_table,
                                 std::complex<MI_F32> *buffer_data, MI_BOOL with_scale, MI_S32 x)
{
    MI_S32 height    = src.GetSizes().m_height;
    MI_S32 row_pitch = src.GetRowPitch();

    MI_U8 *src_row = reinterpret_cast<MI_U8*>(src.GetData());
    for (MI_S32 i = 0; i < height; ++i)
    {
        std::complex<MI_F32> *dst_complex = reinterpret_cast<std::complex<MI_F32>*>(src_row);
        MI_S32 idx  = idx_table[i];
        buffer_data[idx] = dst_complex[x];
        src_row += row_pitch;
    }

    ButterflyTransformNone(buffer_data, 2, height, with_scale, exp_table);

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

template <typename Tp, MI_U8 IS_INVERSE, MI_BOOL IS_DST_C1>
static AURA_VOID DftRadix2ColProcRow2None(Mat &src, Mat &dst)
{
    std::complex<MI_F32> *row0_ptr = src.Ptr<std::complex<MI_F32>>(0);
    std::complex<MI_F32> *row1_ptr = src.Ptr<std::complex<MI_F32>>(1);

    MI_S32 width = src.GetSizes().m_width;
    MI_F32 coef  = IS_INVERSE ? 0.5f : 1.0f;

    if (IS_DST_C1)
    {
        Tp *row0_dst = dst.Ptr<Tp>(0);
        Tp *row1_dst = dst.Ptr<Tp>(1);

        for (MI_S32 x = 0; x < width; x++)
        {
            std::complex<MI_F32> val0 = row0_ptr[x];
            std::complex<MI_F32> val1 = row1_ptr[x];

            row0_dst[x] = SaturateCast<Tp>(((val0 + val1) * coef).real());
            row1_dst[x] = SaturateCast<Tp>(((val0 - val1) * coef).real());
        }
    }
    else
    {
        std::complex<Tp> *row0_dst = dst.Ptr<std::complex<Tp>>(0);
        std::complex<Tp> *row1_dst = dst.Ptr<std::complex<Tp>>(1);

        for (MI_S32 x = 0; x < width; x++)
        {
            std::complex<MI_F32> val0 = row0_ptr[x];
            std::complex<MI_F32> val1 = row1_ptr[x];

            std::complex<MI_F32> row0_result = (val0 + val1) * coef;
            std::complex<MI_F32> row1_result = (val0 - val1) * coef;

            row0_dst[x].real(SaturateCast<Tp>(row0_result.real()));
            row1_dst[x].real(SaturateCast<Tp>(row1_result.real()));
            row0_dst[x].imag(SaturateCast<Tp>(row0_result.imag()));
            row1_dst[x].imag(SaturateCast<Tp>(row1_result.imag()));
        }
    }
}

template <typename Tp, MI_U8 IS_INVERSE, MI_BOOL IS_DST_C1>
static Status DftRadix2ColNoneImpl(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Sizes3 sz     = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    if (1 == height)
    {
        if (IS_DST_C1)
        {
            std::complex<MI_F32> *row0_src = src.Ptr<std::complex<MI_F32>>(0);
            Tp                   *row0_dst = dst.Ptr<Tp>(0);
            for (MI_S32 x = 0; x < width; x++)
            {
                row0_dst[x] = SaturateCast<Tp>(row0_src[x].real());
            }
        }
        else
        {
            std::complex<MI_F32> *row0_src = src.Ptr<std::complex<MI_F32>>(0);
            std::complex<Tp>     *row0_dst = dst.Ptr<std::complex<Tp>>(0);
            for (MI_S32 x = 0; x < width; x++)
            {
                row0_dst[x].real(SaturateCast<Tp>(row0_src[x].real()));
                row0_dst[x].imag(SaturateCast<Tp>(row0_src[x].imag()));
            }
        }

        return Status::OK;
    }

    if (2 == height)
    {
        DftRadix2ColProcRow2None<Tp, IS_INVERSE, IS_DST_C1>(src, dst);
        return Status::OK;
    }

    MI_S32 table_sz = height * sizeof(MI_U16) + (height / 2 + 8 * height) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *exp_table = reinterpret_cast<std::complex<MI_F32>*>(idx_table + height);
    std::complex<MI_F32> *buffer_data = exp_table + height / 2;

    GetReverseIndex(idx_table, height);
    GetDftExpTable<IS_INVERSE>(exp_table, height);

    MI_S32 width_align8 = (width & (-8));

    MI_S32 x = 0;
    for (; x < width_align8; x += 8)
    {
        DftRadix2Col8None<Tp, IS_DST_C1>(src, dst, idx_table, exp_table, buffer_data, with_scale, x);
    }
    for (; x < width; x++)
    {
        DftRadix2Col1None<Tp, IS_DST_C1>(src, dst, idx_table, exp_table, buffer_data, with_scale, x);
    }

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
    return Status::OK;
}

template <typename Tp>
static Status IDftRadix2ColNoneHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Status ret = Status::ERROR;

    MI_BOOL is_dst_c1 = (1 == dst.GetSizes().m_channel);
    if (is_dst_c1)
    {
        ret = DftRadix2ColNoneImpl<Tp, 1, MI_TRUE>(ctx, src, dst, with_scale);
    }
    else
    {
        ret = DftRadix2ColNoneImpl<Tp, 1, MI_FALSE>(ctx, src, dst, with_scale);
    }

    AURA_RETURN(ctx, ret);
}

static Status IDftRadix2ColNoneHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDftRadix2ColNoneHelper<MI_U8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S8:
        {
            ret = IDftRadix2ColNoneHelper<MI_S8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U16:
        {
            ret = IDftRadix2ColNoneHelper<MI_U16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S16:
        {
            ret = IDftRadix2ColNoneHelper<MI_S16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U32:
        {
            ret = IDftRadix2ColNoneHelper<MI_U32>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S32:
        {
            ret = IDftRadix2ColNoneHelper<MI_S32>(ctx, src, dst, with_scale);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = IDftRadix2ColNoneHelper<MI_F16>(ctx, src, dst, with_scale);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = IDftRadix2ColNoneHelper<MI_F32>(ctx, src, dst, with_scale);
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
static Status DftBluesteinRowNoneImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale)
{
    Sizes3 sz     = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    MI_S32 width_padding = 1;
    while (width_padding / 2 <= width)
    {
        width_padding *= 2;
    }

    MI_S32 table_sz = width_padding * sizeof(MI_U16) + (3 * width_padding + width) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table                    = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *a_hor          = reinterpret_cast<std::complex<MI_F32>*>(idx_table + width_padding);
    std::complex<MI_F32> *b_hor          = a_hor + width_padding;
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

    FFTRadix2None(b_hor, width_padding, dft_exp_table, idx_table, MI_FALSE);

    for (MI_S32 y = 0; y < height; y++)
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

        FFTRadix2None(a_hor, width_padding, dft_exp_table, idx_table, MI_FALSE);
        for (MI_S32 x = 0; x < width_padding; ++x)
        {
            a_hor[x] *= b_hor[x];
        }
        FFTRadix2None(a_hor, width_padding, idft_exp_table, idx_table, MI_TRUE);

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

    AURA_FREE(ctx, temp_buffer);
    return Status::OK;
}

static Status DftBluesteinRowNone(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = DftBluesteinRowNoneImpl<MI_U8, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S8:
        {
            ret = DftBluesteinRowNoneImpl<MI_S8, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::U16:
        {
            ret = DftBluesteinRowNoneImpl<MI_U16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S16:
        {
            ret = DftBluesteinRowNoneImpl<MI_S16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::U32:
        {
            ret = DftBluesteinRowNoneImpl<MI_U32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::S32:
        {
            ret = DftBluesteinRowNoneImpl<MI_S32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = DftBluesteinRowNoneImpl<MI_F16, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
        case ElemType::F32:
        {
            ret = DftBluesteinRowNoneImpl<MI_F32, 0>(ctx, src, dst, MI_FALSE);
            break;
        }
#endif
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
AURA_INLINE AURA_VOID FftConvolveColNone(Mat &src, Mat &dst, MI_S32 h_padding, MI_U16 *idx_table,
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

    FFTRadix2None(a_ver, h_padding, dft_exp_table, idx_table, MI_FALSE);
    for (MI_S32 i = 0; i < h_padding; ++i)
    {
        a_ver[i] *= b_ver[i];
    }
    FFTRadix2None(a_ver, h_padding, idft_exp_table, idx_table, MI_TRUE);

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
static Status DftBluesteinColNoneImpl(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale = MI_FALSE)
{
    Sizes3 sz     = src.GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    MI_S32 height_padding = 1;
    while (height_padding / 2 <= height)
    {
        height_padding *= 2;
    }

    MI_S32 table_sz = height_padding * sizeof(MI_U16) + (3 * height_padding + height) * sizeof(std::complex<MI_F32>);
    AURA_VOID *temp_buffer = AURA_ALLOC(ctx, table_sz);
    if (MI_NULL == temp_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    MI_U16 *idx_table                    = reinterpret_cast<MI_U16*>(temp_buffer);
    std::complex<MI_F32> *a_ver          = reinterpret_cast<std::complex<MI_F32>*>(idx_table + height_padding);
    std::complex<MI_F32> *b_ver          = a_ver + height_padding;
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
    FFTRadix2None(b_ver, height_padding, dft_exp_table, idx_table, MI_FALSE);

    for (MI_S32 x = 0; x < width; x++)
    {
        FftConvolveColNone<Tp, IS_DST_C1>(src, dst, height_padding, idx_table, a_ver, b_ver, exp_table, dft_exp_table, idft_exp_table, with_scale, x);
    }

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
    return Status::OK;
}

template <typename Tp>
static Status IDftBluesteinColNoneHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale)
{
    Status ret = Status::ERROR;

    MI_BOOL is_dst_c1 = (1 == dst.GetSizes().m_channel);
    if (is_dst_c1)
    {
        ret = DftBluesteinColNoneImpl<Tp, 1, MI_TRUE>(ctx, src, dst, with_scale);
    }
    else
    {
        ret = DftBluesteinColNoneImpl<Tp, 1, MI_FALSE>(ctx, src, dst, with_scale);
    }

    AURA_RETURN(ctx, ret);
}

static Status IDftBluesteinColNoneHelper(Context *ctx, Mat &src, Mat &dst, MI_BOOL with_scale)
{
    Status ret = Status::ERROR;

    ElemType elem_type = dst.GetElemType();
    switch (elem_type)
    {
        case ElemType::U8:
        {
            ret = IDftBluesteinColNoneHelper<MI_U8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S8:
        {
            ret = IDftBluesteinColNoneHelper<MI_S8>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U16:
        {
            ret = IDftBluesteinColNoneHelper<MI_U16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S16:
        {
            ret = IDftBluesteinColNoneHelper<MI_S16>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::U32:
        {
            ret = IDftBluesteinColNoneHelper<MI_U32>(ctx, src, dst, with_scale);
            break;
        }
        case ElemType::S32:
        {
            ret = IDftBluesteinColNoneHelper<MI_S32>(ctx, src, dst, with_scale);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = IDftBluesteinColNoneHelper<MI_F16>(ctx, src, dst, with_scale);
            break;
        }
#endif // AURA_BUILD_HOST
        case ElemType::F32:
        {
            ret = IDftBluesteinColNoneHelper<MI_F32>(ctx, src, dst, with_scale);
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

DftNone::DftNone(Context *ctx, const OpTarget &target) : DftImpl(ctx, target)
{}

Status DftNone::SetArgs(const Array *src, Array *dst)
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

    if (ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support MI_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DftNone::Run()
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
        ret = DftRadix2RowNone(m_ctx, *src, *dst);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2RowNone failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }
    else
    {
        ret = DftBluesteinRowNone(m_ctx, *src, *dst);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinRowNone failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }

    if (IsPowOf2(height))
    {
        ret = DftRadix2ColNoneImpl<MI_F32, 0, MI_FALSE>(m_ctx, *dst, *dst, MI_FALSE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2ColNoneImpl<MI_F32, 0> failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }
    else
    {
        ret = DftBluesteinColNoneImpl<MI_F32, 0, MI_FALSE>(m_ctx, *dst, *dst, MI_FALSE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinColNoneImpl<MI_F32, 0> failed.");
            AURA_RETURN(m_ctx, ret);
        }
    }

    AURA_RETURN(m_ctx, ret);
}

InverseDftNone::InverseDftNone(Context *ctx, const OpTarget &target) : InverseDftImpl(ctx, target)
{}

Status InverseDftNone::SetArgs(const Array *src, Array *dst, MI_BOOL with_scale)
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

Status InverseDftNone::Run()
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
        if (DftRadix2RowNoneImpl<MI_F32, 1>(m_ctx, *src, *mid, m_with_scale) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftRadix2RowNoneImpl<MI_F32, 1> failed");
            return Status::ERROR;
        }
    }
    else
    {
        if (DftBluesteinRowNoneImpl<MI_F32, 1>(m_ctx, *src, *mid, m_with_scale) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DftBluesteinRowNoneImpl<MI_F32, 1> failed");
            return Status::ERROR;
        }
    }

    if (IsPowOf2(src_size.m_height))
    {
        if (IDftRadix2ColNoneHelper(m_ctx, *mid, *dst, m_with_scale) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDftRadix2ColNoneHelper failed");
            return Status::ERROR;
        }
    }
    else
    {
        if (IDftBluesteinColNoneHelper(m_ctx, *mid, *dst, m_with_scale) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "IDftBluesteinColNoneHelper failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

} // namespace aura