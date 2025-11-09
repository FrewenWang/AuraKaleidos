#include "aura/ops/matrix/dft.hpp"
#include "grid_dft_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/mat.h"

#include <complex>

namespace aura
{

#define SRC(y, x)  irow##y[x]
#define DST(y, x)  orow##y[x]

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID Dft4x4(const Tp *src, DT_S32 istep, std::complex<DT_F32> *dst, DT_S32 ostep)
{
    const Tp *irow0 = src;
    const Tp *irow1 = src + istep;
    const Tp *irow2 = src + istep * 2;
    const Tp *irow3 = src + istep * 3;

    std::complex<DT_F32> *orow0 = dst;
    std::complex<DT_F32> *orow1 = dst + ostep;
    std::complex<DT_F32> *orow2 = dst + ostep * 2;
    std::complex<DT_F32> *orow3 = dst + ostep * 3;

    DT_F32 sum00_02 = (DT_F32)(SRC(0, 0)) + (DT_F32)(SRC(0, 2));
    DT_F32 sub00_02 = (DT_F32)(SRC(0, 0)) - (DT_F32)(SRC(0, 2));
    DT_F32 sum01_03 = (DT_F32)(SRC(0, 1)) + (DT_F32)(SRC(0, 3));
    DT_F32 sub01_03 = (DT_F32)(SRC(0, 1)) - (DT_F32)(SRC(0, 3));
    DT_F32 sum10_12 = (DT_F32)(SRC(1, 0)) + (DT_F32)(SRC(1, 2));
    DT_F32 sub10_12 = (DT_F32)(SRC(1, 0)) - (DT_F32)(SRC(1, 2));
    DT_F32 sum11_13 = (DT_F32)(SRC(1, 1)) + (DT_F32)(SRC(1, 3));
    DT_F32 sub11_13 = (DT_F32)(SRC(1, 1)) - (DT_F32)(SRC(1, 3));
    DT_F32 sum20_22 = (DT_F32)(SRC(2, 0)) + (DT_F32)(SRC(2, 2));
    DT_F32 sub20_22 = (DT_F32)(SRC(2, 0)) - (DT_F32)(SRC(2, 2));
    DT_F32 sum21_23 = (DT_F32)(SRC(2, 1)) + (DT_F32)(SRC(2, 3));
    DT_F32 sub21_23 = (DT_F32)(SRC(2, 1)) - (DT_F32)(SRC(2, 3));
    DT_F32 sum30_32 = (DT_F32)(SRC(3, 0)) + (DT_F32)(SRC(3, 2));
    DT_F32 sub30_32 = (DT_F32)(SRC(3, 0)) - (DT_F32)(SRC(3, 2));
    DT_F32 sum31_33 = (DT_F32)(SRC(3, 1)) + (DT_F32)(SRC(3, 3));
    DT_F32 sub31_33 = (DT_F32)(SRC(3, 1)) - (DT_F32)(SRC(3, 3));

    // 0. calc F(2u, 2v)  (u, v) = (0, 0)/(0, 1)/(1, 0)/(1, 1)  -> DST(0, 0), DST(0, 2), DST(2, 0), DST(2, 2)
    {
        DT_F32 t0 = sum00_02 + sum20_22; // (m, n) = (0, 0)
        DT_F32 t1 = sum01_03 + sum21_23; // (m, n) = (0, 1)
        DT_F32 t2 = sum10_12 + sum30_32; // (m, n) = (1, 0)
        DT_F32 t3 = sum11_13 + sum31_33; // (m, n) = (1, 1)

        DT_F32 sum01 = t0 + t1;
        DT_F32 sub01 = t0 - t1;
        DT_F32 sum23 = t2 + t3;
        DT_F32 sub23 = t2 - t3;

        DST(0, 0) = {sum01 + sum23, 0};
        DST(0, 2) = {sub01 + sub23, 0};
        DST(2, 0) = {sum01 - sum23, 0};
        DST(2, 2) = {sub01 - sub23, 0};
    }

    // 1. calc F(2u, 2v + 1)  (u, v) = (0, 0)/(0, 1)/(1, 0)/(1, 1)  -> DST(0, 1), DST(0, 3), DST(2, 1), DST(2, 3)
    {
        DT_F32 t0 = sub00_02 + sub20_22; // (m, n) = (0, 0)
        DT_F32 t1 = sub01_03 + sub21_23; // (m, n) = (0, 1)
        DT_F32 t2 = sub10_12 + sub30_32; // (m, n) = (1, 0)
        DT_F32 t3 = sub11_13 + sub31_33; // (m, n) = (1, 1)

        DT_F32 sum02 = t0 + t2;
        DT_F32 sub02 = t0 - t2;
        DT_F32 sum13 = t1 + t3;
        DT_F32 sub13 = t1 - t3;

        DST(0, 1) = {sum02, -sum13};
        DST(0, 3) = {sum02,  sum13};
        DST(2, 1) = {sub02, -sub13};
        DST(2, 3) = {sub02,  sub13};
    }

    // 2. calc F(2*u + 1, 2v)
    {
        DT_F32 t0 = sum00_02 - sum20_22; // (m, n) = (0, 0)
        DT_F32 t1 = sum01_03 - sum21_23; // (m, n) = (0, 1)
        DT_F32 t2 = sum10_12 - sum30_32; // (m, n) = (1, 0)
        DT_F32 t3 = sum11_13 - sum31_33; // (m, n) = (1, 1)

        DT_F32 sum01 = t0 + t1;
        DT_F32 sub01 = t0 - t1;
        DT_F32 sum23 = t2 + t3;
        DT_F32 sub23 = t2 - t3;

        DST(1, 0) = {sum01, -sum23};
        DST(1, 2) = {sub01, -sub23};
        DST(3, 0) = {sum01,  sum23};
        DST(3, 2) = {sub01,  sub23};
    }

    // 3. calc F(2*u + 1, 2*v + 1)
    {
        DT_F32 t0 = sub00_02 - sub20_22; // (m, n) = (0, 0)
        DT_F32 t1 = sub01_03 - sub21_23; // (m, n) = (0, 1)
        DT_F32 t2 = sub10_12 - sub30_32; // (m, n) = (1, 0)
        DT_F32 t3 = sub11_13 - sub31_33; // (m, n) = (1, 1)

        DT_F32 sum03 = t0 + t3;
        DT_F32 sub03 = t0 - t3;
        DT_F32 sum12 = t1 + t2;
        DT_F32 sub12 = t1 - t2;

        DST(1, 1) = {sub03, -sum12};
        DST(1, 3) = {sum03,  sub12};
        DST(3, 1) = {sum03, -sub12};
        DST(3, 3) = {sub03,  sum12};
    }
}

#undef SRC
#undef DST

template <typename Tp>
static Status GridDft4x4NoneImpl(const Mat &src, Mat &dst)
{
    DT_S32 istep = src.GetRowPitch() / sizeof(Tp);
    DT_S32 ostep = dst.GetRowPitch() / sizeof(DT_F32) / 2;

    for (DT_S32 h = 0; h < src.GetSizes().m_height; h += 4)
    {
        const Tp *src_row = src.Ptr<Tp>(h);
        std::complex<DT_F32> *dst_row = dst.Ptr<std::complex<DT_F32>>(h);
        for (DT_S32 w = 0; w < src.GetSizes().m_width; w += 4)
        {
            const Tp *src_data = src_row + w;
            std::complex<DT_F32> *dst_data = dst_row + w;

            Dft4x4(src_data, istep, dst_data, ostep);
        }
    }

    return Status::OK;
}

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID RowGridDft1x8(Tp *src, std::complex<DT_F32> *dst, std::complex<DT_F32> *dft_row_exp_table,
                                         std::complex<DT_F32> *row_real_exp_table)
{
    std::complex<DT_F32> src0, src1, src2, src3;
    src0.real((DT_F32)src[0]); src0.imag((DT_F32)src[1]);
    src1.real((DT_F32)src[4]); src1.imag((DT_F32)src[5]);
    src2.real((DT_F32)src[2]); src2.imag((DT_F32)src[3]);
    src3.real((DT_F32)src[6]); src3.imag((DT_F32)src[7]);

    std::complex<DT_F32> temp;
    temp = src1;
    src1 = src0 - temp;
    src0 += temp;
    temp = src3;
    src3 = src2 - temp;
    src2 += temp;

    temp = src2;
    src2 = src0 - temp;
    src0 += temp;
    temp = src3 * dft_row_exp_table[1];
    src3 = src1 - temp;
    src1 += temp;
    // cal conj row dft
    {
        std::complex<DT_F32> yk_conj = std::conj(src3);
        std::complex<DT_F32> fk = std::complex<DT_F32>(0.5f, 0.0f) * (src1 + yk_conj);
        std::complex<DT_F32> gk = std::complex<DT_F32>(0.0f, 0.5f) * (yk_conj - src1);
        std::complex<DT_F32> result = fk + row_real_exp_table[1] * gk;

        dst[1] = result;
        dst[7] = std::conj(result);

        fk  = std::complex<DT_F32>(0.5f, 0.0f) * (src2 + std::conj(src2));
        gk  = std::complex<DT_F32>(0.0f, 0.5f) * (std::conj(src2) - src2);
        result = fk + row_real_exp_table[2] * gk;

        dst[2] = result;
        dst[6] = std::conj(result);

        fk  = std::complex<DT_F32>(0.5f, 0.0f) * (src3 + std::conj(src1));
        gk  = std::complex<DT_F32>(0.0f, 0.5f) * (std::conj(src1) - src3);
        result = fk + row_real_exp_table[3] * gk;

        dst[3] = result;
        dst[5] = std::conj(result);
    }

    std::complex<DT_F32> y0 = src0;
    std::complex<DT_F32> y0_conj = std::conj(y0);
    std::complex<DT_F32> f0 = std::complex<DT_F32>(0.5f, 0.0f) * (y0 + y0_conj);
    std::complex<DT_F32> g0 = std::complex<DT_F32>(0.0f, 0.5f) * (y0_conj - y0);
    dst[0] = f0 + g0;
    dst[4] = f0 - g0;
    // clear
    dst[0].imag(0.0f);
    dst[4].imag(0.0f);
}

AURA_ALWAYS_INLINE DT_VOID ColGridDft8x8(std::complex<DT_F32> src[][8], std::complex<DT_F32> *exp_table, std::complex<DT_F32> *dst, DT_S32 ostep)
{
    // butterfly size is 2, and update src data
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[1][j];
        src[1][j] = src[0][j] - temp;
        src[0][j] += temp;
        temp = src[3][j];
        src[3][j] = src[2][j] - temp;
        src[2][j] += temp;
        temp = src[5][j];
        src[5][j] = src[4][j] - temp;
        src[4][j] += temp;
        temp = src[7][j];
        src[7][j] = src[6][j] - temp;
        src[6][j] += temp;
    }

    // butterfly size is 4, and update src data
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[2][j];
        src[2][j] = src[0][j] - temp;
        src[0][j] += temp;

        temp = src[3][j] * exp_table[2];
        src[3][j] = src[1][j] - temp;
        src[1][j] += temp;

        temp = src[6][j];
        src[6][j] = src[4][j] - temp;
        src[4][j] += temp;

        temp = src[7][j] * exp_table[2];
        src[7][j] = src[5][j] - temp;
        src[5][j] += temp;
    }

    // butterfly size is 8, and store result to dst
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[4][j];
        dst[4 * ostep + j] = src[0][j] - temp;
        dst[j] = src[0][j] + temp;

        temp = src[5][j] * exp_table[1];
        dst[5 * ostep + j] = src[1][j] - temp;
        dst[1 * ostep + j] = src[1][j] + temp;

        temp = src[6][j] * exp_table[2];
        dst[6 * ostep + j] = src[2][j] - temp;
        dst[2 * ostep + j] = src[2][j] + temp;

        temp = src[7][j] * exp_table[3];
        dst[7 * ostep + j] = src[3][j] - temp;
        dst[3 * ostep + j] = src[3][j] + temp;
    }
}

template <typename Tp>
static Status GridDft8x8NoneImpl(const Mat &src, Mat &dst)
{
    const Sizes3 sz     = src.GetSizes();
    const DT_S32 width  = sz.m_width;
    const DT_S32 height = sz.m_height;
    const DT_S32 ostep  = dst.GetRowStep();

    std::complex<DT_F32> row_dft_result[8][8];
    std::complex<DT_F32> exp_table[4];
    std::complex<DT_F32> dft_row_exp_table[2];
    std::complex<DT_F32> row_real_exp_table[4];
    GetDftExpTable<0>(dft_row_exp_table,  4);
    GetDftExpTable<0>(row_real_exp_table, 8);
    GetDftExpTable<0>(exp_table,          8);

    for (DT_S32 y = 0; y < height; y += 8)
    {
        const Tp *src_row0 = src.Ptr<Tp>(y);
        const Tp *src_row1 = src.Ptr<Tp>(y + 1);
        const Tp *src_row2 = src.Ptr<Tp>(y + 2);
        const Tp *src_row3 = src.Ptr<Tp>(y + 3);
        const Tp *src_row4 = src.Ptr<Tp>(y + 4);
        const Tp *src_row5 = src.Ptr<Tp>(y + 5);
        const Tp *src_row6 = src.Ptr<Tp>(y + 6);
        const Tp *src_row7 = src.Ptr<Tp>(y + 7);
        std::complex<DT_F32> *dst_ptr = dst.Ptr<std::complex<DT_F32>>(y);
        for (DT_S32 x = 0; x < width; x += 8)
        {
            RowGridDft1x8(src_row0 + x, row_dft_result[0], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row1 + x, row_dft_result[4], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row2 + x, row_dft_result[2], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row3 + x, row_dft_result[6], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row4 + x, row_dft_result[1], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row5 + x, row_dft_result[5], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row6 + x, row_dft_result[3], dft_row_exp_table, row_real_exp_table);
            RowGridDft1x8(src_row7 + x, row_dft_result[7], dft_row_exp_table, row_real_exp_table);

            ColGridDft8x8(row_dft_result, exp_table, dst_ptr + x, ostep);
        }
    }

    return Status::OK;
}

template <typename Tp, DT_S32 GRID_LEN>
static Status GridDftNxN(const Mat &src, Mat &dst, DT_U16 *idx_x_table, DT_U16 *idx_y_table, 
                  std::complex<DT_F32> *dft_row_exp_table, std::complex<DT_F32> *row_real_exp_table,
                  DT_S32 width_offset, DT_S32 height_offset)
{
    constexpr DT_S32 HALF_GRID = GRID_LEN / 2;
    const DT_S32 ostep  = dst.GetRowStep();

    std::complex<DT_F32> *dst_ptr = dst.Ptr<std::complex<DT_F32>>(height_offset);
    std::complex<DT_F32> buffer[HALF_GRID];
    std::complex<DT_F32> dst_complex[GRID_LEN][GRID_LEN];

    for (DT_S32 y_grid = 0; y_grid < GRID_LEN; y_grid++)
    {
        const Tp *src_row = src.Ptr<Tp>(y_grid + height_offset);
        DT_S32 y_grid_index = idx_y_table[y_grid];

        for (DT_S32 x_grid = 0; x_grid < HALF_GRID; x_grid++)
        {
            DT_S32 idx = idx_x_table[x_grid];
            buffer[x_grid].real(SaturateCast<DT_F32>(src_row[width_offset + 2 * idx]));
            buffer[x_grid].imag(SaturateCast<DT_F32>(src_row[width_offset + 2 * idx + 1]));
        }

        ButterflyTransformNone(buffer, 2, HALF_GRID, DT_FALSE, dft_row_exp_table);

        for (DT_S32 x_grid = 1; x_grid < HALF_GRID; x_grid++)
        {
            std::complex<DT_F32> yk = buffer[x_grid];
            std::complex<DT_F32> yk_conj = std::conj(buffer[HALF_GRID - x_grid]);

            std::complex<DT_F32> fk = std::complex<DT_F32>(0.5f, 0.0f) * (yk + yk_conj);
            std::complex<DT_F32> gk = std::complex<DT_F32>(0.0f, 0.5f) * (yk_conj - yk);

            std::complex<DT_F32> result = fk + row_real_exp_table[x_grid] * gk;
            dst_complex[y_grid_index][x_grid] = result;
            dst_complex[y_grid_index][GRID_LEN - x_grid] = std::conj(result);
        }

        std::complex<DT_F32> y0 = buffer[0];
        std::complex<DT_F32> y0_conj = std::conj(y0);
        std::complex<DT_F32> f0 = std::complex<DT_F32>(0.5f, 0.0f) * (y0 + y0_conj);
        std::complex<DT_F32> g0 = std::complex<DT_F32>(0.0f, 0.5f) * (y0_conj - y0);
        dst_complex[y_grid_index][0] = f0 + g0;
        dst_complex[y_grid_index][HALF_GRID] = f0 - g0;

        // clear
        dst_complex[y_grid_index][0].imag(0.0f);
        dst_complex[y_grid_index][HALF_GRID].imag(0.0f);
    }
    std::complex<DT_F32> *row_result = &dst_complex[0][0];

    // col grid dft
    std::complex<DT_F32> exp_table[HALF_GRID];
    GetDftExpTable<0>(exp_table, GRID_LEN);

    for (DT_S32 size = 2; size < GRID_LEN; size *= 2)
    {
        DT_S32 half_size = size / 2;
        DT_S32 table_step = GRID_LEN / size;

        for (DT_S32 i = 0; i < GRID_LEN; i += size)
        {
            for (DT_S32 j = i, k = 0; j < i + half_size; j++, k += table_step)
            {
                for (DT_S32 row_index = 0; row_index < GRID_LEN; row_index++)
                {
                    std::complex<DT_F32> temp = dst_complex[j + half_size][row_index] * exp_table[k];
                    dst_complex[j + half_size][row_index] = dst_complex[j][row_index] - temp;
                    dst_complex[j][row_index] += temp;
                }
            }
        }
    }

    // last butterfly
    for (DT_S32 j = 0, k = 0; j < HALF_GRID; j++, k++)
    {
        std::complex<DT_F32> *dst_shift     = dst_ptr + j * ostep;
        std::complex<DT_F32> *row_result_shift = row_result + j * GRID_LEN;

        std::complex<DT_F32> *dst_half_shift     = dst_ptr + (j + HALF_GRID) * ostep;
        std::complex<DT_F32> *row_result_half_shift = row_result + (j + HALF_GRID) * GRID_LEN;

        for (DT_S32 row_index = 0; row_index < GRID_LEN; row_index++)
        {
            std::complex<DT_F32> temp = row_result_half_shift[row_index] * exp_table[k];
            dst_half_shift[width_offset + row_index] = row_result_shift[row_index] - temp;
            dst_shift[width_offset + row_index]      = row_result_shift[row_index] + temp;
        }
    }
        
    return Status::OK;
}

template <typename Tp, DT_S32 GRID_LEN>
static Status GridDftNoneImpl(const Mat &src, Mat &dst)
{
    const Sizes3 sz = src.GetSizes();
    const DT_S32 width = sz.m_width;
    const DT_S32 height = sz.m_height;

    constexpr DT_S32 HALF_GRID = GRID_LEN / 2;

    DT_U16 idx_x_table[HALF_GRID] = {0};
    DT_U16 idx_y_table[GRID_LEN]  = {0};
    GetReverseIndex(idx_x_table, HALF_GRID);
    GetReverseIndex(idx_y_table, GRID_LEN);

    std::complex<DT_F32> dft_row_exp_table[HALF_GRID];
    std::complex<DT_F32> row_real_exp_table[GRID_LEN];
    GetDftExpTable<0>(dft_row_exp_table, HALF_GRID);
    GetDftExpTable<0>(row_real_exp_table, GRID_LEN);

    for (DT_S32 y = 0; y < height; y += GRID_LEN)
    {
        for (DT_S32 x = 0; x < width; x += GRID_LEN)
        {
            GridDftNxN<Tp, GRID_LEN>(src, dst, idx_x_table, idx_y_table, dft_row_exp_table, row_real_exp_table, x, y);
        }
    }

    return Status::OK;
}

static Status GridDftCommonNoneImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::OK;

    DT_S32 height = src.GetSizes().m_height;
    DT_S32 width  = src.GetSizes().m_width;
    DT_S32 grid_rows = height / grid_len;
    DT_S32 grid_cols = width  / grid_len;
    for (DT_S32 i = 0; i < grid_rows; i++)
    {
        for (DT_S32 j = 0; j < grid_cols; j++)
        {
            DT_S32 x = j * grid_len;
            DT_S32 y = i * grid_len;
            Rect roi(x, y, grid_len, grid_len);
            Mat src_grid = src.Roi(roi);
            Mat dst_grid = dst.Roi(roi);
            ret = IDft(ctx, src_grid, dst_grid, OpTarget::None());
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, ("GridDftCommonNoneImpl failed, i = " + std::to_string(i) + ", j = " + std::to_string(j)).c_str());
                return ret;
            }
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status GridDftNoneHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, const OpTarget &target)
{
    Status ret = Status::ERROR;
    switch (grid_len)
    {
        case 4:
        {
            ret = GridDft4x4NoneImpl<Tp>(src, dst);
            break;
        }
        case 8:
        {
            ret = GridDft8x8NoneImpl<Tp>(src, dst);
            break;
        }
        case 16:
        {
            ret = GridDftNoneImpl<Tp, 16>(src, dst);
            break;
        }
        case 32:
        {
            ret = GridDftNoneImpl<Tp, 32>(src, dst);
            break;
        }
        default:
        {
            ret = GridDftCommonNoneImpl(ctx, src, dst, grid_len, target);
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

GridDftNone::GridDftNone(Context *ctx, const OpTarget &target) : GridDftImpl(ctx, target)
{}

Status GridDftNone::SetArgs(const Array *src, Array *dst, DT_S32 grid_len)
{
    if (GridDftImpl::SetArgs(src, dst, grid_len) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support DT_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridDftNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GridDftNoneHelper<DT_U8>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::S8:
        {
            ret = GridDftNoneHelper<DT_S8>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::U16:
        {
            ret = GridDftNoneHelper<DT_U16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::S16:
        {
            ret = GridDftNoneHelper<DT_S16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::U32:
        {
            ret = GridDftNoneHelper<DT_U32>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::S32:
        {
            ret = GridDftNoneHelper<DT_S32>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = GridDftNoneHelper<MI_F16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::F32:
        {
            ret = GridDftNoneHelper<DT_F32>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
#endif // AURA_BUILD_HOST
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID ConvertComplex(Tp *dst, DT_S32 dst_offset, std::complex<DT_F32> complex_src)
{
    if (1 == C)
    {
        dst[dst_offset] = SaturateCast<Tp>(complex_src.real());
    }
    else
    {
        dst[dst_offset]     = SaturateCast<Tp>(complex_src.real());
        dst[dst_offset + 1] = SaturateCast<Tp>(complex_src.imag());
    }
}

template <typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID IDft4x4(const std::complex<DT_F32> *src_row0, const std::complex<DT_F32> *src_row1,
                                   const std::complex<DT_F32> *src_row2, const std::complex<DT_F32> *src_row3,
                                   DT_BOOL with_scale, Tp *dst, DT_S32 ostep)
{
    DT_F32 n = with_scale ? 4.0f : 1.0f;
    std::complex<DT_F32> exp_table = {0, 1};
    std::complex<DT_F32> row_dft_result[4][4];
    // butterfly size is 2
    row_dft_result[0][0] = src_row0[0] + src_row0[2];
    row_dft_result[0][1] = src_row0[0] - src_row0[2];
    row_dft_result[0][2] = src_row0[1] + src_row0[3];
    row_dft_result[0][3] = src_row0[1] - src_row0[3];

    row_dft_result[1][0] = src_row1[0] + src_row1[2];
    row_dft_result[1][1] = src_row1[0] - src_row1[2];
    row_dft_result[1][2] = src_row1[1] + src_row1[3];
    row_dft_result[1][3] = src_row1[1] - src_row1[3];

    row_dft_result[2][0] = src_row2[0] + src_row2[2];
    row_dft_result[2][1] = src_row2[0] - src_row2[2];
    row_dft_result[2][2] = src_row2[1] + src_row2[3];
    row_dft_result[2][3] = src_row2[1] - src_row2[3];

    row_dft_result[3][0] = src_row3[0] + src_row3[2];
    row_dft_result[3][1] = src_row3[0] - src_row3[2];
    row_dft_result[3][2] = src_row3[1] + src_row3[3];
    row_dft_result[3][3] = src_row3[1] - src_row3[3];

    // butterfly size is 4
    std::complex<DT_F32> temp;
    temp = row_dft_result[0][2];
    row_dft_result[0][2] = (row_dft_result[0][0] - temp) / n;
    row_dft_result[0][0] = (row_dft_result[0][0] + temp) / n;
    temp = row_dft_result[0][3] * exp_table;
    row_dft_result[0][3] = (row_dft_result[0][1] - temp) / n;
    row_dft_result[0][1] = (row_dft_result[0][1] + temp) / n;

    temp = row_dft_result[1][2];
    row_dft_result[1][2] = (row_dft_result[1][0] - temp) / n;
    row_dft_result[1][0] = (row_dft_result[1][0] + temp) / n;
    temp = row_dft_result[1][3] * exp_table;
    row_dft_result[1][3] = (row_dft_result[1][1] - temp) / n;
    row_dft_result[1][1] = (row_dft_result[1][1] + temp) / n;

    temp = row_dft_result[2][2];
    row_dft_result[2][2] = (row_dft_result[2][0] - temp) / n;
    row_dft_result[2][0] = (row_dft_result[2][0] + temp) / n;
    temp = row_dft_result[2][3] * exp_table;
    row_dft_result[2][3] = (row_dft_result[2][1] - temp) / n;
    row_dft_result[2][1] = (row_dft_result[2][1] + temp) / n;

    temp = row_dft_result[3][2];
    row_dft_result[3][2] = (row_dft_result[3][0] - temp) / n;
    row_dft_result[3][0] = (row_dft_result[3][0] + temp) / n;
    temp = row_dft_result[3][3] * exp_table;
    row_dft_result[3][3] = (row_dft_result[3][1] - temp) / n;
    row_dft_result[3][1] = (row_dft_result[3][1] + temp) / n;

    // col idft
    for (DT_S32 j = 0; j < 4; j++)
    {
        temp = row_dft_result[1][j];
        row_dft_result[1][j] = row_dft_result[0][j] - temp;
        row_dft_result[0][j] += temp;
        temp = row_dft_result[3][j];
        row_dft_result[3][j] = row_dft_result[2][j] - temp;
        row_dft_result[2][j] += temp;
    }

    for (DT_S32 j = 0; j < 4; j++)
    {
        temp = row_dft_result[2][j];
        ConvertComplex<Tp, C>(dst, 2 * ostep + j * C, (row_dft_result[0][j] - temp) / n);
        ConvertComplex<Tp, C>(dst,             j * C, (row_dft_result[0][j] + temp) / n);

        temp = row_dft_result[3][j] * exp_table;
        ConvertComplex<Tp, C>(dst, 3 * ostep + j * C, (row_dft_result[1][j] - temp) / n);
        ConvertComplex<Tp, C>(dst, 1 * ostep + j * C, (row_dft_result[1][j] + temp) / n);
    }
}

template <typename Tp, DT_S32 C>
static Status GridIDft4x4NoneImpl(const Mat &src, Mat &dst, DT_BOOL with_scale)
{
    Sizes3 size   = dst.GetSizes();
    DT_S32 width  = size.m_width;
    DT_S32 height = size.m_height;
    DT_S32 ostep  = dst.GetRowPitch() / sizeof(Tp);

    for (DT_S32 h = 0; h < height; h += 4)
    {
        const std::complex<DT_F32> *src_row0 = src.Ptr<std::complex<DT_F32>>(h);
        const std::complex<DT_F32> *src_row1 = src.Ptr<std::complex<DT_F32>>(h + 1);
        const std::complex<DT_F32> *src_row2 = src.Ptr<std::complex<DT_F32>>(h + 2);
        const std::complex<DT_F32> *src_row3 = src.Ptr<std::complex<DT_F32>>(h + 3);

        Tp *dst_row = dst.Ptr<Tp>(h);

        for (DT_S32 w = 0; w < width; w += 4)
        {
            IDft4x4<Tp, C>(src_row0 + w, src_row2 + w, src_row1 + w, src_row3 + w, with_scale,
                           dst_row + w * C, ostep);
        }
    }

    return Status::OK;
}

static DT_VOID RowGridIDft1x8(const std::complex<DT_F32> *src, std::complex<DT_F32> *dst, DT_BOOL with_scale,
                              DT_U16 *idx_table, std::complex<DT_F32> *dft_row_exp_table)
{
    // butterfly size is 2
    for (DT_S32 x = 0; x < 8; x += 2)
    {
        DT_U16 idx0 = idx_table[x];
        DT_U16 idx1 = idx_table[x + 1];
        dst[x]      = src[idx0] + src[idx1];
        dst[x + 1]  = src[idx0] - src[idx1];
    }

    ButterflyTransformNone(dst, 4, 8, with_scale, dft_row_exp_table);
}

template <typename Tp, DT_S32 C>
AURA_ALWAYS_INLINE DT_VOID ColGridIDft8x8(std::complex<DT_F32> src[][8], std::complex<DT_F32> *exp_table,
                                          DT_BOOL with_scale, Tp *dst, DT_S32 ostep)
{
    DT_F32 n = with_scale ? 8.0f : 1.0f;

    // butterfly size is 2, and update src data
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[1][j];
        src[1][j] = src[0][j] - temp;
        src[0][j] += temp;
        temp = src[3][j];
        src[3][j] = src[2][j] - temp;
        src[2][j] += temp;
        temp = src[5][j];
        src[5][j] = src[4][j] - temp;
        src[4][j] += temp;
        temp = src[7][j];
        src[7][j] = src[6][j] - temp;
        src[6][j] += temp;
    }

    // butterfly size is 4, and update src data
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[2][j];
        src[2][j] = src[0][j] - temp;
        src[0][j] += temp;

        temp = src[3][j] * exp_table[2];
        src[3][j] = src[1][j] - temp;
        src[1][j] += temp;

        temp = src[6][j];
        src[6][j] = src[4][j] - temp;
        src[4][j] += temp;

        temp = src[7][j] * exp_table[2];
        src[7][j] = src[5][j] - temp;
        src[5][j] += temp;
    }

    // butterfly size is 8, and store result to dst
    for (DT_S32 j = 0; j < 8; j++)
    {
        std::complex<DT_F32> temp = src[4][j];
        ConvertComplex<Tp, C>(dst, 4 * ostep + j * C, (src[0][j] - temp) / n);
        ConvertComplex<Tp, C>(dst,             j * C, (src[0][j] + temp) / n);

        temp = src[5][j] * exp_table[1];
        ConvertComplex<Tp, C>(dst, 5 * ostep + j * C, (src[1][j] - temp) / n);
        ConvertComplex<Tp, C>(dst, 1 * ostep + j * C, (src[1][j] + temp) / n);

        temp = src[6][j] * exp_table[2];
        ConvertComplex<Tp, C>(dst, 6 * ostep + j * C, (src[2][j] - temp) / n);
        ConvertComplex<Tp, C>(dst, 2 * ostep + j * C, (src[2][j] + temp) / n);

        temp = src[7][j] * exp_table[3];
        ConvertComplex<Tp, C>(dst, 7 * ostep + j * C, (src[3][j] - temp) / n);
        ConvertComplex<Tp, C>(dst, 3 * ostep + j * C, (src[3][j] + temp) / n);
    }
}

template <typename Tp, DT_S32 C>
static Status GridIDft8x8NoneImpl(const Mat &src, Mat &dst, DT_BOOL with_scale)
{
    Sizes3 size    = dst.GetSizes();
    DT_S32 width   = size.m_width;
    DT_S32 height  = size.m_height;
    DT_S32 channel = size.m_channel;

    DT_S32 ostep = dst.GetRowPitch() / sizeof(Tp);

    std::complex<DT_F32> row_dft_result[8][8];
    std::complex<DT_F32> dft_row_exp_table[4];
    GetDftExpTable<1>(dft_row_exp_table, 8);

    DT_U16 idx_table[8];
    GetReverseIndex(idx_table, 8);

    for (DT_S32 h = 0; h < height; h += 8)
    {
        const std::complex<DT_F32> *src_row0 = src.Ptr<std::complex<DT_F32>>(h);
        const std::complex<DT_F32> *src_row1 = src.Ptr<std::complex<DT_F32>>(h + 1);
        const std::complex<DT_F32> *src_row2 = src.Ptr<std::complex<DT_F32>>(h + 2);
        const std::complex<DT_F32> *src_row3 = src.Ptr<std::complex<DT_F32>>(h + 3);
        const std::complex<DT_F32> *src_row4 = src.Ptr<std::complex<DT_F32>>(h + 4);
        const std::complex<DT_F32> *src_row5 = src.Ptr<std::complex<DT_F32>>(h + 5);
        const std::complex<DT_F32> *src_row6 = src.Ptr<std::complex<DT_F32>>(h + 6);
        const std::complex<DT_F32> *src_row7 = src.Ptr<std::complex<DT_F32>>(h + 7);

        Tp *dst_row = dst.Ptr<Tp>(h);

        for (DT_S32 w = 0; w < width; w += 8)
        {
            RowGridIDft1x8(src_row0 + w, row_dft_result[0], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row1 + w, row_dft_result[4], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row2 + w, row_dft_result[2], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row3 + w, row_dft_result[6], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row4 + w, row_dft_result[1], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row5 + w, row_dft_result[5], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row6 + w, row_dft_result[3], with_scale, idx_table, dft_row_exp_table);
            RowGridIDft1x8(src_row7 + w, row_dft_result[7], with_scale, idx_table, dft_row_exp_table);

            ColGridIDft8x8<Tp, C>(row_dft_result, dft_row_exp_table, with_scale, dst_row + w * channel, ostep);
        }
    }

    return Status::OK;
}

template <typename Tp, DT_S32 GRID_LEN, DT_S32 C>
static Status GridIDftNxN(const Mat &src, Mat &dst, DT_U16 *idx_table, std::complex<DT_F32> *exp_table, 
                   DT_BOOL with_scale, DT_S32 width_offset, DT_S32 height_offset)
{
    constexpr DT_S32 HALF_GRID = GRID_LEN / 2;
    const DT_S32 ostep  = dst.GetRowPitch() / sizeof(Tp);
    
    DT_F32 n = with_scale ? (DT_F32)GRID_LEN : 1.0f;

    Tp *dst_ptr = dst.Ptr<Tp>(height_offset);
    std::complex<DT_F32> row_dft_result[GRID_LEN][GRID_LEN];

    for (DT_S32 y_grid = 0; y_grid < GRID_LEN; y_grid++)
    {
        const std::complex<DT_F32> *src_row = src.Ptr<std::complex<DT_F32>>(y_grid + height_offset);
        DT_S32 y_grid_index = idx_table[y_grid];

        for (DT_S32 x = 0; x < GRID_LEN; x += 2)
        {
            DT_U16 idx0 = idx_table[x];
            DT_U16 idx1 = idx_table[x + 1];
            row_dft_result[y_grid_index][x]     = src_row[width_offset + idx0] + src_row[width_offset + idx1];
            row_dft_result[y_grid_index][x + 1] = src_row[width_offset + idx0] - src_row[width_offset + idx1];
        }

        ButterflyTransformNone(&row_dft_result[y_grid_index][0], 4, GRID_LEN, with_scale, exp_table);
    }

    std::complex<DT_F32> *row_result = &row_dft_result[0][0];

    // col grid dft
    for (DT_S32 size = 2; size < GRID_LEN; size *= 2)
    {
        DT_S32 half_size = size / 2;
        DT_S32 table_step = GRID_LEN / size;

        for (DT_S32 i = 0; i < GRID_LEN; i += size)
        {
            for (DT_S32 j = i, k = 0; j < i + half_size; j++, k += table_step)
            {
                for (DT_S32 row_index = 0; row_index < GRID_LEN; row_index++)
                {
                    std::complex<DT_F32> temp = row_dft_result[j + half_size][row_index] * exp_table[k];
                    row_dft_result[j + half_size][row_index] = row_dft_result[j][row_index] - temp;
                    row_dft_result[j][row_index] += temp;
                }
            }
        }
    }

    // last butterfly
    for (DT_S32 j = 0, k = 0; j < HALF_GRID; j++, k++)
    {
        Tp *dst_shift      = dst_ptr + j * ostep;
        Tp *dst_half_shift = dst_ptr + (j + HALF_GRID) * ostep;

        std::complex<DT_F32> *row_result_shift      = row_result + j * GRID_LEN;
        std::complex<DT_F32> *row_result_half_shift = row_result + (j + HALF_GRID) * GRID_LEN;

        for (DT_S32 row_index = 0; row_index < GRID_LEN; row_index++)
        {
            std::complex<DT_F32> temp = row_result_half_shift[row_index] * exp_table[k];
            ConvertComplex<Tp, C>(dst_half_shift, (width_offset + row_index) * C, (row_result_shift[row_index] - temp) / n);
            ConvertComplex<Tp, C>(dst_shift,      (width_offset + row_index) * C, (row_result_shift[row_index] + temp) / n);
        }
    }

    return Status::OK;
}

template <typename Tp, DT_S32 GRID_LEN, DT_S32 C>
static Status GridIDftNoneImpl(const Mat &src, Mat &dst, DT_BOOL with_scale)
{
    const Sizes3 sz = dst.GetSizes();
    const DT_S32 width  = sz.m_width;
    const DT_S32 height = sz.m_height;

    DT_U16 idx_table[GRID_LEN] = {0};
    GetReverseIndex(idx_table, GRID_LEN);

    std::complex<DT_F32> dft_row_exp_table[GRID_LEN];
    GetDftExpTable<1>(dft_row_exp_table, GRID_LEN);

    for (DT_S32 y = 0; y < height; y += GRID_LEN)
    {
        for (DT_S32 x = 0; x < width; x += GRID_LEN)
        {
            GridIDftNxN<Tp, GRID_LEN, C>(src, dst, idx_table, dft_row_exp_table, with_scale, x, y);
        }
    }

    return Status::OK;
}

template <typename Tp, DT_S32 C>
static Status GridIDft16x16NoneImpl(const Mat &src, Mat &dst, DT_BOOL with_scale)
{
    return GridIDftNoneImpl<Tp, 16, C>(src, dst, with_scale);
}

template <typename Tp, DT_S32 C>
static Status GridIDft32x32NoneImpl(const Mat &src, Mat &dst, DT_BOOL with_scale)
{
    return GridIDftNoneImpl<Tp, 32, C>(src, dst, with_scale);
}

template <typename Tp>
static Status GridIDftNoneHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, DT_BOOL with_scale)
{
    Status ret = Status::OK;
    DT_S32 channel = dst.GetSizes().m_channel;
    switch (grid_len)
    {
        case 4:
        {   
            if (1 == channel)
            {
                ret = GridIDft4x4NoneImpl<Tp, 1>(src, dst, with_scale);
            }
            else
            {
                ret = GridIDft4x4NoneImpl<Tp, 2>(src, dst, with_scale);
            }
            break;
        }
        case 8:
        {
            if (1 == channel)
            {
                ret = GridIDft8x8NoneImpl<Tp, 1>(src, dst, with_scale);
            }
            else
            {
                ret = GridIDft8x8NoneImpl<Tp, 2>(src, dst, with_scale);
            }
            break;
        }
        case 16:
        {
            if (1 == channel)
            {
                ret = GridIDftNoneImpl<Tp, 16, 1>(src, dst, with_scale);
            }
            else
            {
                ret = GridIDftNoneImpl<Tp, 16, 2>(src, dst, with_scale);
            }
            break;
        }
        case 32:
        {
            if (1 == channel)
            {
                ret = GridIDftNoneImpl<Tp, 32, 1>(src, dst, with_scale);
            }
            else
            {
                ret = GridIDftNoneImpl<Tp, 32, 2>(src, dst, with_scale);
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported grid length");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

GridIDftNone::GridIDftNone(Context *ctx, const OpTarget &target) : GridIDftImpl(ctx, target)
{}

Status GridIDftNone::SetArgs(const Array *src, Array *dst, DT_S32 grid_len, DT_BOOL with_scale)
{
    if (GridIDftImpl::SetArgs(src, dst, grid_len, with_scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDftNone::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    DT_S32 dst_channel    = dst->GetSizes().m_channel;
    ElemType dst_elemtype = dst->GetElemType();

    if (1 == dst_channel)
    {
        if (ElemType::F64 == dst_elemtype)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "when dst channel is 1, current dst does not support DT_F64 type.");
            return Status::ERROR;
        }
    }
    else if (2 == dst_channel)
    {
        if (dst_elemtype != ElemType::F32)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "when dst channel is 2, current dst only support DT_F32 type.");
            return Status::ERROR;
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst channel must be 1 or 2");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridIDftNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;
    switch (dst->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GridIDftNoneHelper<DT_U8>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::S8:
        {
            ret = GridIDftNoneHelper<DT_S8>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::U16:
        {
            ret = GridIDftNoneHelper<DT_U16>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::S16:
        {
            ret = GridIDftNoneHelper<DT_S16>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::U32:
        {
            ret = GridIDftNoneHelper<DT_U32>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::S32:
        {
            ret = GridIDftNoneHelper<DT_S32>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = GridIDftNoneHelper<MI_F16>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
        case ElemType::F32:
        {
            ret = GridIDftNoneHelper<DT_F32>(m_ctx, *src, *dst, m_grid_len, m_with_scale);
            break;
        }
#endif // AURA_BUILD_HOST
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
