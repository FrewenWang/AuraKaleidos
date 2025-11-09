#include "mipi_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static constexpr DT_S8 tbl_buffer0[8] = {0, 1, 2, 3,  5,  6,  7,  8};
static constexpr DT_S8 tbl_buffer1[8] = {2, 3, 4, 5,  7,  8,  9, 10};
static constexpr DT_S8 tbl_buffer2[8] = {4, 5, 6, 7,  9, 10, 11, 12};
static constexpr DT_S8 tbl_buffer3[8] = {6, 7, 8, 9, 11, 12, 13, 14};

static constexpr DT_U8 operand_buffer[8]        = {0x03, 0x0C, 0x30, 0xC0, 0x03, 0x0C, 0x30, 0xC0};
static constexpr DT_S8 unpack_shift_buffer[8]   = {0, -2, -4, -6, 0, -2, -4, -6};
static constexpr DT_S8 pack_shift_buffer[8]     = {0,  2,  4,  6, 0,  2,  4,  6};

static Status MipiPackNeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width       = src.GetSizes().m_width;
    DT_S32 width_align = width & (-8);

    uint8x8_t vdu8_shift = neon::vload1(pack_shift_buffer);
    uint8x8_t vdu8_mask;
    neon::vdup(vdu8_mask, static_cast<DT_U8>(3));

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U16 *src_row = src.Ptr<DT_U16>(y);
        DT_U8        *dst_row = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 8)
        {
            uint16x8_t vqu16_src;
            uint8x8_t  vdu8_dst, vdu8_src;

            vqu16_src = neon::vload1q(src_row);
            vdu8_dst  = neon::vshrn_n<2>(vqu16_src); // 8 u8 output
            vdu8_src  = neon::vmovn(vqu16_src);

            uint8x8_t vdu8_sum = neon::vand(vdu8_src, vdu8_mask);
            vdu8_sum = neon::vshl(vdu8_sum, vdu8_shift);
            vdu8_sum = neon::vpadd(vdu8_sum, vdu8_sum);
            vdu8_sum = neon::vpadd(vdu8_sum, vdu8_sum);

            DT_U8 pixel5  = neon::vgetlane<0>(vdu8_sum);
            DT_U8 pixel10 = neon::vgetlane<1>(vdu8_sum);

            neon::vstore(dst_row, vdu8_dst);
            vdu8_dst = neon::vext<4>(vdu8_dst, vdu8_dst);
            *(dst_row + 4) = pixel5;
            neon::vstore(dst_row + 5, vdu8_dst);
            *(dst_row + 9) = pixel10;

            src_row += 8;
            dst_row += 10;
        }

        for (; x < width; x += 4)
        {
            dst_row[0] = src_row[0] >> 2;
            dst_row[1] = src_row[1] >> 2;
            dst_row[2] = src_row[2] >> 2;
            dst_row[3] = src_row[3] >> 2;

            DT_U8 t0 = (src_row[0] & 0x03);
            DT_U8 t1 = (src_row[1] & 0x03) << 2;
            DT_U8 t2 = (src_row[2] & 0x03) << 4;
            DT_U8 t3 = (src_row[3] & 0x03) << 6;

            dst_row[4] = t0 + t1 + t2 + t3;
            src_row += 4;
            dst_row += 5;
        }
    }

    return Status::OK;
}

MipiPackNeon::MipiPackNeon(Context *ctx, const OpTarget &target) : MipiPackImpl(ctx, target)
{}

Status MipiPackNeon::SetArgs(const Array *src, Array *dst)
{
    Status ret = Status::ERROR;

    if (MipiPackImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MipiPackImpl::SetArgs failed");
        return ret;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return ret;
    }

    return Status::OK;
}

Status MipiPackNeon::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return ret;
    }

    WorkerPool *wp = m_ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height = dst->GetSizes().m_height;

    ret = wp->ParallelFor(0, height, MipiPackNeonImpl, std::cref(*src), std::ref(*dst));

    AURA_RETURN(m_ctx, ret);
}

static Status MipiUnpackU8NeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    int8x8_t vds8_idx_x0 = neon::vload1(tbl_buffer0);
    int8x8_t vds8_idx_x1 = neon::vload1(tbl_buffer1);
    int8x8_t vds8_idx_x2 = neon::vload1(tbl_buffer2);
    int8x8_t vds8_idx_x3 = neon::vload1(tbl_buffer3);

    DT_S32 width = dst.GetSizes().m_width;
    DT_S32 width_align = width & (-32);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U8       *dst_row = dst.Ptr<DT_U8>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 32)
        {
            uint8x8x2_t v2du8_table_vec;
            uint8x16_t  vqu8_dst_x0;
            uint8x16_t  vqu8_dst_x1;

            uint8x8_t vdu8_src_x0 = neon::vload1(src_row);
            uint8x8_t vdu8_src_x1 = neon::vload1(src_row + 8);
            uint8x8_t vdu8_src_x2 = neon::vload1(src_row + 16);
            uint8x8_t vdu8_src_x3 = neon::vload1(src_row + 24);
            uint8x8_t vdu8_src_x4 = neon::vload1(src_row + 32);

            v2du8_table_vec.val[0] = vdu8_src_x0;
            v2du8_table_vec.val[1] = vdu8_src_x1;
            uint8x8_t vdu8_tbl_lo  = neon::vtbl(v2du8_table_vec, vds8_idx_x0);

            v2du8_table_vec.val[0] = vdu8_src_x1;
            v2du8_table_vec.val[1] = vdu8_src_x2;
            uint8x8_t vdu8_tbl_hi  = neon::vtbl(v2du8_table_vec, vds8_idx_x1);

            vqu8_dst_x0 = neon::vcombine(vdu8_tbl_lo, vdu8_tbl_hi);

            v2du8_table_vec.val[0] = vdu8_src_x2;
            v2du8_table_vec.val[1] = vdu8_src_x3;
            vdu8_tbl_lo = neon::vtbl(v2du8_table_vec, vds8_idx_x2);

            v2du8_table_vec.val[0] = vdu8_src_x3;
            v2du8_table_vec.val[1] = vdu8_src_x4;
            vdu8_tbl_hi = neon::vtbl(v2du8_table_vec, vds8_idx_x3);

            vqu8_dst_x1 = neon::vcombine(vdu8_tbl_lo, vdu8_tbl_hi);

            neon::vstore(dst_row, vqu8_dst_x0);
            neon::vstore(dst_row + 16, vqu8_dst_x1);

            src_row += 40;
            dst_row += 32;
        }

        for (; x < width; x += 4)
        {
            *(dst_row)     = *(src_row);
            *(dst_row + 1) = *(src_row + 1);
            *(dst_row + 2) = *(src_row + 2);
            *(dst_row + 3) = *(src_row + 3);

            src_row += 5;
            dst_row += 4;
        }
    }

    return Status::OK;;
}

static Status MipiUnpackU8Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 height = dst.GetSizes().m_height;

    Status ret = Status::ERROR;
    ret = wp->ParallelFor(0, height, MipiUnpackU8NeonImpl, std::cref(src), std::ref(dst));

    AURA_RETURN(ctx, ret);
}

static Status MipiUnpackU16NeonImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width       = dst.GetSizes().m_width;
    DT_S32 width_align = width & (-32);

    int8x8_t  vds8_idx_x0  = neon::vload1(tbl_buffer0);
    int8x8_t  vds8_idx_x1  = neon::vload1(tbl_buffer1);
    int8x8_t  vds8_idx_x2  = neon::vload1(tbl_buffer2);
    int8x8_t  vds8_idx_x3  = neon::vload1(tbl_buffer3);
    uint8x8_t vdu8_operand = neon::vload1(operand_buffer);
    uint8x8_t vdu8_shift   = neon::vload1(unpack_shift_buffer);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        DT_U16      *dst_row = dst.Ptr<DT_U16>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += 32)
        {
            uint8x8x2_t v2du8_table_vec;
            uint8x8_t vdu8_src_x0 = neon::vload1(src_row);
            uint8x8_t vdu8_src_x1 = neon::vload1(src_row + 8);
            uint8x8_t vdu8_src_x2 = neon::vload1(src_row + 16);
            uint8x8_t vdu8_src_x3 = neon::vload1(src_row + 24);
            uint8x8_t vdu8_src_x4 = neon::vload1(src_row + 32);

            //1st 8 output
            v2du8_table_vec.val[0] = vdu8_src_x0;
            v2du8_table_vec.val[1] = vdu8_src_x1;
            uint8x8_t vdu8_top8 = neon::vtbl(v2du8_table_vec, vds8_idx_x0);

            uint8x8_t vdu8_btm2_lo, vdu8_btm2_hi;
            neon::vdup(vdu8_btm2_lo, neon::vgetlane<4>(vdu8_src_x0));
            neon::vdup(vdu8_btm2_hi, neon::vgetlane<1>(vdu8_src_x1));
            uint8x8_t vdu8_btm2 = neon::vext<4>(vdu8_btm2_lo, vdu8_btm2_hi);
            vdu8_btm2 = neon::vand(vdu8_btm2, vdu8_operand);
            vdu8_btm2 = neon::vshl(vdu8_btm2, vdu8_shift);

            uint16x8_t vqu16_result_x0 = neon::vshll_n<2>(vdu8_top8);
            vqu16_result_x0 = neon::vaddw(vqu16_result_x0, vdu8_btm2);

            //2nd 8 output
            v2du8_table_vec.val[0] = vdu8_src_x1;
            v2du8_table_vec.val[1] = vdu8_src_x2;
            vdu8_top8 = neon::vtbl(v2du8_table_vec, vds8_idx_x1);

            neon::vdup(vdu8_btm2_lo, neon::vgetlane<6>(vdu8_src_x1));
            neon::vdup(vdu8_btm2_hi, neon::vgetlane<3>(vdu8_src_x2));
            vdu8_btm2 = neon::vext<4>(vdu8_btm2_lo, vdu8_btm2_hi);
            vdu8_btm2 = neon::vand(vdu8_btm2, vdu8_operand);
            vdu8_btm2 = neon::vshl(vdu8_btm2, vdu8_shift);

            uint16x8_t vqu16_result_x1 = neon::vshll_n<2>(vdu8_top8);
            vqu16_result_x1 = neon::vaddw(vqu16_result_x1, vdu8_btm2);

            //3rd 8 output
            v2du8_table_vec.val[0] = vdu8_src_x2;
            v2du8_table_vec.val[1] = vdu8_src_x3;
            vdu8_top8 = neon::vtbl(v2du8_table_vec, vds8_idx_x2);

            neon::vdup(vdu8_btm2_lo, neon::vgetlane<0>(vdu8_src_x3));
            neon::vdup(vdu8_btm2_hi, neon::vgetlane<5>(vdu8_src_x3));
            vdu8_btm2 = neon::vext<4>(vdu8_btm2_lo, vdu8_btm2_hi);
            vdu8_btm2 = neon::vand(vdu8_btm2, vdu8_operand);
            vdu8_btm2 = neon::vshl(vdu8_btm2, vdu8_shift);

            uint16x8_t vqu16_result_x2 = neon::vshll_n<2>(vdu8_top8);
            vqu16_result_x2 = neon::vaddw(vqu16_result_x2, vdu8_btm2);

            //4th 8 output
            v2du8_table_vec.val[0] = vdu8_src_x3;
            v2du8_table_vec.val[1] = vdu8_src_x4;
            vdu8_top8 = neon::vtbl(v2du8_table_vec, vds8_idx_x3);

            neon::vdup(vdu8_btm2_lo, neon::vgetlane<2>(vdu8_src_x4));
            neon::vdup(vdu8_btm2_hi, neon::vgetlane<7>(vdu8_src_x4));
            vdu8_btm2 = neon::vext<4>(vdu8_btm2_lo, vdu8_btm2_hi);
            vdu8_btm2 = neon::vand(vdu8_btm2, vdu8_operand);
            vdu8_btm2 = neon::vshl(vdu8_btm2, vdu8_shift);

            uint16x8_t vqu16_result_x3 = neon::vshll_n<2>(vdu8_top8);
            vqu16_result_x3 = neon::vaddw(vqu16_result_x3, vdu8_btm2);

            neon::vstore(dst_row, vqu16_result_x0);
            neon::vstore(dst_row + 8, vqu16_result_x1);
            neon::vstore(dst_row + 16, vqu16_result_x2);
            neon::vstore(dst_row + 24, vqu16_result_x3);

            src_row += 40;
            dst_row += 32;
        }

        for (; x < width; x += 4)
        {
            DT_U16 data0 = *(src_row + 0);
            DT_U16 data1 = *(src_row + 1);
            DT_U16 data2 = *(src_row + 2);
            DT_U16 data3 = *(src_row + 3);
            DT_U16 data4 = *(src_row + 4);

            *(dst_row    ) = (data0 << 2) + (data4 & 0x03);
            *(dst_row + 1) = (data1 << 2) + ((data4 & 0x0C) >> 2);
            *(dst_row + 2) = (data2 << 2) + ((data4 & 0x30) >> 4);
            *(dst_row + 3) = (data3 << 2) + ((data4 & 0xC0) >> 6);

            src_row += 5;
            dst_row += 4;
        }
    }

    return Status::OK;
}

static Status MipiUnpackU16Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 oheight = dst.GetSizes().m_height;

    Status ret = Status::ERROR;
    ret = wp->ParallelFor(0, oheight, MipiUnpackU16NeonImpl, std::cref(src), std::ref(dst));

    AURA_RETURN(ctx, ret);
}

MipiUnPackNeon::MipiUnPackNeon(Context *ctx, const OpTarget &target) : MipiUnPackImpl(ctx, target)
{}

Status MipiUnPackNeon::SetArgs(const Array *src, Array *dst)
{
    if (MipiUnPackImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MipiUnPackImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MipiUnPackNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (dst->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MipiUnpackU8Neon(m_ctx, *src, *dst, m_target);
            break;
        }

        case ElemType::U16:
        {
            ret = MipiUnpackU16Neon(m_ctx, *src, *dst, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
