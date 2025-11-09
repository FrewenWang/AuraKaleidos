#include "mipi_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status MipiPackNoneImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U16 *src_row = src.Ptr<DT_U16>(y);
        DT_U8        *dst_row = dst.Ptr<DT_U8>(y);

        for (DT_S32 x = 0; x < width; x += 5)
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

MipiPackNone::MipiPackNone(Context *ctx, const OpTarget &target) : MipiPackImpl(ctx, target)
{}

Status MipiPackNone::SetArgs(const Array *src, Array *dst)
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

Status MipiPackNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return ret;
    }

    DT_S32 height = dst->GetSizes().m_height;

    if (m_target.m_data.none.enable_mt)
    {
        WorkerPool *wp = m_ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");
            return ret;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MipiPackNoneImpl, *src, *dst);
    }
    else
    {
        ret = MipiPackNoneImpl(*src, *dst, 0, height);
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
static Status MipiUnpackNoneImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp       *dst_row = dst.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < width; x += 4)
        {
            *(dst_row    ) = *(src_row);
            *(dst_row + 1) = *(src_row + 1);
            *(dst_row + 2) = *(src_row + 2);
            *(dst_row + 3) = *(src_row + 3);

            src_row += 5;
            dst_row += 4;
        }
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
static Status MipiUnpackNoneImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *src_row = src.Ptr<DT_U8>(y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < width; x += 4)
        {
            Tp data0 = *(src_row + 0);
            Tp data1 = *(src_row + 1);
            Tp data2 = *(src_row + 2);
            Tp data3 = *(src_row + 3);
            Tp data4 = *(src_row + 4);

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

MipiUnPackNone::MipiUnPackNone(Context *ctx, const OpTarget &target) : MipiUnPackImpl(ctx, target)
{}

Status MipiUnPackNone::SetArgs(const Array *src, Array *dst)
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

Status MipiUnPackNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return ret;
    }

    DT_S32 height = dst->GetSizes().m_height;

#define UNPACK_NONE_IMPL(type)                                                                          \
    if (m_target.m_data.none.enable_mt)                                                                 \
    {                                                                                                   \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                        \
        if (DT_NULL == wp)                                                                              \
        {                                                                                               \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                       \
            return Status::ERROR;                                                                       \
        }                                                                                               \
                                                                                                        \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MipiUnpackNoneImpl<type>, *src, *dst);    \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
        ret = MipiUnpackNoneImpl<type>(*src, *dst, 0, height);                                          \
    }                                                                                                   \
    if (ret != Status::OK)                                                                              \
    {                                                                                                   \
        DT_CHAR error_msg[128];                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "MipiUnpackNoneImpl<%s> failed", #type);            \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                        \
    }

    switch (dst->GetElemType())
    {
        case ElemType::U8:
        {
            UNPACK_NONE_IMPL(DT_U8);
            break;
        }

        case ElemType::U16:
        {
            UNPACK_NONE_IMPL(DT_U16);
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