#include "aura/ops/misc/mipi.hpp"
#include "mipi_impl.hpp"
#include "misc_comm.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static const MI_U8 ctrl_perm_2mipi[] __attribute__((aligned(128))) = 
{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
    0x01,0x03,0x04,0x06,0x0B,0x09,0x0E,0x0C,0x15,0x17,0x10,0x12,0x1F,0x1D,0x1A,0x18,
    0x09,0x0B,0x0C,0x0E,0x23,0x21,0x26,0x24,0x3D,0x3F,0x38,0x3A,0x37,0x35,0x32,0x30,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,

    0x00,0x00,0x00,0x00,0x01,0x03,0x01,0x0B,0x03,0x03,0x06,0x06,0x02,0x12,0x17,0x1F,
    0x07,0x05,0x07,0x05,0x0C,0x0C,0x0C,0x0C,0x05,0x07,0x25,0x23,0x2F,0x3F,0x3E,0x3A,
    0x0E,0x0A,0x0B,0x0B,0x0F,0x09,0x0B,0x19,0x18,0x18,0x18,0x18,0x19,0x1B,0x19,0x1B,
    0x0B,0x0B,0x0E,0x0E,0x0A,0x02,0x07,0x17,0x1F,0x1D,0x3F,0x3D,0x3C,0x34,0x34,0x34,
    0x1D,0x1F,0x15,0x13,0x17,0x17,0x16,0x12,0x1E,0x1A,0x13,0x13,0x17,0x11,0x13,0x11,
    0x10,0x10,0x10,0x10,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,
};

AURA_ALWAYS_INLINE AURA_VOID ProcessRawFrames(HVX_Vector &vu16_src, HVX_Vector &vu8_dst)
{
    HVX_Vector vu16_zero  = Q6_V_vsplat_R(0x0);
    HVX_Vector vu16_const = Q6_V_vsplat_R(0x30003);

    HVX_Vector *ctrl_data = (HVX_Vector *)ctrl_perm_2mipi;
    HVX_Vector vu8_ctrl0  = ctrl_data[0];
    HVX_Vector vu8_ctrl1  = ctrl_data[1];

    HVX_Vector vu8_pack, vu8_res0, vu8_res1;
    HVX_Vector vu16_shift, vu16_and, vu32_sum;

    vu16_shift = Q6_Vh_vasr_VhR(vu16_src, 0x2);
    vu16_and   = Q6_V_vand_VV(vu16_src, vu16_const);

    vu8_pack = Q6_Vb_vpacke_VhVh(vu16_zero, vu16_and);
    vu32_sum = Q6_Vuw_vrmpy_VubRub(vu8_pack, 0x40100401);
    vu8_res0 = Q6_Vb_vpacke_VhVh(vu32_sum, vu16_shift);

    vu8_res1 = Q6_V_vrdelta_VV(vu8_res0, vu8_ctrl0);
    vu8_dst  = Q6_V_vdelta_VV(vu8_res1, vu8_ctrl1);
}

AURA_ALWAYS_INLINE AURA_VOID PackData(HVX_Vector &vu8_src0, HVX_Vector &vu8_src1, HVX_Vector &vu8_src2, HVX_Vector &vu8_src3,
                                    HVX_Vector &vu8_src4, HVX_Vector &vu8_src5, HVX_Vector &vu8_src6, HVX_Vector &vu8_src7,
                                    HVX_Vector &vu8_dst0, HVX_Vector &vu8_dst1, HVX_Vector &vu8_dst2,
                                    HVX_Vector &vu8_dst3, HVX_Vector &vu8_dst4)
{
    HVX_Vector vu8_zero, vu8_res0, vu8_res1, vu8_res2, vu8_res3;

    vu8_zero = Q6_V_vsplat_R(0x0);
    vu8_res0 = Q6_V_valign_VVR(vu8_src1, vu8_zero, 0x30);

    vu8_dst0 = Q6_V_vor_VV(vu8_src0, vu8_res0);

    vu8_res0 = Q6_V_valign_VVR(vu8_src3, vu8_zero, 0x30);
    vu8_res1 = Q6_V_vor_VV(vu8_src2, vu8_res0);
    vu8_res3 = Q6_V_valign_VVR(vu8_zero, vu8_src1, 0x30);
    vu8_res2 = Q6_V_valign_VVR(vu8_res1, vu8_zero, 0x60);

    vu8_dst1 = Q6_V_vor_VV(vu8_res3, vu8_res2);

    vu8_res3 = Q6_V_valign_VVR(vu8_zero, vu8_src3, 0x10);
    vu8_res2 = Q6_V_valign_VVR(vu8_src4, vu8_zero, 0x40);

    vu8_dst2 = Q6_V_vor_VV(vu8_res3, vu8_res2);

    vu8_res0 = Q6_V_valign_VVR(vu8_src6, vu8_zero, 0x30);
    vu8_res1 = Q6_V_vor_VV(vu8_src5, vu8_res0);
    vu8_res3 = Q6_V_valign_VVR(vu8_zero, vu8_src4, 0x40);
    vu8_res2 = Q6_V_valign_VVR(vu8_res1, vu8_zero, 0x70);

    vu8_dst3 = Q6_V_vor_VV(vu8_res3, vu8_res2);

    vu8_res3 = Q6_V_valign_VVR(vu8_zero, vu8_src6, 0x20);
    vu8_res2 = Q6_V_valign_VVR(vu8_src7, vu8_zero, 0x50);

    vu8_dst4 = Q6_V_vor_VV(vu8_res3, vu8_res2);
}

AURA_ALWAYS_INLINE AURA_VOID MipiPackRowCore(MI_U8 *src, MI_U8 *dst)
{
    HVX_Vector vu16_src0, vu16_src1, vu16_src2, vu16_src3;
    HVX_Vector vu16_src4, vu16_src5, vu16_src6, vu16_src7;

    HVX_Vector vu8_out0, vu8_out1, vu8_out2, vu8_out3;
    HVX_Vector vu8_out4, vu8_out5, vu8_out6, vu8_out7;

    HVX_Vector vu8_dst0, vu8_dst1, vu8_dst2, vu8_dst3, vu8_dst4;

    vload(src                 , vu16_src0);
    vload(src + AURA_HVLEN * 1, vu16_src1);
    vload(src + AURA_HVLEN * 2, vu16_src2);
    vload(src + AURA_HVLEN * 3, vu16_src3);
    vload(src + AURA_HVLEN * 4, vu16_src4);
    vload(src + AURA_HVLEN * 5, vu16_src5);
    vload(src + AURA_HVLEN * 6, vu16_src6);
    vload(src + AURA_HVLEN * 7, vu16_src7);

    ProcessRawFrames(vu16_src0, vu8_out0);
    ProcessRawFrames(vu16_src1, vu8_out1);
    ProcessRawFrames(vu16_src2, vu8_out2);
    ProcessRawFrames(vu16_src3, vu8_out3);
    ProcessRawFrames(vu16_src4, vu8_out4);
    ProcessRawFrames(vu16_src5, vu8_out5);
    ProcessRawFrames(vu16_src6, vu8_out6);
    ProcessRawFrames(vu16_src7, vu8_out7);

    PackData(vu8_out0, vu8_out1, vu8_out2, vu8_out3, vu8_out4, vu8_out5, vu8_out6, vu8_out7, 
             vu8_dst0, vu8_dst1, vu8_dst2, vu8_dst3, vu8_dst4);

    vstore(dst                 , vu8_dst0);
    vstore(dst + AURA_HVLEN    , vu8_dst1);
    vstore(dst + AURA_HVLEN * 2, vu8_dst2);
    vstore(dst + AURA_HVLEN * 3, vu8_dst3);
    vstore(dst + AURA_HVLEN * 4, vu8_dst4);
}

static AURA_VOID MipiPackRow(const MI_U16 *src, MI_U8 *dst, MI_S32 iwidth, MI_S32 owidth)
{
    MI_U8 *src_u8 = reinterpret_cast<MI_U8 *>(const_cast<MI_U16 *>(src));
    MI_U8 *dst_u8 = (MI_U8 *)dst;

    MI_S32 loop = iwidth / (AURA_HALF_HVLEN * 8);
    MI_S32 tail = iwidth % (AURA_HALF_HVLEN * 8);

    for (MI_S32 index_loop = 0; index_loop < loop; index_loop++)
    {
        MipiPackRowCore(src_u8, dst_u8);

        src_u8 += 8 * AURA_HVLEN;
        dst_u8 += 5 * AURA_HVLEN;
    }
    // remain
    if (tail)
    {
        src_u8 = reinterpret_cast<MI_U8 *>(const_cast<MI_U16 *>(src + iwidth - AURA_HVLEN * 4));
        dst_u8 = dst + owidth - AURA_HVLEN * 5;

        MipiPackRowCore(src_u8, dst_u8);
    }
}

static Status MipiPackHvxImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;
    MI_S32 owidth = dst.GetSizes().m_width;

    MI_U64 L2fetch_param = L2PfParam(istride, iwidth * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U16>(y + 1)), L2fetch_param);
        }

        const MI_U16 *src_row = src.Ptr<MI_U16>(y);
        MI_U8 *dst_row = dst.Ptr<MI_U8>(y);
        MipiPackRow(src_row, dst_row, iwidth, owidth);
    }

    return Status::OK;
}

MipiPackHvx::MipiPackHvx(Context *ctx, const OpTarget &target) : MipiPackImpl(ctx, target)
{}

Status MipiPackHvx::SetArgs(const Array *src, Array *dst)
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

Status MipiPackHvx::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return ret;
    }

    switch (src->GetElemType())
    {
        case ElemType::U16:
        {
            MI_S32 height = src->GetSizes().m_height;

            WorkerPool *wp = m_ctx->GetWorkerPool();
            if (MI_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "GetWorkpool failed");
                break;
            }

            ret = wp->ParallelFor((MI_S32)0, height, MipiPackHvxImpl, std::cref(*src), std::ref(*dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MipiPackHvxImpl failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MipiPackHvx::ToString() const
{
    return MipiPackImpl::ToString() + m_profiling_string;
}

Status MipiPackRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;

    MipiPackInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    MipiPack mipi_pack(ctx, OpTarget::Hvx());

    return OpCall(ctx, mipi_pack, &src, &dst);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MISC_PACKAGE_NAME, AURA_OPS_MISC_MIPIPACK_OP_NAME, MipiPackRpc);

} // namespace aura