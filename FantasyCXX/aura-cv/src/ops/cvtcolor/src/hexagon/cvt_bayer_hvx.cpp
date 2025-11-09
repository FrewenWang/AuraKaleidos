#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#include <vector>

namespace aura
{

AURA_ALWAYS_INLINE HVX_Vector CvtMeanX4U8RndSat(const HVX_Vector &vu8_vec0, const HVX_Vector &vu8_vec1, const HVX_Vector &vu8_vec2, const HVX_Vector &vu8_vec3)
{
    HVX_VectorPair ws16_sum0 = Q6_Wh_vadd_VubVub(vu8_vec0, vu8_vec1);
    HVX_VectorPair ws16_sum1 = Q6_Wh_vadd_VubVub(vu8_vec2, vu8_vec3);
    HVX_VectorPair ws16_sum  = Q6_Wh_vadd_WhWh(ws16_sum0, ws16_sum1);

    return Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(ws16_sum), Q6_V_lo_W(ws16_sum), 2);
}

AURA_ALWAYS_INLINE HVX_Vector CvtMeanX4U16RndSat(const HVX_Vector &vu16_vec0, const HVX_Vector &vu16_vec1, const HVX_Vector &vu16_vec2, const HVX_Vector &vu16_vec3)
{
    HVX_VectorPair ws16_sum0 = Q6_Ww_vadd_VuhVuh(vu16_vec0, vu16_vec1);
    HVX_VectorPair ws16_sum1 = Q6_Ww_vadd_VuhVuh(vu16_vec2, vu16_vec3);
    HVX_VectorPair ws16_sum  = Q6_Ww_vadd_WwWw(ws16_sum0, ws16_sum1);

    return Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws16_sum), Q6_V_lo_W(ws16_sum), 2);
}

AURA_ALWAYS_INLINE DT_VOID CvtShuffVectorX3(HVX_VectorX3 &v3_vec0, HVX_VectorX3 &v3_vec1, DT_S32 rt)
{
    HVX_VectorPair w_val0 = Q6_W_vshuff_VVR(v3_vec1.val[0], v3_vec0.val[0], rt);
    HVX_VectorPair w_val1 = Q6_W_vshuff_VVR(v3_vec1.val[1], v3_vec0.val[1], rt);
    HVX_VectorPair w_val2 = Q6_W_vshuff_VVR(v3_vec1.val[2], v3_vec0.val[2], rt);

    v3_vec0.val[0] = Q6_V_lo_W(w_val0);
    v3_vec0.val[1] = Q6_V_lo_W(w_val1);
    v3_vec0.val[2] = Q6_V_lo_W(w_val2);

    v3_vec1.val[0] = Q6_V_hi_W(w_val0);
    v3_vec1.val[1] = Q6_V_hi_W(w_val1);
    v3_vec1.val[2] = Q6_V_hi_W(w_val2);
}

AURA_ALWAYS_INLINE DT_VOID CvtSplatVectorX2(const DT_U8 *src_p, const DT_U8 *src_c, const DT_U8 *src_n0, const DT_U8 *src_n1, DT_S32 offset,
                                            HVX_VectorX2 &v2u8_src_p, HVX_VectorX2 &v2u8_src_c, HVX_VectorX2 &v2u8_src_n0, HVX_VectorX2 &v2u8_src_n1)
{
    v2u8_src_p.val[0]  = Q6_Vb_vsplat_R(src_p[offset]);
    v2u8_src_c.val[0]  = Q6_Vb_vsplat_R(src_c[offset]);
    v2u8_src_n0.val[0] = Q6_Vb_vsplat_R(src_n0[offset]);
    v2u8_src_n1.val[0] = Q6_Vb_vsplat_R(src_n1[offset]);

    v2u8_src_p.val[1]  = Q6_Vb_vsplat_R(src_p[offset + 1]);
    v2u8_src_c.val[1]  = Q6_Vb_vsplat_R(src_c[offset + 1]);
    v2u8_src_n0.val[1] = Q6_Vb_vsplat_R(src_n0[offset + 1]);
    v2u8_src_n1.val[1] = Q6_Vb_vsplat_R(src_n1[offset + 1]);
}

AURA_ALWAYS_INLINE DT_VOID CvtSplatVectorX2(const DT_U16 *src_p, const DT_U16 *src_c, const DT_U16 *src_n0, const DT_U16 *src_n1, DT_S32 offset,
                                            HVX_VectorX2 &v2u8_src_p, HVX_VectorX2 &v2u8_src_c, HVX_VectorX2 &v2u8_src_n0, HVX_VectorX2 &v2u8_src_n1)
{
    v2u8_src_p.val[0]  = Q6_Vh_vsplat_R(src_p[offset]);
    v2u8_src_c.val[0]  = Q6_Vh_vsplat_R(src_c[offset]);
    v2u8_src_n0.val[0] = Q6_Vh_vsplat_R(src_n0[offset]);
    v2u8_src_n1.val[0] = Q6_Vh_vsplat_R(src_n1[offset]);

    v2u8_src_p.val[1]  = Q6_Vh_vsplat_R(src_p[offset + 1]);
    v2u8_src_c.val[1]  = Q6_Vh_vsplat_R(src_c[offset + 1]);
    v2u8_src_n0.val[1] = Q6_Vh_vsplat_R(src_n0[offset + 1]);
    v2u8_src_n1.val[1] = Q6_Vh_vsplat_R(src_n1[offset + 1]);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtBayer2BgrCore(const HVX_VectorX2 &v2u8_src_p_x0, const HVX_VectorX2 &v2u8_src_c_x0, const HVX_VectorX2 &v2u8_src_n0_x0, const HVX_VectorX2 &v2u8_src_n1_x0,
                                            const HVX_VectorX2 &v2u8_src_p_x1, const HVX_VectorX2 &v2u8_src_c_x1, const HVX_VectorX2 &v2u8_src_n0_x1, const HVX_VectorX2 &v2u8_src_n1_x1,
                                            HVX_VectorX3 &v2u8_dst_c_x0, HVX_VectorX3 &v2u8_dst_c_x1, HVX_VectorX3 &v2u8_dst_n_x0, HVX_VectorX3 &v2u8_dst_n_x1,
                                            DT_S32 b_idx, DT_S32 r_idx)
{
    HVX_VectorX2 v2u8_src_p_c, v2u8_src_c_c, v2u8_src_n0_c, v2u8_src_n1_c;
    v2u8_src_p_c.val[0]  = Q6_V_valign_VVR(v2u8_src_p_x1.val[0], v2u8_src_p_x0.val[0], 1);
    v2u8_src_c_c.val[0]  = Q6_V_valign_VVR(v2u8_src_c_x1.val[0], v2u8_src_c_x0.val[0], 1);
    v2u8_src_n0_c.val[0] = Q6_V_valign_VVR(v2u8_src_n0_x1.val[0], v2u8_src_n0_x0.val[0], 1);
    v2u8_src_n1_c.val[0] = Q6_V_valign_VVR(v2u8_src_n1_x1.val[0], v2u8_src_n1_x0.val[0], 1);

    v2u8_src_p_c.val[1]  = Q6_V_valign_VVR(v2u8_src_p_x1.val[1], v2u8_src_p_x0.val[1], 1);
    v2u8_src_c_c.val[1]  = Q6_V_valign_VVR(v2u8_src_c_x1.val[1], v2u8_src_c_x0.val[1], 1);
    v2u8_src_n0_c.val[1] = Q6_V_valign_VVR(v2u8_src_n0_x1.val[1], v2u8_src_n0_x0.val[1], 1);
    v2u8_src_n1_c.val[1] = Q6_V_valign_VVR(v2u8_src_n1_x1.val[1], v2u8_src_n1_x0.val[1], 1);

    v2u8_dst_c_x0.val[b_idx] = Q6_Vub_vavg_VubVub_rnd(v2u8_src_c_x0.val[0], v2u8_src_c_c.val[0]);
    v2u8_dst_c_x0.val[1]     = v2u8_src_c_x0.val[1];
    v2u8_dst_c_x0.val[r_idx] = Q6_Vub_vavg_VubVub_rnd(v2u8_src_p_x0.val[1], v2u8_src_n0_x0.val[1]);

    v2u8_dst_c_x1.val[b_idx] = v2u8_src_c_c.val[0];
    v2u8_dst_c_x1.val[1]     = CvtMeanX4U8RndSat(v2u8_src_p_c.val[0], v2u8_src_c_x0.val[1], v2u8_src_c_c.val[1], v2u8_src_n0_c.val[0]);
    v2u8_dst_c_x1.val[r_idx] = CvtMeanX4U8RndSat(v2u8_src_p_x0.val[1], v2u8_src_p_c.val[1], v2u8_src_n0_x0.val[1], v2u8_src_n0_c.val[1]);

    v2u8_dst_n_x0.val[b_idx] = CvtMeanX4U8RndSat(v2u8_src_c_x0.val[0], v2u8_src_c_c.val[0], v2u8_src_n1_x0.val[0], v2u8_src_n1_c.val[0]);
    v2u8_dst_n_x0.val[1]     = CvtMeanX4U8RndSat(v2u8_src_c_x0.val[1], v2u8_src_n0_x0.val[0], v2u8_src_n0_c.val[0], v2u8_src_n1_x0.val[1]);
    v2u8_dst_n_x0.val[r_idx] = v2u8_src_n0_x0.val[1];

    v2u8_dst_n_x1.val[b_idx] = Q6_Vub_vavg_VubVub_rnd(v2u8_src_c_c.val[0], v2u8_src_n1_c.val[0]);
    v2u8_dst_n_x1.val[1]     = v2u8_src_n0_c.val[0];
    v2u8_dst_n_x1.val[r_idx] = Q6_Vub_vavg_VubVub_rnd(v2u8_src_n0_x0.val[1], v2u8_src_n0_c.val[1]);

    CvtShuffVectorX3(v2u8_dst_c_x0, v2u8_dst_c_x1, -1);
    CvtShuffVectorX3(v2u8_dst_n_x0, v2u8_dst_n_x1, -1);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID CvtBayer2BgrCore(const HVX_VectorX2 &v2u16_src_p_x0, const HVX_VectorX2 &v2u16_src_c_x0, const HVX_VectorX2 &v2u16_src_n0_x0, const HVX_VectorX2 &v2u16_src_n1_x0,
                                            const HVX_VectorX2 &v2u16_src_p_x1, const HVX_VectorX2 &v2u16_src_c_x1, const HVX_VectorX2 &v2u16_src_n0_x1, const HVX_VectorX2 &v2u16_src_n1_x1,
                                            HVX_VectorX3 &v3u16_dst_c_x0, HVX_VectorX3 &v3u16_dst_c_x1, HVX_VectorX3 &v3u16_dst_n_x0, HVX_VectorX3 &v3u16_dst_n_x1,
                                            DT_S32 b_idx, DT_S32 r_idx)
{
    HVX_VectorX2 v2u16_src_p_c, v2u16_src_c_c, v2u16_src_n0_c, v2u16_src_n1_c;
    v2u16_src_p_c.val[0]  = Q6_V_valign_VVR(v2u16_src_p_x1.val[0], v2u16_src_p_x0.val[0], 2);
    v2u16_src_c_c.val[0]  = Q6_V_valign_VVR(v2u16_src_c_x1.val[0], v2u16_src_c_x0.val[0], 2);
    v2u16_src_n0_c.val[0] = Q6_V_valign_VVR(v2u16_src_n0_x1.val[0], v2u16_src_n0_x0.val[0], 2);
    v2u16_src_n1_c.val[0] = Q6_V_valign_VVR(v2u16_src_n1_x1.val[0], v2u16_src_n1_x0.val[0], 2);

    v2u16_src_p_c.val[1]  = Q6_V_valign_VVR(v2u16_src_p_x1.val[1], v2u16_src_p_x0.val[1], 2);
    v2u16_src_c_c.val[1]  = Q6_V_valign_VVR(v2u16_src_c_x1.val[1], v2u16_src_c_x0.val[1], 2);
    v2u16_src_n0_c.val[1] = Q6_V_valign_VVR(v2u16_src_n0_x1.val[1], v2u16_src_n0_x0.val[1], 2);
    v2u16_src_n1_c.val[1] = Q6_V_valign_VVR(v2u16_src_n1_x1.val[1], v2u16_src_n1_x0.val[1], 2);

    v3u16_dst_c_x0.val[b_idx] = Q6_Vuh_vavg_VuhVuh_rnd(v2u16_src_c_x0.val[0], v2u16_src_c_c.val[0]);
    v3u16_dst_c_x0.val[1]     = v2u16_src_c_x0.val[1];
    v3u16_dst_c_x0.val[r_idx] = Q6_Vuh_vavg_VuhVuh_rnd(v2u16_src_p_x0.val[1], v2u16_src_n0_x0.val[1]);

    v3u16_dst_c_x1.val[b_idx] = v2u16_src_c_c.val[0];
    v3u16_dst_c_x1.val[1]     = CvtMeanX4U16RndSat(v2u16_src_p_c.val[0], v2u16_src_c_x0.val[1], v2u16_src_c_c.val[1], v2u16_src_n0_c.val[0]);
    v3u16_dst_c_x1.val[r_idx] = CvtMeanX4U16RndSat(v2u16_src_p_x0.val[1], v2u16_src_p_c.val[1], v2u16_src_n0_x0.val[1], v2u16_src_n0_c.val[1]);

    v3u16_dst_n_x0.val[b_idx] = CvtMeanX4U16RndSat(v2u16_src_c_x0.val[0], v2u16_src_c_c.val[0], v2u16_src_n1_x0.val[0], v2u16_src_n1_c.val[0]);
    v3u16_dst_n_x0.val[1]     = CvtMeanX4U16RndSat(v2u16_src_c_x0.val[1], v2u16_src_n0_x0.val[0], v2u16_src_n0_c.val[0], v2u16_src_n1_x0.val[1]);
    v3u16_dst_n_x0.val[r_idx] = v2u16_src_n0_x0.val[1];

    v3u16_dst_n_x1.val[b_idx] = Q6_Vuh_vavg_VuhVuh_rnd(v2u16_src_c_c.val[0], v2u16_src_n1_c.val[0]);
    v3u16_dst_n_x1.val[1]     = v2u16_src_n0_c.val[0];
    v3u16_dst_n_x1.val[r_idx] = Q6_Vuh_vavg_VuhVuh_rnd(v2u16_src_n0_x0.val[1], v2u16_src_n0_c.val[1]);

    CvtShuffVectorX3(v3u16_dst_c_x0, v3u16_dst_c_x1, -2);
    CvtShuffVectorX3(v3u16_dst_n_x0, v3u16_dst_n_x1, -2);
}

template <typename Tp>
static Status CvtBayer2BgrHvxImpl(const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 ichannel = src.GetSizes().m_channel;
    DT_S32 istride  = src.GetStrides().m_width;

    const Tp *src_p  = DT_NULL;
    const Tp *src_c  = DT_NULL;
    const Tp *src_n0 = DT_NULL;
    const Tp *src_n1 = DT_NULL;

    Tp *dst_c = DT_NULL;
    Tp *dst_n = DT_NULL;

    DT_S32 offset = 3 * width;
    DT_S32 b_idx  = swapb ? 2 : 0;
    DT_S32 r_idx  = swapb ? 0 : 2;

    DT_S32 elem_counts = AURA_HVLEN / ElemTypeSize(dst.GetElemType());
    DT_S32 width_align = (width - 2) & (-(elem_counts * 2));

    HVX_VectorX2 v2_src_p_x0, v2_src_c_x0, v2_src_n0_x0, v2_src_n1_x0;
    HVX_VectorX2 v2_src_p_x1, v2_src_c_x1, v2_src_n0_x1, v2_src_n1_x1;
    HVX_VectorX3 v3_dst_c_x0, v3_dst_c_x1, v3_dst_n_x0, v3_dst_n_x1;

    DT_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 4, 0);

    for (DT_S32 y = start_row * 2; y < end_row * 2; y += 2)
    {
        if (y < (end_row * 2 - 2))
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(y + 2)), L2fetch_param);
        }

        if (swapg)
        {
            src_p  = src.Ptr<Tp>(y + 3);
            src_c  = src.Ptr<Tp>(y + 2);
            src_n0 = src.Ptr<Tp>(y + 1);
            src_n1 = src.Ptr<Tp>(y);
            dst_c  = dst.Ptr<Tp>(y + 2);
            dst_n  = dst.Ptr<Tp>(y + 1);
        }
        else
        {
            src_p  = src.Ptr<Tp>(y);
            src_c  = src.Ptr<Tp>(y + 1);
            src_n0 = src.Ptr<Tp>(y + 2);
            src_n1 = src.Ptr<Tp>(y + 3);
            dst_c  = dst.Ptr<Tp>(y + 1);
            dst_n  = dst.Ptr<Tp>(y + 2);
        }

        vload(src_p,  v2_src_p_x0);
        vload(src_c,  v2_src_c_x0);
        vload(src_n0, v2_src_n0_x0);
        vload(src_n1, v2_src_n1_x0);

        DT_S32 x = 0;
        for (; x < (width_align - (elem_counts * 2)); x += (elem_counts * 2))
        {
            vload(src_p + x + (elem_counts * 2),  v2_src_p_x1);
            vload(src_c + x + (elem_counts * 2),  v2_src_c_x1);
            vload(src_n0 + x + (elem_counts * 2), v2_src_n0_x1);
            vload(src_n1 + x + (elem_counts * 2), v2_src_n1_x1);

            CvtBayer2BgrCore<Tp>(v2_src_p_x0, v2_src_c_x0, v2_src_n0_x0, v2_src_n1_x0,
                                 v2_src_p_x1, v2_src_c_x1, v2_src_n0_x1, v2_src_n1_x1,
                                 v3_dst_c_x0, v3_dst_c_x1, v3_dst_n_x0, v3_dst_n_x1,
                                 b_idx, r_idx);

            vstore(dst_c + (x + 1) * 3,              v3_dst_c_x0);
            vstore(dst_c + (x + 1 + elem_counts) * 3, v3_dst_c_x1);
            vstore(dst_n + (x + 1) * 3,              v3_dst_n_x0);
            vstore(dst_n + (x + 1 + elem_counts) * 3, v3_dst_n_x1);

            v2_src_p_x0  = v2_src_p_x1;
            v2_src_c_x0  = v2_src_c_x1;
            v2_src_n0_x0 = v2_src_n0_x1;
            v2_src_n1_x0 = v2_src_n1_x1;
        }

        {
            x = width_align;
LOOP_BODY:
            CvtSplatVectorX2(src_p, src_c, src_n0, src_n1, x,
                             v2_src_p_x1, v2_src_c_x1, v2_src_n0_x1, v2_src_n1_x1);

            CvtBayer2BgrCore<Tp>(v2_src_p_x0, v2_src_c_x0, v2_src_n0_x0, v2_src_n1_x0,
                                 v2_src_p_x1, v2_src_c_x1, v2_src_n0_x1, v2_src_n1_x1,
                                 v3_dst_c_x0, v3_dst_c_x1, v3_dst_n_x0, v3_dst_n_x1,
                                 b_idx, r_idx);

            vstore(dst_c + (x + 1 - elem_counts * 2) * 3, v3_dst_c_x0);
            vstore(dst_c + (x + 1 - elem_counts) * 3,     v3_dst_c_x1);
            vstore(dst_n + (x + 1 - elem_counts * 2) * 3, v3_dst_n_x0);
            vstore(dst_n + (x + 1 - elem_counts) * 3,     v3_dst_n_x1);
        }

        if (x < (width - 2))
        {
            x = (width - 2);

            vload(src_p + x - (elem_counts * 2),  v2_src_p_x0);
            vload(src_c + x - (elem_counts * 2),  v2_src_c_x0);
            vload(src_n0 + x - (elem_counts * 2), v2_src_n0_x0);
            vload(src_n1 + x - (elem_counts * 2), v2_src_n1_x0);

            goto LOOP_BODY;
        }

        dst_c[0]          = dst_c[3];
        dst_c[1]          = dst_c[4];
        dst_c[2]          = dst_c[5];
        dst_c[offset - 3] = dst_c[offset - 6];
        dst_c[offset - 2] = dst_c[offset - 5];
        dst_c[offset - 1] = dst_c[offset - 4];

        dst_n[0]          = dst_n[3];
        dst_n[1]          = dst_n[4];
        dst_n[2]          = dst_n[5];
        dst_n[offset - 3] = dst_n[offset - 6];
        dst_n[offset - 2] = dst_n[offset - 5];
        dst_n[offset - 1] = dst_n[offset - 4];
    }

    return Status::OK;
}

static Status CvtBayer2BgrRemainHvxImpl(Mat &dst)
{
    DT_S32 height = dst.GetSizes().m_height;
    DT_S32 width  = dst.GetSizes().m_width;

    DT_VOID *dst_c = dst.Ptr<DT_VOID>(height - 1);
    DT_VOID *dst_n = dst.Ptr<DT_VOID>(height - 2);
    AuraMemCopy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    dst_c = dst.Ptr<DT_VOID>(0);
    dst_n = dst.Ptr<DT_VOID>(1);
    AuraMemCopy(dst_c, dst_n, 3 * width * ElemTypeSize(dst.GetElemType()));

    return Status::OK;
}

Status CvtBayer2BgrHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_height & 1 || src.GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst size only support even");
        return ret;
    }

    if (dst.GetSizes().m_height != src.GetSizes().m_height || dst.GetSizes().m_width != src.GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst must have the same height and width");
        return ret;
    }

    if (src.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src channel must be 1 and dst channel must be 3");
        return ret;
    }

    DT_S32 height = (src.GetSizes().m_height - 2) / 2;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBayer2BgrHvxImpl<DT_U8>, std::cref(src), std::ref(dst), swapb, swapg);
            break;
        }

        case ElemType::U16:
        {
            ret = wp->ParallelFor((DT_S32)0, height, CvtBayer2BgrHvxImpl<DT_U16>, std::cref(src), std::ref(dst), swapb, swapg);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    ret |= CvtBayer2BgrRemainHvxImpl(dst);

    AURA_RETURN(ctx, ret);
}

} // namespace aura