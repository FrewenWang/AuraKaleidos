/*
 * xm_cascade_utils_padding.c
 *
 *  Created on: Feb 5, 2024
 *      Author: zhonganyu
 */

#include <float.h>
#include <math.h>
#include <limits.h>
#include <xtensa/tie/xt_ivpn.h>

#include "tileManager.h"
#include "tileManager_api.h"
#include "xm_cascade_utils.h"

typedef xb_vecNx16 vsaN;
#define XVTM_IVP_SCATTERNX16T(a, b, c, d) IVP_SCATTERNX16T((a), (short int *)(b), (c), (d))
#define XVTM_IVP_SCATTERNX8UT(a, b, c, d) IVP_SCATTERNX8UT((a), (unsigned char *)(b), (c), (d))
#define XVTM_IVP_GATHERANX16T_V(a, b, c, d) IVP_GATHERANX16T_V((const short int *)(a), (b), (c), (d))
#define XVTM_IVP_GATHERANX8UT_V(a, b, c, d) IVP_GATHERANX8UT_V((const unsigned char *)(a), (b), (c), (d))
#define XVTM_IVP_SAVNX16POS_FP IVP_SAPOSNX16_FP
#define XVTM_IVP_MOVVSV(vr,sa) (vr) // sa is always zero in XI, if not zero -> use IVP_MOVVSELNX16
#define XVTM_OFFSET_PTR_2NX8U(  ptr, nrows, stride, in_row_offset) ((xb_vec2Nx8U*)   ((uint8_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))
#define XVTM_OFFSET_PTR_NX8U(   ptr, nrows, stride, in_row_offset) ((xb_vecNx8U*)    ((uint8_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))
#define XVTM_OFFSET_PTR_NX16(   ptr, nrows, stride, in_row_offset) ((xb_vecNx16*)    ((int16_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))

#define XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH  5
#define XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH 8
#define XVTM_IVP_SAV2NX8UPOS_FP IVP_SAPOS2NX8U_FP

static void xvGet2NX8SelVec(int32_t channel, int32_t wid, xb_vec2Nx8U &vu8_sel)
{
    xb_vec2Nx8U vu8_seq = IVP_SEQ2NX8U();
    xb_vecNx16 vs16_seq = IVP_SEQNX16();
    switch (channel)
    {
        case 1:
        {
            vu8_sel = IVP_SUB2NX8U(wid - 1, vu8_seq);
            break;
        }
        case 2:
        {
            vu8_sel = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, IVP_SLLI2NX8U(IVP_SRLI2NX8U(vu8_seq, 1), 1)), IVP_AND2NX8U(IVP_ADD2NX8U(vu8_seq, 1), 1));
            break;
        }
        case 3:
        {
            xb_vecNx16 vs16_thr       = IVP_MOVVA16(3);
            xb_vecNx16 vs16_rem_thr   = IVP_REMNX16(vs16_seq, vs16_thr);
            xb_vecNx16 vs16_div_thr   = IVP_QUONX16(vs16_seq, vs16_thr);
            xb_vec2Nx8U vu8_div_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_div_thr), IVP_MOV2NX8U_FROMNX16(vs16_div_thr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx24 vs24_mul_thr  = IVP_MULUS2NX8(3, vu8_div_thr);
            xb_vecNx16 vs16_mul_thr_l = IVP_CVT16U2NX24L(vs24_mul_thr);
            xb_vecNx16 vs16_mul_thr_h = IVP_CVT16U2NX24H(vs24_mul_thr);
            xb_vec2Nx8U vu8_mul_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_mul_thr_h), IVP_MOV2NX8U_FROMNX16(vs16_mul_thr_l), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx8U vu8_rem_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_rem_thr), IVP_MOV2NX8U_FROMNX16(vs16_rem_thr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx8U vu8_sub       = IVP_SUB2NX8U(2, vu8_rem_thr);
            vu8_sel                   = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, vu8_mul_thr), vu8_sub);
            break;
        }
        case 4:
        {
            // a = [0,0,0,0,4,4,4,4,8,8,8,8...] -> [3,3,3,3...], b = [0,1,2,3,4,5,6,7,8...] -> [3,2,1,0,3,2,1,0,3,2,1,0...],a-b=[0,1,2,3...]
            vu8_sel = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, IVP_SLLI2NX8U(IVP_SRLI2NX8U(vu8_seq, 2), 2)), IVP_SUB2NX8U(3, IVP_AND2NX8U(vu8_seq, 3)));
            break;
        }
        default:
        {
            return;
        }
    }
}

static void xvGetNX16SelVec(int32_t channel, int32_t wid, xb_vecNx16 &vs16_sel)
{
    xb_vecNx16 vs16_seq = IVP_SEQNX16();
    switch (channel)
    {
        case 1:
        {
            vs16_sel = IVP_SUBNX16(wid - 1, vs16_seq);
            break;
        }
        case 2:
        {
            vs16_sel = IVP_SUBNX16(IVP_SUBNX16(wid - 1, IVP_SLLINX16(IVP_SRLINX16(vs16_seq, 1), 1)), IVP_ANDNX16(IVP_ADDNX16(vs16_seq, 1), 1));
            break;
        }
        case 3:
        {
            xb_vecNx16 vs16_div_thr      = IVP_QUONX16(vs16_seq, IVP_MOVVA16(3));
            xb_vecNx48 vs48_mul_thr      = IVP_MULNX16(3, vs16_div_thr);
            xb_vecN_2x32v vs32_mul_thr_l = IVP_CVT32UNX48L(vs48_mul_thr);
            xb_vecN_2x32v vs32_mul_thr_h = IVP_CVT32UNX48H(vs48_mul_thr);
            xb_vecNx16 vs16_mul_thr      = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32(vs32_mul_thr_h), IVP_MOVNX16_FROMN_2X32(vs32_mul_thr_l), IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
            xb_vecNx16 vs16_sub          = IVP_SUBNX16(2, IVP_REMNX16(vs16_seq, IVP_MOVVA16(3)));
            vs16_sel                     = IVP_SUBNX16(IVP_SUBNX16(wid - 1, vs16_mul_thr), vs16_sub);
            break;
        }
        case 4:
        {
            // a = [0,0,0,0,4,4,4,4,8,8,8,8...] -> [3,3,3,3...], b = [0,1,2,3,4,5,6,7,8...] -> [3,2,1,0,3,2,1,0,3,2,1,0...],a-b=[0,1,2,3...]
            vs16_sel = IVP_SUBNX16(IVP_SUBNX16(wid - 1, IVP_SLLINX16(IVP_SRLINX16(vs16_seq, 2), 2)), IVP_SUBNX16(3, IVP_ANDNX16(vs16_seq, 3)));
            break;
        }
        default:
        {
            return;
        }
    }
}

static void xvExtendEdgesReflect101_I8(xvTile const *tile, int32_t frame_width, int32_t frame_height)
{
    int32_t channel = XV_FRAME_GET_NUM_CHANNELS(tile->pFrame);

    int32_t w = XV_TILE_GET_EDGE_WIDTH(tile) * channel;
    int32_t h = XV_TILE_GET_EDGE_HEIGHT(tile);

    uint8_t *__restrict src = (uint8_t *)XV_TILE_GET_DATA_PTR(tile);
    xb_vec2Nx8U *vpdst;

    int32_t stride = XV_TILE_GET_PITCH(tile);

    int32_t start_x = XV_TILE_GET_X_COORD(tile) * channel;
    int32_t start_y = XV_TILE_GET_Y_COORD(tile);

    int32_t W_Local = XV_TILE_GET_WIDTH(tile) * channel;
    int32_t H_Local = XV_TILE_GET_HEIGHT(tile);

    frame_width = frame_width * channel;

    // find intersection of tile/frame
    int32_t ixmin = XT_MAX(start_x - w, 0);
    int32_t ixmax = XT_MIN(start_x + W_Local + w - 1 * channel, frame_width - 1 * channel);
    int32_t iymin = XT_MAX(start_y - h, 0);
    int32_t iymax = XT_MIN(start_y + H_Local + h - 1, frame_height - 1);

    int32_t p0x = ixmin - start_x;
    int32_t ps0y = iymin - start_y;
    int32_t pd0y = iymin - start_y - 1;
    int32_t p0w  = (ixmax - ixmin) + 1 * channel;
    int32_t p0h  = iymin - (start_y - h);

    uint8_t *curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, ps0y, stride, p0x);
    uint8_t *dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, pd0y, stride, p0x);
    valign a_store = IVP_ZALIGN();
    valign a_load;

    int32_t p = XT_MAX(1, 2 * (iymax - iymin + 1) - 2);
    int32_t pmod = (p << 16) + 1;

    int32_t x = 0;
    if (p0h > 0)
    {
        for (; x < (p0w - (2 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
        {
            int32_t z = 0;
            for (int32_t y = 0; y < p0h; y++)
            {
                z = IVP_ADDMOD16U(z, pmod);
                int32_t k = XT_MIN(z, p - z);

                xb_vec2Nx8U color, color1;
                xb_vec2Nx8U *src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, k, stride, x);
                a_load = IVP_LA2NX8U_PP(src_1);
                IVP_LA2NX8U_IP(color, a_load, src_1);
                IVP_LAV2NX8U_XP(color1, a_load, src_1, sizeof(uint8_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

                xb_vec2Nx8U *vdst = XVTM_OFFSET_PTR_2NX8U(dst, -y, stride, x);
                IVP_SA2NX8U_IP(color, a_store, vdst);
                IVP_SAV2NX8U_XP(color1, a_store, vdst, sizeof(uint8_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
            }
        }
        if (x < p0w)
        {
            int32_t z = 0;
            for (int32_t y = 0; y < p0h; y++)
            {
                z = IVP_ADDMOD16U(z, pmod);
                int32_t k = XT_MIN(z, p - z);

                xb_vec2Nx8U color;
                xb_vec2Nx8U *src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, k, stride, x);
                a_load = IVP_LA2NX8U_PP(src_1);
                IVP_LAV2NX8U_XP(color, a_load, src_1, sizeof(uint8_t) * (p0w - x));

                xb_vec2Nx8U *vdst = XVTM_OFFSET_PTR_2NX8U(dst, -y, stride, x);
                IVP_SAV2NX8U_XP(color, a_store, vdst, sizeof(uint8_t) * (p0w - x));
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
            }
        }
    }

    int32_t p1x = ixmin - start_x;
    int32_t ps1y = iymax - start_y;
    int32_t pd1y = (iymax + 1) - start_y;
    int32_t p1w = (ixmax - ixmin) + 1 * channel;
    int32_t p1h = ( start_y + H_Local + h ) - 1 - iymax;

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, ps1y, stride, p1x);
    dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, pd1y, stride, p1x);
    x = 0;
    if (p1h > 0)
    {
        for (; x < (p1w - (2 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
        {
            int32_t z = 0;
            for (int32_t y = 0; y < p1h; y++)
            {
                z = IVP_ADDMOD16U(z, pmod);
                int32_t k = XT_MIN(z, p - z);

                xb_vec2Nx8U color, color1;
                xb_vec2Nx8U *src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, -k, stride, x);
                a_load = IVP_LA2NX8U_PP(src_1);
                IVP_LA2NX8U_IP(color, a_load, src_1);
                IVP_LAV2NX8U_XP(color1, a_load, src_1, sizeof(uint8_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

                xb_vec2Nx8U *vdst = XVTM_OFFSET_PTR_2NX8U(dst, y, stride, x);
                IVP_SA2NX8U_IP(color, a_store, vdst);
                IVP_SAV2NX8U_XP(color1, a_store, vdst, sizeof(uint8_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
            }
        }
        if (x < p1w)
        {
            int32_t z = 0;
            for (int32_t y = 0; y < p1h; y++)
            {
                z = IVP_ADDMOD16U(z, pmod);
                int32_t k = XT_MIN(z, p - z);

                xb_vec2Nx8U color;
                xb_vec2Nx8U *src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, -k, stride, x);
                a_load = IVP_LA2NX8U_PP(src_1);
                IVP_LAV2NX8U_XP(color, a_load, src_1, sizeof(uint8_t) * (p1w - x));

                xb_vec2Nx8U *vdst = XVTM_OFFSET_PTR_2NX8U(dst, y, stride, x);
                IVP_SAV2NX8U_XP(color, a_store, vdst, sizeof(uint8_t) * (p1w - x));
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
            }
        }
    }
    int32_t ps2x = XT_MIN(ixmin - start_x + 1 * channel, ixmax - start_x);
    int32_t pd2x = (ixmin - start_x);
    int32_t p2y = -h;
    int32_t p2w = ixmin - (start_x - w);
    int32_t p2h = H_Local + (2 * h);

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, ps2x);
    dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, pd2x);

    x = 0;

    while (x < p2w - XT_MIN(XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH, ixmax - ixmin))
    {
        int32_t wid;
        xb_vec2Nx8U color, color1, color2, color3;
        int32_t ytmp = 0;
        int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p2w-x);
        while ( xtmp < loop_width )
        {

            wid = XT_MIN(p2w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*2 * XCHAL_IVPN_SIMD_WIDTH, 2 * XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);
            xb_vec2Nx8U shuffle_vec;

            xvGet2NX8SelVec(channel, wid, shuffle_vec);

            xb_vec2Nx8U *vpsrc0 = XVTM_OFFSET_PTR_2NX8U(curr_src, 0, 0, (ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8U *vpsrc1 = XVTM_OFFSET_PTR_2NX8U(curr_src, 1, stride, (ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH);
            vpdst = XVTM_OFFSET_PTR_2NX8U(dst, 0, 0, -x - wid);
            int32_t y = 0;
            for (; y < (p2h - 3); y += 4)
            {
                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc1);
                IVP_LAV2NX8U_XP(color1, a_load, vpsrc1, sizeof(uint8_t) * wid);
                vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color2, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc1);
                IVP_LAV2NX8U_XP(color3, a_load, vpsrc1, sizeof(uint8_t) * wid);
                vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

                color = IVP_SHFL2NX8(color, shuffle_vec);
                color1 = IVP_SHFL2NX8(color1, shuffle_vec);
                color2 = IVP_SHFL2NX8(color2, shuffle_vec);
                color3 = IVP_SHFL2NX8(color3, shuffle_vec);

                IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color1, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color2, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color3, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
            }

            for (; y < p2h; y++)
            {
                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, stride - wid, 0);

                color = IVP_SHFL2NX8(color, shuffle_vec);
                IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
            }
            x += wid;
            ytmp += 1;
            xtmp += wid;
        }
        curr_src = curr_src - (ixmax);
    }

    if (x < p2w)
    {
        curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, ps2x);
        dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, pd2x);

        int32_t wid = XT_MIN(p2w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
        xb_vecNx8U * lsrc = XVTM_OFFSET_PTR_NX8U(curr_src, 0, 0, -x);
        xb_vecNx8U * ldst = XVTM_OFFSET_PTR_NX8U(dst,      0, 0, -x - wid);

        int32_t q15_inv_w = 1 + ((1 << 15) / wid);
        int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 65536 / stride; // check for gather bound (I8)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;

        xvGetNX16SelVec(channel, wid, shd);

        xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
        IVP_MULANX16PACKL(shs, stride - wid, shy);
        IVP_MULANX16PACKL(shd, stride + wid, shy);

        for (int32_t s = 0; s < p2h; s += ystep)
        {
            int32_t line_num = (p2h - s) < ystep ? (p2h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
            uint8_t *src0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(lsrc, s, stride, 0);
            uint8_t *dst0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(ldst, s, stride, 0);

            xb_gsr gr0;
            shs = IVP_MOVNX16T(shs, 0, vb);
            shd = IVP_MOVNX16T(shd, 0, vb);
            gr0 = XVTM_IVP_GATHERANX8UT_V(src0, shs, vb, 1);
            xb_vecNx16U v0 = IVP_GATHERDNX8U(gr0);
            XVTM_IVP_SCATTERNX8UT(v0, dst0, shd, vb);
        }
    }

    int32_t ps3x = ixmax - start_x;
    int32_t pd3x = (ixmax + 1 * channel) - start_x;
    int32_t p3y = -h;
    int32_t p3w = ((start_x + W_Local + w) - 1 * channel - ixmax);
    int32_t p3h = H_Local + (2 * h);

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, ps3x);
    dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, pd3x);
    x = 0;

    while (x < p3w - XT_MIN(XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH, ixmax - ixmin))
    {
        int32_t wid;
        xb_vec2Nx8U color, color1, color2, color3;
        int32_t ytmp = 0;
        int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p3w-x);
        while ( xtmp < loop_width)
        {
            wid = XT_MIN(p3w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-(ytmp*2 * XCHAL_IVPN_SIMD_WIDTH), 2 * XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);

            xb_vec2Nx8U shuffle_vec;

            xvGet2NX8SelVec(channel, wid, shuffle_vec);

            xb_vec2Nx8U *vpsrc0 = XVTM_OFFSET_PTR_2NX8U(curr_src, 0, 0, -wid - ((ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH));
            xb_vec2Nx8U *vpsrc1 = XVTM_OFFSET_PTR_2NX8U(curr_src, 1, stride, -wid - ((ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH));
            vpdst = XVTM_OFFSET_PTR_2NX8U(dst, 0, 0, x);
            int32_t y = 0;
            for (; y < (p3h - 3); y += 4)
            {
                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc1);
                IVP_LAV2NX8U_XP(color1, a_load, vpsrc1, sizeof(uint8_t) * wid);
                vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color2, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

                a_load = IVP_LA2NX8U_PP(vpsrc1);
                IVP_LAV2NX8U_XP(color3, a_load, vpsrc1, sizeof(uint8_t) * wid);
                vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

                color = IVP_SHFL2NX8(color, shuffle_vec);
                color1 = IVP_SHFL2NX8(color1, shuffle_vec);
                color2 = IVP_SHFL2NX8(color2, shuffle_vec);
                color3 = IVP_SHFL2NX8(color3, shuffle_vec);

                IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color1, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color2, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

                IVP_SAV2NX8U_XP(color3, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
            }

            for (; y < p3h; y++)
            {
                a_load = IVP_LA2NX8U_PP(vpsrc0);
                IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
                vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, stride - wid, 0);

                color = IVP_SHFL2NX8(color, shuffle_vec);
                IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
                XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
                vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
            }
            x += wid;
            ytmp += 1;
            xtmp += wid;
        }
        curr_src = curr_src + (ixmax);
    }

    if (x < p3w)
    {
        curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, ps3x);
        dst = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, pd3x);

        int32_t wid = XT_MIN(p3w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
        xb_vecNx8U * lsrc = XVTM_OFFSET_PTR_NX8U(curr_src, 0, 0, x - wid);
        xb_vecNx8U * ldst = XVTM_OFFSET_PTR_NX8U(dst,      0, 0, x);

        int32_t q15_inv_w = 1 + ((1 << 15) / wid);
        int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 65536 / stride; // check for gather bound (I8)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;

        xvGetNX16SelVec(channel, wid, shd);

        xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
        IVP_MULANX16PACKL(shs, stride - wid, shy);
        IVP_MULANX16PACKL(shd, stride + wid, shy);

        for (int32_t s = 0; s < p3h; s += ystep)
        {
            int32_t line_num = (p3h - s) < ystep ? (p3h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
            uint8_t *src0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(lsrc, s, stride, 0);
            uint8_t *dst0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(ldst, s, stride, 0);

            xb_gsr gr0;
            shs = IVP_MOVNX16T(shs, 0, vb);
            shd = IVP_MOVNX16T(shd, 0, vb);
            gr0 = XVTM_IVP_GATHERANX8UT_V(src0, shs, vb, 1);
            xb_vecNx16U v0 = IVP_GATHERDNX8U(gr0);
            XVTM_IVP_SCATTERNX8UT(v0, dst0, shd, vb);
        }
    }
    IVP_SCATTERW();
}

static void xvExtendEdgesReflect101_I16(xvTile const * tile, int32_t frame_width, int32_t frame_height)
{
    int32_t channel = XV_FRAME_GET_NUM_CHANNELS(tile->pFrame);

    int32_t w = XV_TILE_GET_EDGE_WIDTH(tile) * channel;
    int32_t h = XV_TILE_GET_EDGE_HEIGHT(tile);

	int16_t* __restrict src = (int16_t *)XV_TILE_GET_DATA_PTR(tile);
	int32_t stride = XV_TILE_GET_PITCH(tile);

    int32_t start_x = XV_TILE_GET_X_COORD(tile) * channel;
    int32_t start_y = XV_TILE_GET_Y_COORD(tile);

    int32_t W_Local = XV_TILE_GET_WIDTH(tile) * channel;
    int32_t H_Local = XV_TILE_GET_HEIGHT(tile);

    frame_width = frame_width * channel;

    int32_t usr_tmp =0;
    // find intersection of tile/frame
    int32_t ixmin = XT_MAX(start_x - w, 0);
    int32_t ixmax = XT_MIN(start_x + W_Local + w - 1 * channel, frame_width - 1 * channel);
    int32_t iymin = XT_MAX(start_y - h, 0);
    int32_t iymax = XT_MIN(start_y + H_Local + h - 1, frame_height - 1);


    int32_t p0x = ixmin - start_x;
    int32_t ps0y = iymin - start_y;
    int32_t pd0y = iymin - start_y - 1;
    int32_t p0w = (ixmax - ixmin) + 1 * channel;
    int32_t p0h = iymin - (start_y - h);

	int16_t* curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, ps0y, stride, p0x);
	int16_t* dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, pd0y, stride, p0x);

	xb_vecNx16 *vpdst;

	valign a_store = IVP_ZALIGN();
	valign a_load;

	int32_t p = XT_MAX(1, 2 * (iymax - iymin + 1) - 2);
	int32_t pmod = (p << 16) + 1;
	int32_t x = 0;
	if(p0h > 0)
	{
		for (; x < (p0w - (3 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2, color3;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color, a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LANX16_IP (color2, a_load, src_1);
				IVP_LAVNX16_XP(color3, a_load, src_1, sizeof(int16_t) * (p0w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color, a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SANX16_IP (color2, a_store, vdst);
				IVP_SAVNX16_XP(color3, a_store, vdst, sizeof(int16_t) * (p0w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		if(x < (p0w - (2 * XCHAL_IVPN_SIMD_WIDTH)))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LAVNX16_XP(color2, a_load, src_1, sizeof(int16_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SAVNX16_XP(color2, a_store, vdst, sizeof(int16_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < (p0w - XCHAL_IVPN_SIMD_WIDTH))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color, a_load, src_1);
				IVP_LAVNX16_XP(color1, a_load, src_1, sizeof(int16_t) * (p0w - x - XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color, a_store, vdst);
				IVP_SAVNX16_XP(color1, a_store, vdst, sizeof(int16_t) * (p0w - x - XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < p0w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LAVNX16_XP(color, a_load, src_1, sizeof(int16_t) * (p0w - x));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SAVNX16_XP(color, a_store, vdst, sizeof(int16_t) * (p0w - x));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else
		{
			//do nothing
		}
	}
	int32_t p1x = ixmin - start_x;
	int32_t ps1y = iymax - start_y;
	int32_t pd1y = (iymax + 1) - start_y;
    int32_t p1w = (ixmax - ixmin) + 1 * channel;
	int32_t p1h = (start_y + H_Local + h) - 1 - iymax;

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, ps1y, stride, p1x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, pd1y, stride, p1x);
	x = 0;
	if(p1h > 0)
	{
		for (; x < (p1w - (3 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2, color3;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LANX16_IP (color2, a_load, src_1);
				IVP_LAVNX16_XP(color3, a_load, src_1, sizeof(int16_t) * (p1w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SANX16_IP (color2, a_store, vdst);
				IVP_SAVNX16_XP(color3, a_store, vdst, sizeof(int16_t) * (p1w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		if(x < (p1w - (2 * XCHAL_IVPN_SIMD_WIDTH)))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LAVNX16_XP(color2, a_load, src_1, sizeof(int16_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SAVNX16_XP(color2, a_store, vdst, sizeof(int16_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < (p1w - XCHAL_IVPN_SIMD_WIDTH))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1;
				xb_vecNx16 * src_0_t = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_0_t);
				IVP_LANX16_IP (color,  a_load, src_0_t);
				IVP_LAVNX16_XP(color1, a_load, src_0_t, sizeof(int16_t) * (p1w - x - XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SAVNX16_XP(color1, a_store, vdst, sizeof(int16_t) * (p1w - x - XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < p1w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LAVNX16_XP(color, a_load, src_1, sizeof(int16_t) * (p1w - x));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SAVNX16_XP(color, a_store, vdst, sizeof(int16_t) * (p1w - x));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else
		{
			//do nothing
		}
	}
    int32_t ps2x = XT_MIN(ixmin - start_x + 1 * channel, ixmax - start_x);
	int32_t pd2x = ixmin - start_x;
	int32_t p2y = -h;
	int32_t p2w = ixmin - (start_x - w);
	int32_t p2h = H_Local + (2 * h);

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, ps2x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, pd2x);
	x = 0;


	while (x < p2w- XT_MIN(XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH, ixmax - ixmin))
	{
		int32_t wid;
		xb_vecNx16 color, color1, color2, color3;

		xb_vecNx16 * vpsrc0;
		xb_vecNx16 * vpsrc1;

		int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p2w-x);
    	while ( xtmp < loop_width)
		{
			wid = XT_MIN(p2w - x, XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*XCHAL_IVPN_SIMD_WIDTH, XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);

            xb_vecNx16 ind;
            xvGetNX16SelVec(channel, wid, ind);
			vsaN index = XVTM_IVP_MOVVSV(ind, 0);
			vpsrc0 = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, (ytmp)*XCHAL_IVPN_SIMD_WIDTH);
			vpsrc1 = XVTM_OFFSET_PTR_NX16(curr_src, 1, stride, (ytmp)*XCHAL_IVPN_SIMD_WIDTH);
			vpdst  = XVTM_OFFSET_PTR_NX16(dst,      0, 0, -x - wid);

			int32_t y = 0;
			for (; y < (p2h - 3); y += 4)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color,  a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color1, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color2, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color3, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				color  =  IVP_SELNX16(color,  color,  index);
				color1 =  IVP_SELNX16(color1, color1, index);
				color2 =  IVP_SELNX16(color2, color2, index);
				color3 =  IVP_SELNX16(color3, color3, index);

				IVP_SAVNX16_XP(color,  a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color1, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color2, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color3, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			for (; y < p2h; y++)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, stride - wid, 0);

				color =  IVP_SELNX16(color, color, index);

				IVP_SAVNX16_XP(color, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			x += wid;
			ytmp+=1;
			xtmp+=wid;

		}
		curr_src = curr_src - (ixmax);


	}
	if(x < p2w)
	{
		curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, ps2x);
		dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, pd2x);

		int32_t wid = XT_MIN(p2w - x, XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
		xb_vecNx16 * lsrc = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, -x + ( XCHAL_IVPN_SIMD_WIDTH * usr_tmp));
		xb_vecNx16 * ldst  = XVTM_OFFSET_PTR_NX16(dst,      0, 0, -x - wid);

		int32_t q15_inv_w = 1 + ((1<<15)/wid);
		int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 32768 / stride; // check for gather bound (I16)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;
        xvGetNX16SelVec(channel, wid, shd);
		xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
		IVP_MULANX16PACKL(shs, stride - wid, shy);
		IVP_MULANX16PACKL(shd, stride + wid, shy);

		shd = IVP_SLLINX16(shd, 1);
		shs = IVP_SLLINX16(shs, 1);

		for (int32_t s = 0; s < p2h; s += ystep)
		{
            int32_t line_num = (p2h - s) < ystep ? (p2h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
			int16_t * src0 = (int16_t *)XVTM_OFFSET_PTR_NX16(lsrc, s, stride, 0);
			int16_t * dst0 = (int16_t *)XVTM_OFFSET_PTR_NX16(ldst, s, stride, 0);

			xb_gsr gr0;
			shs = IVP_MOVNX16T(shs, 0, vb);
			shd = IVP_MOVNX16T(shd, 0, vb);
			gr0 = XVTM_IVP_GATHERANX16T_V(src0, shs, vb, 1);
			xb_vecNx16 v0 = IVP_GATHERDNX16(gr0);
			XVTM_IVP_SCATTERNX16T(v0, dst0, shd, vb);
		}
	}

	int32_t ps3x = ixmax - start_x;
    int32_t pd3x = (ixmax + 1 * channel) - start_x;
	int32_t p3y = -h;
    int32_t p3w = (start_x + W_Local + w) - 1 * channel - ixmax;
	int32_t p3h = H_Local + (2 * h);

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, ps3x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, pd3x);

	x = 0;

	while( x < p3w- XT_MIN(XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH, ixmax - ixmin))
	{
		int32_t wid;
		xb_vecNx16 color, color1, color2, color3;

		xb_vecNx16 * vpsrc0;
		xb_vecNx16 * vpsrc1;

		int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p3w-x);
    	while ( xtmp < loop_width)
		{
			wid = XT_MIN(p3w - x, XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*XCHAL_IVPN_SIMD_WIDTH, XCHAL_IVPN_SIMD_WIDTH),wid),1 * channel);

            xb_vecNx16 ind;
            xvGetNX16SelVec(channel, wid, ind);
			vsaN index = XVTM_IVP_MOVVSV(ind, 0);

			vpdst = XVTM_OFFSET_PTR_NX16(dst, 0, 0, x);
			vpsrc0 = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, -wid-((ytmp)*XCHAL_IVPN_SIMD_WIDTH));
			vpsrc1 = XVTM_OFFSET_PTR_NX16(curr_src, 1, stride, -wid-((ytmp)*XCHAL_IVPN_SIMD_WIDTH));

			int32_t y = 0;

			for (; y < (p3h - 3); y += 4)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color,  a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color1, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color2, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color3, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				color  =  IVP_SELNX16(color,  color,  index);
				color1 =  IVP_SELNX16(color1, color1, index);
				color2 =  IVP_SELNX16(color2, color2, index);
				color3 =  IVP_SELNX16(color3, color3, index);

				IVP_SAVNX16_XP(color,  a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color1, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color2, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color3, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			for (; y < p3h; y++)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, stride - wid, 0);

				color =  IVP_SELNX16(color, color, index);

				IVP_SAVNX16_XP(color, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}

			x += wid;
			ytmp+=1;
			xtmp+=wid;
		}
		curr_src = curr_src+(ixmax);
	}

	if(x < p3w)
	{
		curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, ps3x);
		dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, pd3x);

		int32_t wid = XT_MIN(p3w - x, XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
		xb_vecNx16 * lsrc = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, x - wid);
		xb_vecNx16 * ldst = XVTM_OFFSET_PTR_NX16(dst, 0, 0, x);

		int32_t q15_inv_w = 1 + ((1<<15)/wid);
		int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 32768 / stride; // check for gather bound (I16)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;
        xvGetNX16SelVec(channel, wid, shd);
		xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
		IVP_MULANX16PACKL(shs, stride - wid, shy);
		IVP_MULANX16PACKL(shd, stride + wid, shy);

		shd = IVP_SLLINX16(shd, 1);
		shs = IVP_SLLINX16(shs, 1);

		for (int32_t s = 0; s < p3h; s += ystep)
		{
            int32_t line_num = (p3h - s) < ystep ? (p3h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
			int16_t * src0 = (int16_t *)XVTM_OFFSET_PTR_NX16(lsrc, s, stride, 0);
			int16_t * dst0 = (int16_t *)XVTM_OFFSET_PTR_NX16(ldst, s, stride, 0);

			xb_gsr gr0;
			shs = IVP_MOVNX16T(shs, 0, vb);
			shd = IVP_MOVNX16T(shd, 0, vb);
			gr0 = XVTM_IVP_GATHERANX16T_V(src0, shs, vb, 1);
			xb_vecNx16 v0 = IVP_GATHERDNX16(gr0);
			XVTM_IVP_SCATTERNX16T(v0, dst0, shd, vb);
		}
	}
	IVP_SCATTERW();
}

static XI_ERR_TYPE xvTilePadding(xvTile *pTile)
{
    int32_t x1, x2, y1, y2, indy, copyHeight, copyRowBytes, wb, padVal;
    int32_t tileWidth, tilePitch, frameWidth, frameHeight;
    uint32_t tileHeight;
    uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
    int32_t extraEdgeLeft, extraEdgeRight, extraEdgeTop = 0, extraEdgeBottom = 0;
    uint16_t *__restrict srcPtr_16b, *__restrict dstPtr_16b;
    uint32_t *__restrict srcPtr_32b, *__restrict dstPtr_32b;
    uint8_t *__restrict srcPtr, *__restrict dstPtr;
    xvFrame *pFrame;
    xb_vecNx16 vec1, *__restrict pvecDst;
    xb_vec2Nx8U dvec1, *__restrict pdvecDst;
    xb_vecN_2x32Uv hvec1, *__restrict phvecDst;
    valign vas1;
    valign ald1;

    pFrame = pTile->pFrame;

    int32_t channel = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
    int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
    int32_t bytePerPel;
    bytePerPel = bytesPerPix / channel;

    tileEdgeTop = pTile->tileEdgeTop;
    tileEdgeBottom = pTile->tileEdgeBottom;
    tileEdgeLeft = pTile->tileEdgeLeft;
    tileEdgeRight = pTile->tileEdgeRight;
    tilePitch = pTile->pitch;
    tileHeight = pTile->height;
    tileWidth = pTile->width;
    frameWidth = pFrame->frameWidth;
    frameHeight = pFrame->frameHeight;

    int32_t edgeflags = 0;
    // edgeflags for padding is different from tile reseting
    if(pTile->x - tileEdgeLeft <= 0)
    {
        edgeflags = edgeflags | PADDING_LEFT;
    }
    if(pTile->x + pTile->width + tileEdgeRight >= frameWidth)
    {
        edgeflags = edgeflags | PADDING_RIGHT;
    }
    if(pTile->y - tileEdgeTop <= 0)
    {
        edgeflags = edgeflags | PADDING_TOP;
    }
    if(pTile->y + pTile->height + tileEdgeBottom >= frameHeight)
    {
        edgeflags = edgeflags | PADDING_DOWN;
    }

    if (pFrame->paddingType == FRAME_EDGE_PADDING)
    {
        if ((edgeflags & PADDING_TOP) == PADDING_TOP)
        {
            y1 = pTile->y - (int32_t)tileEdgeTop;
            if (y1 > frameHeight)
            {
                y1 = frameHeight;
            }
            if (y1 < 0)
            {
                extraEdgeTop = -y1;
                y1 = 0;
            }
            srcPtr = &((uint8_t *)pTile->pData)[-((((int32_t)tileEdgeTop - (int32_t)extraEdgeTop) * tilePitch) * (int32_t)bytePerPel) - ((int32_t)tileEdgeLeft * (int32_t)bytesPerPix)];
            dstPtr = &srcPtr[-(extraEdgeTop * tilePitch) * (int32_t)bytePerPel];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;
            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeTop; indy++)
            {
                xb_vec2Nx8U *__restrict srcPtr1 = (xb_vec2Nx8U *)srcPtr;
                for (wb = 0; wb < copyRowBytes; wb += (2 * IVP_SIMD_WIDTH))
                {
                    xb_vec2Nx8U vec;
                    int32_t offset = XT_MIN(copyRowBytes - wb, 2 * IVP_SIMD_WIDTH);
                    ald1 = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(vec, ald1, srcPtr1, offset);
                    IVP_SAV2NX8U_XP(vec, vas1, pdvecDst, offset);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }
        if ((edgeflags & PADDING_DOWN) == PADDING_DOWN)
        {
            y2 = ((int32_t)pTile->y) + (((int32_t)tileHeight) - 1) + (((int32_t)tileEdgeBottom));
            if (y2 < 0)
            {
                y2 = -1;
            }
            if (y2 > (frameHeight - 1))
            {
                extraEdgeBottom = (y2 - frameHeight) + 1;
                y2 = frameHeight - 1;
            }
            srcPtr = &((uint8_t *)pTile->pData)[-((int32_t)tileEdgeLeft * (int32_t)bytesPerPix) + ((((int32_t)tileHeight + (int32_t)tileEdgeBottom) - extraEdgeBottom - 1) * (int32_t)tilePitch * bytePerPel)];
            dstPtr = &srcPtr[tilePitch * (int32_t)bytePerPel];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;
            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeBottom; indy++)
            {
                xb_vec2Nx8U *__restrict srcPtr1 = (xb_vec2Nx8U *)srcPtr;
                for (wb = 0; wb < copyRowBytes; wb += (2 * IVP_SIMD_WIDTH))
                {
                    xb_vec2Nx8U vec;
                    int32_t offset = XT_MIN(copyRowBytes - wb, 2 * IVP_SIMD_WIDTH);
                    ald1 = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(vec, ald1, srcPtr1, offset);
                    IVP_SAV2NX8U_XP(vec, vas1, pdvecDst, offset);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                srcPtr = &srcPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }

        xb_vecNx16 vs16_sel;
	    if (1 == bytePerPel || 2 == bytePerPel)
        {
            switch (channel)
            {
                case 2:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(2));
                    break;
                }
                case 3:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(3));
                    break;
                }
                case 4:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(4));
                    break;
                }
                default:
                {
                    // XV_CHECK_ERROR((channel > 4), XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
                    if (channel > 4)
                    {
                        return XV_ERROR_BAD_ARG;
                    }
                }
            }
        }
        if (bytePerPel == 1)
        {
            xb_vec2Nx8 vs8_sel;
            xb_vec2Nx8U srcVec;

            if (channel == 1)
            {
               vs8_sel = IVP_ZERO2NX8();
            }
            else
            {
               vs8_sel = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(vs16_sel), IVP_MOV2NX8_FROMNX16(vs16_sel), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            }
            if ((edgeflags & PADDING_LEFT) == PADDING_LEFT)
            {
                extraEdgeLeft = ((int32_t)tileEdgeLeft) - pTile->x;
                dstPtr = &((uint8_t *)pTile->pData)[-(((int32_t)tileEdgeTop * (int32_t)tilePitch) + (int32_t)tileEdgeLeft * channel)];
                srcPtr = &dstPtr[extraEdgeLeft * channel];
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
                xb_vec2Nx8U *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    pdvecDst = (xb_vec2Nx8U *)dstPtr;
                    srcPtr1  = (xb_vec2Nx8U*)srcPtr;
                    ald1     = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
                    dvec1 = IVP_SHFL2NX8U(srcVec, vs8_sel);
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, extraEdgeLeft * bytesPerPix);
                    IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                    dstPtr = &dstPtr[tilePitch];
                    srcPtr = &srcPtr[tilePitch];
                }
            }

            if ((edgeflags & PADDING_RIGHT) == PADDING_RIGHT)
            {
                x2 = pTile->x + ((int32_t)tileWidth - 1) + (int32_t)tileEdgeRight;
                extraEdgeRight = x2 - (frameWidth - 1);
                dstPtr = &((uint8_t *)pTile->pData)[-((int32_t)tileEdgeTop * (int32_t)tilePitch) + (((int32_t)tileWidth + (int32_t)tileEdgeRight) - extraEdgeRight)];
                srcPtr = &dstPtr[-channel];
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
                xb_vec2Nx8U *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    srcPtr1 = (xb_vec2Nx8U*)srcPtr;
                    pdvecDst = (xb_vec2Nx8U *)dstPtr;
                    ald1 = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
                    dvec1 = IVP_SHFL2NX8U(srcVec, vs8_sel);
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, extraEdgeRight * bytesPerPix);
                    IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                    dstPtr = &dstPtr[tilePitch];
                    srcPtr = &srcPtr[tilePitch];
                }
            }
        }
        else if (bytePerPel == 2)
        {
            xb_vecNx16U srcVec;
            if (channel == 1)
            {
                vs16_sel = IVP_ZERONX16();
            }
            if ((edgeflags & PADDING_LEFT) == PADDING_LEFT)
            {
                extraEdgeLeft = (int32_t)tileEdgeLeft - pTile->x;
                dstPtr_16b = &((uint16_t *)pTile->pData)[-(((int32_t)tileEdgeTop * (int32_t)pTile->pitch) + (int32_t)tileEdgeLeft * channel)]; // No need of multiplying by 2
                srcPtr_16b = &dstPtr_16b[extraEdgeLeft * channel];                                                                             // No need of multiplying by 2 as pointers are uint16_t *
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
                xb_vecNx16 *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    srcPtr1 = (xb_vecNx16 *)srcPtr_16b;
                    pvecDst = (xb_vecNx16 *) dstPtr_16b;
                    ald1 = IVP_LANX16U_PP(srcPtr1);
                    IVP_LAVNX16U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
                    vec1 = IVP_SHFLNX16(srcVec, vs16_sel);
                    IVP_SAVNX16_XP(vec1, vas1, pvecDst, extraEdgeLeft * bytesPerPix);
                    IVP_SAPOSNX16_FP(vas1, pvecDst);
                    dstPtr_16b = &dstPtr_16b[pTile->pitch];
                    srcPtr_16b = &srcPtr_16b[pTile->pitch];
                }
            }

            if ((edgeflags & PADDING_RIGHT) == PADDING_RIGHT)
            {
                x2 = pTile->x + (int32_t)tileWidth + (int32_t)tileEdgeRight;
                extraEdgeRight = x2 - frameWidth;
                dstPtr_16b = &((uint16_t *)pTile->pData)[-((int32_t)tileEdgeTop * (int32_t)pTile->pitch) + (((int32_t)tileWidth + (int32_t)tileEdgeRight) - extraEdgeRight)* channel];
                srcPtr_16b = &dstPtr_16b[-channel];
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
                xb_vecNx16 *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    srcPtr1 = (xb_vecNx16 *)srcPtr_16b;
                    pvecDst = (xb_vecNx16 *) dstPtr_16b;
                    ald1 = IVP_LANX16U_PP(srcPtr1);
                    IVP_LAVNX16U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
                    vec1 = IVP_SHFLNX16(srcVec, vs16_sel);
                    IVP_SAVNX16_XP(vec1, vas1, pvecDst, extraEdgeRight * bytesPerPix);
                    IVP_SAPOSNX16_FP(vas1, pvecDst);
                    dstPtr_16b = &dstPtr_16b[pTile->pitch];
                    srcPtr_16b = &srcPtr_16b[pTile->pitch];
                }
            }
        }
        else
        {
            if ((edgeflags & PADDING_LEFT) == PADDING_LEFT)
            {
                extraEdgeLeft = (int32_t)tileEdgeLeft - pTile->x;
                dstPtr_32b = &((uint32_t *)pTile->pData)[-(((int32_t)tileEdgeTop * (int32_t)pTile->pitch) + (int32_t)tileEdgeLeft * channel)]; // No need of multiplying by 2
                srcPtr_32b = &dstPtr_32b[channel];                                                                             // No need of multiplying by 2 as pointers are uint32_t *
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
          	xb_vecN_2x32Uv *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    srcPtr1 = (xb_vecN_2x32Uv *)srcPtr_32b;
               	    phvecDst = (xb_vecN_2x32Uv *) dstPtr_32b;
	            ald1 = IVP_LAN_2X32U_PP(srcPtr1);
	            IVP_LAVN_2X32U_XP(hvec1, ald1, srcPtr1, extraEdgeLeft * bytesPerPix);
	            IVP_SAVN_2X32_XP(hvec1, vas1, phvecDst, extraEdgeLeft * bytesPerPix);
                    IVP_SAPOSN_2X32_FP(vas1, (xb_vecN_2x32v *)phvecDst);
                    dstPtr_32b = &dstPtr_32b[pTile->pitch];
                    srcPtr_32b = &srcPtr_32b[pTile->pitch];
                }
            }

            if ((edgeflags & PADDING_RIGHT) == PADDING_RIGHT)
            {
                x2 = (int32_t)pTile->x + (int32_t)tileWidth + (int32_t)tileEdgeRight;
                extraEdgeRight = x2 - frameWidth;
                dstPtr_32b = &((uint32_t *)pTile->pData)[-(((int32_t)tileEdgeTop * (int32_t)pTile->pitch)) + (((int32_t)tileWidth + (int32_t)tileEdgeRight) - extraEdgeRight) * channel];
                srcPtr_32b = &dstPtr_32b[-channel];
                copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

                vas1 = IVP_ZALIGN();
        	    xb_vecN_2x32Uv *__restrict srcPtr1;
                for (indy = 0; indy < copyHeight; indy++)
                {
                    srcPtr1 = (xb_vecN_2x32Uv *)srcPtr_32b;
	            phvecDst = (xb_vecN_2x32Uv *) dstPtr_32b;
	            ald1 = IVP_LAN_2X32U_PP(srcPtr1);
	            IVP_LAVN_2X32U_XP(hvec1, ald1, srcPtr1, extraEdgeRight * bytesPerPix);
	            IVP_SAVN_2X32_XP(hvec1, vas1, phvecDst, extraEdgeRight * bytesPerPix);
                    IVP_SAPOSN_2X32_FP(vas1, (xb_vecN_2x32v *)phvecDst);
                    dstPtr_32b = &dstPtr_32b[pTile->pitch];
                    srcPtr_32b = &srcPtr_32b[pTile->pitch];
                }
            }
        }
    }
    else if (pFrame->paddingType == FRAME_PADDING_REFLECT_101)
    {
        if (bytePerPel == 1)
        {
            xvExtendEdgesReflect101_I8(pTile, frameWidth, frameHeight);
        }
        else if (bytePerPel == 2)
        {
            xvExtendEdgesReflect101_I16(pTile, frameWidth, frameHeight);
        }
        else
        {
            // default comment for MISRA-C
        }
    }
    else
    {
        padVal = 0;
        if (pFrame->paddingType == FRAME_CONSTANT_PADDING)
        {
            padVal = (int32_t)pFrame->paddingVal;
        }
        dvec1 = padVal;
        if (bytePerPel == 1)
        {
            dvec1 = padVal;
        }
        else if (bytePerPel == 2)
        {
            xb_vecNx16U vec = padVal;
            dvec1 = IVP_MOV2NX8U_FROMNX16(vec);
        }
        else
        {
            xb_vecN_2x32Uv hvec = padVal;
            dvec1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32U(hvec));
        }

        if ((edgeflags & PADDING_TOP) == PADDING_TOP)
        {
            y1 = pTile->y - (int32_t)tileEdgeTop;
            extraEdgeTop = -y1;
            dstPtr       = &((uint8_t *) pTile->pData)[ -((int32_t) tileEdgeTop * (int32_t) tilePitch * (int32_t) bytePerPel) - (int32_t) tileEdgeLeft * (int32_t) bytesPerPix];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;

            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeTop; indy++)
            {
                for (wb = copyRowBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
                {
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }

        if ((edgeflags & PADDING_DOWN) == PADDING_DOWN)
        {
            y2 = (int32_t)pTile->y + (int32_t)tileHeight + (int32_t)tileEdgeBottom;
            extraEdgeBottom = y2 - (int32_t)frameHeight;
            dstPtr          = &((uint8_t *) pTile->pData)[(((frameHeight - pTile->y) * (int32_t) tilePitch * (int32_t) bytePerPel) - (int32_t) tileEdgeLeft * (int32_t) bytesPerPix)];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;

            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeBottom; indy++)
            {
                for (wb = copyRowBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
                {
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }
        if ((edgeflags & PADDING_LEFT) == PADDING_LEFT)
        {
            x1 = pTile->x - (int32_t)tileEdgeLeft;
            extraEdgeLeft = -x1;
            dstPtr        = &((uint8_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) tilePitch * (int32_t) bytePerPel) + (int32_t) tileEdgeLeft * (int32_t) bytesPerPix)];
            copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < copyHeight; indy++)
            {
                for (wb = extraEdgeLeft * bytesPerPix; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
                {
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }

        if ((edgeflags & PADDING_RIGHT) == PADDING_RIGHT)
        {
            x2 = pTile->x + ((int32_t)tileWidth - 1) + (int32_t)tileEdgeRight;
            extraEdgeRight = x2 - (frameWidth - 1);
            x2 = frameWidth - 1;
            dstPtr         = &((uint8_t *) pTile->pData)[ -(((int32_t) tileEdgeTop) * ((int32_t) tilePitch) * (int32_t) bytePerPel) + ((((int32_t) tileWidth + ((int32_t) tileEdgeRight)) - extraEdgeRight) * ((int32_t) bytesPerPix))];
            copyHeight = (int32_t)tileEdgeTop + (int32_t)tileHeight + (int32_t)tileEdgeBottom;

            pdvecDst = (xb_vec2Nx8U *)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < copyHeight; indy++)
            {
                for (wb = extraEdgeRight * bytesPerPix; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
                {
                    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U *)dstPtr;
            }
        }

        
    }

    return (XVTM_SUCCESS);
}

XI_ERR_TYPE TilePadding(xi_tile *tile, int32_t paddingType, int32_t paddingValue)
{
#ifdef XI_XV_TILE_COMPATIBILITY
    ((xvTile *)tile)->pFrame->paddingType = paddingType;
    ((xvTile *)tile)->pFrame->paddingVal = paddingValue;
    return xvTilePadding((xvTile *)tile);
#else
    ((xvTile *)tile)->pFrame->paddingType = paddingType;
    ((xvTile *)tile)->pFrame->paddingVal = paddingValue;
    return xvTilePadding((xvTile *)tile);
    // without modify because cadence has XI_XV_TILE_COMPATIBILITY marco.
#endif
}

XI_ERR_TYPE TilePaddingWithSize(xi_tile* p_tile, int32_t padding_type, int32_t padding_width, int32_t padding_height, int32_t padding_value)
{
    XI_TILE_SET_EDGE_WIDTH(p_tile, padding_width);
    XI_TILE_SET_EDGE_HEIGHT(p_tile, padding_height);
    ((xvTile *)p_tile)->pFrame->paddingType = padding_type;
    ((xvTile *)p_tile)->pFrame->paddingVal = padding_value;
    return xvTilePadding((xvTile *)p_tile);
}

XI_ERR_TYPE TileResetting(xi_tile* p_tile, TileInfo* p_tileinfo, int32_t extra_width, int32_t extra_height, int32_t edge_width, int32_t edge_height, int32_t edgeflags)
{
    int32_t top   = ((edgeflags & PADDING_TOP) == PADDING_TOP)     ?  0 : extra_height;
    int32_t down  = ((edgeflags & PADDING_DOWN) == PADDING_DOWN)   ?  0 : extra_height;
    int32_t right = ((edgeflags & PADDING_RIGHT) == PADDING_RIGHT) ?  0 : extra_width;
    int32_t left  = ((edgeflags & PADDING_LEFT) == PADDING_LEFT)   ?  0 : extra_width;
    p_tile->width =  p_tileinfo->base_width + left + right;
    p_tile->height = p_tileinfo->base_height + top + down;
    int32_t channel = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(p_tile));

    if (p_tileinfo->tiletype == TILE_I8)
    {
        p_tile->pData = (void *)(((uint8_t*)(p_tileinfo->base_ptr)) - (top * p_tile->pitch + left * channel));
    }
    if (p_tileinfo->tiletype == TILE_I16)
    {
        p_tile->pData = (void *)(((uint16_t*)(p_tileinfo->base_ptr)) - (top * p_tile->pitch + left * channel));
    }
    if (p_tileinfo->tiletype == TILE_I32)
    {
        p_tile->pData = (void *)(((uint32_t*)(p_tileinfo->base_ptr)) - (top * p_tile->pitch + left * channel));
    }
#ifdef XI_XV_TILE_COMPATIBILITY
    XI_TILE_SET_EDGE_WIDTH(p_tile, edge_width); // set edge size for tile check
    XI_TILE_SET_EDGE_HEIGHT(p_tile, edge_height);
    p_tile->x = p_tileinfo->base_x - left;
    p_tile->y = p_tileinfo->base_y - top;
#endif
    return XVF_SUCCESS;
}

void TileResetToOrigin(xi_tile *p_tile, TileInfo* p_baseinfo)
{
    p_tile->width = p_baseinfo->base_width;
    p_tile->height = p_baseinfo->base_height;
    p_tile->x = p_baseinfo->base_x;
    p_tile->y = p_baseinfo->base_y;
    p_tile->pData = p_baseinfo->base_ptr;
#ifdef XI_XV_TILE_COMPATIBILITY
    XI_TILE_SET_EDGE_WIDTH(p_tile, p_baseinfo->edge_width);
    XI_TILE_SET_EDGE_HEIGHT(p_tile, p_baseinfo->edge_height);
    ((xvTile *)p_tile)->pFrame->paddingType = p_baseinfo->padding_type;
    ((xvTile *)p_tile)->pFrame->paddingVal = p_baseinfo->padding_value;
#endif
}

void ExtractTileInfo(TileInfo* p_tileinfo, xi_tile* p_tile, int32_t tiletype)
{
    p_tileinfo->base_height = p_tile->height;
    p_tileinfo->base_width = p_tile->width;
    p_tileinfo->base_x = p_tile->x;
    p_tileinfo->base_y = p_tile->y;
    p_tileinfo->base_ptr = p_tile->pData;
    p_tileinfo->tiletype = tiletype;
    p_tileinfo->padding_type = ((xvTile *)p_tile)->pFrame->paddingType;
    p_tileinfo->padding_value = ((xvTile *)p_tile)->pFrame->paddingVal;
    p_tileinfo->edge_width = p_tile->edgeWidth;
    p_tileinfo->edge_height = p_tile->edgeHeight;
}

int32_t GetEdgeFlags(xi_tile* p_tile, xi_frame* p_frame)
{
    int32_t edgeFlags = 0;
    if(p_tile->x == 0)
    {
        edgeFlags = edgeFlags | PADDING_LEFT;
    }
    if(p_tile->x + p_tile->width == p_frame->frameWidth)
    {
        edgeFlags = edgeFlags | PADDING_RIGHT;
    }
    if(p_tile->y == 0)
    {
        edgeFlags = edgeFlags | PADDING_TOP;
    }
    if(p_tile->y + p_tile->height == p_frame->frameHeight)
    {
        edgeFlags = edgeFlags | PADDING_DOWN;
    }
    return edgeFlags;
}
