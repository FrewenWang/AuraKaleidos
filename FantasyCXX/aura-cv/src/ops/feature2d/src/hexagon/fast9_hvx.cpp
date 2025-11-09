#include "fast_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/thread_object.h"

namespace aura
{

#define LOG2VLEN                       7

struct FastCorner
{
    FastCorner(DT_U16 x, DT_U16 y, DT_U16 score) : x(x), y(y), score(score)
    {}

    DT_U16 x;
    DT_U16 y;
    DT_U16 score;
};

struct FastParam
{
    FastParam(Context *ctx, DT_U32 width, DT_U32 max_num_corners)
    {
        m_ctx = ctx;

        do
        {
            // Calculate memory size
            DT_U32 num_pixels = width - 3;
            num_pixels = (num_pixels + 8 * AURA_HVLEN - 1) & (-8 * AURA_HVLEN); // roundup to 8*VLEN
            DT_U32 num_pixels_32 = num_pixels >> 5;
            DT_U32 max_num_corners_align128 = (max_num_corners + AURA_HVLEN - 1) & (-AURA_HVLEN);
            DT_U32 total_mem_u8 = num_pixels_32 * sizeof(DT_U32) + width * sizeof(DT_S16) * 2 +
                                  width * sizeof(DT_U8) * 3 + max_num_corners_align128 * sizeof(FastCorner);

            //Allocate memory, break on failure
            data = AURA_ALLOC(ctx, total_mem_u8 * sizeof(DT_U8));
            if (NULL == data)
            {
                DT_CHAR err_str[64];
                snprintf(err_str, sizeof(err_str), "allocation %s byte error", std::to_string(total_mem_u8).c_str());
                AURA_ADD_ERROR_STRING(ctx, err_str);
                break;
            }

            corners     = (FastCorner *)data;
            bit_mask    = (DT_U32 *)(corners + max_num_corners_align128);
            pos_x[0]    = (DT_S16 *)(bit_mask + num_pixels_32);
            pos_x[1]    = (DT_S16 *)(pos_x[0] + width);
            score[0]    = (DT_U8 *)(pos_x[1] + width);
            score[1]    = (DT_U8 *)(score[0] + width);
            score[2]    = (DT_U8 *)(score[1] + width);

            // Assignment
            memset(data, 0, total_mem_u8 * sizeof(DT_U8));

            break;

        } while (DT_TRUE);
    }

    ~FastParam()
    {
        AURA_FREE(m_ctx,data);
        data = NULL;
    }

    Context *m_ctx;
    DT_S32 num_coarse_p;
    DT_S32 num_coarse_c;
    DT_S32 height;
    DT_S32 width;
    DT_S32 stride;
    DT_U32 num_corners;
    DT_U32 *bit_mask;
    DT_S16 *pos_x[2];
    DT_U8 *score[3];
    FastCorner *corners;

    DT_VOID *data;
};

AURA_ALWAYS_INLINE DT_U8 CornerScore16(DT_U64 d0_7, DT_U64 d8_15)
{
    DT_U64 q0 = HEXAGON_V64_CREATE_W(Q6_R_vsplatb_R(0), Q6_R_vsplatb_R(0));

    //k == 0
    DT_U64 v0k0 = Q6_P_valignb_PPI(d0_7, d8_15, 7);
    DT_U64 v1k0 = Q6_P_valignb_PPI(d0_7, d8_15, 6);
    DT_U64 ak0  = Q6_P_vminub_PP(v0k0, v1k0);

    v0k0 = Q6_P_valignb_PPI(d0_7, d8_15, 5);
    ak0  = Q6_P_vminub_PP(ak0, v0k0);

    v1k0 = Q6_P_valignb_PPI(d0_7, d8_15, 4);
    ak0  = Q6_P_vminub_PP(ak0, v1k0);

    v0k0 = Q6_P_valignb_PPI(d0_7, d8_15, 3);
    ak0  = Q6_P_vminub_PP(ak0, v0k0);

    v1k0 = Q6_P_valignb_PPI(d0_7, d8_15, 2);
    ak0  = Q6_P_vminub_PP(ak0, v1k0);

    v0k0 = Q6_P_valignb_PPI(d0_7, d8_15, 1);
    ak0  = Q6_P_vminub_PP(ak0, v0k0);

    ak0  = Q6_P_vminub_PP(ak0, d8_15);
    q0   = Q6_P_vmaxub_PP(q0, Q6_P_vminub_PP(ak0, d0_7));
    v1k0 = Q6_P_valignb_PPI(d8_15, d0_7, 7);
    q0   = Q6_P_vmaxub_PP(q0, Q6_P_vminub_PP(ak0, v1k0));

    DT_U64 v0k8 = v1k0;
    DT_U64 v1k8 = Q6_P_valignb_PPI(d8_15, d0_7, 6);
    DT_U64 ak8  = Q6_P_vminub_PP(v0k8, v1k8);

    v0k8 = Q6_P_valignb_PPI(d8_15, d0_7, 5);
    ak8  = Q6_P_vminub_PP(ak8, v0k8);

    v1k8 = Q6_P_valignb_PPI(d8_15, d0_7, 4);
    ak8  = Q6_P_vminub_PP(ak8, v1k8);

    v0k8 = Q6_P_valignb_PPI(d8_15, d0_7, 3);
    ak8  = Q6_P_vminub_PP(ak8, v0k8);

    v1k8 = Q6_P_valignb_PPI(d8_15, d0_7, 2);
    ak8  = Q6_P_vminub_PP(ak8, v1k8);

    v0k8 = Q6_P_valignb_PPI(d8_15, d0_7, 1);
    ak8  = Q6_P_vminub_PP(ak8, v0k8);

    ak8  = Q6_P_vminub_PP(ak8, d0_7);
    q0   = Q6_P_vmaxub_PP(q0, Q6_P_vminub_PP(ak8, d8_15));
    v1k8 = Q6_P_valignb_PPI(d0_7, d8_15, 7);
    q0   = Q6_P_vmaxub_PP(q0, Q6_P_vminub_PP(ak8, v1k8));

    DT_U8 score = Max<DT_U8>(HEXAGON_V64_GET_UB0(q0), HEXAGON_V64_GET_UB1(q0));
    score = Max<DT_U8>(HEXAGON_V64_GET_UB2(q0), score);
    score = Max<DT_U8>(HEXAGON_V64_GET_UB3(q0), score);
    score = Max<DT_U8>(HEXAGON_V64_GET_UB4(q0), score);
    score = Max<DT_U8>(HEXAGON_V64_GET_UB5(q0), score);
    score = Max<DT_U8>(HEXAGON_V64_GET_UB6(q0), score);
    score = Max<DT_U8>(HEXAGON_V64_GET_UB7(q0), score);

    return score - 1;
}

static DT_VOID Fast9U8DetectCoarse(const DT_U8 *img, DT_U32 iwidth, DT_U32 istride,
                                   DT_U32 *bitmask, DT_U32 threshold, DT_U32 border)
{
    DT_S32 num, numpixels;
    DT_U32 idx = 0x80808080;

    HVX_Vector vu8_pixel00, vu8_pixel04, vu8_pixel08, vu8_pixel12;
    HVX_Vector vu8_pv0, vu8_pv1, vu8_add, vu8_sub, vu8_threshold, vu8_bit_mask;
    HVX_Vector vu8_max00_08, vu8_min00_08, vu8_max04_12, vu8_min04_12, vu8_min_max, vu8_max_min;
    HVX_Vector vu8_mask, vu8_mask_l, vu8_mask_r, vu8_mask_ff;
    HVX_VectorPred q0, q1, q2;

    HVX_Vector *vu8_src_c  = (HVX_Vector *)img;
    HVX_Vector *vu8_src_p3 = (HVX_Vector *)(img - 3 * istride);
    HVX_Vector *vu8_src_n3 = (HVX_Vector *)(img + 3 * istride);
    HVX_Vector *vu32_dst   = (HVX_Vector *)bitmask;

    numpixels = iwidth - 2 * border + (border % AURA_HVLEN);

    vu8_mask_ff = Q6_V_vsplat_R(-1);
    vu8_mask_l  = Q6_V_vnot_V(Q6_V_vand_QR(Q6_Q_vsetq_R(border), 0x01010101));

    DT_U32 nb  = (numpixels >> LOG2VLEN) & 7;
    DT_U32 a   = (numpixels & (8 * AURA_HVLEN - 1)) == 0 ? 0 : ((-1) << nb);
    vu8_mask_r = Q6_V_vnot_V(Q6_V_vsplat_R(Q6_R_vsplatb_R(a)));
    vu8_mask_r = Q6_V_vandor_VQR(vu8_mask_r, Q6_Q_vsetq_R(numpixels), Q6_R_vsplatb_R(1 << nb));

    vu8_threshold = Q6_V_vsplat_R(Q6_R_vsplatb_R(threshold));
    vu8_mask = vu8_mask_l;
    vload(vu8_src_c - 1, vu8_pv0);
    vload(vu8_src_c++,   vu8_pv1);

    num = 8 * AURA_HVLEN;

    for (DT_S32 i = numpixels; i > 0; i -= 8 * AURA_HVLEN)
    {
        num = (i < num) ? i : num;

        vu8_bit_mask = Q6_V_vzero();

        for (DT_S32 j = num; j > 0; j -= AURA_HVLEN)
        {
            vload(vu8_src_p3++, vu8_pixel00);
            vload(vu8_src_n3++, vu8_pixel08);
            vu8_pixel12 = Q6_V_vlalign_VVI(vu8_pv1, vu8_pv0, 3);

            vu8_pv0     = vu8_pv1;
            vload(vu8_src_c++, vu8_pv1);
            vu8_pixel04 = Q6_V_valign_VVI(vu8_pv1, vu8_pv0, 3);

            vu8_sub = Q6_Vub_vsub_VubVub_sat(vu8_pv0, vu8_threshold);
            vu8_add = Q6_Vub_vadd_VubVub_sat(vu8_pv0, vu8_threshold);

            vu8_max00_08 = Q6_Vub_vmax_VubVub(vu8_pixel00, vu8_pixel08);
            vu8_min00_08 = Q6_Vub_vmin_VubVub(vu8_pixel00, vu8_pixel08);

            vu8_max04_12 = Q6_Vub_vmax_VubVub(vu8_pixel04, vu8_pixel12);
            vu8_min04_12 = Q6_Vub_vmin_VubVub(vu8_pixel04, vu8_pixel12);

            vu8_min_max  = Q6_Vub_vmin_VubVub(vu8_max00_08, vu8_max04_12);
            vu8_max_min  = Q6_Vub_vmax_VubVub(vu8_min00_08, vu8_min04_12);

            q0 = Q6_Q_vcmp_gt_VubVub(vu8_min_max, vu8_add);
            q1 = Q6_Q_vcmp_gt_VubVub(vu8_sub, vu8_max_min);
            q2 = Q6_Q_or_QQ(q0, q1);

            idx = Q6_R_rol_RI(idx, 1);
            vu8_bit_mask = Q6_V_vandor_VQR(vu8_bit_mask, q2, idx);
        }

        vstore(vu32_dst++, Q6_V_vand_VV(vu8_bit_mask, vu8_mask));
        vu8_mask = vu8_mask_ff;
    }

    vu32_dst--;
    vstore(vu32_dst, Q6_V_vand_VV(vu32_dst[0], vu8_mask_r));
}

static DT_U32 Fast9U8DetectFine(const DT_U8 *img, DT_U32 istride, DT_U32 *bitmask,
                                DT_S16 *x_pos, DT_U8 *score, DT_U32 num_pixels_32, DT_U32 threshold)
{
    const DT_U8 *p_src = img;
    DT_S32 num_corners = 0;
    DT_U32 bit_masks_v32 = 0, pr = 0;
    DT_U64 pixel_ref = 0, pixel03_12 = 0, pixel11_04 = 0, bright_thr = 0, dark_thr = 0, thresholds = 0;
    DT_S32 q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    DT_S32 bitpos = 0, k = 0, m = 0, x = 0;

    thresholds = HEXAGON_V64_CREATE_W(Q6_R_vsplatb_R(threshold), Q6_R_vsplatb_R(threshold));

    for (DT_U32 i = 0; i < num_pixels_32; i++)
    {
        bit_masks_v32 = *bitmask++;

        while (bit_masks_v32 != 0)
        {
            bitpos = Q6_R_ct0_R(bit_masks_v32);

            k = i * 32 + bitpos;
            m = k & (8 * AURA_HVLEN - 1);
            x = (k & (-8 * AURA_HVLEN)) | ((m & 7) << LOG2VLEN) | (m >> 3);

            pr = Q6_R_vsplatb_R(p_src[x]);

            pixel_ref  = HEXAGON_V64_CREATE_W(pr, pr);

            pixel03_12 = HEXAGON_V64_PUT_B7(pixel03_12, p_src[x + 0 * istride - 3]); // load pixel #12
            pixel11_04 = HEXAGON_V64_PUT_B7(pixel11_04, p_src[x + 0 * istride + 3]); // load pixel #4

            pixel03_12 = HEXAGON_V64_PUT_B6(pixel03_12, p_src[x - 1 * istride - 3]); // load pixel #13
            pixel11_04 = HEXAGON_V64_PUT_B6(pixel11_04, p_src[x + 1 * istride + 3]); // load pixel #5

            pixel03_12 = HEXAGON_V64_PUT_B5(pixel03_12, p_src[x - 2 * istride - 2]); // load pixel #14
            pixel11_04 = HEXAGON_V64_PUT_B5(pixel11_04, p_src[x + 2 * istride + 2]); // load pixel #6

            pixel03_12 = HEXAGON_V64_PUT_B4(pixel03_12, p_src[x - 3 * istride - 1]); // load pixel #15
            pixel11_04 = HEXAGON_V64_PUT_B4(pixel11_04, p_src[x + 3 * istride + 1]); // load pixel #7

            pixel03_12 = HEXAGON_V64_PUT_B3(pixel03_12, p_src[x - 3 * istride + 0]); // load pixel #0
            pixel11_04 = HEXAGON_V64_PUT_B3(pixel11_04, p_src[x + 3 * istride + 0]); // load pixel #8

            pixel03_12 = HEXAGON_V64_PUT_B2(pixel03_12, p_src[x - 3 * istride + 1]); // load pixel #1
            pixel11_04 = HEXAGON_V64_PUT_B2(pixel11_04, p_src[x + 3 * istride - 1]); // load pixel #9

            pixel03_12 = HEXAGON_V64_PUT_B1(pixel03_12, p_src[x - 2 * istride + 2]); // load pixel #2
            pixel11_04 = HEXAGON_V64_PUT_B1(pixel11_04, p_src[x + 2 * istride - 2]); // load pixel #10

            pixel03_12 = HEXAGON_V64_PUT_B0(pixel03_12, p_src[x - 1 * istride + 3]); // load pixel #3
            pixel11_04 = HEXAGON_V64_PUT_B0(pixel11_04, p_src[x + 1 * istride - 3]); // load pixel #11

            bright_thr = Q6_P_vaddub_PP_sat(pixel_ref, thresholds);
            dark_thr   = Q6_P_vsubub_PP_sat(pixel_ref, thresholds);

            q0 = Q6_p_vcmpb_gtu_PP(pixel11_04, bright_thr);
            q1 = Q6_p_vcmpb_gtu_PP(pixel03_12, bright_thr);
            q2 = Q6_p_vcmpb_gtu_PP(dark_thr, pixel11_04);
            q3 = Q6_p_vcmpb_gtu_PP(dark_thr, pixel03_12);

            if (Q6_p_fastcorner9_pp(q1, q0))
            {
                *x_pos++ = x;
                num_corners++;
                pixel11_04 = Q6_P_vsubub_PP_sat(pixel11_04, pixel_ref);
                pixel03_12 = Q6_P_vsubub_PP_sat(pixel03_12, pixel_ref);
                score[x] = CornerScore16(pixel11_04, pixel03_12);
            }
            else if (Q6_p_fastcorner9_pp(q3, q2))
            {
                *x_pos++ = x;
                num_corners++;
                pixel11_04 = Q6_P_vsubub_PP_sat(pixel_ref, pixel11_04);
                pixel03_12 = Q6_P_vsubub_PP_sat(pixel_ref, pixel03_12);
                score[x] = CornerScore16(pixel11_04, pixel03_12);
            }

            bit_masks_v32 = Q6_R_clrbit_RR(bit_masks_v32, bitpos);
        }
    }

    return num_corners;
}

static DT_VOID Fast9U8Row(const DT_U8 *src_row, FastParam *fast_param, DT_U32 max_num_corners,
                          DT_S32 threshold, DT_BOOL nonmax_suppression, DT_S32 cur_row, DT_S32 start_row)
{
    DT_S32 pos_x = 0;
    DT_U8 score  = 0;
    DT_S32 corners_num = 0, corners_diff = 0;
    DT_S32 num_fine = 0;
    DT_S16 *pos_swap_x_ptr = NULL;
    DT_U8  *score_swap_ptr = NULL;

    DT_U32 border = 3;
    DT_U32 num_pixels = fast_param->width - border;
    num_pixels = (num_pixels + 8 * AURA_HVLEN - 1) & (-8 * AURA_HVLEN); // roundup to 8*VLEN
    DT_U32 num_pixels_32 = num_pixels >> 5;

    if (fast_param->num_corners > max_num_corners)
    {
        return;
    }

    memset(fast_param->score[2], 0, fast_param->width);

    if(cur_row < fast_param->height - 3)
    {
        Fast9U8DetectCoarse(src_row, fast_param->width, fast_param->stride, fast_param->bit_mask, threshold, border);
        fast_param->num_coarse_c = Fast9U8DetectFine(src_row, fast_param->stride, fast_param->bit_mask, fast_param->pos_x[1], fast_param->score[2], num_pixels_32, threshold);
    }

    if ((cur_row != start_row) && ((cur_row == 4 || cur_row != start_row + 1)))
    {
        for (DT_S32 i = 0; i < fast_param->num_coarse_p; i++)
        {
            pos_x = fast_param->pos_x[0][i];
            score = fast_param->score[1][pos_x];
            if (!nonmax_suppression ||
                (score > fast_param->score[1][pos_x+1]  && score > fast_param->score[1][pos_x-1] &&
                score > fast_param->score[0][pos_x-1] && score > fast_param->score[0][pos_x]  && score >fast_param->score[0][pos_x+1] &&
                score > fast_param->score[2][pos_x-1]  && score > fast_param->score[2][pos_x]   && score > fast_param->score[2][pos_x+1]))
            {
                num_fine++;
                (*(fast_param->corners)).x = pos_x;
                (*(fast_param->corners)).y = cur_row - 1;
                (*(fast_param->corners)).score = nonmax_suppression ? score : 0;
                fast_param->corners++;
            }
        }

        corners_diff = max_num_corners - fast_param->num_corners;
        corners_num = (corners_diff > num_fine) ? num_fine : corners_diff;
        fast_param->num_corners += corners_num;
    }

    fast_param->num_coarse_p  = fast_param->num_coarse_c;
    score_swap_ptr            = fast_param->score[0];
    fast_param->score[0]      = fast_param->score[1];
    fast_param->score[1]      = fast_param->score[2];
    fast_param->score[2]      = score_swap_ptr;
    pos_swap_x_ptr            = fast_param->pos_x[0];
    fast_param->pos_x[0]      = fast_param->pos_x[1];
    fast_param->pos_x[1]      = pos_swap_x_ptr;
}

static Status Fast9U8HvxImpl(Context *ctx, const Mat &src, ThreadObject<FastParam> &share_fast_param, DT_U32 max_num_corners, DT_S32 threshold,
                             DT_BOOL nonmax_suppression, DT_S32 start_row, DT_S32 end_row)
{
    FastParam *fast_param = share_fast_param.GetObject();
    if (DT_NULL == fast_param)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get fast_param failed");
        return Status::ERROR;
    }

    fast_param->width  = src.GetSizes().m_width;
    fast_param->height = src.GetSizes().m_height;
    fast_param->stride = src.GetStrides().m_width;
    fast_param->num_corners  = 0;
    fast_param->num_coarse_p = 0;
    fast_param->num_coarse_c = 0;

    DT_U64 L2fetch_param = L2PfParam(fast_param->stride, fast_param->width, 1, 0);
    DT_S32 start_y = 0;
    DT_S32 end_y   = 0;

    start_y = (3 == start_row) ? start_row : (start_row - 1);
    end_y = ((fast_param->height - 2) == end_row) ? end_row : (end_row + 1);

    const DT_U8 *src_row = src.Ptr<DT_U8>(start_y);

    for (DT_S32 y = start_y; y < end_y; y++)
    {
        if (y + 4 < end_y)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<DT_U8>(y + 4)), L2fetch_param);
        }

        Fast9U8Row(src_row, fast_param, max_num_corners, threshold, nonmax_suppression, y, start_y);

        src_row = src.Ptr<DT_U8>(y + 1);
    }

    fast_param->corners = fast_param->corners - fast_param->num_corners;

    return Status::OK;
}

static Status Fast9U8Hvx(Context *ctx, const Mat &src, std::vector<KeyPoint> *key_points, DT_U32 max_num_corners,
                         DT_S32 threshold, DT_BOOL nonmax_suppression)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height = src.GetSizes().m_height;

    auto thread_ids = wp->GetComputeThreadIDs();
    ThreadObject<FastParam> share_fast_param(ctx, thread_ids, src.GetSizes().m_width, max_num_corners);

    ret = wp->ParallelFor((DT_S32)3, height - 2, Fast9U8HvxImpl, ctx, std::cref(src), std::ref(share_fast_param), max_num_corners, threshold,
                           nonmax_suppression);

    FastParam *fast_param = NULL;
    DT_U32 rem_corners = max_num_corners;
    DT_U32 corners_num = 0;

    for (auto &id : thread_ids)
    {
        fast_param = share_fast_param.GetObject(id);
        if (NULL == fast_param)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetObject failed");
            ret = Status::ERROR;
            break;
        }

        // Determine the remaining number and update the remaining number.
        corners_num = rem_corners >= fast_param->num_corners ? fast_param->num_corners : rem_corners;
        rem_corners -= corners_num;
        FastCorner *fast_corners = fast_param->corners;

        for (DT_U32 i = 0; i < corners_num; i++)
        {
            key_points->emplace_back(KeyPoint(static_cast<DT_F32>(fast_corners[i].x),
                                              static_cast<DT_F32>(fast_corners[i].y), 7.f, -1,
                                              static_cast<DT_F32>(fast_corners[i].score)));
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Fast9Hvx(Context *ctx, const Mat &src, std::vector<KeyPoint> *key_points, DT_S32 threshold,
                DT_BOOL nonmax_suppression, DT_U32 max_num_corners)
{
    Status ret = Status::ERROR;

    if (src.GetElemType() == ElemType::U8)
    {
        ret = Fast9U8Hvx(ctx, src, key_points, max_num_corners, threshold, nonmax_suppression);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "src only support U8");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
