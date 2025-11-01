#include "fast_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

static AURA_VOID MakeOffsets9Neon(MI_S32 pixel[], MI_S32 stride)
{
    const MI_S32 d_stride = stride + stride;
    const MI_S32 t_stride = d_stride + stride;
    pixel[16] = pixel[0]  = t_stride;
    pixel[17] = pixel[1]  = 1 + t_stride;
    pixel[18] = pixel[2]  = 2 + d_stride;
    pixel[19] = pixel[3]  = 3 + stride;
    pixel[20] = pixel[4]  = 3;
    pixel[21] = pixel[5]  = 3 - stride;
    pixel[22] = pixel[6]  = 2 - d_stride;
    pixel[23] = pixel[7]  = 1 - t_stride;
    pixel[24] = pixel[8]  = -t_stride;
    pixel[9]  = -1 - t_stride;
    pixel[10] = -2 - d_stride;
    pixel[11] = -3 - stride;
    pixel[12] = -3;
    pixel[13] = -3 + stride;
    pixel[14] = -2 + d_stride;
    pixel[15] = -1 + t_stride;
}

static MI_U8 CornerScore9Neon(const MI_U8 *src, const MI_S32 pixel[])
{
    MI_S32 k, center = src[0];
    MI_S16 diff[32];

    for (k = 0; k < 25; k++)
    {
        diff[k] = static_cast<MI_S16>(center - src[pixel[k]]);
    }

    int16x8_t vqs16_q0;
    neon::vdup(vqs16_q0, static_cast<MI_S16>(-1000));
    int16x8_t vqs16_q1;
    neon::vdup(vqs16_q1, static_cast<MI_S16>(1000));

    int16x8_t vqs16_diff0  = neon::vload1q(diff +  0);
    int16x8_t vqs16_diff8  = neon::vload1q(diff +  8);
    int16x8_t vqs16_diff16 = neon::vload1q(diff + 16);
    int16x8_t vqs16_diff24 = neon::vload1q(diff + 24);

    // k == 0
    int16x8_t vqs16_diff0_r0  = neon::vext<1>(vqs16_diff0, vqs16_diff8);
    int16x8_t vqs16_diff0_r1  = neon::vext<2>(vqs16_diff0, vqs16_diff8);
    int16x8_t vqs16_diff0_min = neon::vmin(vqs16_diff0_r0, vqs16_diff0_r1);
    int16x8_t vqs16_diff0_max = neon::vmax(vqs16_diff0_r0, vqs16_diff0_r1);

    vqs16_diff0_r0  = neon::vext<3>(vqs16_diff0, vqs16_diff8);
    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff0_r0);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff0_r0);

    vqs16_diff0_r1  = neon::vext<4>(vqs16_diff0, vqs16_diff8);
    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff0_r1);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff0_r1);

    vqs16_diff0_r0  = neon::vext<5>(vqs16_diff0, vqs16_diff8);
    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff0_r0);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff0_r0);

    vqs16_diff0_r1  = neon::vext<6>(vqs16_diff0, vqs16_diff8);
    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff0_r1);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff0_r1);

    vqs16_diff0_r0  = neon::vext<7>(vqs16_diff0, vqs16_diff8);
    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff0_r0);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff0_r0);

    vqs16_diff0_min = neon::vmin(vqs16_diff0_min, vqs16_diff8);
    vqs16_diff0_max = neon::vmax(vqs16_diff0_max, vqs16_diff8);

    vqs16_q0 = neon::vmax(vqs16_q0, neon::vmin(vqs16_diff0_min, vqs16_diff0));
    vqs16_q1 = neon::vmin(vqs16_q1, neon::vmax(vqs16_diff0_max, vqs16_diff0));

    vqs16_diff0_r1 = neon::vext<1>(vqs16_diff8, vqs16_diff16);
    vqs16_q0       = neon::vmax(vqs16_q0, neon::vmin(vqs16_diff0_min, vqs16_diff0_r1)); // min[(0-9),(1-10),(2-11), ... (7-16)]
    vqs16_q1       = neon::vmin(vqs16_q1, neon::vmax(vqs16_diff0_max, vqs16_diff0_r1)); // max[(0-9),(1-10),(2-11), ... (7-16)]

    // k == 8
    int16x8_t vqs16_diff8_r0  = vqs16_diff0_r1;
    int16x8_t vqs16_diff8_r1  = neon::vext<2>(vqs16_diff8, vqs16_diff16);
    int16x8_t vqs16_diff8_min = neon::vmin(vqs16_diff8_r0, vqs16_diff8_r1);
    int16x8_t vqs16_diff8_max = neon::vmax(vqs16_diff8_r0, vqs16_diff8_r1);

    vqs16_diff8_r0  = neon::vext<3>(vqs16_diff8, vqs16_diff16);
    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff8_r0);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff8_r0);

    vqs16_diff8_r1  = neon::vext<4>(vqs16_diff8, vqs16_diff16);
    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff8_r1);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff8_r1);

    vqs16_diff8_r0  = neon::vext<5>(vqs16_diff8, vqs16_diff16);
    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff8_r0);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff8_r0);

    vqs16_diff8_r1  = neon::vext<6>(vqs16_diff8, vqs16_diff16);
    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff8_r1);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff8_r1);

    vqs16_diff8_r0  = neon::vext<7>(vqs16_diff8, vqs16_diff16);
    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff8_r0);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff8_r0);

    vqs16_diff8_min = neon::vmin(vqs16_diff8_min, vqs16_diff16);
    vqs16_diff8_max = neon::vmax(vqs16_diff8_max, vqs16_diff16);

    vqs16_q0 = neon::vmax(vqs16_q0, neon::vmin(vqs16_diff8_min, vqs16_diff8));
    vqs16_q1 = neon::vmin(vqs16_q1, neon::vmax(vqs16_diff8_max, vqs16_diff8));

    vqs16_diff8_r1 = neon::vext<1>(vqs16_diff16, vqs16_diff24);
    vqs16_q0       = neon::vmax(vqs16_q0, neon::vmin(vqs16_diff8_min, vqs16_diff8_r1));
    vqs16_q1       = neon::vmin(vqs16_q1, neon::vmax(vqs16_diff8_max, vqs16_diff8_r1));

    // fin
    int16x8_t vqs16_q    = neon::vmax(vqs16_q0, neon::vsub(neon::vmovq((MI_S16)0), vqs16_q1));
    int16x4_t vqs16_q2   = neon::vmax(neon::vgetlow(vqs16_q), neon::vgethigh(vqs16_q));
    int32x4_t vqs32_q2_w = neon::vmovl(vqs16_q2);
    int32x2_t vds32_q4   = neon::vmax(neon::vgetlow(vqs32_q2_w), neon::vgethigh(vqs32_q2_w));
    int32x2_t vds32_q8   = neon::vmax(vds32_q4, neon::vreinterpret_64<MI_S32>(neon::vshr_n<32>(neon::vreinterpret_64<MI_S64>(vds32_q4))));

    return static_cast<MI_U8>(neon::vgetlane<0>(vds32_q8) - 1);
}

static Status Fast9NeonImpl(Context *ctx, const Mat &mat, std::vector<KeyPoint> &key_points, MI_S32 threshold, MI_BOOL nonmax_suppression,
                            ThreadBuffer &thread_buffer, std::mutex &mutex, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth  = mat.GetSizes().m_width;
    MI_S32 iheight = mat.GetSizes().m_height;
    MI_S32 istride = mat.GetRowPitch();

    constexpr MI_S32 KSIZE = 16 / 2;
    constexpr MI_S32 NSIZE = 16 + KSIZE + 1;

    MI_S32 x, y, k, pixel[25];
    MakeOffsets9Neon(pixel, istride);

    threshold = Clamp(threshold, 0, 255);

    MI_U8 threshold_tab[512];
    for (x = -255; x <= 255; x++)
    {
        threshold_tab[x + 255] = static_cast<MI_U8>(x < -threshold ? 1 : x > threshold ? 2 : 0);
    }

    MI_U8 *buffer = thread_buffer.GetThreadData<MI_U8>();

    if (!buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    MI_U8 *row_ptr[3];
    row_ptr[0] = buffer;
    row_ptr[1] = row_ptr[0] + iwidth;
    row_ptr[2] = row_ptr[1] + iwidth;

    MI_S32 *cprow[3];
    cprow[0] = reinterpret_cast<MI_S32*>(((MI_UPTR_T)(row_ptr[2] + iwidth) + sizeof(MI_S32) - 1) & -(sizeof(MI_S32))) + 1;
    cprow[1] = cprow[0] + iwidth + 1;
    cprow[2] = cprow[1] + iwidth + 1;

    memset(row_ptr[0], 0, iwidth * 3);

    uint8x16_t vqu8_delta;
    neon::vdup(vqu8_delta, static_cast<MI_U8>(128));
    uint8x16_t vqu8_thresh;
    neon::vdup(vqu8_thresh, static_cast<MI_U8>(threshold));

    const MI_S32 width_align16 = ((iwidth - 6) & (~15)) + 3;

    MI_S32 start_y = 0;
    MI_S32 end_y   = 0;

    if (3 == start_row)
    {
        start_y = start_row;
    }
    else
    {
        start_y = start_row - 1;
    }

    if ((iheight - 2) == end_row)
    {
        end_y = end_row;
    }
    else
    {
        end_y = end_row + 1;
    }

    std::vector<KeyPoint> key_points_tmp;
    key_points_tmp.reserve(1024);

    for (y = start_y; y < end_y; y++)
    {
        const MI_U8 *src_row = mat.Ptr<MI_U8>(y);
        MI_U8 *curr          = row_ptr[(y - 3) % 3];
        MI_S32 *cornerpos    = cprow[(y - 3) % 3];

        memset(curr, 0, iwidth);
        MI_S32 ncorners = 0;

        if (y < (iheight - 3))
        {
            x = 3;
            for (; x < width_align16; x += 16)
            {
                uint8x16_t vqu8_src_cc    = neon::vload1q(src_row + x);
                int8x16_t vqs8_down       = neon::vreinterpret(neon::vsub(neon::vload1q(src_row + x + pixel[0]),  vqu8_delta));
                int8x16_t vqs8_right      = neon::vreinterpret(neon::vsub(neon::vload1q(src_row + x + pixel[4]),  vqu8_delta));
                int8x16_t vqs8_up         = neon::vreinterpret(neon::vsub(neon::vload1q(src_row + x + pixel[8]),  vqu8_delta));
                int8x16_t vqs8_left       = neon::vreinterpret(neon::vsub(neon::vload1q(src_row + x + pixel[12]), vqu8_delta));

                int8x16_t vqs8_ls_cc      = neon::vreinterpret(neon::veor(neon::vqsub(vqu8_src_cc, vqu8_thresh), vqu8_delta));
                int8x16_t vqs8_gt_cc      = neon::vreinterpret(neon::veor(neon::vqadd(vqu8_src_cc, vqu8_thresh), vqu8_delta));

                uint8x16_t vqu8_gt_thresh = neon::vand(neon::vcgt(vqs8_down,  vqs8_gt_cc), neon::vcgt(vqs8_right, vqs8_gt_cc));
                uint8x16_t vqu8_ls_thresh = neon::vand(neon::vcgt(vqs8_ls_cc, vqs8_down),  neon::vcgt(vqs8_ls_cc, vqs8_right));

                uint8x16_t vqu8_gt_val    = neon::vand(neon::vcgt(vqs8_right, vqs8_gt_cc), neon::vcgt(vqs8_up,    vqs8_gt_cc));
                uint8x16_t vqu8_ls_val    = neon::vand(neon::vcgt(vqs8_ls_cc, vqs8_right), neon::vcgt(vqs8_ls_cc, vqs8_up));
                vqu8_gt_thresh            = neon::vorr(vqu8_gt_thresh, vqu8_gt_val);
                vqu8_ls_thresh            = neon::vorr(vqu8_ls_thresh, vqu8_ls_val);

                vqu8_gt_val               = neon::vand(neon::vcgt(vqs8_up,    vqs8_gt_cc), neon::vcgt(vqs8_left,  vqs8_gt_cc));
                vqu8_ls_val               = neon::vand(neon::vcgt(vqs8_ls_cc, vqs8_up),    neon::vcgt(vqs8_ls_cc, vqs8_left));
                vqu8_gt_thresh            = neon::vorr(vqu8_gt_thresh, vqu8_gt_val);
                vqu8_ls_thresh            = neon::vorr(vqu8_ls_thresh, vqu8_ls_val);

                vqu8_gt_val               = neon::vand(neon::vcgt(vqs8_left,  vqs8_gt_cc), neon::vcgt(vqs8_down,  vqs8_gt_cc));
                vqu8_ls_val               = neon::vand(neon::vcgt(vqs8_ls_cc, vqs8_left),  neon::vcgt(vqs8_ls_cc, vqs8_down));
                vqu8_gt_thresh            = neon::vorr(vqu8_gt_thresh, vqu8_gt_val);
                vqu8_ls_thresh            = neon::vorr(vqu8_ls_thresh, vqu8_ls_val);

                uint64x2_t mask = neon::vreinterpret_64<MI_U64>(neon::vorr(vqu8_gt_thresh, vqu8_ls_thresh));

                if (mask[0] || mask[1])
                {
                    uint8x16_t vqu8_gt_val = neon::vmovq(static_cast<MI_U8>(0));
                    uint8x16_t vqu8_ls_val = neon::vmovq(static_cast<MI_U8>(0));
                    uint8x16_t vqu8_gt_result = neon::vmovq(static_cast<MI_U8>(0));
                    uint8x16_t vqu8_ls_result = neon::vmovq(static_cast<MI_U8>(0));

                    for(k = 0; k < NSIZE; k++)
                    {
                        int8x16_t vqs8_round = neon::vreinterpret(neon::veor(neon::vload1q(src_row + x + pixel[k]), vqu8_delta));

                        vqu8_gt_thresh = neon::vcgt(vqs8_round, vqs8_gt_cc);
                        vqu8_ls_thresh = neon::vcgt(vqs8_ls_cc, vqs8_round);

                        vqu8_gt_val    = neon::vand(neon::vsub(vqu8_gt_val, vqu8_gt_thresh), vqu8_gt_thresh);
                        vqu8_ls_val    = neon::vand(neon::vsub(vqu8_ls_val, vqu8_ls_thresh), vqu8_ls_thresh);

                        vqu8_gt_result = neon::vmax(vqu8_gt_result, vqu8_gt_val);
                        vqu8_ls_result = neon::vmax(vqu8_ls_result, vqu8_ls_val);
                    }

                    uint8x16_t vqu8_ksize;
                    neon::vdup(vqu8_ksize, static_cast<MI_U8>(KSIZE));
                    uint8x16_t m = neon::vcgt(neon::vmax(vqu8_gt_result, vqu8_ls_result), vqu8_ksize);

                    k = (mask[0] == 0 ? 8 : 0);
                    for (; k < 16; k++)
                    {
                        if (m[k])
                        {
                            cornerpos[ncorners++] = x + k;
                            curr[x + k] = nonmax_suppression ? CornerScore9Neon(src_row + x + k, pixel) : 0;
                        }
                    }
                }
            }

            for (x = width_align16; x < (iwidth - 3); x++)
            {
                const MI_U8 *src_ofx = src_row + x;

                MI_S32 center    = src_ofx[0];
                const MI_U8 *tab = &threshold_tab[0] - center + 255;
                MI_S32 cmp_val   = tab[src_ofx[pixel[0]]] | tab[src_ofx[pixel[8]]];

                if (0 == cmp_val)
                {
                    continue;
                }

                cmp_val &= tab[src_ofx[pixel[2]]] | tab[src_ofx[pixel[10]]];
                cmp_val &= tab[src_ofx[pixel[4]]] | tab[src_ofx[pixel[12]]];
                cmp_val &= tab[src_ofx[pixel[6]]] | tab[src_ofx[pixel[14]]];

                if (0 == cmp_val)
                {
                    continue;
                }

                cmp_val &= tab[src_ofx[pixel[1]]] | tab[src_ofx[pixel[9]]];
                cmp_val &= tab[src_ofx[pixel[3]]] | tab[src_ofx[pixel[11]]];
                cmp_val &= tab[src_ofx[pixel[5]]] | tab[src_ofx[pixel[13]]];
                cmp_val &= tab[src_ofx[pixel[7]]] | tab[src_ofx[pixel[15]]];

                if (cmp_val & 1)
                {
                    MI_S32 bound = center - threshold, count = 0;
                    for (k = 0; k < NSIZE; k++)
                    {
                        MI_S32 neighbor = src_ofx[pixel[k]];
                        if (neighbor < bound)
                        {
                            if (++count > KSIZE)
                            {
                                cornerpos[ncorners++] = x;
                                curr[x] = nonmax_suppression ? CornerScore9Neon(src_ofx, pixel) : 0;
                                break;
                            }
                        }
                        else
                        {
                            count = 0;
                        }
                    }
                }

                if (cmp_val & 2)
                {
                    MI_S32 bound = center + threshold, count = 0;
                    for (k = 0; k < NSIZE; k++)
                    {
                        MI_S32 neighbor = src_ofx[pixel[k]];
                        if (neighbor > bound)
                        {
                            if (++count > KSIZE)
                            {
                                cornerpos[ncorners++] = x;
                                curr[x] = nonmax_suppression ? CornerScore9Neon(src_ofx, pixel) : 0;
                                break;
                            }
                        }
                        else
                        {
                            count = 0;
                        }
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if (3 == y || (start_row != 3 && y <= start_row))
        {
            continue;
        }

        const MI_U8 *prev  = row_ptr[(y - 4 + 3) % 3];
        const MI_U8 *pprev = row_ptr[(y - 5 + 3) % 3];

        cornerpos = cprow[(y - 4 + 3) % 3];
        ncorners  = cornerpos[-1];

        for (k = 0; k < ncorners; k++)
        {
            x = cornerpos[k];
            MI_S32 score = prev[x];

            if (!nonmax_suppression ||
               (score > prev[x + 1]  && score > prev[x - 1] && score > pprev[x - 1] && score > pprev[x] &&
                score > pprev[x + 1] && score > curr[x - 1] && score > curr[x]      && score > curr[x + 1]))
            {
                key_points_tmp.emplace_back(KeyPoint(static_cast<MI_F32>(x), static_cast<MI_F32>(y - 1), 7.f, -1, static_cast<MI_F32>(score)));
            }
        }
    }

    std::lock_guard<std::mutex> guard(mutex);
    key_points.insert(key_points.end(), key_points_tmp.begin(), key_points_tmp.end());

    return Status::OK;
}

static Status Fast9Neon(Context *ctx, const Mat &src, std::vector<KeyPoint> &key_points, MI_S32 threshold, MI_BOOL nonmax_suppression, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    MI_S32 buffer_size = (src.GetSizes().m_width + 16) * 3 * (sizeof(MI_S32) + sizeof(MI_U8)) + 128;
    ThreadBuffer thread_buffer(ctx, buffer_size);
    std::mutex mutex;

    key_points.clear();

    Status ret = wp->ParallelFor(3, (src.GetSizes().m_height - 2), Fast9NeonImpl, ctx, std::cref(src), std::ref(key_points),
                                 threshold, nonmax_suppression, std::ref(thread_buffer), std::ref(mutex));

    AURA_RETURN(ctx, ret);
}

FastNeon::FastNeon(Context *ctx, const OpTarget &target) : FastImpl(ctx, target)
{}

Status FastNeon::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, MI_S32 threshold,
                         MI_BOOL nonmax_suppression, FastDetectorType type, MI_U32 max_num_corners)
{
    if (FastImpl::SetArgs(src, key_points, threshold, nonmax_suppression, type, max_num_corners) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FastImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status FastNeon::Run()
{
    Status ret = Status::ERROR;
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    switch (m_detector_type)
    {
        case FastDetectorType::FAST_9_16:
        {
            ret = Fast9Neon(m_ctx, *src, *m_key_points, m_threshold, m_nonmax_suppression, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Fast9Neon failed, FastDetectorType: FAST_9_16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "fast detector type error");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura