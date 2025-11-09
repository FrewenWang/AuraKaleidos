#include "fast_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static DT_VOID MakeOffsetsNone(DT_S32 pixel[25], DT_S32 stride, DT_S32 pattern_size)
{
    const DT_S32 offsets16[][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    const DT_S32 offsets12[][2] =
    {
        {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
        {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
    };

    const DT_S32 offsets8[][2] =
    {
        {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
    };

    const DT_S32 (*offsets)[2] = pattern_size == 16 ? offsets16 :
                                 pattern_size == 12 ? offsets12 :
                                 pattern_size == 8  ? offsets8  : 0;

    DT_S32 k = 0;
    for (; k < pattern_size; k++)
    {
        pixel[k] = offsets[k][0] + offsets[k][1] * stride;
    }

    for (; k < 25; k++)
    {
        pixel[k] = pixel[k - pattern_size];
    }
}

template<DT_S32 PATTERN_SIZE>
static DT_S32 CornerScoreNone(const DT_U8 *src_c, const DT_S32 pixel[], DT_S32 threshold);

template<>
DT_S32 CornerScoreNone<16>(const DT_U8 *src_c, const DT_S32 pixel[], DT_S32 threshold)
{
    constexpr DT_S32 ksize = 8;
    constexpr DT_S32 nsize = ksize * 3 + 1;

    DT_S32 k, center = src_c[0];
    DT_S16 diff[nsize];

    for (k = 0; k < nsize; k++)
    {
        diff[k] = static_cast<DT_S16>(center - src_c[pixel[k]]);
    }

    DT_S32 a0 = threshold;
    for (k = 0; k < 16; k += 2)
    {
        DT_S32 a = Min(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        a = Min(a, static_cast<DT_S32>(diff[k + 3]));
        if (a <= a0)
        {
            continue;
        }
        a  = Min(a, static_cast<DT_S32>(diff[k + 4]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 5]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 6]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 7]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 8]));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k])));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k + 9])));
    }

    DT_S32 b0 = -a0;
    for (k = 0; k < 16; k += 2)
    {
        DT_S32 b = Max(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        b = Max(b, static_cast<DT_S32>(diff[k + 3]));
        b = Max(b, static_cast<DT_S32>(diff[k + 4]));
        b = Max(b, static_cast<DT_S32>(diff[k + 5]));
        if (b >= b0)
        {
            continue;
        }
        b  = Max(b, static_cast<DT_S32>(diff[k + 6]));
        b  = Max(b, static_cast<DT_S32>(diff[k + 7]));
        b  = Max(b, static_cast<DT_S32>(diff[k + 8]));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k])));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k + 9])));
    }

    threshold = -b0 - 1;

    return threshold;
}

template<>
DT_S32 CornerScoreNone<12>(const DT_U8 *src_c, const DT_S32 pixel[], DT_S32 threshold)
{
    constexpr DT_S32 ksize = 6;
    constexpr DT_S32 nsize = ksize * 3 + 1;

    DT_S32 k, center = src_c[0];
    DT_S16 diff[nsize + 4];

    for (k = 0; k < nsize; k++)
    {
        diff[k] = static_cast<DT_S16>(center - src_c[pixel[k]]);
    }

    DT_S32 a0 = threshold;
    for (k = 0; k < 12; k += 2)
    {
        DT_S32 a = Min(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        if (a <= a0)
        {
            continue;
        }
        a  = Min(a, static_cast<DT_S32>(diff[k + 3]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 4]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 5]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 6]));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k])));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k + 7])));
    }

    DT_S32 b0 = -a0;
    for (k = 0; k < 12; k += 2)
    {
        DT_S32 b = Max(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        b = Max(b, static_cast<DT_S32>(diff[k + 3]));
        b = Max(b, static_cast<DT_S32>(diff[k + 4]));
        if (b >= b0)
        {
            continue;
        }
        b  = Max(b, static_cast<DT_S32>(diff[k + 5]));
        b  = Max(b, static_cast<DT_S32>(diff[k + 6]));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k])));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k + 7])));
    }

    threshold = -b0 - 1;

    return threshold;
}

template<>
DT_S32 CornerScoreNone<8>(const DT_U8 *src_c, const DT_S32 pixel[], DT_S32 threshold)
{
    constexpr DT_S32 ksize = 4;
    constexpr DT_S32 nsize = ksize * 3 + 1;

    DT_S32 k, center = src_c[0];
    DT_S16 diff[nsize];

    for (k = 0; k < nsize; k++)
    {
        diff[k] = static_cast<DT_S16>(center - src_c[pixel[k]]);
    }

    DT_S32 a0 = threshold;
    for (k = 0; k < 8; k += 2)
    {
        DT_S32 a = Min(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        if (a <= a0)
        {
            continue;
        }
        a  = Min(a, static_cast<DT_S32>(diff[k + 3]));
        a  = Min(a, static_cast<DT_S32>(diff[k + 4]));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k])));
        a0 = Max(a0, Min(a, static_cast<DT_S32>(diff[k + 5])));
    }

    DT_S32 b0 = -a0;
    for (k = 0; k < 8; k += 2)
    {
        DT_S32 b = Max(static_cast<DT_S32>(diff[k + 1]), static_cast<DT_S32>(diff[k + 2]));
        b = Max(b, static_cast<DT_S32>(diff[k + 3]));
        if (b >= b0)
        {
            continue;
        }
        b  = Max(b, static_cast<DT_S32>(diff[k + 4]));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k])));
        b0 = Min(b0, Max(b, static_cast<DT_S32>(diff[k + 5])));
    }

    threshold = -b0 - 1;

    return threshold;
}

template <DT_S32 PATTERN_SIZE>
static Status FastNoneImpl(Context *ctx, const Mat &mat, std::vector<KeyPoint> &key_points, DT_S32 threshold, DT_BOOL nonmax_suppression)
{
    DT_S32 iwidth  = mat.GetSizes().m_width;
    DT_S32 iheight = mat.GetSizes().m_height;
    DT_S32 istride = mat.GetRowPitch();

    constexpr DT_S32 ksize = PATTERN_SIZE / 2;
    constexpr DT_S32 nsize = PATTERN_SIZE + ksize + 1;

    DT_S32 x, y, k, pixel[25];
    MakeOffsetsNone(pixel, istride, PATTERN_SIZE);

    key_points.clear();

    threshold = Clamp<DT_U8>(threshold, 0, 255);

    DT_U8 threshold_tab[512];
    for (x = -255; x <= 255; x++)
    {
        threshold_tab[x + 255] = static_cast<DT_U8>(x < -threshold ? 1 : x > threshold ? 2 : 0);
    }

    DT_S32 buffer_size = (iwidth + 16) * 3 * (sizeof(DT_S32) + sizeof(DT_U8)) + 128;
    DT_U8 *buffer      = static_cast<DT_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, buffer_size, 0));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    DT_U8 *row_ptr[3];
    row_ptr[0] = buffer;
    row_ptr[1] = row_ptr[0] + iwidth;
    row_ptr[2] = row_ptr[1] + iwidth;

    DT_S32 *cprow[3];
    cprow[0] = reinterpret_cast<DT_S32*>(((DT_UPTR_T)(row_ptr[2] + iwidth) + sizeof(DT_S32) - 1) & -(sizeof(DT_S32))) + 1;
    cprow[1] = cprow[0] + iwidth + 1;
    cprow[2] = cprow[1] + iwidth + 1;

    memset(row_ptr[0], 0, iwidth * 3);

    y = 3;
    for (; y < (iheight - 2); y++)
    {
        const DT_U8 *src_row = mat.Ptr<DT_U8>(y) + 3;
        DT_U8 *curr          = row_ptr[(y - 3) % 3];
        DT_S32 *cornerpos    = cprow[(y - 3) % 3];

        memset(curr, 0, iwidth);
        DT_S32 ncorners = 0;

        if (y < (iheight - 3))
        {
            x = 3;
            for (; x < (iwidth - 3); x++, src_row++)
            {
                DT_S32 center    = src_row[0];
                const DT_U8 *tab = &threshold_tab[0] - center + 255;
                DT_S32 cmp_val   = tab[src_row[pixel[0]]] | tab[src_row[pixel[8]]];

                if (0 == cmp_val)
                {
                    continue;
                }

                cmp_val &= tab[src_row[pixel[2]]] | tab[src_row[pixel[10]]];
                cmp_val &= tab[src_row[pixel[4]]] | tab[src_row[pixel[12]]];
                cmp_val &= tab[src_row[pixel[6]]] | tab[src_row[pixel[14]]];

                if (0 == cmp_val)
                {
                    continue;
                }

                cmp_val &= tab[src_row[pixel[1]]] | tab[src_row[pixel[9]]];
                cmp_val &= tab[src_row[pixel[3]]] | tab[src_row[pixel[11]]];
                cmp_val &= tab[src_row[pixel[5]]] | tab[src_row[pixel[13]]];
                cmp_val &= tab[src_row[pixel[7]]] | tab[src_row[pixel[15]]];

                if (cmp_val & 1)
                {
                    DT_S32 bound = center - threshold, count = 0;
                    for (k = 0; k < nsize; k++)
                    {
                        DT_S32 neighbor = src_row[pixel[k]];
                        if (neighbor < bound)
                        {
                            if (++count > ksize)
                            {
                                cornerpos[ncorners++] = x;
                                curr[x] = nonmax_suppression ? (static_cast<DT_U8>(CornerScoreNone<PATTERN_SIZE>(src_row, pixel, threshold))) : 0;
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
                    DT_S32 bound = center + threshold, count = 0;
                    for (k = 0; k < nsize; k++)
                    {
                        DT_S32 neighbor = src_row[pixel[k]];
                        if (neighbor > bound)
                        {
                            if (++count > ksize)
                            {
                                cornerpos[ncorners++] = x;
                                curr[x] = nonmax_suppression ? (static_cast<DT_U8>(CornerScoreNone<PATTERN_SIZE>(src_row, pixel, threshold))) : 0;
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

        if (3 == y)
        {
            continue;
        }

        const DT_U8 *prev  = row_ptr[(y - 4 + 3) % 3];
        const DT_U8 *pprev = row_ptr[(y - 5 + 3) % 3];

        cornerpos = cprow[(y - 4 + 3) % 3];
        ncorners  = cornerpos[-1];

        for (k = 0; k < ncorners; k++)
        {
            x = cornerpos[k];
            DT_S32 score = prev[x];
            if (!nonmax_suppression ||
               (score > prev[x + 1]  && score > prev[x - 1] && score > pprev[x - 1] && score > pprev[x] &&
                score > pprev[x + 1] && score > curr[x - 1] && score > curr[x]      && score > curr[x + 1]))
            {
                key_points.emplace_back(KeyPoint(static_cast<DT_F32>(x), static_cast<DT_F32>(y - 1), 7.f, -1, static_cast<DT_F32>(score)));
            }
        }
    }

    AURA_FREE(ctx, buffer);

    return Status::OK;
}

FastNone::FastNone(Context *ctx, const OpTarget &target) : FastImpl(ctx, target)
{}

Status FastNone::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 threshold,
                         DT_BOOL nonmax_suppression, FastDetectorType type, DT_U32 max_num_corners)
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

    return Status::OK;
}

Status FastNone::Run()
{
    Status ret = Status::ERROR;
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mat is null");
        return Status::ERROR;
    }

    switch (m_detector_type)
    {
        case FastDetectorType::FAST_5_8:
        {
            ret = FastNoneImpl<8>(m_ctx, *src, *m_key_points, m_threshold, m_nonmax_suppression);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FastNoneImpl<8> failed");
            }
            break;
        }

        case FastDetectorType::FAST_7_12:
        {
            ret = FastNoneImpl<12>(m_ctx, *src, *m_key_points, m_threshold, m_nonmax_suppression);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FastNoneImpl<12> failed");
            }
            break;
        }

        case FastDetectorType::FAST_9_16:
        {
            ret = FastNoneImpl<16>(m_ctx, *src, *m_key_points, m_threshold, m_nonmax_suppression);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "FastNoneImpl<16> failed");
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