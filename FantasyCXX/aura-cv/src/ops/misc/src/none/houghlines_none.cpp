#include "houghlines_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

struct HoughCmpGt
{
    HoughCmpGt(const MI_S32 *aux) : aux(aux) {}
    MI_BOOL operator()(MI_S32 l1, MI_S32 l2) const
    {
        return (aux[l1] > aux[l2]) || (aux[l1] == aux[l2] && l1 < l2);
    }
    const MI_S32 *aux;
};

static AURA_VOID CreateTrigTable(MI_S32 numangle, MI_F64 min_theta, MI_F64 theta_step, MI_F32 irho, MI_F32 *tab_sin, MI_F32 *tab_cos)
{
    MI_F32 ang = static_cast<MI_F32>(min_theta);
    for (MI_S32 n = 0; n < numangle; ang += theta_step, n++)
    {
        tab_sin[n] = static_cast<MI_F32>(Sin(static_cast<MI_F64>(ang)) * irho);
        tab_cos[n] = static_cast<MI_F32>(Cos(static_cast<MI_F64>(ang)) * irho);
    }
}

static AURA_VOID FindLocalMaximums(MI_S32 numrho, MI_S32 numangle, MI_S32 threshold, const Mat &accum_mat, std::vector<MI_S32> &sort_buffer)
{
    for (MI_S32 n = 0; n < numangle; n++)
    {
        const MI_S32 *accum = accum_mat.Ptr<MI_S32>(n + 1);

        for (MI_S32 r = 0; r < numrho; r++)
        {
            MI_S32 base = r + 1;
            if ((accum[base] > threshold) && (accum[base] > accum[base - 1]) && (accum[base] >= accum[base + 1]) &&
                (accum[base] > accum[base - numrho - 2]) && (accum[base] >= accum[base + numrho + 2]))
            {
                sort_buffer.emplace_back((n + 1) * (numrho + 2) + base);
            }
        }
    }
}

// Multi-Scale variant of Classical Hough Transform
struct HoughIndex
{
    HoughIndex() : value(0), rho(0.f), theta(0.f) {}
    HoughIndex(MI_S32 value, MI_F32 rho, MI_F32 theta) : value(value), rho(rho), theta(theta) {}

    MI_S32 value;
    MI_F32 rho, theta;
};

AURA_INLINE MI_F32 CvAtanf(MI_F32 y, MI_F32 x)
{
    constexpr MI_F32 atan2_p1 =  0.9997878412794807f  * static_cast<MI_F32>(180 / AURA_PI);
    constexpr MI_F32 atan2_p3 = -0.3258083974640975f  * static_cast<MI_F32>(180 / AURA_PI);
    constexpr MI_F32 atan2_p5 =  0.1555786518463281f  * static_cast<MI_F32>(180 / AURA_PI);
    constexpr MI_F32 atan2_p7 = -0.04432655554792128f * static_cast<MI_F32>(180 / AURA_PI);

    MI_F32 ax = Abs(x), ay = Abs(y);
    MI_F32 a, c, c2;

    if (ax >= ay)
    {
        c = ay / (ax + static_cast<MI_F32>(DBL_EPSILON));
        c2 = c * c;
        a = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
    }
    else
    {
        c = ax / (ay + static_cast<MI_F32>(DBL_EPSILON));
        c2 = c * c;
        a = 90.f - (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c;
    }

    if (x < 0)
    {
        a = 180.f - a;
    }

    if (y < 0)
    {
        a = 360.f - a;
    }

    return a;
}

static Status HoughLinesStandardNoneImpl(Context *ctx, const Mat &mat, std::vector<Scalar> &lines, LinesType line_type, MI_F32 rho,
                                         MI_F32 theta, MI_S32 threshold, MI_S32 lines_max, MI_F64 min_theta, MI_F64 max_theta)
{
    if (lines_max <= 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "lines_max should big than 0");
        return Status::ERROR;
    }

    MI_S32 width   = mat.GetSizes().m_width;
    MI_S32 height  = mat.GetSizes().m_height;
    MI_S32 max_rho = width + height;
    MI_S32 min_rho = -max_rho;

    if (min_theta > max_theta)
    {
        AURA_ADD_ERROR_STRING(ctx, "max_theta must be greater than min_theta");
        return Status::ERROR;
    }

    MI_S32 numangle = Round((max_theta - min_theta) / theta);
    MI_S32 numrho   = Round(((max_rho - min_rho) + 1) / rho);

    Mat accum_mat(ctx, ElemType::S32, Sizes3((numangle + 2), (numrho + 2), 1));
    if (!(accum_mat.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "mat create failed");
        return Status::ERROR;
    }
    MI_S32 *accum = accum_mat.Ptr<MI_S32>(0);
    memset(accum, 0, (numangle + 2) * (numrho + 2) * sizeof(MI_S32));

    MI_F32 irho = 1 / rho;
    MI_F32 *tab_sin = static_cast<MI_F32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, numangle * sizeof(MI_F32), 0));
    MI_F32 *tab_cos = static_cast<MI_F32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, numangle * sizeof(MI_F32), 0));
    if ((MI_NULL == tab_sin) || (MI_NULL == tab_cos))
    {
        AURA_FREE(ctx, tab_sin);
        AURA_FREE(ctx, tab_cos);
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    // create sin and cos table
    CreateTrigTable(numangle, min_theta, theta, irho, tab_sin, tab_cos);

    // stage 1. fill accumulator
    for (MI_S32 y = 0; y < height; y++)
    {
        const MI_U8 *iaura = mat.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            if (iaura[x] != 0)
            {
                for (MI_S32 n = 0; n < numangle; n++)
                {
                    MI_S32 r = Round(x * tab_cos[n] + y * tab_sin[n]);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
            }
        }
    }

    // stage 2. find local maximums
    std::vector<MI_S32> sort_buffer;
    sort_buffer.reserve(1024);

    FindLocalMaximums(numrho, numangle, threshold, accum_mat, sort_buffer);

    // stage 3. sort the detected lines by accumulator value
    std::sort(sort_buffer.begin(), sort_buffer.end(), HoughCmpGt(accum));

    // stage 4. store the first min(total,lines_max) lines to the output buffer
    lines_max = Min(lines_max, static_cast<MI_S32>(sort_buffer.size()));
    MI_F64 scale = 1. / (numrho + 2);

    if (LinesType::VEC2F == line_type)
    {
        for (MI_S32 i = 0; i < lines_max; i++)
        {
            MI_S32 idx        = sort_buffer[i];
            MI_S32 n          = Floor(idx * scale) - 1;
            MI_S32 r          = idx - (n + 1) * (numrho + 2) - 1;
            MI_F32 line_rho   = (r - (numrho - 1) * 0.5f) * rho;
            MI_F32 line_angle = static_cast<MI_F32>(min_theta) + n * theta;

            lines.emplace_back(static_cast<MI_F64>(line_rho), static_cast<MI_F64>(line_angle), 0.f, 0.f);
        }
    }
    else if (LinesType::VEC3F == line_type)
    {
        for (MI_S32 i = 0; i < lines_max; i++)
        {
            MI_S32 idx        = sort_buffer[i];
            MI_S32 n          = Floor(idx * scale) - 1;
            MI_S32 r          = idx - (n + 1) * (numrho + 2) - 1;
            MI_F32 line_rho   = (r - (numrho - 1) * 0.5f) * rho;
            MI_F32 line_angle = static_cast<MI_F32>(min_theta) + n * theta;

            lines.emplace_back(static_cast<MI_F64>(line_rho), static_cast<MI_F64>(line_angle), static_cast<MI_F64>(accum[idx]), 0.f);
        }
    }

    AURA_FREE(ctx, tab_sin);
    AURA_FREE(ctx, tab_cos);

    return Status::OK;
}

static Status HoughLinesSDivNoneImpl(Context *ctx, const Mat &mat, std::vector<Scalar> &lines, LinesType line_type, MI_F32 rho, MI_F32 theta,
                                     MI_S32 threshold, MI_S32 srn, MI_S32 stn, MI_S32 lines_max, MI_F64 min_theta, MI_F64 max_theta)
{
    if (lines_max < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "lines_max should > 0");
        return Status::ERROR;
    }

    constexpr MI_F32 d2r = static_cast<MI_F32>((AURA_PI / 180));
    MI_S32 sfn     = srn * stn;
    MI_F32 irho    = 1 / rho;
    MI_F32 itheta  = 1 / theta;
    MI_F32 srho    = rho / srn;
    MI_F32 stheta  = theta / stn;
    MI_F32 isrho   = 1 / srho;
    MI_F32 istheta = 1 / stheta;

    MI_S32 width      = mat.GetSizes().m_width;
    MI_S32 height     = mat.GetSizes().m_height;
    MI_S32 rho_num    = Floor(Sqrt(static_cast<MI_F64>(width) * width + static_cast<MI_F64>(height) * height) * irho);
    MI_S32 theta_num  = Floor(2 * AURA_PI * itheta);
    MI_S32 accum_size = rho_num * theta_num;

    std::vector<HoughIndex> lst;
    lst.reserve(1024);
    threshold = Min(threshold, static_cast<MI_S32>(255));
    lst.emplace_back(threshold, -1.f, 0.f);

    // Precalculate sin table
    std::vector<MI_F32> sin_table(5 * theta_num * stn, 0.f);

    for (MI_S32 index = 0; index < 5 * theta_num * stn; index++)
    {
        sin_table[index] = Cos(stheta * index * 0.2f);
    }

    // Counting all feature pixels
    MI_S32 fn = 0;

    for (MI_S32 y = 0; y < height; y++)
    {
        const MI_U8 *src = mat.Ptr<MI_U8>(y);
        for(MI_S32 x = 0; x < width; x++)
        {
            fn += (src[x] != 0);
        }
    }

    std::vector<MI_S32> point_x(fn, 0), point_y(fn, 0);  //store feature point
    std::vector<MI_U8> caccum(accum_size, static_cast<MI_U8>(0));
    MI_S32 fi = 0;

    // Full Hough Transform (it's accumulator update part)
    for (MI_S32 y = 0; y < height; y++)
    {
        const MI_U8 *src = mat.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            if (src[x])
            {
                // Remember the feature point
                point_x[fi] = x;
                point_y[fi] = y;
                fi++;

                /* Update the accumulator */
                MI_F32 xc        = static_cast<MI_F32>(x) + 0.5f;
                MI_F32 yc        = static_cast<MI_F32>(y) + 0.5f;
                MI_F32 cur_theta = Abs(CvAtanf(yc, xc) * d2r);
                MI_F32 cur_rho   = static_cast<MI_F32>(Sqrt(static_cast<MI_F64>(xc) * xc + static_cast<MI_F64>(yc) * yc));
                MI_F32 r0        = cur_rho * irho;
                MI_S32 ti0       = Floor((cur_theta + AURA_PI * 0.5) * itheta);
                caccum[ti0]++;

                MI_F32 theta_it     = (rho / cur_rho) < theta ? (rho / cur_rho) : theta;  // Value of theta for iterating
                MI_F32 scale_factor = theta_it * itheta;
                MI_S32 halftn       = Floor(AURA_PI / theta_it);
                MI_F32 phi          = theta_it - static_cast<MI_F32>(AURA_PI * 0.5);
                MI_F32 phi1         = (theta_it + cur_theta) * itheta;
                MI_S32 iprev        = -1;
                MI_S32 cmax         = 0;

                for (MI_S32 ti1 = 1; ti1 < halftn; ti1++)
                {
                    MI_F32 rv = r0 * Cos(phi);   // Some temporary rho value
                    MI_S32 i = static_cast<MI_S32>(rv) * theta_num;
                    i += Floor(phi1);

                    if ((i < 0) || (i >= accum_size))
                    {
                        AURA_ADD_ERROR_STRING(ctx, "i should >= 0 and < (rho_num * theta_num)");
                        return Status::ERROR;
                    }

                    caccum[i] = static_cast<MI_U8>(caccum[i] + ((i ^ iprev) != 0));
                    iprev = i;

                    if (cmax < caccum[i])
                    {
                        cmax = caccum[i];
                    }

                    phi += theta_it;
                    phi1 += scale_factor;
                }
            }
        }
    }

    // Starting additional analysis
    MI_S32 count = 0;
    for (MI_S32 ri = 0; ri < rho_num; ri++)
    {
        for (MI_S32 ti = 0; ti < theta_num; ti++)
        {
            if (caccum[ri * theta_num + ti] > threshold)
            {
                count++;
            }
        }
    }

    if ((count * 100) > accum_size)
    {
        if (Status::OK == HoughLinesStandardNoneImpl(ctx, mat, lines, line_type, rho, theta, threshold, lines_max, min_theta, max_theta))
        {
            return Status::OK;
        }
        else
        {
            AURA_ADD_ERROR_STRING(ctx, "HoughLinesStandardNoneImpl failed");
            return Status::ERROR;
        }
    }

    std::vector<MI_U8> buffer_vec(srn * stn + 2, 0);
    MI_U8 *buffer = &buffer_vec[0];
    MI_U8 *mcaccum = buffer + 1;

    for (MI_S32 ri = 0; ri < rho_num; ri++)
    {
        for (MI_S32 ti = 0; ti < theta_num; ti++)
        {
            if (caccum[ri * theta_num + ti] > threshold)
            {
                memset(mcaccum, 0, sfn * sizeof(MI_U8));

                for (MI_S32 index = 0; index < fn; index++)
                {
                    MI_F32 xc = static_cast<MI_F32>(point_x[index]) + 0.5f;
                    MI_F32 yc = static_cast<MI_F32>(point_y[index]) + 0.5f;

                    // Update the accumulator
                    MI_F32 cur_theta = Abs(CvAtanf(yc, xc) * d2r);
                    MI_F32 cur_rho   = static_cast<MI_F32>(Sqrt(static_cast<MI_F64>(xc) * xc + static_cast<MI_F64>(yc) * yc)) * isrho;
                    MI_S32 ti0       = Floor((cur_theta + AURA_PI * 0.5) * istheta);
                    MI_S32 ti2       = (ti * stn - ti0) * 5;
                    MI_F32 r0        = static_cast<MI_F32>(ri) * srn;

                    for (MI_S32 ti1 = 0; ti1 < stn; ti1++, ti2 += 5)
                    {
                        MI_F32 rv = cur_rho * sin_table[static_cast<MI_S32>((Abs(ti2)))] - r0;
                        MI_S32 i = Floor(rv) * stn + ti1;

                        i = Max(i, static_cast<MI_S32>(-1));
                        i = Min(i, sfn);
                        mcaccum[i]++;

                        if ((i < -1) || (i > sfn))
                        {
                            AURA_ADD_ERROR_STRING(ctx, "i should >= -1 and <= sfn");
                            return Status::ERROR;
                        }
                    }
                }

                // Find peaks in maccum...
                for (MI_S32 index = 0; index < sfn; index++)
                {
                    MI_S32 pos = static_cast<MI_S32>(lst.size() - 1);

                    if ((pos < 0) || (lst[pos].value < mcaccum[index]))
                    {
                        HoughIndex vi(mcaccum[index],
                                      index / stn * srho + ri * rho,
                                      index % stn * stheta + ti * theta - static_cast<MI_F32>(AURA_PI * 0.5));
                        lst.emplace_back(vi);

                        for (; pos >= 0; pos--)
                        {
                            if (lst[pos].value > vi.value)
                            {
                                break;
                            }

                            lst[pos + 1] = lst[pos];
                        }

                        lst[pos + 1] = vi;

                        if (static_cast<MI_S32>(lst.size()) > lines_max)
                        {
                            lst.pop_back();
                        }
                    }
                }
            }
        }
    }

    MI_S32 pos = static_cast<MI_S32>(lst.size() - 1);
    if ((pos >= 0) && (lst[pos].rho < 0))
    {
        lst.pop_back();
    }

    if (LinesType::VEC2F == line_type)
    {
        for (MI_U64 idx = 0; idx < lst.size(); idx++)
        {
            lines.emplace_back(static_cast<MI_F64>(lst[idx].rho), static_cast<MI_F64>(lst[idx].theta), 0.f, 0.f);
        }
    }
    else if (LinesType::VEC3F == line_type)
    {
        for (MI_U64 idx = 0; idx < lst.size(); idx++)
        {
            lines.emplace_back(static_cast<MI_F64>(lst[idx].rho), static_cast<MI_F64>(lst[idx].theta),
                               static_cast<MI_F64>(lst[idx].value), 0.f);
        }
    }

    return Status::OK;
}

static Status HoughLinesNoneImpl(Context *ctx, const Mat &mat, std::vector<Scalar> &lines, LinesType line_type, MI_F64 rho, MI_F64 theta,
                                 MI_S32 threshold, MI_F64 srn, MI_F64 stn, MI_F64 min_theta, MI_F64 max_theta)
{
    Status ret = Status::ERROR;

    if ((0 == srn) && (0 == stn))
    {
        ret = HoughLinesStandardNoneImpl(ctx, mat, lines, line_type, static_cast<MI_F32>(rho), static_cast<MI_F32>(theta),
                                         threshold, INT_MAX, min_theta, max_theta);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "HoughLinesStandardNoneImpl excute failed");
            return Status::ERROR;
        }
    }
    else
    {
        ret = HoughLinesSDivNoneImpl(ctx, mat, lines, line_type, static_cast<MI_F32>(rho), static_cast<MI_F32>(theta),
                                     threshold, Round(srn), Round(stn), INT_MAX, min_theta, max_theta);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "HoughLinesSDivNoneImpl excute failed");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

AURA_INLINE MI_U32 Next(MI_U64 &state)
{
    state = static_cast<MI_U64>(static_cast<MI_U32>(state)) * 4164903690U + static_cast<MI_U32>(state >> 32);
    return static_cast<MI_U32>(state);
}

AURA_INLINE MI_S32 Uniform(MI_S32 a, MI_S32 b, MI_U64 &state)
{
    return a == b ? a : static_cast<MI_S32>(Next(state) % (b - a) + a);
}

static Status HoughLinesPNoneImpl(Context *ctx, const Mat &mat, std::vector<Scalari> &lines, MI_F64 rho, MI_F64 theta,
                                  MI_S32 threshold, MI_F64 min_line_length, MI_F64 max_gap)
{
    MI_U64 rng      = static_cast<MI_U64>(-1);
    MI_S32 width    = mat.GetSizes().m_width;
    MI_S32 height   = mat.GetSizes().m_height;
    MI_S32 numangle = Round(AURA_PI / static_cast<MI_F32>(theta));
    MI_S32 numrho   = Round(((width + height) * 2 + 1) / static_cast<MI_F32>(rho));

    Mat accum_mat(ctx, ElemType::S32, Sizes3(numangle, numrho, 1));
    if (!(accum_mat.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "mat create failed");
        return Status::ERROR;
    }
    MI_S32 *accum = accum_mat.Ptr<MI_S32>(0);
    memset(accum, 0, numangle * numrho * sizeof(MI_S32));

    std::vector<MI_F32> trig_tab(numangle * 2, 0.f);
    MI_F32 irho = 1 / static_cast<MI_F32>(rho);

    for (MI_S32 n = 0; n < numangle; n++)
    {
        trig_tab[n * 2]     = static_cast<MI_F32>(Cos(static_cast<MI_F64>(n) * static_cast<MI_F32>(theta)) * irho);
        trig_tab[n * 2 + 1] = static_cast<MI_F32>(Sin(static_cast<MI_F64>(n) * static_cast<MI_F32>(theta)) * irho);
    }

    Mat mask_mat(ctx, ElemType::U8, Sizes3(height, width, 1));
    if (!(mask_mat.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "mat create failed");
        return Status::ERROR;
    }
    MI_U8 *mdata0 = mask_mat.Ptr<MI_U8>(0);
    std::vector<Point2i> nzloc;
    nzloc.reserve(1024);

    // stage 1. collect non-zero iaura points
    for (MI_S32 y = 0; y < height; y++)
    {
        const MI_U8 *data = mat.Ptr<MI_U8>(y);
        MI_U8 *mdata = mask_mat.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            if (data[x])
            {
                mdata[x] = 1;
                nzloc.emplace_back(x, y);
            }
            else
            {
                mdata[x] = 0;
            }
        }
    }

    MI_S32 count       = static_cast<MI_S32>(nzloc.size());
    MI_S32 line_gap    = Round(max_gap);
    MI_S32 line_length = Round(min_line_length);
    MI_S32 lines_max   = INT_MAX;

    // stage 2. process all the points in random order
    for (; count > 0; count--)
    {
        constexpr MI_S32 SHIFT = 16;
        Point2i line_end[2];

        // choose random point out of the remaining ones
        MI_S32  idx     = Uniform(0, count, rng);
        MI_S32  max_val = threshold - 1, max_n = 0;
        Point2i point   = nzloc[idx];
        MI_S32 *adata   = accum_mat.Ptr<MI_S32>(0);
        MI_S32 i        = point.m_y, j = point.m_x, dx0, dy0, xflag;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count - 1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if (!mdata0[i * width + j])
        {
            continue;
        }

        // update accumulator, find the most probable line
        for (MI_S32 n = 0; n < numangle; n++, adata += numrho)
        {
            MI_S32 r = Round(j * trig_tab[n * 2] + i * trig_tab[n * 2 + 1]);
            r += (numrho - 1) / 2;
            MI_S32 val = ++adata[r];
            if (max_val < val)
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if (max_val < threshold)
        {
            continue;
        }

        // from the current point walk in each direction
        // along the found line and extract the line segment
        MI_F32 a = -trig_tab[max_n * 2 + 1];
        MI_F32 b = trig_tab[max_n * 2];
        MI_S32 x0 = j;
        MI_S32 y0 = i;

        if (Abs(a) > Abs(b))
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = Round(b * (1 << SHIFT) / Abs(a));
            y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = Round(a * (1 << SHIFT) / Abs(b));
            x0 = (x0 << SHIFT) + (1 << (SHIFT - 1));
        }

        for (MI_S32 k = 0; k < 2; k++)
        {
            MI_S32 gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
            {
                dx = -dx;
                dy = -dy;
            }

            // walk along the line using fixed-point arithmetic,
            // stop at the iaura border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                MI_U8 *mdata;
                MI_S32 i1, j1;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> SHIFT;
                }
                else
                {
                    j1 = x >> SHIFT;
                    i1 = y;
                }

                if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                {
                    break;
                }

                mdata = mdata0 + i1 * width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    gap = 0;
                    line_end[k].m_y = i1;
                    line_end[k].m_x = j1;
                }
                else if (++gap > line_gap)
                {
                    break;
                }
            }
        }

        MI_S32 good_line = (Abs(line_end[1].m_x - line_end[0].m_x) >= line_length) || (Abs(line_end[1].m_y - line_end[0].m_y) >= line_length);

        for (MI_S32 k = 0; k < 2; k++)
        {
            MI_S32 x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
            {
                dx = -dx;
                dy = -dy;
            }

            // walk along the line using fixed-point arithmetic,
            // stop at the iaura border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                MI_U8 *mdata;
                MI_S32 i1, j1;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> SHIFT;
                }
                else
                {
                    j1 = x >> SHIFT;
                    i1 = y;
                }

                mdata = mdata0 + i1 * width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    if (good_line)
                    {
                        adata = accum_mat.Ptr<MI_S32>(0);
                        for (MI_S32 n = 0; n < numangle; n++, adata += numrho)
                        {
                            MI_S32 r = Round(j1 * trig_tab[n * 2] + i1 * trig_tab[n * 2 + 1]);
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if (i1 == line_end[k].m_y && j1 == line_end[k].m_x)
                {
                    break;
                }
            }
        }

        if (good_line)
        {
            lines.emplace_back(line_end[0].m_x, line_end[0].m_y, line_end[1].m_x, line_end[1].m_y);
            if (static_cast<MI_S32>(lines.size()) >= lines_max)
            {
                return Status::OK;
            }
        }
    }

    return Status::OK;
}

HoughLinesNone::HoughLinesNone(Context *ctx, const OpTarget &target) : HoughLinesImpl(ctx, target)
{}

Status HoughLinesNone::SetArgs(const Array *src, std::vector<Scalar> &lines, LinesType line_type, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                               MI_F64 srn, MI_F64 stn, MI_F64 min_theta, MI_F64 max_theta)
{
    if (HoughLinesImpl::SetArgs(src, lines, line_type, rho, theta, threshold, srn, stn, min_theta, max_theta) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "HoughLinesImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status HoughLinesNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if ((MI_NULL == src))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    m_lines->clear();

    ret = HoughLinesNoneImpl(m_ctx, *src, *m_lines, m_line_type, m_rho, m_theta, m_threshold, m_srn, m_stn, m_min_theta, m_max_theta);

    AURA_RETURN(m_ctx, ret);
}

HoughLinesPNone::HoughLinesPNone(Context *ctx, const OpTarget &target) : HoughLinesPImpl(ctx, target)
{}

Status HoughLinesPNone::SetArgs(const Array *src, std::vector<Scalari> &lines, MI_F64 rho, MI_F64 theta, MI_S32 threshold,
                                MI_F64 min_line_length, MI_F64 max_gap)
{
    if (HoughLinesPImpl::SetArgs(src, lines, rho, theta, threshold, min_line_length, max_gap) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "HoughLinesPImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status HoughLinesPNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if ((MI_NULL == src))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    m_lines->clear();

    ret = HoughLinesPNoneImpl(m_ctx, *src, *m_lines, m_rho, m_theta, m_threshold, m_min_line_length, m_max_gap);

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura