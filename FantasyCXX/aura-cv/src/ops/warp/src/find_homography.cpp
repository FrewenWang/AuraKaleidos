#include "aura/ops/warp/find_homography.hpp"
#include "aura/runtime/logger.h"

#define MODEL_POINTS_NUM           (4)
#define MAX_ATTEMPTS               (10000)

namespace aura
{

AURA_ALWAYS_INLINE DT_U64 RandomNext(DT_U64 &state)
{
    state = state * 4164903690U + (state >> 32);
    return state;
}

AURA_ALWAYS_INLINE DT_VOID ElemRotate(DT_F64 &v0, DT_F64 &v1, DT_F64 s, DT_F64 c)
{
    DT_F64 a0 = v0;
    DT_F64 b0 = v1;
    v0 = a0 * c - b0 * s;
    v1 = a0 * s + b0 * c;
}

static DT_BOOL CollinearPoints(std::vector<Point2> &sub_points, DT_S32 count)
{
    DT_S32 i = count - 1;

    for (DT_S32 j = 0; j < i; j++)
    {
        DT_F32 dx1 = sub_points[j].m_x - sub_points[i].m_x;
        DT_F32 dy1 = sub_points[j].m_y - sub_points[i].m_y;

        for (DT_S32 k = 0; k < j; k++)
        {
            DT_F32 dx2 = sub_points[k].m_x - sub_points[i].m_x;
            DT_F32 dy2 = sub_points[k].m_y - sub_points[i].m_y;

            if (Abs(dx2 * dy1 - dy2 * dx1) <= FLT_EPSILON * (Abs(dx1) + Abs(dy1) + Abs(dx2) + Abs(dy2)))
            {
                return DT_TRUE;
            }
        }
    }

    return DT_FALSE;
}

static DT_BOOL CheckSubset(std::vector<Point2> &sub_src_points, std::vector<Point2> &sub_dst_points, DT_S32 count)
{
    if (CollinearPoints(sub_src_points, count) || CollinearPoints(sub_dst_points, count))
    {
        return DT_FALSE;
    }

    if (4 == count)
    {
        const DT_S32 permute[4][3] = { { 0, 1, 2 },{ 1, 2, 3 },{ 0, 2, 3 },{ 0, 1, 3 } };
        DT_S32 negative = 0;

        for (DT_S32 i = 0; i < 4; i++)
        {
            const DT_S32 *t = permute[i];
            DT_F64 det_a = sub_src_points[t[0]].m_x * sub_src_points[t[1]].m_y + sub_src_points[t[0]].m_y * sub_src_points[t[2]].m_x +
                           sub_src_points[t[1]].m_x * sub_src_points[t[2]].m_y - sub_src_points[t[1]].m_y * sub_src_points[t[2]].m_x -
                           sub_src_points[t[0]].m_x * sub_src_points[t[2]].m_y - sub_src_points[t[0]].m_y * sub_src_points[t[1]].m_x;

            DT_F64 det_b = sub_dst_points[t[0]].m_x * sub_dst_points[t[1]].m_y + sub_dst_points[t[0]].m_y * sub_dst_points[t[2]].m_x +
                           sub_dst_points[t[1]].m_x * sub_dst_points[t[2]].m_y - sub_dst_points[t[1]].m_y * sub_dst_points[t[2]].m_x -
                           sub_dst_points[t[0]].m_x * sub_dst_points[t[2]].m_y - sub_dst_points[t[0]].m_y * sub_dst_points[t[1]].m_x;
            negative += (det_a * det_b < 0);
        }

        if ((negative != 0) && (negative != 4))
        {
            return DT_FALSE;
        }
    }

    return DT_TRUE;
}

static DT_BOOL GetSubPoints(const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, std::vector<Point2> &sub_src_points,
                            std::vector<Point2> &sub_dst_points, DT_S32 max_attempts, DT_U64 &state)
{
    DT_U32 idx[MODEL_POINTS_NUM] = { 0 };
    DT_U32 idx_i      = 0;
    DT_S32 i          = 0;
    DT_S32 j          = 0;
    DT_S32 iters      = 0;
    DT_S32 points_num = (DT_S32)src_points.size();

    for (iters = 0; iters < max_attempts; iters++)
	{
        for (i = 0; i < MODEL_POINTS_NUM && iters < max_attempts; i++)
        {
            idx_i = 0;
            for(;;)
            {
                idx_i  = (DT_U32)RandomNext(state) % points_num;
                idx[i] = idx_i;

                for (j = 0; j < i; j++)
                {
                    if (idx_i == idx[j])
                    {
                        break;
                    }
                }

                if (j == i)
                {
                    break;
                }
            }

            sub_src_points[i].m_x = src_points[idx_i].m_x;
            sub_src_points[i].m_y = src_points[idx_i].m_y;
            sub_dst_points[i].m_x = dst_points[idx_i].m_x;
            sub_dst_points[i].m_y = dst_points[idx_i].m_y;
        }

        if ((i == MODEL_POINTS_NUM) && (!CheckSubset(sub_src_points, sub_dst_points, i)))
        {
            continue;
        }

        break;
    }

    return (i == MODEL_POINTS_NUM && iters < max_attempts) ? DT_FALSE : DT_TRUE;
}

static DT_S32 RansacUpdateNumIters(DT_F32 p, DT_F32 ep, DT_S32 model_points_num, DT_S32 max_iters)
{
    p  = p  > 0. ? p  : 0.;
    p  = p  < 1. ? p  : 1.;
    ep = ep > 0. ? ep : 0.;
    ep = ep < 1. ? ep : 1.;

    DT_F32 num   = (1.0 - p) > FLT_MIN ? (1.0 - p) : FLT_MIN;
    DT_F32 denom = 1.0 - pow(1.0 - ep, model_points_num);
    if (denom < FLT_MIN)
    {
        return 0;
    }

    num   = log(num);
    denom = log(denom);

    return ((denom >= 0) || (-num >= max_iters * (-denom))) ? max_iters : Round(num / denom);
}

static DT_VOID SvdJacobiN8(DT_F64 a[][8], DT_F64 w[], DT_F64 v[][8])
{
    const DT_F64 eps = DBL_EPSILON;
    const DT_S32 n   = 8;
    DT_S32 i         = 0;
    DT_S32 j         = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n;j ++)
        {
            v[i][j]= 0.0;
        }
        v[i][i]= 1.0;
    }

    DT_S32 m            = 0;
    DT_S32 k            = 0;
    DT_S32 iters        = 0;
    DT_S32 max_iters    = n * n * 30;
    DT_S32 ind_r[n - 1] = { 0 };
    DT_S32 ind_c[n]     = { 0 };

    DT_F64 mv = 0.0;
    for (k = 0; k < n; k++)
    {
        w[k] = a[k][k];
        if (k < n - 1)
        {
            m  = k + 1;
            mv = Abs(a[k][m]);
            for (i = k + 2; i < n; i++)
            {
                DT_F64 val = Abs(a[k][i]);
                if (mv < val)
                {
                    mv = val;
                    m  = i;
                }
            }
            ind_r[k] = m;
        }

        if (k > 0)
        {
            m  = 0;
            mv = Abs(a[0][k]);

            for (i = 1; i < k; i++)
            {
                DT_F64 val = Abs(a[i][k]);
                if (mv < val)
                {
                    mv = val,
                    m  = i;
                }
            }
            ind_c[k] = m;
        }
    }

    for (iters = 0; iters < max_iters; iters++)
    {
        // find index (k,l) of pivot p
        k  = 0;
        mv = Abs(a[0][ind_r[0]]);
        for (i = 1; i < n - 1; i++)
        {
            DT_F64 val = Abs(a[i][ind_r[i]]);
            if (mv < val)
            {
                mv = val;
                k  = i;
            }
        }

        DT_S32 l = ind_r[k];
        for (i = 1; i < n; i++)
        {
            DT_F64 val = Abs(a[ind_c[i]][i]);
            if (mv < val)
            {
                mv = val;
                k  = ind_c[i];
                l  = i;
            }
        }

        DT_F64 p = a[k][l];
        if (Abs(p) <= eps)
        {
            break;
        }

        DT_F64 y = (w[l] - w[k]) * 0.5;
        DT_F64 t = Abs(y) + hypot(p, y);
        DT_F64 s = hypot(p, t);
        DT_F64 c = t / s;
        s        = p / s;
        t        = (p / t) * p;

        if (y < 0)
        {
            s = -s;
            t = -t;
        }

        a[k][l] = 0;
        w[k] -= t;
        w[l] += t;

        // rotate rows and columns k and l
        for (i = 0; i < k; i++)
        {
            ElemRotate(a[i][k], a[i][l], s, c);
        }

        for (i = k + 1; i < l; i++)
        {
            ElemRotate(a[k][i], a[i][l], s, c);
        }

        for (i = l + 1; i < n; i++)
        {
            ElemRotate(a[k][i], a[l][i], s, c);
        }

        // rotate eigenvectors
        for (i = 0; i < n; i++)
        {
            ElemRotate(v[k][i], v[l][i], s, c);
        }

        for (j = 0; j < 2; j++)
        {
            DT_S32 idx = j == 0 ? k : l;
            if (idx < n - 1)
            {
                m  = idx + 1;
                mv = Abs(a[idx][m]);
                for (i = idx + 2; i < n; i++)
                {
                    DT_F64 val = Abs(a[idx][i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                ind_r[idx] = m;
            }

            if (idx > 0)
            {
                m  = 0;
                mv = Abs(a[0][idx]);
                for (i = 1; i < idx; i++)
                {
                    DT_F64 val = Abs(a[i][idx]);
                    if (mv < val)
                    {
                        mv = val;
                        m  = i;
                    }
                }
                ind_c[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for (k = 0; k < n - 1; k++)
    {
        m = k;
        for (i = k + 1; i < n; i++)
        {
            if (w[m] < w[i])
            {
                m = i;
            }
        }

        if (k != m)
        {
            Swap(w[m], w[k]);
            for (i = 0; i < n; i++)
            {
                Swap(v[m][i], v[k][i]);
            }
        }
    }

    return;
}

static DT_VOID SvdJacobiN9(DT_F64 a[][9], DT_F64 w[], DT_F64 v[][9])
{
    const DT_F64 eps = DBL_EPSILON;
    const DT_S32 n   = 9;
    DT_S32 i         = 0;
    DT_S32 j         = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n;j ++)
        {
            v[i][j]= 0.0;
        }
        v[i][i]= 1.0;
    }

    DT_S32 m            = 0;
    DT_S32 k            = 0;
    DT_S32 iters        = 0;
    DT_S32 max_iters    = n * n * 30;
    DT_S32 ind_r[n - 1] = { 0 };
    DT_S32 ind_c[n]     = { 0 };

    DT_F64 mv = 0.0;
    for (k = 0; k < n; k++)
    {
        w[k] = a[k][k];
        if (k < n - 1)
        {
            m  = k + 1;
            mv = Abs(a[k][m]);
            for (i = k + 2; i < n; i++)
            {
                DT_F64 val = Abs(a[k][i]);
                if (mv < val)
                {
                    mv = val;
                    m  = i;
                }
            }
            ind_r[k] = m;
        }

        if (k > 0)
        {
            m  = 0;
            mv = Abs(a[0][k]);
            for (i = 1; i < k; i++)
            {
                DT_F64 val = Abs(a[i][k]);
                if (mv < val)
                {
                    mv = val,
                    m  = i;
                }

            }
            ind_c[k] = m;
        }
    }

    for (iters = 0; iters < max_iters; iters++)
    {
        // find index (k,l) of pivot p
        k  = 0;
        mv = Abs(a[0][ind_r[0]]);
        for (i = 1; i < n - 1; i++)
        {
            DT_F64 val = Abs(a[i][ind_r[i]]);
            if (mv < val)
            {
                mv = val;
                k  = i;
            }
        }

        DT_S32 l = ind_r[k];
        for (i = 1; i < n; i++)
        {
            DT_F64 val = Abs(a[ind_c[i]][i]);
            if (mv < val)
            {
                mv = val;
                k  = ind_c[i];
                l  = i;
            }
        }

        DT_F64 p = a[k][l];
        if (Abs(p) <= eps)
        {
            break;
        }

        DT_F64 y = (w[l] - w[k]) * 0.5;
        DT_F64 t = Abs(y) + hypot(p, y);
        DT_F64 s = hypot(p, t);
        DT_F64 c = t / s;
        s        = p / s;
        t        = (p / t) * p;

        if (y < 0)
        {
            s = -s;
            t = -t;
        }

        a[k][l] = 0;
        w[k] -= t;
        w[l] += t;

        // rotate rows and columns k and l
        for (i = 0; i < k; i++)
        {
            ElemRotate(a[i][k], a[i][l], s, c);
        }

        for (i = k + 1; i < l; i++)
        {
            ElemRotate(a[k][i], a[i][l], s, c);
        }

        for (i = l + 1; i < n; i++)
        {
            ElemRotate(a[k][i], a[l][i], s, c);
        }

        // rotate eigenvectors
        for (i = 0; i < n; i++)
        {
            ElemRotate(v[k][i], v[l][i], s, c);
        }

        for (j = 0; j < 2; j++)
        {
            DT_S32 idx = j == 0 ? k : l;
            if (idx < n - 1)
            {
                m  = idx + 1;
                mv = Abs(a[idx][m]);
                for (i = idx + 2; i < n; i++)
                {
                    DT_F64 val = Abs(a[idx][i]);
                    if (mv < val)
                    {
                        mv = val;
                        m  = i;
                    }
                }
                ind_r[idx] = m;
            }

            if (idx > 0)
            {
                m  = 0;
                mv = Abs(a[0][idx]);
                for (i = 1; i < idx; i++)
                {
                    DT_F64 val = Abs(a[i][idx]);
                    if (mv < val)
                    {
                        mv = val;
                        m  = i;
                    }
                }
                ind_c[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for (k = 0; k < n - 1; k++)
    {
        m = k;
        for (i = k + 1; i < n; i++)
        {
            if (w[m] < w[i])
            {
                m = i;
            }
        }

        if (k != m)
        {
            Swap(w[m], w[k]);
            for (i = 0; i < n; i++)
            {
                Swap(v[m][i], v[k][i]);
            }
        }
    }

    return;
}

static DT_S32 FindInliers(const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, Mat &dst_h, DT_S32 *mask,
                          DT_F64 reproj_threshold)
{
    DT_S32 inliers_num = 0;
    DT_S32 count       = (DT_S32)src_points.size();
    DT_F64 per_error   = 0.0f;
    reproj_threshold  *= reproj_threshold;

    DT_F64 *h = (DT_F64*)dst_h.Ptr<DT_F64>(0);
    for (DT_S32 i = 0; i < count; i++)
    {
        DT_F64 ww = 1.0f / (h[6] * src_points[i].m_x + h[7] * src_points[i].m_y + 1.0);
        DT_F64 dx = ((h[0] * src_points[i].m_x + h[1] * src_points[i].m_y + h[2]) * ww - dst_points[i].m_x);
        DT_F64 dy = ((h[3] * src_points[i].m_x + h[4] * src_points[i].m_y + h[5]) * ww - dst_points[i].m_y);

        per_error = dx * dx + dy * dy;
        mask[i]   = (per_error <= reproj_threshold) ? 1 : 0;
        inliers_num += mask[i];
    }

    return inliers_num;
}

static DT_VOID ComputeErrorAndJacobi(std::vector<Point2> &src_points, std::vector<Point2> &dst_points, DT_F64 h[8], DT_F64 err[], DT_F64 *jac)
{
    DT_S32 in_num = (DT_S32)src_points.size();
    for (DT_S32 i = 0; i < in_num; i++)
    {
        DT_F64 mx = src_points[i].m_x;
        DT_F64 my = src_points[i].m_y;
        DT_F64 ww = h[6] * mx + h[7] * my + 1.0;

        ww        = Abs(ww) > DBL_EPSILON ? 1.0 / ww : 0.0;
        DT_F64 xi = (h[0] * mx + h[1] * my + h[2])*ww;
        DT_F64 yi = (h[3] * mx + h[4] * my + h[5])*ww;

        err[i * 2]     = xi - dst_points[i].m_x;
        err[i * 2 + 1] = yi - dst_points[i].m_y;

        DT_S32 index   = (i * 2) * 8;
        jac[index    ] = mx * ww;
        jac[index + 1] = my * ww;
        jac[index + 2] = ww;
        jac[index + 3] = 0.0;
        jac[index + 4] = 0.0;
        jac[index + 5] = 0.0;
        jac[index + 6] = -mx * ww * xi;
        jac[index + 7] = -my * ww * xi;

        index          = (i * 2 + 1) * 8;
        jac[index    ] = 0.0;
        jac[index + 1] = 0.0;
        jac[index + 2] = 0.0;
        jac[index + 3] = mx * ww;
        jac[index + 4] = my * ww;
        jac[index + 5] = ww;
        jac[index + 6] = -mx * ww * yi;
        jac[index + 7] = -my * ww * yi;
    }

    return;
}

static DT_F64 NormL2sqr(DT_F64 src[], DT_S32 num)
{
    DT_F64 res = 0.0;
    for (DT_S32 i = 0; i < num; i++)
    {
        res += (src[i] * src[i]);
    }

    return res;
}

static DT_F64 NormInf(DT_F64 src[], DT_S32 num)
{
    DT_F64 res = 0.0;
    for (DT_S32 i = 0; i < num; i++)
    {
        res = Max(res, Abs(src[i]));
    }

    return res;
}

static DT_F64 VectorDot(DT_F64 src1[], DT_F64 src2[])
{
    DT_F64 res = 0.0;
    for (DT_S32 i = 0; i < 8; i++)
    {
        res += src1[i] * src2[i];
    }

    return res;
}

static Status MulTransposed(Context *ctx, std::vector<Point2> &inliers, DT_F64 *src, DT_F64 dst[8][8])
{
    DT_S32 cols = (DT_S32)inliers.size() * 2;
    DT_F64 *src_trans = reinterpret_cast<DT_F64*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 8 * cols * sizeof(DT_F64), 0));
    if (DT_NULL == src_trans)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    DT_S32 i = 0;
    DT_S32 j = 0;
    DT_S32 k = 0;
    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < 8; j++)
        {
            dst[i][j] = 0.0;
        }
    }

    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < cols; j++)
        {
            src_trans[i * cols + j] = src[j * 8 + i];
        }
    }

    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < 8; j++)
        {
            for (k = 0; k < cols; k++)
            {
                dst[i][j] += src_trans[i * cols + k] * src[k * 8 + j];
            }
        }
    }

    AURA_FREE(ctx, src_trans);
    return Status::OK;
}

static Status Gemm1Trans(Context *ctx, std::vector<Point2> &inliers, DT_F64 *j, DT_F64 b[], DT_F64 d[8])
{
    DT_S32 cols = (DT_S32)inliers.size() * 2;
    DT_F64 *j_trans = reinterpret_cast<DT_F64*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 8 * cols * sizeof(DT_F64), 0));
    if (DT_NULL == j_trans)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    DT_S32 x = 0;
    DT_S32 y = 0;
    for (x = 0; x < 8; x++)
    {
        d[x] = 0.0;
    }

    for (x = 0; x < 8; x++)
    {
        for (y = 0; y < cols; y++)
        {
            j_trans[x * cols + y] = j[y * 8 + x];
        }
    }

    for (x = 0; x < 8; x++)
    {
        for (y = 0; y < cols; y++)
        {
            d[x] += j_trans[x * cols + y] * b[y];
        }
    }

    AURA_FREE(ctx, j_trans);
    return Status::OK;
}

static DT_VOID SVBkSbnb1(DT_S32 n, DT_F64 w[], DT_F64 u[][8], DT_F64 v[][8], DT_F64 b[], DT_F64 x[])
{
    DT_F64 eps       = DBL_EPSILON * 2;
    DT_F64 threshold = 0.0;
    DT_S32 i         = 0;
    DT_S32 j         = 0;

    for (i = 0; i < n; i++)
    {
        x[i] = 0.0;
        threshold += w[i];
    }
    threshold *= eps;

    for (i = 0; i < n; i++)
    {
        DT_F64 wi = w[i];
        if (Abs(wi) <= threshold)
        {
            continue;
        }

        wi       = 1.0 / wi;
        DT_F64 s = 0.0;
        for (j = 0; j < n; j++)
        {
            s += u[i][j] * b[j];
        }

        s *= wi;
        for (j = 0; j < n; j++)
        {
            x[j] = x[j] + s * v[i][j];
        }
    }

    return;
}

static DT_VOID SolveDelta(DT_F64 src[8][8], DT_F64 src2[8], DT_F64 dst[8])
{
    const DT_S32 n = 8;
    DT_F64 w[n]    = { 0.0 };
    DT_F64 v[n][n] = {{ 0.0 }};

    SvdJacobiN8(src, w, v);

    DT_F64 u[n][n];
    for (DT_S32 i = 0; i < n; i++)
    {
        for (DT_S32 j = 0; j < n; j++)
        {
            u[i][j] = v[i][j];
        }
    }

    SVBkSbnb1(n, w, u, v, src2, dst);

    return;
}

static DT_VOID Gemm1Trans3Trans(DT_F64 src1[][8], DT_F64 src2[], DT_F64 alpha, DT_F64 src3[], DT_F64 beta, DT_F64 dst[])
{
    const DT_S32 rows          = 8;
    DT_F64 src1_trans[8][rows] = {{ 0 }};
    DT_S32 i                   = 0;
    DT_S32 j                   = 0;

    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < rows; j++)
        {
            src1_trans[i][j] = src1[j][i];
        }
    }

    DT_F64 temp_r[8] = { 0.0 };
    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < rows; j++)
        {
            temp_r[i] += alpha * src1_trans[i][j] * src2[j];
        }
    }

    for (i = 0; i < 8; i++)
    {
        dst[i] = temp_r[i] + beta * src3[i];
    }

    return;
}

static DT_VOID SVBkSbnb8(DT_S32 n, DT_F64 w[], DT_F64 u[][8], DT_F64 v[][8], DT_F64 x[8][8])
{
    DT_F64 buffer[8] = { 0 };
    DT_F64 eps       = DBL_EPSILON * 2;
    DT_F64 threshold = 0.0;
    DT_S32 i         = 0;
    DT_S32 j         = 0;
    DT_S32 k         = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < 8; j++)
        {
            x[i][j] = 0.0;
        }
    }

    for (i = 0; i < 8; i++)
    {
        threshold += w[i];
    }
    threshold *= eps;

    for (i = 0; i < 8; i++)
    {
        DT_F64 wi = w[i];
        if (Abs(wi) <= threshold)
        {
            continue;
        }

        wi = 1.0 / wi;
        for (j = 0; j < 8; j++)
        {
            buffer[j] = u[j][i] * wi;
        }

        for (j = 0; j < n; j++)
        {
            DT_F64 s = v[i][j];
            for (k = 0; k < n; k++)
            {
                x[j][k] += (s * buffer[k]);
            }
        }
    }

    return;
}

static DT_VOID InvertEig(DT_F64 src[][8], DT_F64 dst[][8])
{
    const DT_S32 n  = 8;
    DT_F64 u[n][n]  = {{ 0.0 }};
    DT_F64 w[n]     = { 0.0 };
    DT_F64 vt[n][n] = {{ 0.0 }};

    SvdJacobiN8(src, w, vt);

    for (DT_S32 i = 0; i < n; i++)
    {
        for (DT_S32 j = 0; j < n; j++)
        {
            u[i][j] = vt[j][i];
        }
    }

    SVBkSbnb8(n, w, u, vt, dst);

    return;
}

static Status LMSolver(Context *ctx, std::vector<Point2> &src_points, std::vector<Point2> &dst_points, DT_S32 max_iters, DT_F64 h[8])
{
    DT_S32 in_num  = (DT_S32)src_points.size() * 2;
    DT_F64 *buffer = reinterpret_cast<DT_F64*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 10 * sizeof(DT_F64) * in_num, 0));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    Status ret  = Status::ERROR;
    DT_F64 *pr  = buffer;
    DT_F64 *prd = buffer + in_num;
    DT_F64 *pj  = buffer + in_num * 2;

    const DT_S32 m   = 8;
    DT_F64 x[m]      = { h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7] };
    DT_F64 a[m][m]   = {{ 0.0 }};
    DT_F64 ap[m][m]  = {{ 0.0 }};
    DT_F64 v[m]      = { 0.0 };
    DT_F64 d[m]      = { 0.0 };
    DT_F64 xd[m]     = { 0.0 };
    DT_F64 temp_d[m] = { 0.0 };
    DT_F64 s         = 0.0;
    DT_F64 sd        = 0.0;

    const DT_F64 rlo = 0.25;
    const DT_F64 rhi = 0.75;
    DT_F64 lambda    = 1;
    DT_F64 lc        = 0.75;
    DT_F64 ds        = 0.0;
    DT_F64 r         = 0.0;
    DT_F64 maxval    = DBL_EPSILON;
    DT_S32 iter      = 0;
    DT_S32 proceed   = 1;
    DT_S32 i         = 0;
    DT_S32 j         = 0;

    ComputeErrorAndJacobi(src_points, dst_points, x, pr, pj);
    s = NormL2sqr(pr, in_num);

    ret  = MulTransposed(ctx, src_points, pj, a);
    ret |= Gemm1Trans(ctx, src_points, pj, pr, v);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "MulTransposed/Gemm1Trans failed");
        goto EXIT;
    }

    for (i = 0; i < m; i++)
    {
        d[i] = a[i][i];
    }

    for (;;)
    {
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < m; j++)
            {
                ap[i][j] = a[i][j];
            }
            ap[i][i] += lambda * d[i];
        }

        SolveDelta(ap, v, d);
        for (i = 0; i < m; i++)
        {
            xd[i] = x[i] - d[i];
        }

        ComputeErrorAndJacobi(src_points, dst_points, xd, prd, pj);
        sd = NormL2sqr(prd, in_num);

        Gemm1Trans3Trans(a, d, -1, v, 2, temp_d);

        ds = VectorDot(d, temp_d);
        r  = (s - sd) / (Abs(ds) > DBL_EPSILON ? ds : 1);

        if (r > rhi)
        {
            lambda *= 0.5;
            if (lambda < lc)
            {
                lambda = 0;
            }
        }
        else if (r < rlo)
        {
            // find new nu if r too low
            DT_F64 t  = VectorDot(d, v);
            DT_F64 nu = (sd - s) / (Abs(t) > DBL_EPSILON ? t : 1) + 2;
            nu = Min(Max(nu, (DT_F64)2.0), (DT_F64)10.0);

            if (0 == lambda)
            {
                InvertEig(a, ap);
                maxval = DBL_EPSILON;
                for (i = 0; i < m; i++)
                {
                    maxval = Max(maxval, Abs(ap[i][i]));
                }

                lc = 1.0 / maxval;
                lambda = lc;
                nu *= 0.5;
            }
            lambda *= nu;
        }

        if (sd < s)
        {
            s = sd;
            for(i = 0; i < m; i++)
            {
                Swap(x[i], xd[i]);
            }

            ComputeErrorAndJacobi(src_points, dst_points, x, pr, pj);

            ret  = MulTransposed(ctx, src_points, pj, a);
            ret |= Gemm1Trans(ctx, src_points, pj, pr, v);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "MulTransposed/Gemm1Trans failed");
                goto EXIT;
            }
        }

        iter++;
        proceed = (iter < max_iters) && (NormInf(d, m) >= FLT_EPSILON) && (NormInf(pr, in_num) >= FLT_EPSILON);
        if (!proceed)
        {
            break;
        }
    }

    h[0] = x[0]; h[1] = x[1];
    h[2] = x[2]; h[3] = x[3];
    h[4] = x[4]; h[5] = x[5];
    h[6] = x[6]; h[7] = x[7];

EXIT:
    AURA_FREE(ctx, buffer);
    buffer = DT_NULL;
    pr     = DT_NULL;
    prd    = DT_NULL;
    pj     = DT_NULL;

    return ret;
}

static Status CalHomography(const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, Mat &h_mat)
{
    DT_S32 points_num = (DT_S32)src_points.size();

    const DT_S32 n   = 9;
    DT_F64 w[n]      = { 0.0 };
    DT_F64 ltl[n][n] = {{ 0.0 }};
    DT_F64 v[n][n]   = {{ 0.0 }};
    DT_S32 i = 0;
    DT_S32 j = 0;
    DT_S32 k = 0;

    DT_F64 dst_cm[2] = { 0.0 };
    DT_F64 src_cm[2] = { 0.0 };
    DT_F64 dst_sm[2] = { 0.0 };
    DT_F64 src_sm[2] = { 0.0 };

    for (i = 0; i < points_num; i++)
    {
        src_cm[0] += (DT_F64)src_points[i].m_x;
        src_cm[1] += (DT_F64)src_points[i].m_y;
        dst_cm[0] += (DT_F64)dst_points[i].m_x;
        dst_cm[1] += (DT_F64)dst_points[i].m_y;
    }

    dst_cm[0] /= points_num;
    dst_cm[1] /= points_num;
    src_cm[0] /= points_num;
    src_cm[1] /= points_num;

    for (i = 0; i < points_num; i++)
    {
        src_sm[0] += Abs(src_points[i].m_x - src_cm[0]);
        src_sm[1] += Abs(src_points[i].m_y - src_cm[1]);
        dst_sm[0] += Abs(dst_points[i].m_x - dst_cm[0]);
        dst_sm[1] += Abs(dst_points[i].m_y - dst_cm[1]);
    }

    if (Abs(dst_sm[0]) < DBL_EPSILON || Abs(dst_sm[1]) < DBL_EPSILON ||
        Abs(src_sm[0]) < DBL_EPSILON || Abs(src_sm[1]) < DBL_EPSILON)
    {
        return Status::ERROR;
    }

    dst_sm[0] = points_num / dst_sm[0];
    dst_sm[1] = points_num / dst_sm[1];
    src_sm[0] = points_num / src_sm[0];
    src_sm[1] = points_num / src_sm[1];

    for (i = 0; i < points_num; i++)
    {
        DT_F64 X = (src_points[i].m_x - src_cm[0]) * src_sm[0];
        DT_F64 Y = (src_points[i].m_y - src_cm[1]) * src_sm[1];
        DT_F64 x = (dst_points[i].m_x - dst_cm[0]) * dst_sm[0];
        DT_F64 y = (dst_points[i].m_y - dst_cm[1]) * dst_sm[1];

        DT_F64 lx[n] = { X, Y, 1.f, 0.f, 0.f, 0.f, -x * X, -x * Y, -x };
        DT_F64 ly[n] = { 0.f, 0.f, 0.f, X, Y, 1.f, -y * X, -y * Y, -y };

        for (j = 0; j < n; j++)
        {
            for (k = j; k < n; k++)
            {
                ltl[j][k] += lx[j] * lx[k] + ly[j] * ly[k];
            }
        }
    }

    for (j = 1; j < n; j++)
    {
        for (k = 0; k < j; k++)
        {
            ltl[j][k] = ltl[k][j];
        }
    }

    SvdJacobiN9(ltl, w, v);

    DT_F64 h0[3][3];
    for (i = 0; i < 3; i++)
    {
        h0[i][0] = v[8][i * 3];
        h0[i][1] = v[8][i * 3 + 1];
        h0[i][2] = v[8][i * 3 + 2];
    }

    DT_F64 norm_h0[3][3] = {{ 1. / dst_sm[0], 0, dst_cm[0] }, { 0, 1. / dst_sm[1], dst_cm[1] }, { 0, 0, 1 }};
    DT_F64 norm_h1[3][3] = {{ src_sm[0], 0, -src_cm[0] * src_sm[0] }, { 0, src_sm[1], -src_cm[1] * src_sm[1] }, { 0, 0, 1 }};

    DT_F64 result_h[3][3] = {{ 0.0 }};
    for (i = 0; i < 3; i++)
    {
        result_h[i][0] = norm_h0[i][0] * h0[0][0] + norm_h0[i][1] * h0[1][0] + norm_h0[i][2] * h0[2][0];
        result_h[i][1] = norm_h0[i][0] * h0[0][1] + norm_h0[i][1] * h0[1][1] + norm_h0[i][2] * h0[2][1];
        result_h[i][2] = norm_h0[i][0] * h0[0][2] + norm_h0[i][1] * h0[1][2] + norm_h0[i][2] * h0[2][2];
    }

    for (i = 0; i < 3; i++)
    {
        h0[i][0] = result_h[i][0] * norm_h1[0][0] + result_h[i][1] * norm_h1[1][0] + result_h[i][2] * norm_h1[2][0];
        h0[i][1] = result_h[i][0] * norm_h1[0][1] + result_h[i][1] * norm_h1[1][1] + result_h[i][2] * norm_h1[2][1];
        h0[i][2] = result_h[i][0] * norm_h1[0][2] + result_h[i][1] * norm_h1[1][2] + result_h[i][2] * norm_h1[2][2];
    }

    DT_F64 *H = NULL;
    for (i = 0; i < 3; i++)
    {
        H = (DT_F64*)h_mat.Ptr<DT_F64>(i);
        H[0] = h0[i][0] / h0[2][2];
        H[1] = h0[i][1] / h0[2][2];
        H[2] = h0[i][2] / h0[2][2];
    }

    return Status::OK;
}

static Status FindHomographyRansacImpl(Context *ctx, const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, Mat &h_mat,
                                       DT_F64 reproj_threshold, DT_S32 max_iters, DT_F64 confidence)
{
    Status ret = Status::ERROR;

    DT_S32 max_inliers = MODEL_POINTS_NUM;
    DT_S32 inliers_num = 0;
    DT_U64 state       = 0xffffffffffffffff;
    DT_S32 niters      = max_iters;
    DT_S32 points_num  = (DT_S32)src_points.size();
    std::vector<Point2> src_model_points(MODEL_POINTS_NUM);
    std::vector<Point2> dst_model_points(MODEL_POINTS_NUM);

    Mat mask_mat(ctx, ElemType::S32, Sizes3(2, points_num, 1), AURA_MEM_HEAP);
    if (!mask_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create mat failed");
        return Status::ERROR;
    }

    DT_S32 *mask_iter   = (DT_S32 *)mask_mat.Ptr<DT_S32>(0);
    DT_S32 *mask_result = (DT_S32 *)mask_mat.Ptr<DT_S32>(1);

    if (MODEL_POINTS_NUM == points_num)
    {
        for (DT_S32 i = 0; i < MODEL_POINTS_NUM; i++)
        {
            src_model_points[i].m_x = src_points[i].m_x;
            src_model_points[i].m_y = src_points[i].m_y;
            dst_model_points[i].m_x = dst_points[i].m_x;
            dst_model_points[i].m_y = dst_points[i].m_y;
        }
        niters = 1;
    }

    for (DT_S32 iter = 0; iter < niters; iter++)
    {
        if (points_num > MODEL_POINTS_NUM)
        {
            DT_BOOL found = GetSubPoints(src_points, dst_points, src_model_points, dst_model_points, MAX_ATTEMPTS, state);
            if (found)
            {
                if (0 == iter)
                {
                    AURA_ADD_ERROR_STRING(ctx, "FindHomography GetSubPoints Failed");
                    return Status::ERROR;
                }
                break;
            }
        }

        ret = CalHomography(src_model_points, dst_model_points, h_mat);
        if (ret != Status::OK)
        {
            continue;
        }

        inliers_num = FindInliers(src_points, dst_points, h_mat, mask_iter, reproj_threshold);
        if (inliers_num >= max_inliers)
        {
            max_inliers = inliers_num;
            for (DT_S32 i = 0; i < points_num; i++)
            {
                mask_result[i] = mask_iter[i];
            }
            niters = RansacUpdateNumIters(confidence, (DT_F32)(points_num - inliers_num) / points_num, MODEL_POINTS_NUM, niters);
        }
    }

    std::vector<Point2> src_inliers(max_inliers);
    std::vector<Point2> dst_inliers(max_inliers);

    DT_S32 i, j;
    for (i = 0, j = 0; i < points_num; i++)
    {
        if (1 == mask_result[i])
        {
            src_inliers[j].m_x = src_points[i].m_x;
            src_inliers[j].m_y = src_points[i].m_y;
            dst_inliers[j].m_x = dst_points[i].m_x;
            dst_inliers[j].m_y = dst_points[i].m_y;
            j++;
        }
    }

    src_inliers.resize(j);
    dst_inliers.resize(j);

    ret = CalHomography(src_inliers, dst_inliers, h_mat);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "FindHomography CalHomography Failed");
        return Status::ERROR;
    }

    DT_F64 h8[8]  = {0.0};
    DT_F64 *dst_h = (DT_F64*)h_mat.Ptr<DT_F64>(0);
    h8[0] = dst_h[0]; h8[1] = dst_h[1]; h8[2] = dst_h[2];
    h8[3] = dst_h[3]; h8[4] = dst_h[4]; h8[5] = dst_h[5];
    h8[6] = dst_h[6]; h8[7] = dst_h[7];

    ret = LMSolver(ctx, src_inliers, dst_inliers, 10, h8);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "FindHomography LMSolver Failed");
        return Status::ERROR;
    }

    dst_h[0] = h8[0]; dst_h[1] = h8[1]; dst_h[2] = h8[2];
    dst_h[3] = h8[3]; dst_h[4] = h8[4]; dst_h[5] = h8[5];
    dst_h[6] = h8[6]; dst_h[7] = h8[7]; dst_h[8] = 1.0f;

    return Status::OK;
}

AURA_EXPORTS Mat FindHomography(Context *ctx, const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points,
                                DT_F64 reproj_threshold, DT_S32 max_iters, DT_F64 confidence)
{
    if (DT_NULL == ctx)
    {
        return Mat();
    }

    if (src_points.size() != dst_points.size() || src_points.size() < 4)
    {
        AURA_ADD_ERROR_STRING(ctx, "the number of src points and dst points must be equal and greater than 3");
        return Mat();
    }

    Mat h_mat(ctx, ElemType::F64, Sizes3(3, 3, 1), AURA_MEM_HEAP);
    if (!h_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create mat failed");
        return Mat();
    }

    Status ret = FindHomographyRansacImpl(ctx, src_points, dst_points, h_mat, reproj_threshold, max_iters, confidence);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "FindHomographyRansacImpl failed");
        return Mat();
    }

    return h_mat;
}
} // namespace aura