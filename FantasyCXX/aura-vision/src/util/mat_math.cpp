//
// Created by liruiyang on 19-1-9.
//

#include "mat_math.h"

#include <math.h>
#include <float.h>
#include <string.h>
#include <opencv2/core/core.hpp>

#define MAX_ITERATION_COUNT 30
//#define FLT_EPSILON      0x1.0p-23

namespace aura::vision {

    void MatMath::scale_image_bilinear(const uint8_t *src, params2d_t &src_addr, uint8_t *dst, params2d_t &dst_addr,
                                       float x_scale, float y_scale, float src_offset_x, float src_offset_y,
                                       int16_t dst_offset_x, int16_t dst_offset_y) {
        uint32_t x, y;
        float a[4];
        float src_x_f, src_y_f, xf, yf, dx, dy;
        uint32_t src_x, src_y;
        uint8_t tl, tr, bl, br;

        for (y = 0u; y < dst_addr.dim_y; y++) {
            for (x = 0u; x < dst_addr.dim_x; x++) {

                /* Apply scale factors to find input pixel for each output pixel */
                src_x_f = ((float) (x + dst_offset_x) + 0.5f) * x_scale - 0.5f;
                src_y_f = ((float) (y + dst_offset_y) + 0.5f) * y_scale - 0.5f;

                xf = floorf(src_x_f);
                yf = floorf(src_y_f);
                dx = src_x_f - xf;
                dy = src_y_f - yf;

                src_x = (int32_t) (xf + src_offset_x);
                src_y = (int32_t) (yf + src_offset_y);

                a[0] = (1.0f - dx) * (1.0f - dy);
                a[1] = (1.0f - dx) * (dy);
                a[2] = (dx) * (1.0f - dy);
                a[3] = (dx) * (dy);

                tl = src[src_y * src_addr.stride_y + src_x];
                tr = src[src_y * src_addr.stride_y + src_x + 1];
                bl = src[(src_y + 1) * src_addr.stride_y + src_x];
                br = src[(src_y + 1) * src_addr.stride_y + src_x + 1];

                dst[(y * dst_addr.stride_y) + x] = (uint8_t) (a[0] * tl + a[2] * tr
                                                              + a[1] * bl + a[3] * br + 0.5f);
            }
        }
    }

    void MatMath::mean_stddev(uint8_t *base_ptr, params2d_t &addrs, float *fmean, float *fstddev,
                              uint32_t *pixels_processed, float *fcurrent_sum) {
        uint32_t x, y;
        float sum_diff_sqrs = 0.0f;
        *fmean = *fstddev = 0.0f;

        for (y = 0; y < addrs.dim_y; y++) {
            for (x = 0; x < addrs.dim_x; x++) {
                *fcurrent_sum += (float) base_ptr[y * addrs.stride_y + x];
            }
        }

        *pixels_processed += addrs.dim_x * addrs.dim_y;
        *fmean = *fcurrent_sum / *pixels_processed;

        for (y = 0; y < addrs.dim_y; y++) {
            for (x = 0; x < addrs.dim_x; x++) {
                sum_diff_sqrs += (float) powf(
                        base_ptr[y * addrs.stride_y + x] - *fmean, 2);
            }
        }

        *fstddev = (float) sqrtf(sum_diff_sqrs / *pixels_processed);
    }

    void MatMath::convert_to_bidiag(const int rows, const int cols, float *u, float *v, float *diag,
                                    float *superdiag) {
        int i, j, k;
        float s, s2, si, scale, half_norm_squared;

        /* Householder processing */
        s = 0;
        scale = 0;
        for (i = 0; i < cols; i++) {
            superdiag[i] = scale * s;
            /* process columns */
            scale = 0;
            for (j = i; j < rows; j++) {
                scale += fabs(u[i + j * cols]);
            }
            if (scale > 0) {
                s2 = 0;
                for (j = i; j < rows; j++) {
                    u[i + j * cols] = u[i + j * cols] / scale;
                    s2 += u[i + j * cols] * u[i + j * cols];
                }
                if (u[i + i * cols] < 0) {
                    s = sqrt(s2);
                } else {
                    s = -sqrt(s2);
                }
                half_norm_squared = u[i + i * cols] * s - s2;
                u[i + i * cols] -= s;
                for (j = i + 1; j < cols; j++) {
                    si = 0;
                    for (k = i; k < rows; k++) {
                        si += u[i + k * cols] * u[j + k * cols];
                    }
                    si = si / half_norm_squared;
                    for (k = i; k < rows; k++) {
                        u[j + k * cols] += si * u[i + k * cols];
                    }
                }
            } /* if (scale>0) */
            for (j = i; j < rows; j++) {
                u[i + j * cols] *= scale;
            }
            diag[i] = s * scale;
            /* process rows */
            s = 0;
            scale = 0;
            if ((i < rows) && (i != cols - 1)) {
                for (j = i + 1; j < cols; j++) {
                    scale += fabs(u[j + i * cols]);
                }
                if (scale > 0) {
                    s2 = 0;
                    for (j = i + 1; j < cols; j++) {
                        u[j + i * cols] = u[j + i * cols] / scale;
                        s2 += u[j + i * cols] * u[j + i * cols];
                    }
                    j--;
                    if (u[j + i * cols] < 0) {
                        s = sqrtf(s2);
                    } else {
                        s = -sqrtf(s2);
                    }
                    half_norm_squared = u[i + 1 + i * cols] * s - s2;
                    u[i + 1 + i * cols] -= s;
                    for (k = i + 1; k < cols; k++) {
                        superdiag[k] = u[k + i * cols] / half_norm_squared;
                    }
                    if (i < rows - 1) {
                        for (j = i + 1; j < rows; j++) {
                            si = 0;
                            for (k = i + 1; k < cols; k++) {
                                si += u[k + i * cols] * u[k + j * cols];
                            }
                            for (k = i + 1; k < cols; k++) {
                                u[k + j * cols] += si * superdiag[k];
                            }
                        }
                    }
                } /* if (scale>0) */
                for (k = i + 1; k < cols; k++) {
                    u[k + i * cols] *= scale;
                }
            } /* if ((i<Nrows)&&(i!=Ncols-1)) */
        } /* for (i=0;i<Ncols;i++) */

        /* update V */
        v[cols * cols - 1] = 1;
        s = superdiag[cols - 1];
        for (i = cols - 2; i >= 0; i--) {
            if (s != 0) {
                for (j = i + 1; j < cols; j++) {
                    v[i + j * cols] = u[j + i * cols]
                                      / (u[i + 1 + i * cols] * s);
                }
                for (j = i + 1; j < cols; j++) {
                    si = 0;
                    for (k = i + 1; k < cols; k++) {
                        si += u[k + i * cols] * v[j + k * cols];
                    }
                    for (k = i + 1; k < cols; k++) {
                        v[j + k * cols] += si * v[i + k * cols];
                    }
                }
            } /* if (s!=0) */
            for (j = i + 1; j < cols; j++) {
                v[j + i * cols] = 0;
                v[i + j * cols] = 0;
            }
            v[i + i * cols] = 1;
            s = superdiag[i];
        } /* for (i=Ncols-2;i>=0;i--) */
        /* expand U to from Nrows x Ncols to */
        /*                  Nrows x Nrows    */
        if (rows > cols) {
            for (i = rows - 1; i >= 0; i--) {
                for (j = rows - 1; j >= 0; j--) {
                    if (j <= cols - 1) {
                        u[j + i * rows] = u[j + i * cols];
                    } else {
                        u[j + i * rows] = 0;
                    }
                }
            }
        }
        /* update U */
        for (i = cols - 1; i >= 0; i--) {
            s = diag[i];
            for (j = i + 1; j < cols; j++) {
                u[j + i * rows] = 0;
            }
            if (s != 0) {
                for (j = i + 1; j < rows; j++) {
                    si = 0;
                    for (k = i + 1; k < rows; k++) {
                        si += u[i + k * rows] * u[j + k * rows];
                    }
                    si = si / (u[i + i * rows] * s);
                    for (k = i; k < rows; k++) {
                        u[j + k * rows] += si * u[i + k * rows];
                    }
                }
                /* initial U1 */
                if (i == cols - 1) {
                    for (j = i; j < rows; j++) {
                        for (k = rows - 1; k >= i + 1; k--) {
                            u[k + j * rows] = u[i + j * rows] * u[i + k * rows]
                                              / (u[i + i * rows] * s);
                            if (j == k) {
                                u[k + j * rows] += 1;
                            }
                        }
                    }
                }
                for (j = i; j < rows; j++) {
                    u[i + j * rows] = u[i + j * rows] / s;
                }
            } else { /* if (s!=0) */
                if (i == cols - 1) {
                    for (k = 1; k <= rows - cols; k++) {
                        u[i + k + (i + k) * rows] = 1;
                    }
                }
                for (j = i; j < rows; j++) {
                    u[i + j * rows] = 0;
                }
            } /* if (s!=0) */
            u[i + i * rows] += 1;
        } /* for (i=Ncols-1;i>=0;i--) */
    }

    int MatMath::bidiag_to_diag(const int rows, const int cols, float *u, float *v, float *diag, float *superdiag) {
        int row, i, k, m, rotation_test, iter, total_iter;
        float x, y, z, epsilon;
        float c, s, f, g, h;

        iter = 0;
        total_iter = 0;
        /* ------------------------------------------------------------------- */
        /* find max in col                                                     */
        /* ------------------------------------------------------------------- */
        x = 0;
        for (i = 0; i < cols; i++) {
            y = fabsf(diag[i]) + fabsf(superdiag[i]);
            if (x < y) {
                x = y;
            }
        }
        epsilon = FLT_EPSILON * x;
        for (k = cols - 1; k >= 0; k--) {
            total_iter += iter;
            iter = 0;
            while (1 == 1) {
                rotation_test = 1;
                for (m = k; m >= 0; m--) {
                    if (fabsf(superdiag[m]) <= epsilon) {
                        rotation_test = 0;
                        break;
                    }
                    if (fabsf(diag[m - 1]) <= epsilon) {
                        break;
                    }
                } /* for (m=k;m>=0;m--) */
                if (rotation_test) {
                    c = 0;
                    s = 1;
                    for (i = m; i <= k; i++) {
                        f = s * superdiag[i];
                        superdiag[i] = c * superdiag[i];
                        if (fabsf(f) <= epsilon) {
                            break;
                        }
                        g = diag[i];
                        h = sqrtf(f * f + g * g);
                        diag[i] = h;
                        c = g / h;
                        s = -f / h;
                        for (row = 0; row < rows; row++) {
                            y = u[m - 1 + row * rows];
                            z = u[i + row * rows];
                            u[m - 1 + row * rows] = y * c + z * s;
                            u[i + row * rows] = -y * s + z * c;
                        }
                    } /* for (i=m;i<=k;i++) */
                } /* if (rotation_test) */
                z = diag[k];
                if (m == k) {
                    if (z < 0) {
                        diag[k] = -z;
                        for (row = 0; row < cols; row++) {
                            v[k + row * cols] = -v[k + row * cols];
                        }
                    } /* if (z>0) */
                    break;
                } else { /* if (m==k) */
                    if (iter >= MAX_ITERATION_COUNT) {
                        return -1;
                    }
                    iter++;
                    x = diag[m];
                    y = diag[k - 1];
                    g = superdiag[k - 1];
                    h = superdiag[k];
                    f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
                    g = sqrt(f * f + 1);
                    if (f < 0) {
                        g = -g;
                    }
                    f = ((x - z) * (x + z) + h * (y / (f + g) - h)) / x;
                    /* next QR transformation */
                    c = 1;
                    s = 1;
                    for (i = m + 1; i <= k; i++) {
                        g = superdiag[i];
                        y = diag[i];
                        h = s * g;
                        g = g * c;
                        z = sqrt(f * f + h * h);
                        superdiag[i - 1] = z;
                        c = f / z;
                        s = h / z;
                        f = x * c + g * s;
                        g = -x * s + g * c;
                        h = y * s;
                        y = c * y;
                        for (row = 0; row < cols; row++) {
                            x = v[i - 1 + row * cols];
                            z = v[i + row * cols];
                            v[i - 1 + row * cols] = x * c + z * s;
                            v[i + row * cols] = -x * s + z * c;
                        }
                        z = sqrtf(f * f + h * h);
                        diag[i - 1] = z;
                        if (z != 0) {
                            c = f / z;
                            s = h / z;
                        }
                        f = c * g + s * y;
                        x = -s * g + c * y;
                        for (row = 0; row < rows; row++) {
                            y = u[i - 1 + row * rows];
                            z = u[i + row * rows];
                            u[i - 1 + row * rows] = c * y + s * z;
                            u[i + row * rows] = -s * y + c * z;
                        }
                    } /* for (i=m+1;i<=k;i++) */
                    superdiag[m] = 0;
                    superdiag[k] = f;
                    diag[k] = x;
                } /* if (m==k) */
            } /* while (1==1) */
        } /* for (k=Ncols-1:k>=0;k--) */

        return total_iter;

    }

    void MatMath::sort_singular_values(const int rows, const int cols, float *u, float *v, float *singular_values) {
        int i, j, row, max_index;
        float temp;

        for (i = 0; i < cols - 1; i++) {
            max_index = i;
            for (j = i + 1; j < cols; j++) {
                if (singular_values[j] > singular_values[max_index]) {
                    max_index = j;
                }
            }
            if (max_index != i) {
                temp = singular_values[i];
                singular_values[i] = singular_values[max_index];
                singular_values[max_index] = temp;
                for (row = 0; row < rows; row++) {
                    temp = u[max_index + row * rows];
                    u[max_index + row * rows] = u[i + row * rows];
                    u[i + row * rows] = temp;
                }
                for (row = 0; row < cols; row++) {
                    temp = v[max_index + row * cols];
                    v[max_index + row * cols] = v[i + row * cols];
                    v[i + row * cols] = temp;
                }
            }
        }
    }

    void MatMath::svd(const int rows, const int cols, float *a, float *u, float *v, float *u1, float *diag,
                      float *superdiag) {
        int row, col, Nrows1, Ncols1;

        /* ------------------------------------------------------------------- */
        /* copy A matrix to U                                                  */
        /* ------------------------------------------------------------------- */
        if (rows >= cols) {
            Nrows1 = rows;
            Ncols1 = cols;
            memcpy(u, a, sizeof(float) * rows * cols);
        } else {
            /* transpose matrix */
            for (row = 0; row < rows; row++) {
                for (col = 0; col < cols; col++) {
                    u[row + col * rows] = a[col + row * cols];
                }
            }
            Nrows1 = cols;
            Ncols1 = rows;
        }

        /* ------------------------------------------------------------------- */
        /* convert A to bidiagonal matrix using Householder reflections        */
        /* ------------------------------------------------------------------- */
        convert_to_bidiag(Nrows1, Ncols1, u, v, diag, superdiag);

        /* ------------------------------------------------------------------- */
        /* convert bidiagonal to diagonal using Givens rotations               */
        /* ------------------------------------------------------------------- */
        bidiag_to_diag(Nrows1, Ncols1, u, v, diag, superdiag);

        /* ------------------------------------------------------------------- */
        /* sort singular values in descending order                            */
        /* ------------------------------------------------------------------- */
        sort_singular_values(Nrows1, Ncols1, u, v, diag);

        /* ------------------------------------------------------------------- */
        /* switch U and V                                                      */
        /* ------------------------------------------------------------------- */
        if (cols > rows) {
            memcpy(u1, v, sizeof(double) * rows * rows);
            memcpy(v, u, sizeof(double) * cols * cols);
            memcpy(u, u1, sizeof(double) * rows * rows);
        }
    }

    int MatMath::cholesky(int enable_test, const int order, float *a, float *l) {
        short i, j, k;
        float sum, sum1;

        if (enable_test) {
            /* test A for positive definite matrix:    */
            /* z_transpose*A*z>0 where z=1,2,...order  */
            sum1 = 0;
            for (i = 0; i < order; i++) {
                sum = 0;
                for (j = 0; j < order; j++) {
                    sum += a[i * order + j] * (float) (j + 1);
                }
                sum1 += (float) (i + 1) * sum;
            }
            if (sum1 <= 0) {
                return -1;
            }
        }

        /* generate lower diagonal matrix L */
        for (j = 0; j < order; j++) {

            /* diagonal entry */
            sum = 0.0;
            for (k = 0; k <= j - 1; k++) {
                sum += l[j * order + k] * l[j * order + k];
            }
            l[j * order + j] = sqrtf(a[j * order + j] - sum);

            /* lower triangular entries */
            for (i = j + 1; i < order; i++) {
                sum = 0.0;
                for (k = 0; k <= j - 1; k++) {
                    sum += l[i * order + k] * l[j * order + k];
                }
                l[i * order + j] = (a[i * order + j] - sum) / l[j * order + j];
            }
        }

        return 0;
    }

    void MatMath::cholesky_solver(const int order, float *l, float *y, float *b, float *x) {
        short i, k;
        float sum;

        /* solve L*y=b for y using forward substitution */
        for (i = 0; i < order; i++) {
            if (i == 0) {
                y[i] = b[i] / l[0];
            } else {
                sum = 0.0;
                for (k = 0; k <= i - 1; k++)
                    sum += l[i * order + k] * y[k];
                y[i] = (b[i] - sum) / l[i * order + i];
            }
        }

        /* solve U*x=y for x using backward substitution */
        for (i = order - 1; i >= 0; i--) {
            if (i == order - 1) {
                x[i] = y[i] / l[i * order + i];
            } else {
                sum = 0.0;
                for (k = order - 1; k >= i + 1; k--)
                    sum += l[k * order + i] * x[k];
                x[i] = (y[i] - sum) / l[i * order + i];
            }
        }
    }

    void MatMath::mat_mul(float *x1, const int r1, const int c1, float *x2, const int c2, float *y) {
        int i, j, k;
        float sum;

        for (i = 0; i < r1; i++)
            for (j = 0; j < c2; j++) {
                sum = 0;
                for (k = 0; k < c1; k++)
                    sum += x1[k + i * c1] * x2[j + k * c2];
                y[j + i * c2] = sum;
            }
    }

    void MatMath::mat_trans(const float *x, const int rows, const int cols, float *y) {
        int i, j;

        for (i = 0; i < cols; i++)
            for (j = 0; j < rows; j++)
                y[i * rows + j] = x[i + cols * j];
    }

    void MatMath::find_col_max_min(cv::Mat &mat, int rows, int cols, float *max_arr, float *min_arr) {
        for (int i = 0; i < cols; ++i) {
            float max = mat.at<float>(0, i);
            float min = max;
            for (int j = 0; j < rows; ++j) {
                float tmp = mat.at<float>(j, i);
                if (tmp > max) {
                    max = tmp;
                }
                if (tmp < min) {
                    min = tmp;
                }
            }
            max_arr[i] = max;
            min_arr[i] = min;
        }
    }

    void MatMath::color_normalization(float *input, int rows, int cols, int channels,
                                      std::vector<float> &mean, std::vector<float> &std, NORMAL_ALG algorithm) {
        int p = 0;
        for (int h = 0; h < rows; ++h) {
            for (int w = 0; w < cols; ++w) {
                for (int c = 0; c < channels; ++c) {
                    if (algorithm == MUL) {
                        input[p] = (input[p] - mean[c]) * std[c];
                    } else {
                        input[p] = (input[p] - mean[c]) / std[c];
                    }
                    p++;
                }
            }
        }
    }

    void MatMath::convert_hwc_to_chw(cv::Mat &img, float *pblob) {
        int rows = img.rows;
        int cols = img.cols;
        int img_size = rows * cols;
        if (img.channels() == 1) {
            cv::Mat img_32f;
            img.convertTo(img_32f, CV_32FC1);
            memcpy(pblob, img_32f.ptr(0), img_size * sizeof(float));
        } else {
            float *data = (float *) img.data;
            int count = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < rows * cols; j++) {
                    pblob[count] = data[j * 3 + i];
                    count += 1;
                }
            }
        }
    }
}