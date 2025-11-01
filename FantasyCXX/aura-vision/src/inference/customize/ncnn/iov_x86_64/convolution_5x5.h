// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <xmmintrin.h>
#include "sse_fill_mat.h"

static void conv5x5s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out_mat = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

//        out_mat.fill(bias0);
        iov::fill_mat_sse(out_mat, bias0);

        float* out = out_mat;

        for (int q = 0; q<inch; q++)
        {

            const float* img0 = (bottom_blob.channel(q));
            const float* kernel0 = kernel + p*inch * 25 + q * 25;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

            __m128  xmm0, xmm1,xmm2,xmm3,xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12;
            __m128 xmm13, xmm14, xmm15;

            for (int i=0; i < outh; i++)
            {

                const float* r0 = img0 + i*w;
                const float* r1 = img0 + (i+1)*w;
                const float* r2 = img0 + (i+2)*w;
                const float* r3 = img0 + (i+3)*w;
                const float* r4 = img0 + (i+4)*w;
                float* outptr = out + i*outw;

                xmm0 = _mm_load_ss(k0);
                xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0);
                xmm1 = _mm_load_ss(k0+1);
                xmm1 = _mm_shuffle_ps(xmm1, xmm1, 0);
                xmm2 = _mm_load_ss(k0+2);
                xmm2 = _mm_shuffle_ps(xmm2, xmm2, 0);
                xmm3 = _mm_load_ss(k0+3);
                xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0);
                xmm4 = _mm_load_ss(k0+4);
                xmm4 = _mm_shuffle_ps(xmm4, xmm4, 0);
                xmm5 = _mm_load_ss(k1);
                xmm5 = _mm_shuffle_ps(xmm5, xmm5, 0);
                xmm6 = _mm_load_ss(k1+1);
                xmm6 = _mm_shuffle_ps(xmm6, xmm6, 0);
                xmm7 = _mm_load_ss(k1+2);
                xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
                xmm8 = _mm_load_ss(k1+3);
                xmm8 = _mm_shuffle_ps(xmm8, xmm8, 0);
                xmm9 = _mm_load_ss(k1+4);
                xmm9 = _mm_shuffle_ps(xmm9, xmm9, 0);
                xmm10 = _mm_load_ss(k2);
                xmm10 = _mm_shuffle_ps(xmm10, xmm10, 0);
                xmm11 = _mm_load_ss(k2+1);
                xmm11 = _mm_shuffle_ps(xmm11, xmm11, 0);
                xmm12 = _mm_load_ss(k2+2);
                xmm12 = _mm_shuffle_ps(xmm12, xmm12, 0);

                for ( int t = 0; t + 3 < outw; t += 4)
                {

                    xmm15 = _mm_xor_ps(xmm15,xmm15);

                    xmm13 = _mm_loadu_ps(r0 + t);
                    xmm13 = _mm_mul_ps(xmm13, xmm0);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r0 + t + 1);
                    xmm13 = _mm_mul_ps(xmm13, xmm1);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r0 + t + 2);
                    xmm13 = _mm_mul_ps(xmm13, xmm2);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r0 + t + 3);
                    xmm13 = _mm_mul_ps(xmm13, xmm3);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r0 + t + 4);
                    xmm13 = _mm_mul_ps(xmm13, xmm4);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r1 + t);
                    xmm13 = _mm_mul_ps(xmm13, xmm5);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r1 + t + 1);
                    xmm13 = _mm_mul_ps(xmm13, xmm6);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r1 + t + 2);
                    xmm13 = _mm_mul_ps(xmm13, xmm7);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r1 + t + 3);
                    xmm13 = _mm_mul_ps(xmm13, xmm8);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r1 + t + 4);
                    xmm13 = _mm_mul_ps(xmm13, xmm9);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r2 + t);
                    xmm13 = _mm_mul_ps(xmm13, xmm10);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r2 + t + 1);
                    xmm13 = _mm_mul_ps(xmm13, xmm11);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r2 + t + 2);
                    xmm13 = _mm_mul_ps(xmm13, xmm12);
                    xmm15 = _mm_add_ps(xmm15, xmm13);


                    xmm14 = _mm_loadu_ps(outptr + t);
                    xmm15 = _mm_add_ps(xmm15, xmm14);
                    _mm_storeu_ps(outptr + t, xmm15);

                }

                xmm0 = _mm_load_ss(k2+3);
                xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0);
                xmm1 = _mm_load_ss(k2+4);
                xmm1 = _mm_shuffle_ps(xmm1, xmm1, 0);
                xmm2 = _mm_load_ss(k3);
                xmm2 = _mm_shuffle_ps(xmm2, xmm2, 0);
                xmm3 = _mm_load_ss(k3+1);
                xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0);
                xmm4 = _mm_load_ss(k3+2);
                xmm4 = _mm_shuffle_ps(xmm4, xmm4, 0);
                xmm5 = _mm_load_ss(k3+3);
                xmm5 = _mm_shuffle_ps(xmm5, xmm5, 0);
                xmm6 = _mm_load_ss(k3+4);
                xmm6 = _mm_shuffle_ps(xmm6, xmm6, 0);
                xmm7 = _mm_load_ss(k4);
                xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
                xmm8 = _mm_load_ss(k4+1);
                xmm8 = _mm_shuffle_ps(xmm8, xmm8, 0);
                xmm9 = _mm_load_ss(k4+2);
                xmm9 = _mm_shuffle_ps(xmm9, xmm9, 0);
                xmm10 = _mm_load_ss(k4+3);
                xmm10 = _mm_shuffle_ps(xmm10, xmm10, 0);
                xmm11 = _mm_load_ss(k4+4);
                xmm11 = _mm_shuffle_ps(xmm11, xmm11, 0);


                for ( int t = 0; t + 3 < outw; t += 4)
                {

                    xmm15 = _mm_xor_ps(xmm15,xmm15);

                    xmm13 = _mm_loadu_ps(r2 + t + 3);
                    xmm13 = _mm_mul_ps(xmm13, xmm0);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r2 + t + 4);
                    xmm13 = _mm_mul_ps(xmm13, xmm1);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r3 + t);
                    xmm13 = _mm_mul_ps(xmm13, xmm2);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r3 + t + 1);
                    xmm13 = _mm_mul_ps(xmm13, xmm3);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r3 + t + 2);
                    xmm13 = _mm_mul_ps(xmm13, xmm4);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r3 + t + 3);
                    xmm13 = _mm_mul_ps(xmm13, xmm5);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r3 + t + 4);
                    xmm13 = _mm_mul_ps(xmm13, xmm6);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r4 + t);
                    xmm13 = _mm_mul_ps(xmm13, xmm7);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r4 + t + 1);
                    xmm13 = _mm_mul_ps(xmm13, xmm8);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r4 + t + 2);
                    xmm13 = _mm_mul_ps(xmm13, xmm9);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r4 + t + 3);
                    xmm13 = _mm_mul_ps(xmm13, xmm10);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm13 = _mm_loadu_ps(r4 + t + 4);
                    xmm13 = _mm_mul_ps(xmm13, xmm11);
                    xmm15 = _mm_add_ps(xmm15, xmm13);

                    xmm14 = _mm_loadu_ps(outptr + t);
                    xmm15 = _mm_add_ps(xmm15, xmm14);
                    _mm_storeu_ps(outptr + t, xmm15);

                }

                int t = (outw & (-4));   //the left value
                for (; t < outw ; t++)
                {
                    float sum = 0.0;
                    sum += r0[t]     * k0[0];
                    sum += r0[t + 1] * k0[1];
                    sum += r0[t + 2] * k0[2];
                    sum += r0[t + 3] * k0[3];
                    sum += r0[t + 4] * k0[4];
                    sum += r1[t]     * k1[0];
                    sum += r1[t + 1] * k1[1];
                    sum += r1[t + 2] * k1[2];
                    sum += r1[t + 3] * k1[3];
                    sum += r1[t + 4] * k1[4];
                    sum += r2[t]     * k2[0];
                    sum += r2[t + 1] * k2[1];
                    sum += r2[t + 2] * k2[2];
                    sum += r2[t + 3] * k2[3];
                    sum += r2[t + 4] * k2[4];
                    sum += r3[t]     * k3[0];
                    sum += r3[t + 1] * k3[1];
                    sum += r3[t + 2] * k3[2];
                    sum += r3[t + 3] * k3[3];
                    sum += r3[t + 4] * k3[4];
                    sum += r4[t]     * k4[0];
                    sum += r4[t + 1] * k4[1];
                    sum += r4[t + 2] * k4[2];
                    sum += r4[t + 3] * k4[3];
                    sum += r4[t + 4] * k4[4];

                    outptr[t] += sum;
                }

            }

        }

    }

}
