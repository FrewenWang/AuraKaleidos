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

static void convdw3x3s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

//        out.fill(bias0);
        iov::fill_mat_sse(out, bias0);

        const float* kernel0 = kernel + g*9;
        const float* img0 = bottom_blob.channel(g);

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        for(int i=0; i < outh; i++) {
            const float* r0 = img0 + i * w;
            const float* r1 = img0 + (i+1) * w;
            const float* r2 = img0 + (i+2) * w;
            float* outptr = (float*)out + outw * i;

            __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

            xmm0 = _mm_load_ss(k0);
            xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0);
            xmm1 = _mm_load_ss(k0+1);
            xmm1 = _mm_shuffle_ps(xmm1, xmm1, 0);
            xmm2 = _mm_load_ss(k0+2);
            xmm2 = _mm_shuffle_ps(xmm2, xmm2, 0);
            xmm3 = _mm_load_ss(k1);
            xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0);
            xmm4 = _mm_load_ss(k1+1);
            xmm4 = _mm_shuffle_ps(xmm4, xmm4, 0);

            for ( int t = 0; t + 3 < outw; t += 4)
            {

                __m128 xmm6,xmm7,xmm5;
                xmm7 = _mm_xor_ps(xmm7,xmm7);

                xmm6 = _mm_loadu_ps(r0 + t);
                xmm6 = _mm_mul_ps(xmm6, xmm0);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r0 + t + 1);
                xmm6 = _mm_mul_ps(xmm6, xmm1);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r0 + t +2);
                xmm6 = _mm_mul_ps(xmm6, xmm2);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r1 + t);
                xmm6 = _mm_mul_ps(xmm6, xmm3);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r1 + t +1);
                xmm6 = _mm_mul_ps(xmm6, xmm4);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm5 = _mm_loadu_ps(outptr + t);
                xmm7 = _mm_add_ps(xmm7, xmm5);
                _mm_storeu_ps(outptr + t, xmm7);

            }

            xmm0 = _mm_load_ss(k1 + 2);
            xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0);
            xmm1 = _mm_load_ss(k2);
            xmm1 = _mm_shuffle_ps(xmm1, xmm1, 0);
            xmm2 = _mm_load_ss(k2 + 1);
            xmm2 = _mm_shuffle_ps(xmm2, xmm2, 0);
            xmm3 = _mm_load_ss(k2 +2);
            xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0);


            for (int t = 0; t + 3 < outw; t += 4)
            {

                __m128 xmm6, xmm7, xmm5;
                xmm7 = _mm_xor_ps(xmm7, xmm7);

                xmm6 = _mm_loadu_ps(r1 + t + 2);
                xmm6 = _mm_mul_ps(xmm6, xmm0);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r2 + t);
                xmm6 = _mm_mul_ps(xmm6, xmm1);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r2 + t + 1);
                xmm6 = _mm_mul_ps(xmm6, xmm2);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm6 = _mm_loadu_ps(r2 + t + 2);
                xmm6 = _mm_mul_ps(xmm6, xmm3);
                xmm7 = _mm_add_ps(xmm7, xmm6);

                xmm5 = _mm_loadu_ps(outptr + t);
                xmm7 = _mm_add_ps(xmm7, xmm5);
                _mm_storeu_ps(outptr + t, xmm7);


            }

            int t = (outw & (-4));   //the left value
            for (; t < outw ; t++)
            {
                float sum = 0.0;
                sum += r0[t]     * k0[0];
                sum += r0[t + 1] * k0[1];
                sum += r0[t + 2] * k0[2];
                sum += r1[t]     * k1[0];
                sum += r1[t + 1] * k1[1];
                sum += r1[t + 2] * k1[2];
                sum += r2[t]     * k2[0];
                sum += r2[t + 1] * k2[1];
                sum += r2[t + 2] * k2[2];

                outptr[t] += sum;
            }

        }
    }
}

static void convdw3x3s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g*9;

        float* outptr = out;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        int i = 0;

        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain>0; remain--)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }

    }
}
