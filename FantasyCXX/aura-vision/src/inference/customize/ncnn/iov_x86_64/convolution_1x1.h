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

static void conv1x1s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;
    int size = outw * outh;

    const float* pimage = bottom_blob.channel(0);
    int imagestep=  bottom_blob.cstep;

    const float* kernel = _kernel;
    const float* bias = _bias;

    for (int i=0; i<outch; ++i) {
        Mat out0 = top_blob.channel(i);
        const float bias0 = bias ? bias[i] : 0.f;
//        out0.fill(bias0);
        iov::fill_mat_sse(out0, bias0);
    }

    int q=0;
//    #pragma omp parallel for num_threads(opt.num_threads)
    for(q=0; q<inch-3; q+=4) {

        const float* img0 = pimage + q*imagestep;
        const float* img1 = pimage + (q+1)*imagestep;
        const float* img2 = pimage + (q+2)*imagestep;
        const float* img3 = pimage + (q+3)*imagestep;

        int p=0;
        for(; p+7<outch; p+=8) {
            const float* k0 = kernel + (p+0) * inch;
            const float* k1 = kernel + (p+1) * inch;
            const float* k2 = kernel + (p+2) * inch;
            const float* k3 = kernel + (p+3) * inch;
            const float* k4 = kernel + (p+4) * inch;
            const float* k5 = kernel + (p+5) * inch;
            const float* k6 = kernel + (p+6) * inch;
            const float* k7 = kernel + (p+7) * inch;

            Mat out0 = top_blob.channel(p);
            Mat out1 = top_blob.channel(p+1);
            Mat out2 = top_blob.channel(p+2);
            Mat out3 = top_blob.channel(p+3);
            Mat out4 = top_blob.channel(p+4);
            Mat out5 = top_blob.channel(p+5);
            Mat out6 = top_blob.channel(p+6);
            Mat out7 = top_blob.channel(p+7);

            float* out0ptr = out0;
            float* out1ptr = out1;
            float* out2ptr = out2;
            float* out3ptr = out3;
            float* out4ptr = out4;
            float* out5ptr = out5;
            float* out6ptr = out6;
            float* out7ptr = out7;

            __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
            __m128 xmm9, xmm10, xmm11, xmm12, xmm13, xmm15;
            int t=0;
            for(; t+3 < size; t+=4) {
                const float* r0 = img0 + t;
                const float* r1 = img1 + t;
                const float* r2 = img2 + t;
                const float* r3 = img3 + t;

                xmm10 = _mm_loadu_ps(r0);
                xmm11 = _mm_loadu_ps(r1);
                xmm12 = _mm_loadu_ps(r2);
                xmm13 = _mm_loadu_ps(r3);

                xmm9 = _mm_loadu_ps(out0ptr + t);
                xmm15 = _mm_loadu_ps(k0 + q);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm0 = _mm_mul_ps(xmm10, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm0 = _mm_mul_ps(xmm11, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm0 = _mm_mul_ps(xmm12, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm0 = _mm_mul_ps(xmm13, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                _mm_storeu_ps(out0ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out1ptr + t);
                xmm15 = _mm_loadu_ps(k1 + q);
                xmm1 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm1 = _mm_mul_ps(xmm10, xmm1);
                xmm9 = _mm_add_ps(xmm9, xmm1);
                xmm1 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm1 = _mm_mul_ps(xmm11, xmm1);
                xmm9 = _mm_add_ps(xmm9, xmm1);
                xmm1 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm1 = _mm_mul_ps(xmm12, xmm1);
                xmm9 = _mm_add_ps(xmm9, xmm1);
                xmm1 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm1 = _mm_mul_ps(xmm13, xmm1);
                xmm9 = _mm_add_ps(xmm9, xmm1);
                _mm_storeu_ps(out1ptr + t, xmm9);

                xmm9 = _mm_loadu_ps(out2ptr + t);
                xmm15 = _mm_loadu_ps(k2 + q);
                xmm2 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm2 = _mm_mul_ps(xmm10, xmm2);
                xmm9 = _mm_add_ps(xmm9, xmm2);
                xmm2 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm2 = _mm_mul_ps(xmm11, xmm2);
                xmm9 = _mm_add_ps(xmm9, xmm2);
                xmm2 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm2 = _mm_mul_ps(xmm12, xmm2);
                xmm9 = _mm_add_ps(xmm9, xmm2);
                xmm2 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm2 = _mm_mul_ps(xmm13, xmm2);
                xmm9 = _mm_add_ps(xmm9, xmm2);
                _mm_storeu_ps(out2ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out3ptr + t);
                xmm15 = _mm_loadu_ps(k3 + q);
                xmm3 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm3 = _mm_mul_ps(xmm10, xmm3);
                xmm9 = _mm_add_ps(xmm9, xmm3);
                xmm3 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm3 = _mm_mul_ps(xmm11, xmm3);
                xmm9 = _mm_add_ps(xmm9, xmm3);
                xmm3 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm3 = _mm_mul_ps(xmm12, xmm3);
                xmm9 = _mm_add_ps(xmm9, xmm3);
                xmm3 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm3 = _mm_mul_ps(xmm13, xmm3);
                xmm9 = _mm_add_ps(xmm9, xmm3);
                _mm_storeu_ps(out3ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out4ptr + t);
                xmm15 = _mm_loadu_ps(k4 + q);
                xmm4 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm4 = _mm_mul_ps(xmm10, xmm4);
                xmm9 = _mm_add_ps(xmm9, xmm4);
                xmm4 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm4 = _mm_mul_ps(xmm11, xmm4);
                xmm9 = _mm_add_ps(xmm9, xmm4);
                xmm4 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm4 = _mm_mul_ps(xmm12, xmm4);
                xmm9 = _mm_add_ps(xmm9, xmm4);
                xmm4 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm4 = _mm_mul_ps(xmm13, xmm4);
                xmm9 = _mm_add_ps(xmm9, xmm4);
                _mm_storeu_ps(out4ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out5ptr + t);
                xmm15 = _mm_loadu_ps(k5 + q);
                xmm5 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm5 = _mm_mul_ps(xmm10, xmm5);
                xmm9 = _mm_add_ps(xmm9, xmm5);
                xmm5 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm5 = _mm_mul_ps(xmm11, xmm5);
                xmm9 = _mm_add_ps(xmm9, xmm5);
                xmm5 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm5 = _mm_mul_ps(xmm12, xmm5);
                xmm9 = _mm_add_ps(xmm9, xmm5);
                xmm5 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm5 = _mm_mul_ps(xmm13, xmm5);
                xmm9 = _mm_add_ps(xmm9, xmm5);
                _mm_storeu_ps(out5ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out6ptr + t);
                xmm15 = _mm_loadu_ps(k6 + q);
                xmm6 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm6 = _mm_mul_ps(xmm10, xmm6);
                xmm9 = _mm_add_ps(xmm9, xmm6);
                xmm6 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm6 = _mm_mul_ps(xmm11, xmm6);
                xmm9 = _mm_add_ps(xmm9, xmm6);
                xmm6 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm6 = _mm_mul_ps(xmm12, xmm6);
                xmm9 = _mm_add_ps(xmm9, xmm6);
                xmm6 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm6 = _mm_mul_ps(xmm13, xmm6);
                xmm9 = _mm_add_ps(xmm9, xmm6);
                _mm_storeu_ps(out6ptr+t, xmm9);

                xmm9 = _mm_loadu_ps(out7ptr + t);
                xmm15 = _mm_loadu_ps(k7 + q);
                xmm7 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm7 = _mm_mul_ps(xmm10, xmm7);
                xmm9 = _mm_add_ps(xmm9, xmm7);
                xmm7 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm7 = _mm_mul_ps(xmm11, xmm7);
                xmm9 = _mm_add_ps(xmm9, xmm7);
                xmm7 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm7 = _mm_mul_ps(xmm12, xmm7);
                xmm9 = _mm_add_ps(xmm9, xmm7);
                xmm7 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm7 = _mm_mul_ps(xmm13, xmm7);
                xmm9 = _mm_add_ps(xmm9, xmm7);
                _mm_storeu_ps(out7ptr+t, xmm9);
            }

            //remaining t
            for(; t<size; ++t) {
                out0ptr[t] += k0[q] * img0[t];
                out0ptr[t] += k0[q+1] * img1[t];
                out0ptr[t] += k0[q+2] * img2[t];
                out0ptr[t] += k0[q+3] * img3[t];

                out1ptr[t] += k1[q] * img0[t];
                out1ptr[t] += k1[q+1] * img1[t];
                out1ptr[t] += k1[q+2] * img2[t];
                out1ptr[t] += k1[q+3] * img3[t];

                out2ptr[t] += k2[q] * img0[t];
                out2ptr[t] += k2[q+1] * img1[t];
                out2ptr[t] += k2[q+2] * img2[t];
                out2ptr[t] += k2[q+3] * img3[t];

                out3ptr[t] += k3[q] * img0[t];
                out3ptr[t] += k3[q+1] * img1[t];
                out3ptr[t] += k3[q+2] * img2[t];
                out3ptr[t] += k3[q+3] * img3[t];

                out4ptr[t] += k4[q] * img0[t];
                out4ptr[t] += k4[q+1] * img1[t];
                out4ptr[t] += k4[q+2] * img2[t];
                out4ptr[t] += k4[q+3] * img3[t];

                out5ptr[t] += k5[q] * img0[t];
                out5ptr[t] += k5[q+1] * img1[t];
                out5ptr[t] += k5[q+2] * img2[t];
                out5ptr[t] += k5[q+3] * img3[t];

                out6ptr[t] += k6[q] * img0[t];
                out6ptr[t] += k6[q+1] * img1[t];
                out6ptr[t] += k6[q+2] * img2[t];
                out6ptr[t] += k6[q+3] * img3[t];

                out7ptr[t] += k7[q] * img0[t];
                out7ptr[t] += k7[q+1] * img1[t];
                out7ptr[t] += k7[q+2] * img2[t];
                out7ptr[t] += k7[q+3] * img3[t];
            }
        }

        //remaining p
        for(; p<outch; ++p) {
            const float* k0 = kernel + p * inch;
            float* out0ptr = top_blob.channel(p);

            __m128 xmm0, xmm9, xmm10, xmm11, xmm12, xmm13, xmm15;

            int t=0;
            for(; t+3 < size; t+=4) {
                const float *r0 = img0 + t;
                const float *r1 = img1 + t;
                const float *r2 = img2 + t;
                const float *r3 = img3 + t;

                xmm10 = _mm_loadu_ps(r0);
                xmm11 = _mm_loadu_ps(r1);
                xmm12 = _mm_loadu_ps(r2);
                xmm13 = _mm_loadu_ps(r3);

                xmm9 = _mm_loadu_ps(out0ptr + t);
                xmm15 = _mm_loadu_ps(k0 + q);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b00000000);
                xmm0 = _mm_mul_ps(xmm10, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b01010101);
                xmm0 = _mm_mul_ps(xmm11, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b10101010);
                xmm0 = _mm_mul_ps(xmm12, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                xmm0 = _mm_shuffle_ps(xmm15, xmm15, 0b11111111);
                xmm0 = _mm_mul_ps(xmm13, xmm0);
                xmm9 = _mm_add_ps(xmm9, xmm0);
                _mm_storeu_ps(out0ptr + t, xmm9);
            }

            //remaining t
            for(; t<size; ++t) {
                out0ptr[t] += k0[q] * img0[t];
                out0ptr[t] += k0[q + 1] * img1[t];
                out0ptr[t] += k0[q + 2] * img2[t];
                out0ptr[t] += k0[q + 3] * img3[t];
            }

        }

    }

    // remaining q
    for(; q<inch; ++q) {
        const float* img = bottom_blob.channel(q);
        for(int p=0; p<outch; ++p) {
            const float* k = kernel + p*inch + q;
            float* out0ptr = top_blob.channel(p);

            for(int t=0; t<size; ++t) {
                out0ptr[t] += k[0] * img[t];
            }
        }
    }

}

static void conv1x1s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }

        }
    }

}
