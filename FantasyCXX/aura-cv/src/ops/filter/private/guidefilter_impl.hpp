#ifndef AURA_OPS_FILTER_GUIDEFILTER_IMPL_HPP__
#define AURA_OPS_FILTER_GUIDEFILTER_IMPL_HPP__

#include "make_border_impl.hpp"
#include "aura/ops/filter/guidefilter.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/thread_buffer.h"

#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

#define AURA_OPS_FILTER_GUIDEFILTER_OP_NAME          "GuideFilter"

#define GUIDE_FILTER_EPS (1E-5)

namespace aura
{

AURA_INLINE MI_S32 GetFastKsize(MI_S32 ksize)
{
    //    Normal ksize    |   Fast ksize
    //    3               |   3
    //    5               |   3
    //    7               |   3
    //    9               |   5
    //    11              |   5
    //    13              |   7
    //    15              |   7
    //    17              |   9
    //    ...
    MI_S32 radius     = ksize >> 1;
    MI_S32 radius_sub = Max((radius >> 1), (MI_S32)1);
    MI_S32 ksize_fast = (radius_sub << 1) + 1;

    return ksize_fast;
}

class GuideFilterImpl : public OpImpl
{
public:
    GuideFilterImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_S32 ksize, MI_F32 eps,
                           GuideFilterType type = GuideFilterType::NORMAL,
                           BorderType border_type = BorderType::REPLICATE,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    MI_S32          m_ksize;
    MI_F32          m_eps;
    GuideFilterType m_type;
    BorderType      m_border_type;
    Scalar          m_border_value;

    const Array *m_src0;
    const Array *m_src1;
    Array       *m_dst;
};

class GuideFilterNone : public GuideFilterImpl
{
public:
    GuideFilterNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_S32 ksize, MI_F32 eps,
                   GuideFilterType type = GuideFilterType::NORMAL,
                   BorderType border_type = BorderType::REPLICATE,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_src_border0;
    Mat m_src_border1;
};

#if defined(AURA_ENABLE_NEON)
template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<!is_floating_point<Tp>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(Tp        *row_a,  Tp        *row_b,  // original row data
                                                SumType   *row_aa, SumType   *row_ab, // a * a, a * b
                                                SumType   *sum_a,  SumType   *sum_b,  // sum of a, sum of b
                                                SqSumType *sum_aa, SqSumType *sum_ab, // sum of a * a, sum of a * b
                                                MI_S32     width)
{
    using VType    = typename neon::DVector<Tp>::VType;
    using VSumType = typename neon::QVector<SumType>::VType;
    using VSqType  = typename neon::QVector<SqSumType>::VType;

    constexpr MI_S32 ELEM_COUNTS = 8 / sizeof(Tp);
    const MI_S32 width_align = (width * C) & (-ELEM_COUNTS);

    MI_S32 x = 0;

    for (; x < width_align; x += ELEM_COUNTS)
    {
        VType    vd_a  = neon::vload1(row_a + x);
        VType    vd_b  = neon::vload1(row_b + x);
        VSumType vq_aa = neon::vmull(vd_a, vd_a);
        VSumType vq_ab = neon::vmull(vd_a, vd_b);

        neon::vstore(row_aa + x, vq_aa);
        neon::vstore(row_ab + x, vq_ab);

        VSumType vq_sum_a  = neon::vaddw(neon::vload1q(sum_a  + x), vd_a);
        VSumType vq_sum_b  = neon::vaddw(neon::vload1q(sum_b  + x), vd_b);
        neon::vstore(sum_a + x, vq_sum_a);
        neon::vstore(sum_b + x, vq_sum_b);

        VSqType  vq_sum_aa_l = neon::vaddw(neon::vload1q(sum_aa + x),                   neon::vgetlow(vq_aa));
        VSqType  vq_sum_aa_h = neon::vaddw(neon::vload1q(sum_aa + x + ELEM_COUNTS / 2), neon::vgethigh(vq_aa));
        VSqType  vq_sum_ab_l = neon::vaddw(neon::vload1q(sum_ab + x),                   neon::vgetlow(vq_ab));
        VSqType  vq_sum_ab_h = neon::vaddw(neon::vload1q(sum_ab + x + ELEM_COUNTS / 2), neon::vgethigh(vq_ab));

        neon::vstore(sum_aa + x,               vq_sum_aa_l);
        neon::vstore(sum_aa + x + ELEM_COUNTS / 2, vq_sum_aa_h);
        neon::vstore(sum_ab + x,               vq_sum_ab_l);
        neon::vstore(sum_ab + x + ELEM_COUNTS / 2, vq_sum_ab_h);
    }

    for (; x < width * C; x++)
    {
        Tp a = row_a[x];
        Tp b = row_b[x];
        SumType aa = a * a;
        SumType ab = a * b;

        row_aa[x] = aa;
        row_ab[x] = ab;

        sum_a[x]  += a;
        sum_b[x]  += b;
        sum_aa[x] += aa;
        sum_ab[x] += ab;
    }
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<!is_floating_point<Tp>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(Tp        *row_a,   // original row data
                                                SumType   *row_aa,  // a * a, a * b
                                                SumType   *sum_a,   // sum of a, sum of b
                                                SqSumType *sum_aa,  // sum of a * a, sum of a * b
                                                MI_S32     width)
{
    using VType    = typename neon::DVector<Tp>::VType;
    using VSumType = typename neon::QVector<SumType>::VType;
    using VSqType  = typename neon::QVector<SqSumType>::VType;

    constexpr MI_S32 ELEM_COUNTS = 8 / sizeof(Tp);
    const MI_S32 width_align = (width * C) & (-ELEM_COUNTS);

    MI_S32 x = 0;

    for (; x < width_align; x += ELEM_COUNTS)
    {
        VType    vd_a  = neon::vload1(row_a + x);
        VSumType vq_aa = neon::vmull(vd_a, vd_a);

        neon::vstore(row_aa + x, vq_aa);

        VSumType vq_sum_a  = neon::vaddw(neon::vload1q(sum_a  + x), vd_a);
        neon::vstore(sum_a + x, vq_sum_a);

        VSqType  vq_sum_aa_l = neon::vaddw(neon::vload1q(sum_aa + x),                   neon::vgetlow(vq_aa));
        VSqType  vq_sum_aa_h = neon::vaddw(neon::vload1q(sum_aa + x + ELEM_COUNTS / 2), neon::vgethigh(vq_aa));

        neon::vstore(sum_aa + x,               vq_sum_aa_l);
        neon::vstore(sum_aa + x + ELEM_COUNTS / 2, vq_sum_aa_h);
    }

    for (; x < width * C; x++)
    {
        Tp a = row_a[x];
        SumType aa = a * a;

        row_aa[x] = aa;
        sum_a[x]  += a;
        sum_aa[x] += aa;
    }
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(MI_F16 *row_a,  MI_F16 *row_b,  // original row data
                                                MI_F32 *row_aa, MI_F32 *row_ab, // a * a, a * b
                                                MI_F32 *sum_a,  MI_F32 *sum_b,  // sum of a, sum of b
                                                MI_F32 *sum_aa, MI_F32 *sum_ab, // sum of a * a, sum of a * b
                                                MI_S32  width)
{
    MI_S32 x = 0;
    const MI_S32 width_align4 = (width * C) & (-4);

    for (; x < width_align4; x += 4)
    {
        float16x4_t vdf16_a = neon::vload1(row_a + x);
        float16x4_t vdf16_b = neon::vload1(row_b + x);
        float32x4_t vqf32_a = neon::vcvt<MI_F32>(vdf16_a);
        float32x4_t vqf32_b = neon::vcvt<MI_F32>(vdf16_b);

        float32x4_t vqf32_aa = neon::vmul(vqf32_a, vqf32_a);
        float32x4_t vqf32_ab = neon::vmul(vqf32_a, vqf32_b);

        neon::vstore(row_aa + x, vqf32_aa);
        neon::vstore(row_ab + x, vqf32_ab);

        neon::vstore(sum_a  + x, neon::vadd(neon::vload1q(sum_a  + x), vqf32_a));
        neon::vstore(sum_b  + x, neon::vadd(neon::vload1q(sum_b  + x), vqf32_b));
        neon::vstore(sum_aa + x, neon::vadd(neon::vload1q(sum_aa + x), vqf32_aa));
        neon::vstore(sum_ab + x, neon::vadd(neon::vload1q(sum_ab + x), vqf32_ab));
    }

    for (; x < width * C; x++)
    {
        MI_F16 a = row_a[x];
        MI_F16 b = row_b[x];
        MI_F32 aa = a * a;
        MI_F32 ab = a * b;

        row_aa[x] = aa;
        row_ab[x] = ab;

        sum_a[x]  += a;
        sum_b[x]  += b;
        sum_aa[x] += aa;
        sum_ab[x] += ab;
    }
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(MI_F16 *row_a,  // original row data
                                                MI_F32 *row_aa, // a * a, a * b
                                                MI_F32 *sum_a,  // sum of a, sum of b
                                                MI_F32 *sum_aa, // sum of a * a, sum of a * b
                                                MI_S32  width)
{
    MI_S32 x = 0;
    const MI_S32 width_align4 = (width * C) & (-4);

    for (; x < width_align4; x += 4)
    {
        float16x4_t vdf16_a = neon::vload1(row_a + x);
        float32x4_t vqf32_a = neon::vcvt<MI_F32>(vdf16_a);

        float32x4_t vqf32_aa = neon::vmul(vqf32_a, vqf32_a);

        neon::vstore(row_aa + x, vqf32_aa);

        neon::vstore(sum_a  + x, neon::vadd(neon::vload1q(sum_a  + x), vqf32_a));
        neon::vstore(sum_aa + x, neon::vadd(neon::vload1q(sum_aa + x), vqf32_aa));
    }

    for (; x < width * C; x++)
    {
        MI_F16 a = row_a[x];
        MI_F32 aa = a * a;

        row_aa[x] = aa;
        sum_a[x]  += a;
        sum_aa[x] += aa;
    }
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<std::is_same<Tp, MI_F32>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(MI_F32 *row_a,  MI_F32 *row_b,  // original row data
                                                MI_F32 *row_aa, MI_F32 *row_ab, // a * a, a * b
                                                MI_F32 *sum_a,  MI_F32 *sum_b,  // sum of a, sum of b
                                                MI_F32 *sum_aa, MI_F32 *sum_ab, // sum of a * a, sum of a * b
                                                MI_S32  width)
{
    MI_S32 x = 0;
    const MI_S32 width_align4 = (width * C) & (-4);

    for (; x < width_align4; x += 4)
    {
        float32x4_t vqf32_a = neon::vload1q(row_a + x);
        float32x4_t vqf32_b = neon::vload1q(row_b + x);

        float32x4_t vqf32_aa = neon::vmul(vqf32_a, vqf32_a);
        float32x4_t vqf32_ab = neon::vmul(vqf32_a, vqf32_b);

        neon::vstore(row_aa + x, vqf32_aa);
        neon::vstore(row_ab + x, vqf32_ab);

        neon::vstore(sum_a  + x, neon::vadd(neon::vload1q(sum_a  + x), vqf32_a));
        neon::vstore(sum_b  + x, neon::vadd(neon::vload1q(sum_b  + x), vqf32_b));
        neon::vstore(sum_aa + x, neon::vadd(neon::vload1q(sum_aa + x), vqf32_aa));
        neon::vstore(sum_ab + x, neon::vadd(neon::vload1q(sum_ab + x), vqf32_ab));
    }

    for (; x < width * C; x++)
    {
        MI_F32 a  = row_a[x];
        MI_F32 b  = row_b[x];
        MI_F32 aa = a * a;
        MI_F32 ab = a * b;

        row_aa[x] = aa;
        row_ab[x] = ab;

        sum_a[x]  += a;
        sum_b[x]  += b;
        sum_aa[x] += aa;
        sum_ab[x] += ab;
    }
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C,
          typename std::enable_if<std::is_same<Tp, MI_F32>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterRowDataAccumulateNeon(MI_F32 *row_a,  // original row data
                                                MI_F32 *row_aa, // a * a, a * b
                                                MI_F32 *sum_a,  // sum of a, sum of b
                                                MI_F32 *sum_aa, // sum of a * a, sum of a * b
                                                MI_S32  width)
{
    MI_S32 x = 0;
    const MI_S32 width_align4 = (width * C) & (-4);

    for (; x < width_align4; x += 4)
    {
        float32x4_t vqf32_a  = neon::vload1q(row_a + x);
        float32x4_t vqf32_aa = neon::vmul(vqf32_a, vqf32_a);

        neon::vstore(row_aa + x, vqf32_aa);

        neon::vstore(sum_a  + x, neon::vadd(neon::vload1q(sum_a  + x), vqf32_a));
        neon::vstore(sum_aa + x, neon::vadd(neon::vload1q(sum_aa + x), vqf32_aa));
    }

    for (; x < width * C; x++)
    {
        MI_F32 a  = row_a[x];
        MI_F32 aa = a * a;

        row_aa[x] = aa;
        sum_a[x]  += a;
        sum_aa[x] += aa;
    }
}

template <typename Tp, typename SumType,
          typename std::enable_if<!is_floating_point<Tp>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterSubRowNeon(Tp *src, SumType *sum, const MI_S32 width)
{
    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(SumType);

    MI_S32 x = 0;
    for (; x < width - ELEM_COUNTS; x += ELEM_COUNTS)
    {
        neon::vstore(sum + x, neon::vsubw(neon::vload1q(sum + x), neon::vload1(src + x)));
    }

    for (; x < width; x++)
    {
        sum[x] -= src[x];
    }
}

template <typename Tp, typename SumType,
          typename std::enable_if<std::is_same<Tp, MI_F16>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterSubRowNeon(Tp *src, SumType *sum, const MI_S32 width)
{
    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(SumType);

    MI_S32 x = 0;
    for (; x < width - ELEM_COUNTS; x += ELEM_COUNTS)
    {
        neon::vstore(sum + x, neon::vsub(neon::vload1q(sum + x), neon::vcvt<MI_F32>(neon::vload1(src + x))));
    }

    for (; x < width; x++)
    {
        sum[x] -= src[x];
    }
}

template <typename Tp, typename SumType,
          typename std::enable_if<std::is_same<Tp, MI_F32>::value>::type* = MI_NULL>
static AURA_VOID GuideFilterSubRowNeon(Tp *src, SumType *sum, const MI_S32 width)
{
    constexpr MI_S32 ELEM_COUNTS = 16 / sizeof(SumType);

    MI_S32 x = 0;
    for (; x < width - ELEM_COUNTS; x += ELEM_COUNTS)
    {
        neon::vstore(sum + x, neon::vsub(neon::vload1q(sum + x), neon::vload1q(src + x)));
    }

    for (; x < width; x++)
    {
        sum[x] -= src[x];
    }
}

template <typename SumType, typename SqSumType, MI_S32 C>
static AURA_VOID GuideFilterCalcKernelSum(SumType *a, SumType *b, SqSumType *aa, SqSumType *ab, const MI_S32 n,
                                        const MI_S32 ksize, MI_F32 eps, MI_F32 *dst_a, MI_F32 *dst_b)
{
    SqSumType sum_kernel_a[C]  = {0};
    SqSumType sum_kernel_b[C]  = {0};
    SqSumType sum_kernel_aa[C] = {0};
    SqSumType sum_kernel_ab[C] = {0};

    const MI_S32 ksq = ksize * ksize;

    for (MI_S32 c = 0; c < C; c++)
    {
        for (MI_S32 k = 0; k < ksize; k++)
        {
            sum_kernel_a[c]  +=  a[k * C + c];
            sum_kernel_b[c]  +=  b[k * C + c];
            sum_kernel_aa[c] += aa[k * C + c];
            sum_kernel_ab[c] += ab[k * C + c];
        }

        MI_F32 mean_i  = static_cast<MI_F32>(sum_kernel_a[c]) / ksq;
        MI_F32 mean_p  = static_cast<MI_F32>(sum_kernel_b[c]) / ksq;
        MI_F32 var     = static_cast<MI_F32>(sum_kernel_aa[c]) / ksq - mean_i * mean_i;
        MI_F32 cov     = static_cast<MI_F32>(sum_kernel_ab[c]) / ksq - mean_i * mean_p;
        MI_F32 var_eps = (var + eps);

        dst_a[c] = cov / var_eps;
        dst_b[c] = mean_p - mean_i * dst_a[c];
    }

    for (MI_S32 x = 1; x < n; x++)
    {
        for (MI_S32 c = 0; c < C; c++)
        {
            sum_kernel_a[c]  = sum_kernel_a[c]  +  a[x * C + c + ksize * C - C] -  a[x * C + c - C];
            sum_kernel_b[c]  = sum_kernel_b[c]  +  b[x * C + c + ksize * C - C] -  b[x * C + c - C];
            sum_kernel_aa[c] = sum_kernel_aa[c] + aa[x * C + c + ksize * C - C] - aa[x * C + c - C];
            sum_kernel_ab[c] = sum_kernel_ab[c] + ab[x * C + c + ksize * C - C] - ab[x * C + c - C];

            MI_F32 mean_i  = static_cast<MI_F32>(sum_kernel_a[c])  / ksq;
            MI_F32 mean_p  = static_cast<MI_F32>(sum_kernel_b[c])  / ksq;
            MI_F32 var     = static_cast<MI_F32>(sum_kernel_aa[c]) / ksq - mean_i * mean_i;
            MI_F32 cov     = static_cast<MI_F32>(sum_kernel_ab[c]) / ksq- mean_i * mean_p;
            MI_F32 var_eps = (var + eps);

            dst_a[x * C + c] = cov / var_eps;
            dst_b[x * C + c] = mean_p - mean_i * dst_a[x * C + c];
        }
    }
}

template <typename SumType, typename SqSumType, MI_S32 C>
static AURA_VOID GuideFilterCalcKernelSum(SumType *a, SqSumType *aa, const MI_S32 n,
                                        const MI_S32 ksize, MI_F32 eps, MI_F32 *dst_a, MI_F32 *dst_b)
{
    SqSumType sum_kernel_a[C]  = {0};
    SqSumType sum_kernel_aa[C] = {0};

    const MI_S32 ksq = ksize * ksize;

    for (MI_S32 c = 0; c < C; c++)
    {
        for (MI_S32 k = 0; k < ksize; k++)
        {
            sum_kernel_a[c]  +=  a[k * C + c];
            sum_kernel_aa[c] += aa[k * C + c];
        }

        MI_F32 mean_i  = static_cast<MI_F32>(sum_kernel_a[c]) / ksq;
        MI_F32 var     = static_cast<MI_F32>(sum_kernel_aa[c]) / ksq - mean_i * mean_i;
        MI_F32 var_eps = (var + eps);

        dst_a[c] = var / var_eps;
        dst_b[c] = mean_i - mean_i * dst_a[c];
    }

    for (MI_S32 x = 1; x < n; x++)
    {
        for (MI_S32 c = 0; c < C; c++)
        {
            sum_kernel_a[c]  = sum_kernel_a[c]  +  a[x * C + c + ksize * C - C] -  a[x * C + c - C];
            sum_kernel_aa[c] = sum_kernel_aa[c] + aa[x * C + c + ksize * C - C] - aa[x * C + c - C];

            MI_F32 mean_i  = static_cast<MI_F32>(sum_kernel_a[c])  / ksq;
            MI_F32 var     = static_cast<MI_F32>(sum_kernel_aa[c]) / ksq - mean_i * mean_i;
            MI_F32 var_eps = (var + eps);

            dst_a[x * C + c] = var / var_eps;
            dst_b[x * C + c] = mean_i - mean_i * dst_a[x * C + c];
        }
    }
}

template <typename Tp, typename SumType, typename SqSumType, BorderType BORDER_TYPE, MI_S32 C>
static Status GuideFilterCalcABNeonImpl(const Mat &src_a, const Mat &src_b, Mat &dst_a, Mat &dst_b,
                                        ThreadBuffer &row_a_buffer, ThreadBuffer &row_b_buffer, ThreadBuffer &row_aa_buffer,
                                        ThreadBuffer &row_ab_buffer, ThreadBuffer &sum_a_buffer, ThreadBuffer &sum_b_buffer,
                                        ThreadBuffer &sum_aa_buffer, ThreadBuffer &sum_ab_buffer, ThreadBuffer &arr_ptr_buffer,
                                        MI_S32 ksize, MI_F32 eps, const Scalar &border_value, MI_S32 start_row, MI_S32 end_row)
{
    const Sizes3 sz    = src_a.GetSizes();
    const MI_S32 width = sz.m_width;

    const MI_S32 width_align = AURA_ALIGN((width + ksize) * C, 64);

    Tp      *row_a_data  = row_a_buffer.GetThreadData<Tp>();
    Tp      *row_b_data  = row_b_buffer.GetThreadData<Tp>();
    SumType *row_aa_data = row_aa_buffer.GetThreadData<SumType>();
    SumType *row_ab_data = row_ab_buffer.GetThreadData<SumType>();

    SumType   *sum_a  = sum_a_buffer.GetThreadData<SumType>();
    SumType   *sum_b  = sum_b_buffer.GetThreadData<SumType>();
    SqSumType *sum_aa = sum_aa_buffer.GetThreadData<SqSumType>();
    SqSumType *sum_ab = sum_ab_buffer.GetThreadData<SqSumType>();

    AURA_VOID  **arr_ptr = arr_ptr_buffer.GetThreadData<AURA_VOID*>();

    memset(sum_a,  0, width_align * sizeof(SumType));
    memset(sum_b,  0, width_align * sizeof(SumType));
    memset(sum_aa, 0, width_align * sizeof(SqSumType));
    memset(sum_ab, 0, width_align * sizeof(SqSumType));

    Tp      **row_a  = reinterpret_cast<Tp**>(arr_ptr);
    Tp      **row_b  = reinterpret_cast<Tp**>(arr_ptr + ksize);
    SumType **row_aa = reinterpret_cast<SumType**>(arr_ptr + 2 * ksize);
    SumType **row_ab = reinterpret_cast<SumType**>(arr_ptr + 3 * ksize);

    for (MI_S32 i = 0; i < ksize; i++)
    {
        row_a[i]  = row_a_data  + i * width_align;
        row_b[i]  = row_b_data  + i * width_align;
        row_aa[i] = row_aa_data + i * width_align;
        row_ab[i] = row_ab_data + i * width_align;
    }

    // 1. Init (ksize rows data)
    const MI_S32 ksh = ksize / 2;

    // 0th row is left unchanged on purpose
    memset(row_a[0],  0, width_align * sizeof(Tp));
    memset(row_b[0],  0, width_align * sizeof(Tp));
    memset(row_aa[0], 0, width_align * sizeof(SumType));
    memset(row_ab[0], 0, width_align * sizeof(SumType));

    for (MI_S32 y = 1; y < ksize; y++)
    {
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src_a, start_row + y - 1 - ksh, width, ksize, row_a[y], border_value);
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src_b, start_row + y - 1 - ksh, width, ksize, row_b[y], border_value);

        GuideFilterRowDataAccumulateNeon<Tp, SumType, SqSumType, C>(row_a[y], row_b[y], row_aa[y], row_ab[y],
                                                                    sum_a, sum_b, sum_aa, sum_ab, width + ksize - 1);
    }

    // 2. Loop Rows
    MI_S32 idx_head = 0;
    MI_S32 idx_tail = ksize - 1;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        GuideFilterSubRowNeon<Tp,      SumType  >(row_a[idx_head],  sum_a,  (width + ksize - 1) * C);
        GuideFilterSubRowNeon<Tp,      SumType  >(row_b[idx_head],  sum_b,  (width + ksize - 1) * C);
        GuideFilterSubRowNeon<SumType, SqSumType>(row_aa[idx_head], sum_aa, (width + ksize - 1) * C);
        GuideFilterSubRowNeon<SumType, SqSumType>(row_ab[idx_head], sum_ab, (width + ksize - 1) * C);

        // Update Index
        idx_head = (idx_head + 1) % ksize;
        idx_tail = (idx_tail + 1) % ksize; // new idx_tail should be old idx_head

        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src_a, y + ksh, width, ksize, row_a[idx_tail], border_value);
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src_b, y + ksh, width, ksize, row_b[idx_tail], border_value);

        GuideFilterRowDataAccumulateNeon<Tp, SumType, SqSumType, C>(row_a[idx_tail], row_b[idx_tail], row_aa[idx_tail], row_ab[idx_tail],
                                                                    sum_a, sum_b, sum_aa, sum_ab, width + ksize - 1);

        // 2.2 filter done and write back
        MI_F32 *dst_a_row = dst_a.Ptr<MI_F32>(y);
        MI_F32 *dst_b_row = dst_b.Ptr<MI_F32>(y);

        GuideFilterCalcKernelSum<SumType, SqSumType, C>(sum_a, sum_b, sum_aa, sum_ab,
                                                        width, ksize, eps, dst_a_row, dst_b_row);
    }

    return Status::OK;
}


template <typename Tp, typename SumType, typename SqSumType, BorderType BORDER_TYPE, MI_S32 C>
static Status GuideFilterCalcABSameSrcNeonImpl(const Mat &src, Mat &dst_a, Mat &dst_b,
                                               ThreadBuffer &row_a_buffer, ThreadBuffer &row_aa_buffer, ThreadBuffer &sum_a_buffer,
                                               ThreadBuffer &sum_aa_buffer, ThreadBuffer &arr_ptr_buffer, MI_S32 ksize, MI_F32 eps,
                                               const Scalar &border_value, MI_S32 start_row, MI_S32 end_row)
{
    const Sizes3 sz    = src.GetSizes();
    const MI_S32 width = sz.m_width;

    const MI_S32 width_align = AURA_ALIGN((width + ksize) * C, 64);

    Tp        *row_a_data  = row_a_buffer.GetThreadData<Tp>();
    SumType   *row_aa_data = row_aa_buffer.GetThreadData<SumType>();
    SumType   *sum_a       = sum_a_buffer.GetThreadData<SumType>();
    SqSumType *sum_aa      = sum_aa_buffer.GetThreadData<SqSumType>();

    AURA_VOID  **arr_ptr = arr_ptr_buffer.GetThreadData<AURA_VOID*>();

    memset(sum_a,  0, width_align * sizeof(SumType));
    memset(sum_aa, 0, width_align * sizeof(SqSumType));

    Tp      **row_a  = reinterpret_cast<Tp**>(arr_ptr);
    SumType **row_aa = reinterpret_cast<SumType**>(arr_ptr + ksize);

    for (MI_S32 i = 0; i < ksize; i++)
    {
        row_a[i]  = row_a_data  + i * width_align;
        row_aa[i] = row_aa_data + i * width_align;
    }

    // 1. Init (ksize rows data)
    const MI_S32 ksh = ksize / 2;

    // 0th row is left unchanged on purpose
    memset(row_a[0],  0, width_align * sizeof(Tp));
    memset(row_aa[0], 0, width_align * sizeof(SumType));

    for (MI_S32 y = 1; y < ksize; y++)
    {
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, start_row + y - 1 - ksh, width, ksize, row_a[y], border_value);
        GuideFilterRowDataAccumulateNeon<Tp, SumType, SqSumType, C>(row_a[y],row_aa[y], sum_a, sum_aa, width + ksize - 1);
    }

    // 2. Loop Rows
    MI_S32 idx_head = 0;
    MI_S32 idx_tail = ksize - 1;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        GuideFilterSubRowNeon<Tp,      SumType  >(row_a[idx_head],  sum_a,  (width + ksize - 1) * C);
        GuideFilterSubRowNeon<SumType, SqSumType>(row_aa[idx_head], sum_aa, (width + ksize - 1) * C);

        // Update Index
        idx_head = (idx_head + 1) % ksize;
        idx_tail = (idx_tail + 1) % ksize; // new idx_tail should be old idx_head

        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, y + ksh, width, ksize, row_a[idx_tail], border_value);

        GuideFilterRowDataAccumulateNeon<Tp, SumType, SqSumType, C>(row_a[idx_tail], row_aa[idx_tail], sum_a, sum_aa, width + ksize - 1);

        // 2.2 filter done and write back
        MI_F32 *dst_a_row = dst_a.Ptr<MI_F32>(y);
        MI_F32 *dst_b_row = dst_b.Ptr<MI_F32>(y);

        GuideFilterCalcKernelSum<SumType, SqSumType, C>(sum_a, sum_aa, width, ksize, eps, dst_a_row, dst_b_row);
    }

    return Status::OK;
}

template <typename D8, typename D16, typename D32, MI_S32 C, typename d8x8xC_t = typename neon::MDVector<D8, C>::MVType,
          typename std::enable_if<std::is_same<D8, MI_U8>::value || std::is_same<D8, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d8x8xC_t GuideFilterLinearTransNeonCore(const D8 *src_row, const MI_F32 *mean_a_row, const MI_F32 *mean_b_row, MI_S32 x)
{
    using d16x8_t   = typename neon::QVector<D16>::VType;
    using d16x4_t   = typename neon::DVector<D16>::VType;
    using d32x4_t   = typename neon::QVector<D32>::VType;
    using f32x4xC_t = typename neon::MQVector<MI_F32, C>::MVType;

    d8x8xC_t mvd8_src, mvd8_result;
    f32x4xC_t mvqf32_mean_a_lo, mvqf32_mean_a_hi, mvqf32_mean_b_lo, mvqf32_mean_b_hi;

    neon::vload(src_row + x,            mvd8_src);
    neon::vload(mean_a_row + x,         mvqf32_mean_a_lo);
    neon::vload(mean_a_row + x + 4 * C, mvqf32_mean_a_hi);
    neon::vload(mean_b_row + x,         mvqf32_mean_b_lo);
    neon::vload(mean_b_row + x + 4 * C, mvqf32_mean_b_hi);

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        d16x8_t vqd16_src    = neon::vmovl(mvd8_src.val[ch]);
        d32x4_t vqd32_src_lo = neon::vmovl(neon::vgetlow(vqd16_src));
        d32x4_t vqd32_src_hi = neon::vmovl(neon::vgethigh(vqd16_src));

        d16x4_t vd16_result_lo = neon::vqmovn(neon::vcvt<D32>(neon::vmla(mvqf32_mean_b_lo.val[ch], mvqf32_mean_a_lo.val[ch],
                                              neon::vcvt<MI_F32>(vqd32_src_lo))));
        d16x4_t vd16_result_hi = neon::vqmovn(neon::vcvt<D32>(neon::vmla(mvqf32_mean_b_hi.val[ch], mvqf32_mean_a_hi.val[ch],
                                              neon::vcvt<MI_F32>(vqd32_src_hi))));
        mvd8_result.val[ch] = neon::vqmovn(neon::vcombine(vd16_result_lo, vd16_result_hi));
    }

    return mvd8_result;
}

template <typename D16, typename D32, typename D64, MI_S32 C, typename d16x8xC_t = typename neon::MQVector<D16, C>::MVType,
          typename std::enable_if<std::is_same<D16, MI_U16>::value || std::is_same<D16, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d16x8xC_t GuideFilterLinearTransNeonCore(const D16 *src_row, const MI_F32 *mean_a_row, const MI_F32 *mean_b_row, MI_S32 x)
{
    using d16x4_t   = typename neon::DVector<D16>::VType;
    using d32x4_t   = typename neon::QVector<D32>::VType;
    using f32x4xC_t = typename neon::MQVector<MI_F32, C>::MVType;

    d16x8xC_t mvd16_src, mvd16_result;
    f32x4xC_t mvqf32_mean_a_lo, mvqf32_mean_a_hi, mvqf32_mean_b_lo, mvqf32_mean_b_hi;

    neon::vload(src_row + x, mvd16_src);
    neon::vload(mean_a_row + x, mvqf32_mean_a_lo);
    neon::vload(mean_a_row + x + 4 * C, mvqf32_mean_a_hi);
    neon::vload(mean_b_row + x, mvqf32_mean_b_lo);
    neon::vload(mean_b_row + x + 4 * C, mvqf32_mean_b_hi);

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        d32x4_t vqd32_src_lo   = neon::vmovl(neon::vgetlow(mvd16_src.val[ch]));
        d32x4_t vqd32_src_hi   = neon::vmovl(neon::vgethigh(mvd16_src.val[ch]));
        d16x4_t vd16_result_lo = neon::vqmovn(neon::vcvt<D32>(neon::vmla(mvqf32_mean_b_lo.val[ch], mvqf32_mean_a_lo.val[ch],
                                              neon::vcvt<MI_F32>(vqd32_src_lo))));
        d16x4_t vd16_result_hi = neon::vqmovn(neon::vcvt<D32>(neon::vmla(mvqf32_mean_b_hi.val[ch], mvqf32_mean_a_hi.val[ch],
                                              neon::vcvt<MI_F32>(vqd32_src_hi))));
        mvd16_result.val[ch]   = neon::vcombine(vd16_result_lo, vd16_result_hi);
    }

    return mvd16_result;
}

#if defined(AURA_ENABLE_NEON_FP16)
template <typename F16, typename F32, typename F64, MI_S32 C, typename f16x8xC_t = typename neon::MQVector<F16, C>::MVType,
          typename std::enable_if<std::is_same<F16, MI_F16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE f16x8xC_t GuideFilterLinearTransNeonCore(const F16 *src_row, const MI_F32 *mean_a_row, const MI_F32 *mean_b_row, MI_S32 x)
{
    using f16x4_t   = typename neon::DVector<F16>::VType;
    using f32x4xC_t   = typename neon::MQVector<F32, C>::MVType;

    f16x8xC_t mvf16_src, mvf16_result;
    f32x4xC_t mvqf32_mean_a_lo, mvqf32_mean_a_hi, mvqf32_mean_b_lo, mvqf32_mean_b_hi;

    neon::vload(src_row + x, mvf16_src);
    neon::vload(mean_a_row + x, mvqf32_mean_a_lo);
    neon::vload(mean_a_row + x + 4 * C, mvqf32_mean_a_hi);
    neon::vload(mean_b_row + x, mvqf32_mean_b_lo);
    neon::vload(mean_b_row + x + 4 * C, mvqf32_mean_b_hi);

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        f16x4_t vf16_src_lo   = neon::vgetlow(mvf16_src.val[ch]);
        f16x4_t vf16_src_hi   = neon::vgethigh(mvf16_src.val[ch]);
        f16x4_t vf16_result_lo = neon::vcvt<F16>(neon::vmla(mvqf32_mean_b_lo.val[ch], mvqf32_mean_a_lo.val[ch],
                                                neon::vcvt<MI_F32>(vf16_src_lo)));
        f16x4_t vf16_result_hi = neon::vcvt<F16>(neon::vmla(mvqf32_mean_b_hi.val[ch], mvqf32_mean_a_hi.val[ch],
                                                neon::vcvt<MI_F32>(vf16_src_hi)));
        mvf16_result.val[ch]   = neon::vcombine(vf16_result_lo, vf16_result_hi);
    }

    return mvf16_result;
}
#endif // AURA_ENABLE_NEON_FP16

template <typename F32, typename SumType, typename SqSumType, MI_S32 C, typename f32x4xC_t = typename neon::MQVector<F32, C>::MVType,
          typename std::enable_if<std::is_same<F32, MI_F32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE f32x4xC_t GuideFilterLinearTransNeonCore(const F32 *src_row, const MI_F32 *mean_a_row, const MI_F32 *mean_b_row, MI_S32 x)
{
    f32x4xC_t mvf32_src, mvf32_result;
    f32x4xC_t mvqf32_mean_a, mvqf32_mean_b;

    neon::vload(src_row + x, mvf32_src);
    neon::vload(mean_a_row + x, mvqf32_mean_a);
    neon::vload(mean_b_row + x, mvqf32_mean_b);

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvf32_result.val[ch] = neon::vmla(mvqf32_mean_b.val[ch], mvqf32_mean_a.val[ch], mvf32_src.val[ch]);
    }

    return mvf32_result;
}

template <typename Tp, typename SumType, typename SqSumType, MI_S32 C>
static Status GuideFilterLinearTransNeonImpl(const Mat &src, const Mat &mean_a, const Mat &mean_b, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    const Sizes3 sz      = mean_a.GetSizes();
    const MI_S32 width   = sz.m_width;

    using MVType = typename std::conditional<sizeof(Tp) == 1, typename neon::MDVector<Tp, C>::MVType,
                   typename neon::MQVector<Tp, C>::MVType>::type;

    constexpr MI_S32 ELEM_COUNTS = (sizeof(Tp) == 4) ? 4 : 8;
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;
    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const Tp     *src_row    = src.Ptr<Tp>(y);
        const MI_F32 *mean_a_row = mean_a.Ptr<MI_F32>(y);
        const MI_F32 *mean_b_row = mean_b.Ptr<MI_F32>(y);

        Tp *dst_row = dst.Ptr<Tp>(y);
        MI_S32 x = 0;

        for (; x < width_align - VOFFSET; x += VOFFSET)
        {
            MVType mv_result = GuideFilterLinearTransNeonCore<Tp, SumType, SqSumType, C>(src_row, mean_a_row, mean_b_row, x);
            neon::vstore(dst_row + x, mv_result);
        }

        for (; x < sz.m_width * sz.m_channel; ++x)
        {
            dst_row[x] = SaturateCast<Tp>(static_cast<MI_F32>(src_row[x]) * mean_a_row[x] + mean_b_row[x]);
        }
    }

    return Status::OK;
}

class GuideFilterNeon : public GuideFilterImpl
{
public:
    GuideFilterNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_S32 ksize, MI_F32 eps,
                   GuideFilterType type = GuideFilterType::NORMAL,
                   BorderType border_type = BorderType::REPLICATE,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status GuideFilterNormalNeon(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const MI_S32 &ksize, const MI_F32 &eps,
                             BorderType &border_type, const Scalar &border_value, const OpTarget &target);

Status GuideFilterFastNeon(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const MI_S32 &ksize, const MI_F32 &eps,
                            BorderType &border_type, const Scalar &border_value, const OpTarget &target);

#endif // defined(AURA_ENABLE_NEON)

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
// ...
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_FILTER_GUIDEFILTER_IMPL_HPP__
