#ifndef AURA_OPS_MATRIX_DFT_IMPL_HPP__
#define AURA_OPS_MATRIX_DFT_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

#include <complex>
namespace aura
{

#if defined(AURA_ENABLE_NEON)
template <typename Tp> struct CvtDstVec;

template <> struct CvtDstVec<MI_U8>  { using VType = uint8x8_t;     };
template <> struct CvtDstVec<MI_S8>  { using VType = int8x8_t;      };
template <> struct CvtDstVec<MI_U16> { using VType = uint16x8_t;    };
template <> struct CvtDstVec<MI_S16> { using VType = int16x8_t;     };
template <> struct CvtDstVec<MI_U32> { using VType = uint32x4x2_t;  };
template <> struct CvtDstVec<MI_S32> { using VType = int32x4x2_t;   };
template <> struct CvtDstVec<MI_F16> { using VType = float16x8_t;   };
template <> struct CvtDstVec<MI_F32> { using VType = float32x4x2_t; };

template <typename Tp, typename std::enable_if<(std::is_same<Tp, MI_F32>::value)>::type* = MI_NULL>
AURA_ALWAYS_INLINE float32x4x2_t CvtFromF32Pair(const float32x4x2_t &v)
{
    return v;
}

#  if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp, typename std::enable_if<(std::is_same<Tp, MI_F16>::value)>::type* = MI_NULL>
AURA_ALWAYS_INLINE float16x8_t CvtFromF32Pair(const float32x4x2_t &v)
{
    return neon::vcombine(neon::vcvt<MI_F16>(v.val[0]), neon::vcvt<MI_F16>(v.val[1]));
}
#  endif // AURA_ENABLE_NEON_FP16

template <typename Tp, typename std::enable_if<(std::is_same<Tp, MI_S32>::value) || (std::is_same<Tp, MI_U32>::value)>::type* = MI_NULL>
AURA_ALWAYS_INLINE typename CvtDstVec<Tp>::VType CvtFromF32Pair(const float32x4x2_t &v)
{
    using VType = typename CvtDstVec<Tp>::VType;

    VType v2q;
    v2q.val[0] = neon::vcvt<Tp>(v.val[0]);
    v2q.val[1] = neon::vcvt<Tp>(v.val[1]);

    return v2q;
}

template <typename Tp, typename std::enable_if<(std::is_same<Tp, MI_S16>::value) || (std::is_same<Tp, MI_U16>::value)>::type* = MI_NULL>
AURA_ALWAYS_INLINE typename CvtDstVec<Tp>::VType CvtFromF32Pair(const float32x4x2_t &v)
{
    using Type  = typename Promote<Tp>::Type;
    using VType = typename CvtDstVec<Type>::VType;

    VType v_promote = CvtFromF32Pair<Type>(v);

    return neon::vcombine(neon::vqmovn(v_promote.val[0]), neon::vqmovn(v_promote.val[1]));
}

template <typename Tp, typename std::enable_if<(std::is_same<Tp, MI_S8>::value) || (std::is_same<Tp, MI_U8>::value)>::type* = MI_NULL>
AURA_ALWAYS_INLINE typename CvtDstVec<Tp>::VType CvtFromF32Pair(const float32x4x2_t &v)
{
    using Type  = typename Promote<Tp>::Type;
    using VType = typename CvtDstVec<Type>::VType;

    VType v_promote = CvtFromF32Pair<Type>(v);

    return neon::vqmovn(v_promote);
}

template <typename Dt, typename std::enable_if<((sizeof(Dt) < 4))>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID StoreF32PairAs(Dt *p, const float32x4x2_t &v)
{
    using VType = typename CvtDstVec<Dt>::VType;

    VType vq = CvtFromF32Pair<Dt>(v);
    neon::vstore(p, vq);
}

template <typename Dt, typename std::enable_if<((sizeof(Dt) == 4))>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID StoreF32PairAs(Dt *p, const float32x4x2_t &v)
{
    using VType = typename CvtDstVec<Dt>::VType;

    VType v2q = CvtFromF32Pair<Dt>(v);
    neon::vstore(p,     v2q.val[0]);
    neon::vstore(p + 4, v2q.val[1]);
}
#endif

AURA_ALWAYS_INLINE MI_BOOL IsPowOf2(MI_S32 n)
{
    return n && !(n & (n - 1));
}

AURA_ALWAYS_INLINE MI_U16 FFTBitReverseU16(MI_U16 value, MI_U8 bit_num)
{
    value = ((value & 0x5555) << 1) | ((value & 0xAAAA) >> 1);
    value = ((value & 0x3333) << 2) | ((value & 0xCCCC) >> 2);
    value = ((value & 0x0F0F) << 4) | ((value & 0xF0F0) >> 4);
    value = ((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8);
    return value >> (16 - bit_num);
}

AURA_ALWAYS_INLINE AURA_VOID GetReverseIndex(MI_U16 *idx_table, MI_S32 n)
{
    if (!IsPowOf2(n))
    {
        return;
    }

    MI_U8 levels = static_cast<MI_U8>(Log2((MI_F32) n));

    for (MI_U16 i = 0; i < n; ++i)
    {
        idx_table[i] = FFTBitReverseU16(i, levels);
    }
}

template <MI_U8 IS_INVERSE>
AURA_ALWAYS_INLINE AURA_VOID GetDftExpTable(std::complex<MI_F32> *exp_table, MI_S32 n)
{
    MI_S32 half_n = n / 2;
    MI_F32 pi_pe = (IS_INVERSE ? 2.0f : -2.0f) * AURA_PI / n;

    for (MI_S32 i = 0; i < half_n; ++i)
    {
        MI_F32 theta = pi_pe * i;
        exp_table[i].real(Cos(theta));
        exp_table[i].imag(Sin(theta));
    }
}

template <MI_U8 IS_INVERSE>
AURA_ALWAYS_INLINE AURA_VOID GetBlueSteinExpTable(std::complex<MI_F32> *exp_table, MI_S32 n)
{
    MI_F32 pi_pe = (IS_INVERSE ? 1.0f : -1.0f) * AURA_PI / n;

    for (MI_S32 i = 0; i < n; i++)
    {
        MI_S32 temp = (MI_S32) i * i;
        temp %= (2 * n);
        MI_F32 angle = pi_pe * temp;
        exp_table[i].real(Cos(angle));
        exp_table[i].imag(Sin(angle));
    }
}

template <typename Tp, MI_U8 IS_INVERSE>
AURA_VOID DftRadix2RowProcCol1None(const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    MI_S32 height = sz.m_height;

    for (MI_S32 y = 0; y < height; ++y)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        MI_F32 *dst_row = dst.Ptr<MI_F32>(y);

        dst_row[0] = SaturateCast<MI_F32>(src_row[0]);
        dst_row[1] = IS_INVERSE ? SaturateCast<MI_F32>(src_row[1]) : 0.0f;
    }
}

template <typename Tp, MI_U8 IS_INVERSE>
AURA_VOID DftRadix2RowProcCol2None(const Mat &src, Mat &dst)
{
    Sizes3 sz     = src.GetSizes();
    MI_S32 height = sz.m_height;

    if (IS_INVERSE)
    {
        for (MI_S32 y = 0; y < height; ++y)
        {
            const std::complex<MI_F32> *src_row = src.Ptr<std::complex<MI_F32>>(y);
            std::complex<MI_F32> *dst_row = dst.Ptr<std::complex<MI_F32>>(y);

            dst_row[0] = (src_row[0] + src_row[1]) / 2.0f;
            dst_row[1] = (src_row[0] - src_row[1]) / 2.0f;
        }
    }
    else
    {
        for (MI_S32 y = 0; y < height; ++y)
        {
            const Tp *src_row = src.Ptr<Tp>(y);
            MI_F32 *dst_row = dst.Ptr<MI_F32>(y);

            dst_row[0] = SaturateCast<MI_F32>(src_row[0]) + SaturateCast<MI_F32>(src_row[1]);
            dst_row[1] = 0.0f;
            dst_row[2] = SaturateCast<MI_F32>(src_row[0]) - SaturateCast<MI_F32>(src_row[1]);
            dst_row[3] = 0.0f;
        }
    }
}

AURA_ALWAYS_INLINE AURA_VOID ButterflyTransformNone(std::complex<MI_F32> *src, MI_S32 start_level, MI_S32 n,
                                                  MI_BOOL with_scale, const std::complex<MI_F32> *exp_table)
{
    for (MI_S32 size = start_level; size <= n; size *= 2)
    {
        MI_S32 half_size = size / 2;
        MI_S32 table_step = n / size;
        for (MI_S32 i = 0; i < n; i += size)
        {
            for (MI_S32 j = i, k = 0; j < i + half_size; j++, k += table_step)
            {
                std::complex<MI_F32> temp = src[j + half_size] * exp_table[k];
                src[j + half_size] = src[j] - temp;
                src[j] += temp;
            }
        }
    }

    if (with_scale)
    {
        for (MI_S32 i = 0; i < n; ++i)
        {
            src[i] /= n;
        }
    }
}

class DftImpl : public OpImpl
{
public:
    DftImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
};

class DftNone : public DftImpl
{
public:
    DftNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;

};

#if defined(AURA_ENABLE_NEON)
AURA_VOID ButterflyTransformNeon(std::complex<MI_F32> *src, MI_S32 start_level, MI_S32 n, MI_BOOL with_scale,
                               const std::complex<MI_F32> *dft_exp_table);

class DftNeon : public DftImpl
{
public:
    DftNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class DftCL : public DftImpl
{
public:
    DftCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type);

private:
    MI_S32                m_buffer_pitch;
    MI_S32                m_local_buffer_bytes;
    MI_S32                m_exp_total_bytes;
    std::vector<CLKernel> m_cl_kernels;
    Mat                   m_param;
    CLMem                 m_cl_src;
    CLMem                 m_cl_param;
    CLMem                 m_cl_dst;

    std::string m_profiling_string;
};
#endif

class InverseDftImpl : public OpImpl
{
public:
    InverseDftImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_BOOL with_scale);

    Status Initialize() override;

    Status DeInitialize() override;

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
    Mat         m_mid;
    MI_S32      m_with_scale;
};

class InverseDftNone : public InverseDftImpl
{
public:
    InverseDftNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_BOOL with_scale) override;

    Status Run() override;

};

#if defined(AURA_ENABLE_NEON)
class InverseDftNeon : public InverseDftImpl
{
public:
    InverseDftNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_BOOL with_scale) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class InverseDftCL : public InverseDftImpl
{
public:
    InverseDftCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_BOOL with_scale) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, MI_S32 with_scale, MI_BOOL is_dst_c1);

private:
    MI_S32                m_buffer_pitch;
    MI_S32                m_local_buffer_bytes;
    MI_S32                m_exp_total_bytes;
    std::vector<CLKernel> m_cl_kernels;
    Mat                   m_param;
    CLMem                 m_cl_src;
    CLMem                 m_cl_param;
    CLMem                 m_cl_dst;
    CLMem                 m_cl_mid;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_DFT_IMPL_HPP__
