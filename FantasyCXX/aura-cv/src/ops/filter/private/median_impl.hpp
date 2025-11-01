#ifndef AURA_OPS_FILTER_MEDIAN_IMPL_HPP__
#define AURA_OPS_FILTER_MEDIAN_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class MedianImpl : public OpImpl
{
public:
    MedianImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 ksize);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    MI_S32       m_ksize;
    const Array *m_src;
    Array       *m_dst;
};

class MedianNone : public MedianImpl
{
public:
    MedianNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_src_border;
};

#if defined(AURA_ENABLE_NEON)
class MedianNeon : public MedianImpl
{
public:
    MedianNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize) override;

    Status Run() override;
};

template<typename Tp>
AURA_ALWAYS_INLINE AURA_VOID MinMaxOp(Tp &a, Tp &b)
{
    Tp t = a;
    a = neon::vmin(t, b);
    b = neon::vmax(t, b);
}

Status Median3x3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status Median5x5Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status Median7x7Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class MedianCL : public MedianImpl
{
public:
    MedianCL(Context *ctx, const OpTarget &target);
    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class MedianHvx : public MedianImpl
{
public:
    MedianHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status Median3x3Hvx(Context *ctx, const Mat &src, Mat &dst);
Status Median5x5Hvx(Context *ctx, const Mat &src, Mat &dst);
Status Median7x7Hvx(Context *ctx, const Mat &src, Mat &dst);

template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorMinMax(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    return Q6_vminmax_VubVub(v_src0, v_src1);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorMinMax(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    return Q6_vminmax_VbVb(v_src0, v_src1);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorMinMax(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    return Q6_vminmax_VuhVuh(v_src0, v_src1);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID VectorMinMax(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    return Q6_vminmax_VhVh(v_src0, v_src1);
}
#  endif // AURA_BUILD_HEXAGON

using MedianInParam = HexagonRpcParamType<Mat, Mat, MI_S32>;

#  define AURA_OPS_FILTER_MEDIAN_OP_NAME          "Median"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
}

#endif // AURA_OPS_FILTER_Median_IMPL_HPP__