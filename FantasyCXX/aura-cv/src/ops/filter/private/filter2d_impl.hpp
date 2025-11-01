#ifndef AURA_OPS_FILTER_FILTER2D_IMPL_HPP__
#define AURA_OPS_FILTER_FILTER2D_IMPL_HPP__

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

class Filter2dImpl : public OpImpl
{
public:
    Filter2dImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

private:

protected:
    MI_S32      m_ksize;
    BorderType  m_border_type;
    Scalar      m_border_value;

    const Array *m_src;
    Array       *m_dst;
    const Array *m_kmat;
};

class Filter2dNone : public Filter2dImpl
{
public:
    Filter2dNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_src_border;
};

#if defined(AURA_ENABLE_NEON)
class Filter2dNeon : public Filter2dImpl
{
public:
    Filter2dNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status Filter2d3x3Neon(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_F32> &kdata,
                       BorderType border_type, const Scalar &border_value, const OpTarget &target);
Status Filter2d5x5Neon(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_F32> &kdata,
                       BorderType border_type, const Scalar &border_value, const OpTarget &target);
Status Filter2d7x7Neon(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_F32> &kdata,
                       BorderType border_type, const Scalar &border_value, const OpTarget &target);
#endif

#if defined(AURA_ENABLE_OPENCL)
class Filter2dCL : public Filter2dImpl
{
public:
    Filter2dCL(Context *ctx, const OpTarget &target);
    Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_kmat;

    std::string m_profiling_string;
};
#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class Filter2dHvx : public Filter2dImpl
{
public:
    Filter2dHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, const Array *kmat,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status Filter2d3x3Hvx(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                      BorderType border_type, const Scalar &border_value);
Status Filter2d5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                      BorderType border_type, const Scalar &border_value);
Status Filter2d7x7Hvx(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                      BorderType border_type, const Scalar &border_value);
#  endif // AURA_BUILD_HEXAGON

using Filter2dInParam = HexagonRpcParamType<Mat, Mat, Mat, BorderType, Scalar>;
#  define AURA_OPS_FILTER_FILTER2D_OP_NAME          "Filter2d"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

}

#endif // AURA_OPS_FILTER_FILTER2D_IMPL_HPP__