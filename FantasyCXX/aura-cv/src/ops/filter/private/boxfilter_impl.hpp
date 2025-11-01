#ifndef AURA_OPS_FILTER_BOXFILTER_IMPL_HPP__
#define AURA_OPS_FILTER_BOXFILTER_IMPL_HPP__

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

class BoxFilterImpl : public OpImpl
{
public:
    BoxFilterImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    MI_S32      m_ksize;
    BorderType  m_border_type;
    Scalar      m_border_value;

    const Array *m_src;
    Array       *m_dst;
};

class BoxFilterNone : public BoxFilterImpl
{
public:
    BoxFilterNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_src_border;
};

#if defined(AURA_ENABLE_NEON)
class BoxFilterNeon : public BoxFilterImpl
{
public:
    BoxFilterNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status BoxFilter3x3Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status BoxFilter5x5Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status BoxFilter7x7Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status BoxFilterKxKNeon(Context *ctx, const Mat &src, Mat &dst, const MI_S32 ksize,
                        BorderType border_type, const Scalar &border_value, const OpTarget &target);
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class BoxFilterCL : public BoxFilterImpl
{
public:
    BoxFilterCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
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

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class BoxFilterHvx : public BoxFilterImpl
{
public:
    BoxFilterHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status BoxFilter3x3Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status BoxFilter5x5Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status BoxFilter7x7Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status BoxFilter9x9Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status BoxFilter11x11Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status BoxFilterKxKHvx(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, BorderType border_type, const Scalar &border_value);
#  endif // AURA_BUILD_HEXAGON

using BoxFilterInParam = HexagonRpcParamType<Mat, Mat, MI_S32, BorderType, Scalar>;
#  define AURA_OPS_FILTER_BOXFILTER_OP_NAME          "BoxFilter"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_FILTER_BOXFILTER_IMPL_HPP__