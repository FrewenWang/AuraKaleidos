#ifndef AURA_OPS_FILTER_LAPLACIAN_IMPL_HPP__
#define AURA_OPS_FILTER_LAPLACIAN_IMPL_HPP__

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

class LaplacianImpl : public OpImpl
{
public:
    LaplacianImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32     m_ksize;
    BorderType m_border_type;
    Scalar     m_border_value;

    const Array *m_src;
    Array       *m_dst;
};

class LaplacianNone : public LaplacianImpl
{
public:
    LaplacianNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class LaplacianNeon : public LaplacianImpl
{
public:
    LaplacianNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status Laplacian1x1Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status Laplacian3x3Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status Laplacian5x5Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
Status Laplacian7x7Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target);
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class LaplacianCL : public LaplacianImpl
{
public:
    LaplacianCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class LaplacianHvx : public LaplacianImpl
{
public:
    LaplacianHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status Laplacian1x1Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status Laplacian3x3Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status Laplacian5x5Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
Status Laplacian7x7Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value);
#  endif // AURA_BUILD_HEXAGON

using LaplacianInParam = HexagonRpcParamType<Mat, Mat, DT_S32, BorderType, Scalar>;
#  define AURA_OPS_FILTER_LAPLACIAN_OP_NAME          "Laplacian"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

}; // namespace aura

#endif // AURA_OPS_FILTER_LAPLACIAN_IMPL_HPP__