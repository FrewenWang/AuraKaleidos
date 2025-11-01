#ifndef AURA_OPS_MATRIX_SUM_IMPL_HPP__
#define AURA_OPS_MATRIX_SUM_IMPL_HPP__

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

class SumImpl : public OpImpl
{
public:
    SumImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Scalar *result);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    Scalar *m_result;
    const Array *m_src;
};

class SumNone : public SumImpl
{
public:
    SumNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *result) override;

    Status Run() override;
};

class MeanNone : public SumNone
{
public:
    MeanNone(Context *ctx, const OpTarget &target);

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class SumNeon : public SumImpl
{
public:
    SumNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *result) override;

    Status Run() override;
};

class MeanNeon : public SumNeon
{
public:
    MeanNeon(Context *ctx, const OpTarget &target);

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class SumCL : public SumImpl
{
public:
    SumCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *result) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type);

private:
    MI_S32 m_blk_h = 128;
    MI_S32 m_blk_w = 32;
    MI_S32 m_group_size_x_main;
    MI_S32 m_group_size_y_main;
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_partial;
    CLMem m_cl_dst;
    Mat dst_mat;

    std::string m_profiling_string;
};

class MeanCL : public SumCL
{
public:
    MeanCL(Context *ctx, const OpTarget &target);

    Status Run() override;
};

#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class SumHvx : public SumImpl
{
public:
    SumHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *result) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

class MeanHvx : public SumHvx
{
public:
    MeanHvx(Context *ctx, const OpTarget &target);

    Status Run() override;
};

using SumInParam = HexagonRpcParamType<Mat>;
using SumOutParam = HexagonRpcParamType<Scalar>;
#  define AURA_OPS_MATRIX_SUM_OP_NAME          "Sum"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_MATRIX_SUM_IMPL_HPP__