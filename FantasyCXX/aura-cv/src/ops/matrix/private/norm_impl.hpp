#ifndef AURA_OPS_MATRIX_NORM_IMPL_HPP__
#define AURA_OPS_MATRIX_NORM_IMPL_HPP__

#include "aura/ops/matrix/norm.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class NormImpl : public OpImpl
{
public:
    NormImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, MI_F64 *result, NormType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    MI_F64 *m_result;
    NormType m_type;
};
class NormNone : public NormImpl
{
public:
    NormNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, MI_F64 *result, NormType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
Status AbsSumNeon(Context *ctx, const Mat &src, MI_F64 &result, const OpTarget &target);
Status SqSumNeon(Context *ctx, const Mat &mat, Scalar &result, const OpTarget &target);

class NormNeon : public NormImpl
{
public:
    NormNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, MI_F64 *result, NormType type) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class NormCL : public NormImpl
{
public:
    NormCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, MI_F64 *result, NormType type) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, NormType m_type);

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
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_NORM_IMPL_HPP__
