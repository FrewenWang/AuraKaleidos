#ifndef AURA_OPS_FILTER_CONVERT_TO_IMPL_HPP__
#define AURA_OPS_FILTER_CONVERT_TO_IMPL_HPP__

#include "aura/ops/matrix/convert_to.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class ConvertToImpl : public OpImpl
{
public:
    ConvertToImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_F32 alpha, MI_F32 beta);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
    MI_F32       m_alpha;
    MI_F32       m_beta;
};

class ConvertToNone : public ConvertToImpl
{
public:
    ConvertToNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_F32 alpha, MI_F32 beta) override;

    Status Run() override;

};

#if defined(AURA_ENABLE_NEON)
class ConvertToNeon : public ConvertToImpl
{
public:
    ConvertToNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_F32 alpha, MI_F32 beta) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class ConvertToCL : public ConvertToImpl
{
public:
    ConvertToCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, MI_BOOL scale);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem    m_cl_src;
    CLMem    m_cl_dst;
    MI_BOOL  m_is_same_mat;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_FILTER_CONVERT_TO_IMPL_HPP__
