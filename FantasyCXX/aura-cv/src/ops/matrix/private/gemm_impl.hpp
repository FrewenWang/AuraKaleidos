#ifndef AURA_OPS_MATRIX_GEMM_IMPL_HPP__
#define AURA_OPS_MATRIX_GEMM_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class GemmImpl : public OpImpl
{
public:
    GemmImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src0;
    const Array *m_src1;
    Array       *m_dst;
};

class GemmNone : public GemmImpl
{
public:
    GemmNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class GemmNeon : public GemmImpl
{
public:
    GemmNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class GemmAdrenoCL : public GemmImpl
{
public:
    GemmAdrenoCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, MI_S32 elem_counts, MI_S32 load_size);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem                 m_cl_src0;
    CLMem                 m_cl_src1;
    CLMem                 m_cl_dst;
    MI_S32                m_bm = 64;
    MI_S32                m_bn = 64;
    MI_S32                m_bk = 8;
    MI_S32                m_local_size_x;
    MI_S32                m_local_size_y;
    MI_S32                m_elem_counts;

    std::string m_profiling_string;
};

class GemmMaliCL : public GemmImpl
{
public:
    GemmMaliCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem                 m_cl_src0;
    CLMem                 m_cl_src1;
    CLMem                 m_cl_dst;
    MI_S32                m_elem_counts;

    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_GEMM_IMPL_HPP__
