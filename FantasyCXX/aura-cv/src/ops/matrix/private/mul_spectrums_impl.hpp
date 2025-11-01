#ifndef AURA_OPS_MATRIX_MUL_SPECTRUMS_IMPL_HPP__
#define AURA_OPS_MATRIX_MUL_SPECTRUMS_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class MulSpectrumsImpl : public OpImpl
{
public:
    MulSpectrumsImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src0;
    const Array *m_src1;
    Array *m_dst;
    MI_BOOL m_conj_src1;
};

class MulSpectrumsNone : public MulSpectrumsImpl
{
public:
    MulSpectrumsNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class MulSpectrumsNeon : public MulSpectrumsImpl
{
public:
    MulSpectrumsNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class MulSpectrumsCL : public MulSpectrumsImpl
{
public:
    MulSpectrumsCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_BOOL conj_src1);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src0;
    CLMem m_cl_src1;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

} // namespace aura

#endif // AURA_OPS_MATRIX_MUL_SPECTRUMS_IMPL_HPP__
