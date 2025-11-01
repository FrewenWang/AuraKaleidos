#ifndef AURA_OPS_MATRIX_BINARY_IMPL_HPP__
#define AURA_OPS_MATRIX_BINARY_IMPL_HPP__

#include "aura/ops/matrix/binary.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class BinaryImpl : public OpImpl
{
public:
    BinaryImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src0;
    const Array *m_src1;
    Array *m_dst;
    BinaryOpType m_type;
};

class BinaryNone : public BinaryImpl
{
public:
    BinaryNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class BinaryNeon : public BinaryImpl
{
public:
    BinaryNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type) override;

    Status Run() override;
};
#endif

#if defined(AURA_ENABLE_OPENCL)
class BinaryCL : public BinaryImpl
{
public:
    BinaryCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, BinaryOpType op_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src0;
    CLMem m_cl_src1;
    CLMem m_cl_dst;
    MI_S32 m_elem_counts;

    Mat m_fkmat;
    std::string m_profiling_string;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_BINARY_IMPL_HPP__
