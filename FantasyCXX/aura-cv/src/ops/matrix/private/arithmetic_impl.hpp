#ifndef AURA_OPS_FILTER_ARITHMETIC_IMPL_HPP__
#define AURA_OPS_FILTER_ARITHMETIC_IMPL_HPP__

#include "aura/ops/matrix/arithmetic.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#    include "aura/runtime/hexagon.h"
#endif
namespace aura
{

/**
 * ArithmIntegerTraits table -> WType
 *         ADD    SUB    MUL    DIV
 *  U8     U16    S16    U16    U16
 *  S8     S16    S16    S16    S16
 *  U16    U32    S32    U32    U32
 *  S16    S32    S32    S32    S32
 *  U32    U32    S32    U32    U32
 *  S32    S32    S32    S32    S32
 */

template <typename Tp, ArithmOpType TYPE, typename Tp1 = DT_VOID>
struct ArithmIntegerTraits
{
    using WType = typename Promote<Tp>::Type;
};

template <typename Tp>
struct ArithmIntegerTraits<Tp, ArithmOpType::ADD, typename std::enable_if<(4 == sizeof(Tp))>::type>
{
    using WType = Tp;
};

template <typename Tp>
struct ArithmIntegerTraits<Tp, ArithmOpType::SUB, typename std::enable_if<(1 == sizeof(Tp))>::type>
{
    using WType = DT_S16;
};

template <typename Tp>
struct ArithmIntegerTraits<Tp, ArithmOpType::SUB, typename std::enable_if<(2 <= sizeof(Tp))>::type>
{
    using WType = DT_S32;
};

template <typename Tp>
struct ArithmIntegerTraits<Tp, ArithmOpType::MUL, typename std::enable_if<(4 == sizeof(Tp))>::type>
{
    using WType = Tp;
};

template <typename Tp>
struct ArithmIntegerTraits<Tp, ArithmOpType::DIV, typename std::enable_if<(4 == sizeof(Tp))>::type>
{
    using WType = Tp;
};

class ArithmeticImpl : public OpImpl
{
public:
    ArithmeticImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:

    ArithmOpType m_op_type;
    const Array *m_src0;
    const Array *m_src1;
    Array       *m_dst;
};

class ArithmeticNone : public ArithmeticImpl
{
public:
    ArithmeticNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op) override;

    Status Run() override;

};

#if defined(AURA_ENABLE_NEON)
class ArithmeticNeon : public ArithmeticImpl
{
public:
    ArithmeticNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class ArithmeticCL : public ArithmeticImpl
{
public:
    ArithmeticCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, ArithmOpType op_type);

private:
    DT_S32 m_elem_counts;
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src0;
    CLMem m_cl_src1;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class ArithmeticHvx : public ArithmeticImpl
{
public:
    ArithmeticHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using ArithmeticInParam = HexagonRpcParamType<Mat, Mat, Mat, ArithmOpType>;
#define AURA_OPS_MATRIX_ARITHMETIC_OP_NAME          "Arithmetic"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

class ScalarDivideImpl : public OpImpl
{
public:
    ScalarDivideImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(DT_F32 scalar, const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
    DT_F32       m_scalar;
};

class ScalarDivideNone : public ScalarDivideImpl
{
public:
    ScalarDivideNone(Context *ctx, const OpTarget &target);

    Status SetArgs(DT_F32 scalar, const Array *src, Array *dst) override;

    Status Run() override;

};

#if defined(AURA_ENABLE_NEON)
class ScalarDivideNeon : public ScalarDivideImpl
{
public:
    ScalarDivideNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(DT_F32 scalar, const Array *src, Array *dst) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_FILTER_ARITHMETIC_IMPL_HPP__
