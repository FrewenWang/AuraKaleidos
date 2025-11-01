#ifndef AURA_OPS_MATRIX_INTEGRAL_IMPL_HPP__
#define AURA_OPS_MATRIX_INTEGRAL_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class IntegralImpl : public OpImpl
{
public:
    IntegralImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, Array *dst_sq = MI_NULL);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
    Array       *m_dst_sq;
};

class IntegralNone : public IntegralImpl
{
public:
    IntegralNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, Array *dst_sq = MI_NULL) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class IntegralNeon : public IntegralImpl
{
public:
    IntegralNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, Array *dst_sq = MI_NULL) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class IntegralHvx : public IntegralImpl
{
public:
    IntegralHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, Array *dst_sq = MI_NULL) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

using IntegralInParam = HexagonRpcParamType<Mat, Mat, Mat>;
#  define AURA_OPS_MATRIX_INTEGRAL_OP_NAME          "Integral"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_MATRIX_INTEGRAL_IMPL_HPP__