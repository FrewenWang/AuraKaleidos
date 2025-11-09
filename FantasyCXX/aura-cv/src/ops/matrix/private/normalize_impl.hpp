#ifndef AURA_OPS_MATRIX_NORMALIZE_IMPL_HPP__
#define AURA_OPS_MATRIX_NORMALIZE_IMPL_HPP__

#include "aura/ops/matrix/norm.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class NormalizeImpl : public OpImpl
{
public:
    NormalizeImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
    DT_F32       m_alpha;
    DT_F32       m_beta;
    NormType     m_type;
};

class NormalizeNone : public NormalizeImpl
{
public:
    NormalizeNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class NormalizeNeon : public NormalizeImpl
{
public:
    NormalizeNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_NORMALIZE_IMPL_HPP__
