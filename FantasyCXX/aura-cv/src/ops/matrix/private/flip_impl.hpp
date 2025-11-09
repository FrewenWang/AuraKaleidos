#ifndef AURA_OPS_MATRIX_FLIP_IMPL_HPP__
#define AURA_OPS_MATRIX_FLIP_IMPL_HPP__

#include "aura/ops/matrix/flip.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class FlipImpl : public OpImpl
{
public:
    FlipImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, FlipType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    Array       *m_dst;
    FlipType     m_type;
};

class FlipNone : public FlipImpl
{
public:
    FlipNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, FlipType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class FlipNeon : public FlipImpl
{
public:
    FlipNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, FlipType type) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

} // namespace aura

#endif // AURA_OPS_MATRIX_FLIP_IMPL_HPP__
