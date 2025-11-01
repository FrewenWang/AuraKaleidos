#ifndef AURA_OPS_MATRIX_SPLIT_IMPL_HPP__
#define AURA_OPS_MATRIX_SPLIT_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class SplitImpl : public OpImpl
{
public:
    SplitImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, const std::vector<Array*> &dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    const Array *m_src;
    std::vector<Array*> m_dst;
};

class SplitNone : public SplitImpl
{
public:
    SplitNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, const std::vector<Array*> &dst) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class SplitNeon : public SplitImpl
{
public:
    SplitNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, const std::vector<Array*> &dst) override;

    Status Run() override;
};
#endif // AURA_ENABLE_NEON

} // namespace aura

#endif // AURA_OPS_MATRIX_SPLIT_IMPL_HPP__
