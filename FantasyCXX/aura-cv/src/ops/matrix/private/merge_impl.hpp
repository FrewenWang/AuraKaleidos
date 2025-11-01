#ifndef AURA_OPS_MATRIX_MERGE_IMPL_HPP__
#define AURA_OPS_MATRIX_MERGE_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class MergeImpl : public OpImpl
{
public:
    MergeImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const std::vector<const Array*> &src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:

    std::vector<const Array*> m_src;
    Array *m_dst;
};

class MergeNone : public MergeImpl
{
public:
    MergeNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, Array *dst) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_MATRIX_MERGE_IMPL_HPP__
