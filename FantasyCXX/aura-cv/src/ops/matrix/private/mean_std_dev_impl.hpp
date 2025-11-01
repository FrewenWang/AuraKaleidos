#ifndef AURA_OPS_MATRIX_MEAN_STD_DEV_IMPL_HPP__
#define AURA_OPS_MATRIX_MEAN_STD_DEV_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class MeanStdDevImpl : public OpImpl
{
public:
    MeanStdDevImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Scalar *mean, Scalar *std_dev);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Scalar *m_mean;
    Scalar *m_std_dev;
};

class MeanStdDevNone : public MeanStdDevImpl
{
public:
    MeanStdDevNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *mean, Scalar *std_dev) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class MeanStdDevNeon : public MeanStdDevImpl
{
public:
    MeanStdDevNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Scalar *mean, Scalar *std_dev) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_MEAN_STD_DEV_IMPL_HPP__
