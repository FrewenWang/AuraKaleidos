#ifndef AURA_OPS_MATRIX_PSNR_IMPL_HPP__
#define AURA_OPS_MATRIX_PSNR_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class PsnrImpl : public OpImpl
{
public:
    PsnrImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src0, const Array *src1, DT_F64 coef_r, DT_F64 *result);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src0;
    const Array *m_src1;
    DT_F64 m_coef_r;
    DT_F64 *m_result;
};

class PsnrNone : public PsnrImpl
{
public:
    PsnrNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src0, const Array *src1, DT_F64 coef_r, DT_F64 *result) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_MATRIX_PSNR_IMPL_HPP__
