#ifndef AURA_OPS_MATRIX_MIN_MAX_LOC_IMPL_HPP__
#define AURA_OPS_MATRIX_MIN_MAX_LOC_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class MinMaxLocImpl : public OpImpl
{
public:
    MinMaxLocImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    MI_F64 *m_min_val;
    MI_F64 *m_max_val;
    Point3i *m_min_pos;
    Point3i *m_max_pos;
};

class MinMaxLocNone : public MinMaxLocImpl
{
public:
    MinMaxLocNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class MinMaxLocNeon : public MinMaxLocImpl
{
public:
    MinMaxLocNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_MATRIX_MIN_MAX_LOC_IMPL_HPP__
