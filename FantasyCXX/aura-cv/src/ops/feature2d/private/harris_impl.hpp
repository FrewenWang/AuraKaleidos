#ifndef AURA_OPS_FEATURE2D_HARRIS_IMPL_HPP__
#define AURA_OPS_FEATURE2D_HARRIS_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{
class HarrisImpl : public OpImpl
{
public:
    HarrisImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32     m_block_size;
    DT_S32     m_ksize;
    DT_F64     m_k;
    BorderType m_border_type;
    Scalar     m_border_value;

    const Array *m_src;
    Array *m_dst;
};

class HarrisNone : public HarrisImpl
{
public:
    HarrisNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;

private:
    std::string m_profiling_string;
};

Status CornerEigenValsVecsNone(Context *ctx, const Mat &src, Mat &dst, DT_S32 block_size, DT_S32 aperture_size, DT_BOOL use_harris,
                               DT_F64 k, BorderType border_type, const Scalar &border_value, const OpTarget &target);

#if defined(AURA_ENABLE_NEON)
class HarrisNeon : public HarrisImpl
{
public:
    HarrisNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status CornerEigenValsVecsNeon(Context *ctx, const Mat &src, Mat &dst, DT_S32 block_size, DT_S32 aperture_size,
                               DT_BOOL use_harris, DT_F64 k, BorderType border_type, const Scalar &border_value, const OpTarget &target);
#endif

} // namespace aura

#endif // AURA_OPS_FEATURE2D_HARRIS_IMPL_HPP__