#ifndef AURA_OPS_FEATURE2D_TOMASI_IMPL_HPP__
#define AURA_OPS_FEATURE2D_TOMASI_IMPL_HPP__

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
class TomasiImpl : public OpImpl
{
public:
    TomasiImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                           DT_F64 quality_leve, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                           DT_BOOL use_harris = DT_FALSE, DT_F64 harris_k = 0.04);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32 m_max_corners;
    DT_F64 m_quality_level;
    DT_F64 m_min_distance;
    DT_S32 m_block_size;
    DT_S32 m_gradient_size;
    DT_BOOL m_use_harris;
    DT_F64 m_harris_k;

    const Array *m_src;
    std::vector<KeyPoint> *m_key_points;
};

class TomasiNone : public TomasiImpl
{
public:
    TomasiNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                   DT_F64 quality_leve, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                   DT_BOOL use_harris = DT_FALSE, DT_F64 harris_k = 0.04) override;

    Status Run() override;

private:
    std::string m_profiling_string;
};


#if defined(AURA_ENABLE_NEON)
class TomasiNeon : public TomasiImpl
{
public:
    TomasiNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 max_num_corners,
                   DT_F64 quality_leve, DT_F64 min_distance, DT_S32 block_size, DT_S32 gradient_size,
                   DT_BOOL use_harris = DT_FALSE, DT_F64 harris_k = 0.04) override;

    Status Run() override;
};

#endif

} // namespace aura

#endif // AURA_OPS_FEATURE2D_TOMASI_IMPL_HPP__