#ifndef AURA_OPS_FEATURE2D_FAST_IMPL_HPP__
#define AURA_OPS_FEATURE2D_FAST_IMPL_HPP__

#include "aura/ops/feature2d/fast.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{
class FastImpl : public OpImpl
{
public:
    FastImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 threshold,
                           DT_BOOL nonmax_suppression, FastDetectorType type = FastDetectorType::FAST_9_16,
                           DT_U32 max_num_corners = 3000);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32           m_threshold;
    DT_U32           m_max_num_corners;
    DT_BOOL          m_nonmax_suppression;
    FastDetectorType m_detector_type;

    const Array *m_src;
    std::vector<KeyPoint> *m_key_points;
};

class FastNone : public FastImpl
{
public:
    FastNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points,
                   DT_S32 threshold, DT_BOOL nonmax_suppression,
                   FastDetectorType type = FastDetectorType::FAST_9_16,
                   DT_U32 max_num_corners = 3000) override;

    Status Run() override;

private:
    std::string m_profiling_string;
};

#if defined(AURA_ENABLE_NEON)
class FastNeon : public FastImpl
{
public:
    FastNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points,
                   DT_S32 threshold, DT_BOOL nonmax_suppression,
                   FastDetectorType type = FastDetectorType::FAST_9_16,
                   DT_U32 max_num_corners = 3000) override;

    Status Run() override;
};
#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class FastHvx : public FastImpl
{
public:
    FastHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<KeyPoint> &key_points,
                   DT_S32 threshold, DT_BOOL nonmax_suppression,
                   FastDetectorType type = FastDetectorType::FAST_9_16,
                   DT_U32 max_num_corners = 3000) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if  defined(AURA_BUILD_HEXAGON)
Status Fast9Hvx(Context *ctx, const Mat &src, std::vector<KeyPoint> *key_points, DT_S32 threshold,
                DT_BOOL nonmax_suppression, DT_U32 max_num_corners);
#  endif // AURA_BUILD_HEXAGON

using FastInParam = HexagonRpcParamType<Mat, DT_S32, DT_BOOL, FastDetectorType, DT_U32>;
using FastOutParam = HexagonRpcParamType<std::vector<KeyPoint>>;

#  define AURA_OPS_FEATURE2D_FAST_OP_NAME          "Fast"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_FETURE2D_FAST_IMPL_HPP__