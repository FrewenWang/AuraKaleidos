#ifndef AURA_OPS_FEATURE2D_CANNY_IMPL_HPP__
#define AURA_OPS_FEATURE2D_CANNY_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

#include <deque>

namespace aura
{

AURA_INLINE DT_VOID CannyPush(DT_U8 *map, std::deque<DT_U8*> &stack)
{
    *(map) = 2;
    stack.emplace_back(map);
}

AURA_INLINE DT_VOID CannyCheck(DT_S32 m, DT_S32 high, DT_U8 *map, std::deque<DT_U8*> &stack)
{
    if (m > high)
    {
        CannyPush(map, stack);
    }
    else
    {
        *(map) = 0;
    }
}

class CannyImpl : public OpImpl
{
public:
    CannyImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_F64 low_thresh, DT_F64 high_thresh,
                           DT_S32 aperture_size = 3, DT_BOOL l2_gradient = DT_FALSE);

    virtual Status SetArgs(const Array *dx, const Array *dy, Array *dst, DT_F64 low_thresh,
                           DT_F64 high_thresh, DT_BOOL l2_gradient = DT_FALSE);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;


protected:
    DT_F64 m_low_thresh;
    DT_F64 m_high_thresh;
    DT_S32 m_aperture_size;
    DT_BOOL m_l2_gradient;
    DT_BOOL m_is_aperture;

    const Array *m_src;
    const Array *m_dx;
    const Array *m_dy;
    Array *m_dst;
};

class CannyNone : public CannyImpl
{
public:
    CannyNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F64 low_thresh, DT_F64 high_thresh,
                   DT_S32 aperture_size = 3, DT_BOOL l2_gradient = DT_FALSE) override;

    Status SetArgs(const Array *dx, const Array *dy, Array *dst, DT_F64 low_thresh,
                   DT_F64 high_thresh, DT_BOOL l2_gradient = DT_FALSE) override;

    Status Run() override;

private:
    std::string m_profiling_string;
};

#if defined(AURA_ENABLE_NEON)
class CannyNeon : public CannyImpl
{
public:
    CannyNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F64 low_thresh, DT_F64 high_thresh,
                   DT_S32 aperture_size = 3, DT_BOOL l2_gradient = DT_FALSE) override;

    Status SetArgs(const Array *dx, const Array *dy, Array *dst, DT_F64 low_thresh,
                   DT_F64 high_thresh, DT_BOOL l2_gradient = DT_FALSE) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_FEATURE2D_CANNY_IMPL_HPP__