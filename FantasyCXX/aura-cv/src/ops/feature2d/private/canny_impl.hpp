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

AURA_INLINE AURA_VOID CannyPush(MI_U8 *map, std::deque<MI_U8*> &stack)
{
    *(map) = 2;
    stack.emplace_back(map);
}

AURA_INLINE AURA_VOID CannyCheck(MI_S32 m, MI_S32 high, MI_U8 *map, std::deque<MI_U8*> &stack)
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

    virtual Status SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                           MI_S32 aperture_size = 3, MI_BOOL l2_gradient = MI_FALSE);

    virtual Status SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                           MI_F64 high_thresh, MI_BOOL l2_gradient = MI_FALSE);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;


protected:
    MI_F64 m_low_thresh;
    MI_F64 m_high_thresh;
    MI_S32 m_aperture_size;
    MI_BOOL m_l2_gradient;
    MI_BOOL m_is_aperture;

    const Array *m_src;
    const Array *m_dx;
    const Array *m_dy;
    Array *m_dst;
};

class CannyNone : public CannyImpl
{
public:
    CannyNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                   MI_S32 aperture_size = 3, MI_BOOL l2_gradient = MI_FALSE) override;

    Status SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                   MI_F64 high_thresh, MI_BOOL l2_gradient = MI_FALSE) override;

    Status Run() override;

private:
    std::string m_profiling_string;
};

#if defined(AURA_ENABLE_NEON)
class CannyNeon : public CannyImpl
{
public:
    CannyNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                   MI_S32 aperture_size = 3, MI_BOOL l2_gradient = MI_FALSE) override;

    Status SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                   MI_F64 high_thresh, MI_BOOL l2_gradient = MI_FALSE) override;

    Status Run() override;
};
#endif

} // namespace aura

#endif // AURA_OPS_FEATURE2D_CANNY_IMPL_HPP__