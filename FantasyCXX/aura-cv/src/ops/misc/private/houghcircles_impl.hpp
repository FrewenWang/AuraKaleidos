/** @brief      : houghcircles impl header for aura
 *  @file       : houghcircles_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_HOUGH_CIRCLES_IMPL_HPP__
#define AURA_OPS_MISC_HOUGH_CIRCLES_IMPL_HPP__

#include "aura/ops/misc/houghcircles.hpp"

namespace aura
{

class HoughCirclesImpl : public OpImpl
{
public:
    HoughCirclesImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                           DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius);

    std::vector<const Array*> GetInputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    HoughCirclesMethod m_method;

    DT_F64 m_dp;
    DT_F64 m_min_dist;
    DT_F64 m_canny_thresh;
    DT_F64 m_acc_thresh;
    DT_S32 m_min_radius;
    DT_S32 m_max_radius;

    const Array *m_src;
    std::vector<Scalar> *m_circles;
};

class HoughCirclesNone : public HoughCirclesImpl
{
public:
    HoughCirclesNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                   DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_MISC_HOUGH_CIRCLES_IMPL_HPP__