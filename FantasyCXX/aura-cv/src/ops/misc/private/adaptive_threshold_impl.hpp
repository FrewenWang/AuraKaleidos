/** @brief      : adaptive threshold impl header for aura
 *  @file       : adaptive_threshold_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 18, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_ADAPTIVE_THRESHOLD_IMPL_HPP__
#define AURA_OPS_MISC_ADAPTIVE_THRESHOLD_IMPL_HPP__

#include "aura/ops/misc/adaptive_threshold.hpp"

namespace aura
{

class AdaptiveThresholdImpl : public OpImpl
{
public:
    AdaptiveThresholdImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                           DT_S32 type, DT_S32 block_size, DT_F32 delta);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    AdaptiveThresholdMethod m_method;

    DT_F32       m_max_val;
    DT_S32       m_type;
    DT_S32       m_block_size;
    DT_F32       m_delta;

    const Array *m_src;
    Array       *m_dst;
};

class AdaptiveThresholdNone : public AdaptiveThresholdImpl
{
public:
    AdaptiveThresholdNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                   DT_S32 type, DT_S32 block_size, DT_F32 delta) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_MISC_ADAPTIVE_THRESHOLD_IMPL_HPP__