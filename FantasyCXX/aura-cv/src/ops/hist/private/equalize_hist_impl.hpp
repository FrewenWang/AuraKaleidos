/** @brief      : equalize hist impl for aura
 *  @file       : equalize_hist_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Mar. 20, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HIST_EQUALIZE_HIST_IMPL_HPP__
#define AURA_OPS_HIST_EQUALIZE_HIST_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class EqualizeHistImpl : public OpImpl
{
public:
    EqualizeHistImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array *m_dst;
};

class EqualizeHistNone : public EqualizeHistImpl
{
public:
    EqualizeHistNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_HIST_EQUALIZE_HIST_IMPL_HPP__