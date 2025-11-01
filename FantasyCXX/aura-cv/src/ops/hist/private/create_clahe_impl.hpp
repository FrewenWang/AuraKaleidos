/** @brief      : create clahe impl for aura
 *  @file       : create_clahe_impl.hpp
 *  @author     : lidong11@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Mar. 20, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HIST_CREATE_CLAHE_IMPL_HPP__
#define AURA_OPS_HIST_CREATE_CLAHE_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"

namespace aura
{

class CreateClAHEImpl : public OpImpl
{
public:
    CreateClAHEImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_F64 clip_limit = 40.0, const Sizes &tile_grid_size = Sizes(8, 8));

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    MI_F64 m_clip_limit;
    Sizes  m_tile_grid_size;

    const Array *m_src;
    Array *m_dst;
};

class CreateClAHENone : public CreateClAHEImpl
{
public:
    CreateClAHENone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_F64 clip_limit = 40.0, const Sizes &tile_grid_size = Sizes(8, 8)) override;

    Status Run() override;
};

} // namespace aura

#endif // AURA_OPS_HIST_CREATE_CLAHE_IMPL_HPP__