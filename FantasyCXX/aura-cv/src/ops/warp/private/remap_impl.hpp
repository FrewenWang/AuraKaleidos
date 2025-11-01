/** @brief      : remap impl for aura
 *  @file       : remap_impl.hpp
 *  @author     : jianwen@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Oct. 23, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_WARP_REMAP_IMPL_HPP__
#define AURA_OPS_WARP_REMAP_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class RemapImpl : public OpImpl
{
public:
    RemapImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type = InterpType::LINEAR,
                           BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    const Array *m_map;
    Array       *m_dst;
    InterpType   m_interp_type;
    BorderType   m_border_type;
    Scalar       m_border_value;
};

class RemapNone : public RemapImpl
{
public:
    RemapNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_OPENCL)
class RemapCL : public RemapImpl
{
public:
    RemapCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type = InterpType::LINEAR,
                   BorderType border_type = BorderType::REPLICATE, const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType map_elem_type, ElemType dst_elem_type, MI_S32 channel,
                                              BorderType border_type, InterpType interp_type);
private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem    m_cl_src;
    CLMem    m_cl_map;
    CLMem    m_cl_dst;
    MI_S32   m_elem_counts;
    MI_S32   m_elem_height;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

} // namespace aura

#endif // AURA_OPS_WARP_REMAP_IMPL_HPP__
