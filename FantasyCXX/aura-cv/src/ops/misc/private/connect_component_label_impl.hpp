/** @brief      : connect_component_label impl for aura
 *  @file       : connect_component_label_impl.hpp
 *  @author     : wangshiyu7@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : July. 31, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_MISC_CCL_IMPL_HPP__
#define AURA_OPS_MISC_CCL_IMPL_HPP__

#include "aura/ops/misc/connect_component_label.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class ConnectComponentLabelImpl : public OpImpl
{
public:
    ConnectComponentLabelImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, CCLAlgo algo_type = CCLAlgo::SPAGHETTI,
                           ConnectivityType connectivity_type = ConnectivityType::CROSS,
                           EquivalenceSolver solver_type = EquivalenceSolver::UNION_FIND_PATH_COMPRESS);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;

    ConnectivityType m_connectivity_type;
    CCLAlgo m_algo_type;
};

class ConnectComponentLabelNone : public ConnectComponentLabelImpl
{
public:
    ConnectComponentLabelNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, CCLAlgo algo_type = CCLAlgo::SPAGHETTI,
                   ConnectivityType connectivity_type = ConnectivityType::CROSS,
                   EquivalenceSolver solver_type = EquivalenceSolver::UNION_FIND_PATH_COMPRESS) override;

    Status Run() override;

private:
    EquivalenceSolver m_solver_type;
};

#if defined(AURA_ENABLE_OPENCL)
class ConnectComponentLabelCL : public ConnectComponentLabelImpl
{
public:
    ConnectComponentLabelCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, CCLAlgo algo_type = CCLAlgo::HA_GPU,
                   ConnectivityType connectivity_type = ConnectivityType::CROSS,
                   EquivalenceSolver solver_type = EquivalenceSolver::UNION_FIND_PATH_COMPRESS) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType dst_elem_type, CCLAlgo algo_type, ConnectivityType connectivity_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL
}

#endif // AURA_OPS_MISC_CCL_IMPL_HPP__