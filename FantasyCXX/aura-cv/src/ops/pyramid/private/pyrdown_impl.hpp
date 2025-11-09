/** @brief     : pyrdown_impl header for aura
*  @file       : pyrdown_impl.hpp
*  @author     : zhangpengfei10@xiaomi.com
*  @version    : 1.0.0
*  @date       : Sep. 4, 2023
*  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
*/

#ifndef AURA_OPS_PYRAMID_PYRDOWN_IMPL_HPP__
#define AURA_OPS_PYRAMID_PYRDOWN_IMPL_HPP__

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

template <typename Tp>
struct PyrDownTraits
{
    static_assert(std::is_same<Tp, DT_U8>::value || std::is_same<Tp, DT_S16>::value || std::is_same<Tp, DT_U16>::value,
                  "Tp must be one of DT_U8/DT_S16/DT_U16");

    // Tp = DT_U8 DT_U16 DT_S16
    using KernelType = typename Promote<Tp>::Type;
    // Tp = DT_U8 DT_U16 DT_S16
    static constexpr DT_U32 Q = std::is_same<Tp, DT_U8>::value ? 8 : 14;
};

class PyrDownImpl : public OpImpl
{
public:
    PyrDownImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                           BorderType border_type = BorderType::REFLECT_101);

    Status Initialize() override;

    Status DeInitialize() override;

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

private:
    Status PrepareKmat();

protected:
    DT_S32      m_ksize;
    DT_F32      m_sigma;
    BorderType  m_border_type;
    const Array *m_src;
    Array       *m_dst;
    Mat         m_kmat;
};

class PyrDownNone : public PyrDownImpl
{
public:
    PyrDownNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;
};

Status PyrDown5x5None(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type, const OpTarget &target);

#if defined(AURA_ENABLE_NEON)
class PyrDownNeon : public PyrDownImpl
{
public:
    PyrDownNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;
};

Status PyrDown5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType &border_type, const OpTarget &target);

#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class PyrDownCL : public PyrDownImpl
{
public:
    PyrDownCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);
private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem       m_cl_src;
    CLMem       m_cl_dst;
    CLMem       m_cl_kmat;
    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class PyrDownHvx : public PyrDownImpl
{
public:
    PyrDownHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status PyrDown5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                     BorderType border_type = BorderType::REFLECT_101);
#  endif // AURA_BUILD_HEXAGON

using PyrDownInParam = HexagonRpcParamType<Mat, Mat, DT_S32, DT_F32, BorderType>;
#  define AURA_OPS_PYRAMID_PYRDOWN_OP_NAME          "PyrDown"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_PYRAMID_PYRDOWN_IMPL_HPP__