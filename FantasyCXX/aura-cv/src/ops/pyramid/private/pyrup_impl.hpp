/** @brief     : pyrup_impl header for aura
*  @file       : pyrup_impl.hpp
*  @author     : zhangpengfei10@xiaomi.com
*  @version    : 1.0.0
*  @date       : Sep. 6, 2023
*  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
*/

#ifndef AURA_OPS_PYRAMID_PYRUP_IMPL_HPP__
#define AURA_OPS_PYRAMID_PYRUP_IMPL_HPP__

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
struct PyrUpTraits
{
    static_assert(std::is_same<Tp, MI_U8>::value || std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value,
                  "Tp must be one of MI_U8/MI_S16/MI_U16");

    // Tp = MI_U8 MI_U16 MI_S16
    using KernelType = typename Promote<Tp>::Type;
    // Tp = MI_U8 MI_U16 MI_S16
    static constexpr MI_U32 Q = std::is_same<Tp, MI_U8>::value ? 9 : 13;
};

class PyrUpImpl : public OpImpl
{
public:
    PyrUpImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                           BorderType border_type = BorderType::REFLECT_101);

    Status Initialize() override;

    Status DeInitialize() override;

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

private:
    Status PrepareKmat();

protected:
    MI_S32      m_ksize;
    MI_F32      m_sigma;
    BorderType  m_border_type;
    const Array *m_src;
    Array       *m_dst;
    Mat         m_kmat;
};

class PyrUpNone : public PyrUpImpl
{
public:
    PyrUpNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;
};

Status PyrUp5x5None(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type, const OpTarget &target);

#if defined(AURA_ENABLE_NEON)
class PyrUpNeon : public PyrUpImpl
{
public:
    PyrUpNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;
};

Status PyrUp5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType &border_type, const OpTarget &target);

#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class PyrUpCL : public PyrUpImpl
{
public:
    PyrUpCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem       m_cl_src;
    CLMem       m_cl_dst;
    CLMem       m_cl_kmat;
    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class PyrUpHvx : public PyrUpImpl
{
public:
    PyrUpHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status PyrUp5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                   BorderType border_type = BorderType::REFLECT_101);
#  endif // AURA_BUILD_HEXAGON

using PyrUpInParam = HexagonRpcParamType<Mat, Mat, MI_S32, MI_F32, BorderType>;
#  define AURA_OPS_PYRAMID_PYRUP_OP_NAME          "PyrUp"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_PYRAMID_PYRUP_IMPL_HPP__