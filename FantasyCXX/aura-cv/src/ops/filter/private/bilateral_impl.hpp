#ifndef AURA_OPS_FILTER_BILATERAL_IMPL_HPP__
#define AURA_OPS_FILTER_BILATERAL_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

namespace aura
{

class BilateralImpl : public OpImpl
{
public:
    BilateralImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_F32 sigma_color,
                           DT_F32 sigma_space, DT_S32 ksize,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    Status Initialize() override;

    Status DeInitialize() override;

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    Status PrepareSpaceMat();

    Status PrepareColorMat();

private:
    virtual Sizes GetColorMatStride(Sizes3 color_size);

protected:
    // 高斯核的大小
    DT_S32     m_ksize;
    BorderType m_border_type;
    Scalar     m_border_value;
    DT_F32     m_sigma_color;
    DT_F32     m_sigma_space;
    DT_S32     m_valid_num;
    DT_F32     m_scale_index;

    const Array *m_src;
    Array       *m_dst;
    Mat         m_space_ofs;
    Mat         m_space_weight;
    Mat         m_color_weight;
    Sizes       m_color_stride;
};

class BilateralNone : public BilateralImpl
{
public:
    BilateralNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 sigma_color,
                   DT_F32 sigma_space, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};


#if defined(AURA_ENABLE_NEON)
class BilateralNeon : public BilateralImpl
{
public:
    BilateralNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 sigma_color,
                   DT_F32 sigma_space, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

Status Bilateral3x3Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &space_weight, const Mat &color_weight,
                        DT_S32 valid_num, BorderType border_type, const Scalar &border_value, const OpTarget &target);
#endif// AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class BilateralCL : public BilateralImpl
{
public:
    BilateralCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_F32 sigma_color,
                   DT_F32 sigma_space, DT_S32 ksize,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type, DT_S32 valid_num);

private:
    Sizes GetColorMatStride(Sizes3 color_size) override;

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_space;
    CLMem m_cl_color;

    std::string m_profiling_string;
};
#endif// AURA_ENABLE_OPENCL

} // namespace aura

#endif // AURA_OPS_FILTER_BILATERAL_IMPL_HPP__