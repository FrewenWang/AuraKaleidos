#ifndef AURA_OPS_FILTER_GAUSSIAN_IMPL_HPP__
#define AURA_OPS_FILTER_GAUSSIAN_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif
#if defined(AURA_ENABLE_XTENSA)
#  include "aura/runtime/xtensa.h"
#endif

#define AURA_OPS_FILTER_GAUSSIAN_OP_NAME          "Gaussian"

namespace aura
{

AURA_INLINE Mat GetGaussianKmat(Context *ctx, const std::vector<DT_F32> &kernel)
{
    const DT_S32 ksize = kernel.size();

    Mat kmat(ctx, ElemType::F32, Sizes3(1, ksize, 1));
    DT_F32 *ker_row = kmat.Ptr<DT_F32>(0);

    for (DT_S32 i = 0; i < ksize; i++)
    {
        ker_row[i] = kernel[i];
    }

    return kmat;
}

/**
 * 获取高斯的矩阵数据
 * @tparam Tp
 * @tparam Q
 * @param ctx
 * @param kernel
 * @return
 */
template <typename Tp, DT_U32 Q>
AURA_INLINE Mat GetGaussianKmat(Context *ctx, const std::vector<DT_F32> &kernel)
{
    // 获取高斯核系数的数量
    const DT_S32 ksize = kernel.size();
    //
    Mat kmat(ctx, GetElemType<Tp>(), Sizes3(1, ksize, 1));
    Tp *ker_row = kmat.Ptr<Tp>(0);
    //
    DT_S32 sum = 0;
    DT_F32 err = 0.f;

    for (DT_S32 i = 0; i < ksize / 2; i++)
    {
        DT_F32 tmp             = kernel[i] * (1 << Q) + err;
        Tp result              = static_cast<Tp>(Round(tmp));
        err                    = tmp - (DT_F32)result;
        ker_row[i]             = result;
        ker_row[ksize - 1 - i] = result;
        sum += result;
    }

    ker_row[ksize / 2] = (1 << Q) - sum * 2;

    return kmat;
}

/**
 * TODO 重点：GaussianImpl没有重写来自
 */
class GaussianImpl : public OpImpl
{
public:
    GaussianImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                           BorderType border_type = BorderType::REFLECT_101,
                           const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    // 高斯核的大小。TODO 为什么要用DT_S32
    DT_S32     m_ksize;
    /// sugma的值
    DT_F32     m_sigma;
    /// 边界类型
    BorderType m_border_type;
    Scalar     m_border_value;

    const Array *m_src;
    Array       *m_dst;
    Mat         m_kmat;
};

class GaussianNone : public GaussianImpl
{
public:
    GaussianNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Status PrepareKmat();

private:
    Mat m_src_border;
};

#if defined(AURA_ENABLE_NEON)
class GaussianNeon : public GaussianImpl
{
public:
    GaussianNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Status PrepareKmat();
};

Status Gaussian3x3Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                       BorderType &border_type, const Scalar &border_value, const OpTarget &target);
Status Gaussian5x5Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                       BorderType &border_type, const Scalar &border_value, const OpTarget &target);
Status Gaussian7x7Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                       BorderType &border_type, const Scalar &border_value, const OpTarget &target);
#endif

#if defined(AURA_ENABLE_OPENCL)
/**
 * 高函数滤波函数的逻辑
 **/
class GaussianCL : public GaussianImpl
{
public:
    GaussianCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type);

private:
    Status PrepareKmat();

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_kmat;

    std::string m_profiling_string;
};
#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class GaussianHvx : public GaussianImpl
{
public:
    GaussianHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

#  if defined(AURA_BUILD_HEXAGON)
    Status Initialize() override;

    Status DeInitialize() override;
#  endif // AURA_BUILD_HEXAGON

    Status Run() override;

    std::string ToString() const override;

#  if defined(AURA_BUILD_HEXAGON)
private:
    Status PrepareKmat();
#  endif // AURA_BUILD_HEXAGON

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status Gaussian3x3Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value);
Status Gaussian5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value);
Status Gaussian7x7Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value);
Status Gaussian9x9Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value);
#  endif // AURA_BUILD_HEXAGON

//// 进行高斯滤波的输入参数的类型重定义
using GaussianInParamHvx = HexagonRpcParamType<Mat, Mat, DT_S32, DT_F32, BorderType, Scalar>;

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

#if defined(AURA_ENABLE_XTENSA)
class GaussianVdsp : public GaussianImpl
{
public:
    GaussianVdsp(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                   BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

private:
    XtensaMat m_xtensa_src;
    XtensaMat m_xtensa_dst;
};

using GaussianInParamVdsp = XtensaRpcParamType<XtensaMat, XtensaMat, DT_S32, DT_F32, BorderType, Scalar>;

#endif // defined(AURA_ENABLE_XTENSA)

} // namespace aura

#endif // AURA_OPS_FILTER_GAUSSIAN_IMPL_HPP__