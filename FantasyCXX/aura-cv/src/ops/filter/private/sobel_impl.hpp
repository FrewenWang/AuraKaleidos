#ifndef AURA_OPS_FILTER_SOBEL_IMPL_HPP__
#define AURA_OPS_FILTER_SOBEL_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif // AURA_ENABLE_OPENCL
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

namespace aura
{

class SobelImpl : public OpImpl
{
public:
    SobelImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                           BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar());

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    AURA_VOID Dump(const std::string &prefix) const override;

protected:
    MI_S32     m_dx;
    MI_S32     m_dy;
    MI_S32     m_ksize;
    MI_F32     m_scale;
    BorderType m_border_type;
    Scalar     m_border_value;

    const Array *m_src;
    Array       *m_dst;
};

class SobelNone : public SobelImpl
{
public:
    SobelNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                   BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_src_border;
    Mat m_kx;
    Mat m_ky;
};

#if defined(AURA_ENABLE_NEON)
class SobelNeon : public SobelImpl
{
public:
    SobelNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                   BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar()) override;

    Status Run() override;
};

// WITH_SCALE  = MI_FALSE
// MVTypeInter = int16x8x1_t, int16x8x2_t, int16x8x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE,
          typename std::enable_if<!WITH_SCALE, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqs16_result, MI_S16 *dst, MI_F32 scale)
{
    AURA_UNUSED(scale);

    neon::vstore(dst, mvqs16_result);
}

// WITH_SCALE  = MI_TRUE
// MVTypeInter = int16x8x1_t, int16x8x2_t, int16x8x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE, typename std::enable_if<WITH_SCALE, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqs16_result, MI_S16 *dst, MI_F32 scale)
{
    constexpr MI_S32 channel = static_cast<MI_S32>(sizeof(MVTypeInter) / sizeof(int16x8_t));

    using MVTypeF32 = typename neon::MQVector<MI_F32, channel>::MVType;
    MVTypeF32 mvqf32_result_l, mvqf32_result_h;
    int32x4_t vqs32_result_l, vqs32_result_h;

    for (MI_S32 c = 0; c < channel; c++)
    {
        vqs32_result_l         = neon::vmovl(neon::vgetlow(mvqs16_result.val[c]));
        mvqf32_result_l.val[c] = neon::vcvt<MI_F32>(vqs32_result_l);
        mvqf32_result_l.val[c] = neon::vmul(mvqf32_result_l.val[c], scale);
        vqs32_result_l         = neon::vcvt<MI_S32>(neon::vrndn(mvqf32_result_l.val[c]));

        vqs32_result_h         = neon::vmovl(neon::vgethigh(mvqs16_result.val[c]));
        mvqf32_result_h.val[c] = neon::vcvt<MI_F32>(vqs32_result_h);
        mvqf32_result_h.val[c] = neon::vmul(mvqf32_result_h.val[c], scale);
        vqs32_result_h         = neon::vcvt<MI_S32>(neon::vrndn(mvqf32_result_h.val[c]));

        mvqs16_result.val[c]   = neon::vcombine(neon::vqmovn(vqs32_result_l), neon::vqmovn(vqs32_result_h));
    }

    neon::vstore(dst, mvqs16_result);
}

// WITH_SCALE  = MI_FALSE
// MVTypeInter = int16x8x1_t, int16x8x2_t, int16x8x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE, typename SType = typename neon::Scalar<MVTypeInter>::SType,
          typename std::enable_if<(!WITH_SCALE) && std::is_same<SType, MI_S16>::value, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqs16_result, MI_F32 *dst, MI_F32 scale)
{
    AURA_UNUSED(scale);

    constexpr MI_S32 channel = static_cast<MI_S32>(sizeof(MVTypeInter) / sizeof(int16x8_t));
    constexpr MI_S32 voffset = channel << 2;

    using MVTypeF32 = typename neon::MQVector<MI_F32, channel>::MVType;
    MVTypeF32 mvqf32_result_l, mvqf32_result_h;
    int32x4_t vqs32_result_l, vqs32_result_h;

    for (MI_S32 c = 0; c < channel; c++)
    {
        vqs32_result_l         = neon::vmovl(neon::vgetlow(mvqs16_result.val[c]));
        mvqf32_result_l.val[c] = neon::vcvt<MI_F32>(vqs32_result_l);

        vqs32_result_h         = neon::vmovl(neon::vgethigh(mvqs16_result.val[c]));
        mvqf32_result_h.val[c] = neon::vcvt<MI_F32>(vqs32_result_h);
    }

    neon::vstore(dst, mvqf32_result_l);
    neon::vstore(dst + voffset, mvqf32_result_h);
}

// WITH_SCALE  = MI_TRUE
// MVTypeInter = int16x8x1_t, int16x8x2_t, int16x8x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE, typename SType = typename neon::Scalar<MVTypeInter>::SType,
          typename std::enable_if<WITH_SCALE && std::is_same<SType, MI_S16>::value, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqs16_result, MI_F32 *dst, MI_F32 scale)
{
    constexpr MI_S32 channel = static_cast<MI_S32>(sizeof(MVTypeInter) / sizeof(int16x8_t));
    constexpr MI_S32 voffset = channel << 2;

    using MVTypeF32 = typename neon::MQVector<MI_F32, channel>::MVType;
    MVTypeF32 mvqf32_result_l, mvqf32_result_h;
    int32x4_t vqs32_result_l, vqs32_result_h;

    for (MI_S32 c = 0; c < channel; c++)
    {
        vqs32_result_l         = neon::vmovl(neon::vgetlow(mvqs16_result.val[c]));
        mvqf32_result_l.val[c] = neon::vcvt<MI_F32>(vqs32_result_l);
        mvqf32_result_l.val[c] = neon::vmul(mvqf32_result_l.val[c], scale);

        vqs32_result_h         = neon::vmovl(neon::vgethigh(mvqs16_result.val[c]));
        mvqf32_result_h.val[c] = neon::vcvt<MI_F32>(vqs32_result_h);
        mvqf32_result_h.val[c] = neon::vmul(mvqf32_result_h.val[c], scale);
    }

    neon::vstore(dst, mvqf32_result_l);
    neon::vstore(dst + voffset, mvqf32_result_h);
}

// WITH_SCALE  = MI_FALSE
// MVTypeInter = float32x4x1_t, float32x4x2_t, float32x4x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE, typename SType = typename neon::Scalar<MVTypeInter>::SType,
          typename std::enable_if<(!WITH_SCALE) && std::is_same<SType, MI_F32>::value, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqf32_result, MI_F32 *dst, MI_F32 scale)
{
    AURA_UNUSED(scale);

    neon::vstore(dst, mvqf32_result);
}

// WITH_SCALE  = MI_TRUE
// MVTypeInter = float32x4x1_t, float32x4x2_t, float32x4x3_t
template <typename MVTypeInter, MI_BOOL WITH_SCALE, typename SType = typename neon::Scalar<MVTypeInter>::SType,
          typename std::enable_if<WITH_SCALE && std::is_same<SType, MI_F32>::value, AURA_VOID>::type* = MI_NULL>
AURA_INLINE AURA_VOID SobelPostProcess(MVTypeInter &mvqf32_result, MI_F32 *dst, MI_F32 scale)
{
    constexpr MI_S32 channel = static_cast<MI_S32>(sizeof(MVTypeInter) / sizeof(float32x4_t));

    for (MI_S32 c = 0; c < channel; c++)
    {
        mvqf32_result.val[c] = neon::vmul(mvqf32_result.val[c], scale);
    }

    neon::vstore(dst, mvqf32_result);
}

Status Sobel1x1Neon(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                    BorderType border_type, const Scalar &border_value, const OpTarget &target);
Status Sobel3x3Neon(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                    BorderType border_type, const Scalar &border_value, const OpTarget &target);
Status Sobel5x5Neon(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                    BorderType border_type, const Scalar &border_value, const OpTarget &target);
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class SobelCL : public SobelImpl
{
public:
    SobelCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize,
                   MI_F32 scale, BorderType border_type = BorderType::REFLECT_101,
                   const Scalar &border_value = Scalar()) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale, BorderType border_type,
                                              MI_S32 channel, ElemType src_elem_type, ElemType dst_elem_type);

private:
    std::vector<CLKernel> m_cl_kernels;
    CLMem m_cl_src;
    CLMem m_cl_dst;

    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class SobelHvx : public SobelImpl
{
public:
    SobelHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale = 1.f,
                   BorderType border_type = BorderType::REFLECT_101, const Scalar &border_value = Scalar()) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
AURA_ALWAYS_INLINE AURA_VOID SobelPostProcess(HVX_Vector &vs16_result_l, HVX_Vector &vs16_result_h, MI_F32 scale)
{
    HVX_Vector vf32_scale = vsplat<MI_F32>(scale);

    HVX_VectorPair ws32_result_l = Q6_Ww_vunpack_Vh(vs16_result_l);
    HVX_VectorPair ws32_result_h = Q6_Ww_vunpack_Vh(vs16_result_h);

    HVX_Vector vf32_result_l_l = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(ws32_result_l));
    HVX_Vector vf32_result_l_h = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(ws32_result_l));
    HVX_Vector vf32_result_h_l = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(ws32_result_h));
    HVX_Vector vf32_result_h_h = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(ws32_result_h));

    vf32_result_l_l = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vf32_result_l_l, vf32_scale));
    vf32_result_l_h = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vf32_result_l_h, vf32_scale));
    vf32_result_h_l = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vf32_result_h_l, vf32_scale));
    vf32_result_h_h = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vf32_result_h_h, vf32_scale));

    HVX_Vector vs32_result_l_l = Q6_Vw_vcvt_Vsf(vf32_result_l_l);
    HVX_Vector vs32_result_l_h = Q6_Vw_vcvt_Vsf(vf32_result_l_h);
    HVX_Vector vs32_result_h_l = Q6_Vw_vcvt_Vsf(vf32_result_h_l);
    HVX_Vector vs32_result_h_h = Q6_Vw_vcvt_Vsf(vf32_result_h_h);

    vs16_result_l = Q6_Vh_vpack_VwVw_sat(vs32_result_l_h, vs32_result_l_l);
    vs16_result_h = Q6_Vh_vpack_VwVw_sat(vs32_result_h_h, vs32_result_h_l);
}

Status Sobel1x1Hvx(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                   BorderType border_type, const Scalar &border_value);
Status Sobel3x3Hvx(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                   BorderType border_type, const Scalar &border_value);
Status Sobel5x5Hvx(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                   BorderType border_type, const Scalar &border_value);
#  endif // AURA_BUILD_HEXAGON

using SobelInParam = HexagonRpcParamType<Mat, Mat, MI_S32, MI_S32, MI_S32, MI_F32, BorderType, Scalar>;
#  define AURA_OPS_FILTER_SOBEL_OP_NAME          "Sobel"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_FILTER_SOBEL_IMPL_HPP__