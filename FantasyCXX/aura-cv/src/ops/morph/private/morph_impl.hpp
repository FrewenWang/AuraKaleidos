/** @brief     : morph_impl header for aura
*  @file       : morph_impl.hpp
*  @author     : wuzhiwei3@xiaomi.com
*  @version    : 1.0.0
*  @date       : Sep. 12, 2023
*  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
*/

#ifndef AURA_OPS_MORPH_MORPH_IMPL_HPP__
#define AURA_OPS_MORPH_MORPH_IMPL_HPP__

#include "aura/ops/morph/morph.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

namespace aura
{

class MorphImpl : public OpImpl
{
public:
    MorphImpl(Context *ctx, MorphType type, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    DT_S32     m_ksize;
    MorphType  m_type;
    MorphShape m_shape;
    DT_S32     m_iterations;

    const Array *m_src;
    Array       *m_dst;
};

class MorphNone : public MorphImpl
{
public:
    MorphNone(Context *ctx, MorphType type, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

private:
    Mat m_kmat;
    Mat m_src_border;
};

#if defined(AURA_ENABLE_NEON)
class MorphNeon : public MorphImpl
{
public:
    MorphNeon(Context *ctx, MorphType type, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1) override;

    Status Run() override;
};

template <typename Tp, MorphType MORPH_TYPE, typename VqTpye = typename neon::QVector<Tp>::VType>
AURA_ALWAYS_INLINE VqTpye MorphNeonMinMax(const VqTpye &vq_src_x0, const VqTpye &vq_src_x1, const VqTpye &vq_src_x2)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        VqTpye vq_result = neon::vmin(vq_src_x0, vq_src_x1);
        return neon::vmin(vq_result, vq_src_x2);
    }
    else
    {
        VqTpye vq_result = neon::vmax(vq_src_x0, vq_src_x1);
        return neon::vmax(vq_result, vq_src_x2);
    }
};

template <typename Tp, MorphType MORPH_TYPE, typename VqTpye = typename neon::QVector<Tp>::VType>
AURA_ALWAYS_INLINE VqTpye MorphNeonMinMax(const VqTpye &vq_src_x0, const VqTpye &vq_src_x1, const VqTpye &vq_src_x2,
                                          const VqTpye &vq_src_x3, const VqTpye &vq_src_x4)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        VqTpye vq_result_01 = neon::vmin(vq_src_x0, vq_src_x1);
        VqTpye vq_result_23 = neon::vmin(vq_src_x2, vq_src_x3);
        return neon::vmin(neon::vmin(vq_result_01, vq_result_23), vq_src_x4);
    }
    else
    {
        VqTpye vq_result_01 = neon::vmax(vq_src_x0, vq_src_x1);
        VqTpye vq_result_23 = neon::vmax(vq_src_x2, vq_src_x3);
        return neon::vmax(neon::vmax(vq_result_01, vq_result_23), vq_src_x4);
    }
};

template <typename Tp, MorphType MORPH_TYPE, typename VqTpye = typename neon::QVector<Tp>::VType>
AURA_ALWAYS_INLINE VqTpye MorphNeonMinMax(const VqTpye &vq_src_x0, const VqTpye &vq_src_x1, const VqTpye &vq_src_x2,
                                          const VqTpye &vq_src_x3, const VqTpye &vq_src_x4, const VqTpye &vq_src_x5,
                                          const VqTpye &vq_src_x6)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        VqTpye vq_result_01 = neon::vmin(vq_src_x0, vq_src_x1);
        VqTpye vq_result_23 = neon::vmin(vq_src_x2, vq_src_x3);
        VqTpye vq_result_45 = neon::vmin(vq_src_x4, vq_src_x5);
        vq_result_01        = neon::vmin(vq_result_01, vq_src_x6);
        vq_result_23        = neon::vmin(vq_result_23, vq_result_45);
        return neon::vmin(vq_result_01, vq_result_23);
    }
    else
    {
        VqTpye vq_result_01 = neon::vmax(vq_src_x0, vq_src_x1);
        VqTpye vq_result_23 = neon::vmax(vq_src_x2, vq_src_x3);
        VqTpye vq_result_45 = neon::vmax(vq_src_x4, vq_src_x5);
        vq_result_01        = neon::vmax(vq_result_01, vq_src_x6);
        vq_result_23        = neon::vmax(vq_result_23, vq_result_45);
        return neon::vmax(vq_result_01, vq_result_23);
    }
};

Status Morph3x3Neon(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target);
Status Morph5x5Neon(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target);
Status Morph7x7Neon(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target);
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class MorphCL : public MorphImpl
{
public:
    MorphCL(Context *ctx, MorphType type, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, MorphShape shape, MorphType type);

private:
    Status MorphCLImpl(CLMem &cl_src, CLMem &cl_dst);

private:
    std::vector<CLKernel> m_cl_kernels;
    Mat   m_tmp;
    CLMem m_cl_src;
    CLMem m_cl_dst;
    CLMem m_cl_tmp;
    std::string m_profiling_string;
};
#endif // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class MorphHvx : public MorphImpl
{
public:
    MorphHvx(Context *ctx, MorphType type, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape = MorphShape::RECT, DT_S32 iterations = 1) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

private:
    Mat m_iter_mat;
    const std::string m_buffer_name;
    Buffer m_iter_buffer;
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu8_result = Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1);
    }
    else
    {
        vu8_result = Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1);
    }
}

// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2, HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu8_result = Q6_Vub_vmin_VubVub(Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1), vu8_src_x2);
    }
    else
    {
        vu8_result = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1), vu8_src_x2);
    }
}

// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1,
                                          HVX_Vector &vu8_src_x2, HVX_Vector &vu8_src_x3, HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu8_result = Q6_Vub_vmin_VubVub(Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1), Q6_Vub_vmin_VubVub(vu8_src_x2, vu8_src_x3));
    }
    else
    {
        vu8_result = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1), Q6_Vub_vmax_VubVub(vu8_src_x2, vu8_src_x3));
    }
}

// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2,
                                          HVX_Vector &vu8_src_x3, HVX_Vector &vu8_src_x4, HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmin_VubVub(vu8_src_x2, vu8_src_x3);
        vu8_result = Q6_Vub_vmin_VubVub(Q6_Vub_vmin_VubVub(vu8_result_x0, vu8_result_x1), vu8_src_x4);
    }
    else
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmax_VubVub(vu8_src_x2, vu8_src_x3);
        vu8_result = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(vu8_result_x0, vu8_result_x1), vu8_src_x4);
    }
}

// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2,
                                          HVX_Vector &vu8_src_x3, HVX_Vector &vu8_src_x4, HVX_Vector &vu8_src_x5,
                                          HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmin_VubVub(vu8_src_x2, vu8_src_x3);
        HVX_Vector vu8_result_x2 = Q6_Vub_vmin_VubVub(vu8_src_x4, vu8_src_x5);
        vu8_result = Q6_Vub_vmin_VubVub(Q6_Vub_vmin_VubVub(vu8_result_x0, vu8_result_x1), vu8_result_x2);
    }
    else
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmax_VubVub(vu8_src_x2, vu8_src_x3);
        HVX_Vector vu8_result_x2 = Q6_Vub_vmax_VubVub(vu8_src_x4, vu8_src_x5);
        vu8_result = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(vu8_result_x0, vu8_result_x1), vu8_result_x2);
    }
}

// using Tp = DT_U8
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2, HVX_Vector &vu8_src_x3,
                                          HVX_Vector &vu8_src_x4, HVX_Vector &vu8_src_x5, HVX_Vector &vu8_src_x6, HVX_Vector &vu8_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmin_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmin_VubVub(vu8_src_x2, vu8_src_x3);
        HVX_Vector vu8_result_x2 = Q6_Vub_vmin_VubVub(vu8_src_x4, vu8_src_x5);
        vu8_result = Q6_Vub_vmin_VubVub(Q6_Vub_vmin_VubVub(vu8_result_x0, vu8_result_x1), Q6_Vub_vmin_VubVub(vu8_result_x2, vu8_src_x6));
    }
    else
    {
        HVX_Vector vu8_result_x0 = Q6_Vub_vmax_VubVub(vu8_src_x0, vu8_src_x1);
        HVX_Vector vu8_result_x1 = Q6_Vub_vmax_VubVub(vu8_src_x2, vu8_src_x3);
        HVX_Vector vu8_result_x2 = Q6_Vub_vmax_VubVub(vu8_src_x4, vu8_src_x5);
        vu8_result = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(vu8_result_x0, vu8_result_x1), Q6_Vub_vmax_VubVub(vu8_result_x2, vu8_src_x6));
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu16_result = Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1);
    }
    else
    {
        vu16_result = Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1);
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_src_x2, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu16_result = Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1), vu16_src_x2);
    }
    else
    {
        vu16_result = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1), vu16_src_x2);
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1,
                                          HVX_Vector &vu16_src_x2, HVX_Vector &vu16_src_x3, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vu16_result = Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1), Q6_Vuh_vmin_VuhVuh(vu16_src_x2, vu16_src_x3));
    }
    else
    {
        vu16_result = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1), Q6_Vuh_vmax_VuhVuh(vu16_src_x2, vu16_src_x3));
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_src_x2, HVX_Vector &vu16_src_x3,
                                          HVX_Vector &vu16_src_x4, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmin_VuhVuh(vu16_src_x2, vu16_src_x3);
        vu16_result = Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmin_VuhVuh(vu16_result_x0, vu16_result_x1), vu16_src_x4);
    }
    else
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmax_VuhVuh(vu16_src_x2, vu16_src_x3);
        vu16_result = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(vu16_result_x0, vu16_result_x1), vu16_src_x4);
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_src_x2,
                                          HVX_Vector &vu16_src_x3, HVX_Vector &vu16_src_x4, HVX_Vector &vu16_src_x5, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmin_VuhVuh(vu16_src_x2, vu16_src_x3);
        HVX_Vector vu16_result_x2 = Q6_Vuh_vmin_VuhVuh(vu16_src_x4, vu16_src_x5);
        vu16_result = Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmin_VuhVuh(vu16_result_x0, vu16_result_x1), vu16_result_x2);
    }
    else
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmax_VuhVuh(vu16_src_x2, vu16_src_x3);
        HVX_Vector vu16_result_x2 = Q6_Vuh_vmax_VuhVuh(vu16_src_x4, vu16_src_x5);
        vu16_result = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(vu16_result_x0, vu16_result_x1), vu16_result_x2);
    }
}

// using Tp = DT_U16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_src_x2, HVX_Vector &vu16_src_x3,
                                          HVX_Vector &vu16_src_x4, HVX_Vector &vu16_src_x5, HVX_Vector &vu16_src_x6, HVX_Vector &vu16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmin_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmin_VuhVuh(vu16_src_x2, vu16_src_x3);
        HVX_Vector vu16_result_x2 = Q6_Vuh_vmin_VuhVuh(vu16_src_x4, vu16_src_x5);
        vu16_result = Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmin_VuhVuh(vu16_result_x0, vu16_result_x1), Q6_Vuh_vmin_VuhVuh(vu16_result_x2, vu16_src_x6));
    }
    else
    {
        HVX_Vector vu16_result_x0 = Q6_Vuh_vmax_VuhVuh(vu16_src_x0, vu16_src_x1);
        HVX_Vector vu16_result_x1 = Q6_Vuh_vmax_VuhVuh(vu16_src_x2, vu16_src_x3);
        HVX_Vector vu16_result_x2 = Q6_Vuh_vmax_VuhVuh(vu16_src_x4, vu16_src_x5);
        vu16_result = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(vu16_result_x0, vu16_result_x1), Q6_Vuh_vmax_VuhVuh(vu16_result_x2, vu16_src_x6));
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1, HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vs16_result = Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1);
    }
    else
    {
        vs16_result = Q6_Vuh_vmax_VuhVuh(vs16_src_x0, vs16_src_x1);
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1,
                                          HVX_Vector &vs16_src_x2, HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vs16_result = Q6_Vh_vmin_VhVh(Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1), vs16_src_x2);
    }
    else
    {
        vs16_result = Q6_Vh_vmax_VhVh(Q6_Vh_vmax_VhVh(vs16_src_x0, vs16_src_x1), vs16_src_x2);
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1,
                                          HVX_Vector &vs16_src_x2, HVX_Vector &vs16_src_x3,
                                          HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        vs16_result = Q6_Vh_vmin_VhVh(Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1), Q6_Vh_vmin_VhVh(vs16_src_x2, vs16_src_x3));
    }
    else
    {
        vs16_result = Q6_Vh_vmax_VhVh(Q6_Vh_vmax_VhVh(vs16_src_x0, vs16_src_x1), Q6_Vh_vmax_VhVh(vs16_src_x2, vs16_src_x3));
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1, HVX_Vector &vs16_src_x2,
                                          HVX_Vector &vs16_src_x3, HVX_Vector &vs16_src_x4, HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmin_VhVh(vs16_src_x2, vs16_src_x3);
        vs16_result = Q6_Vh_vmin_VhVh(Q6_Vh_vmin_VhVh(vs16_result_x0, vs16_result_x1), vs16_src_x4);
    }
    else
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmax_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmax_VhVh(vs16_src_x2, vs16_src_x3);
        vs16_result = Q6_Vh_vmax_VhVh(Q6_Vh_vmax_VhVh(vs16_result_x0, vs16_result_x1), vs16_src_x4);
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1, HVX_Vector &vs16_src_x2,
                                          HVX_Vector &vs16_src_x3, HVX_Vector &vs16_src_x4, HVX_Vector &vs16_src_x5,
                                          HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmin_VhVh(vs16_src_x2, vs16_src_x3);
        HVX_Vector vs16_result_x2 = Q6_Vh_vmin_VhVh(vs16_src_x4, vs16_src_x5);
        vs16_result = Q6_Vh_vmin_VhVh(Q6_Vh_vmin_VhVh(vs16_result_x0, vs16_result_x1), vs16_result_x2);
    }
    else
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmax_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmax_VhVh(vs16_src_x2, vs16_src_x3);
        HVX_Vector vs16_result_x3 = Q6_Vh_vmax_VhVh(vs16_src_x4, vs16_src_x5);
        vs16_result = Q6_Vh_vmax_VhVh(Q6_Vh_vmax_VhVh(vs16_result_x0, vs16_result_x1), vs16_result_x3);
    }
}

// using Tp = DT_S16
template <typename Tp, MorphType MORPH_TYPE, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID MorphHvxMinMax(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1, HVX_Vector &vs16_src_x2,
                                          HVX_Vector &vs16_src_x3, HVX_Vector &vs16_src_x4, HVX_Vector &vs16_src_x5,
                                          HVX_Vector &vs16_src_x6, HVX_Vector &vs16_result)
{
    if (MorphType::ERODE == MORPH_TYPE)
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmin_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmin_VhVh(vs16_src_x2, vs16_src_x3);
        HVX_Vector vs16_result_x2 = Q6_Vh_vmin_VhVh(vs16_src_x4, vs16_src_x5);
        vs16_result = Q6_Vh_vmin_VhVh(Q6_Vh_vmin_VhVh(vs16_result_x0, vs16_result_x1), Q6_Vh_vmin_VhVh(vs16_result_x2, vs16_src_x6));
    }
    else
    {
        HVX_Vector vs16_result_x0 = Q6_Vh_vmax_VhVh(vs16_src_x0, vs16_src_x1);
        HVX_Vector vs16_result_x1 = Q6_Vh_vmax_VhVh(vs16_src_x2, vs16_src_x3);
        HVX_Vector vs16_result_x3 = Q6_Vh_vmax_VhVh(vs16_src_x4, vs16_src_x5);
        vs16_result = Q6_Vh_vmax_VhVh(Q6_Vh_vmax_VhVh(vs16_result_x0, vs16_result_x1), Q6_Vh_vmax_VhVh(vs16_result_x3, vs16_src_x6));
    }
}

Status Morph3x3Hvx(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape);
Status Morph5x5Hvx(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape);
Status Morph7x7Hvx(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape);
#  endif // AURA_BUILD_HEXAGON

using MorphInParam = HexagonRpcParamType<Mat, Mat, Buffer, MorphType, DT_S32, MorphShape, DT_S32>;
#  define AURA_OPS_MORPH_MORPH_OP_NAME          "Morph"

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_MORPH_MORPH_IMPL_HPP__