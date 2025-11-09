#ifndef AURA_OPS_CVTCOLOR_CVTCOLOR_IMPL_HPP__
#define AURA_OPS_CVTCOLOR_CVTCOLOR_IMPL_HPP__

#include "aura/ops/cvtcolor/cvtcolor.hpp"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif
#if defined(AURA_ENABLE_XTENSA)
#  include "aura/runtime/xtensa.h"
#endif

#define AURA_OPS_CVTCOLOR_OP_NAME          "CvtColor"

#define CVTCOLOR_COEF_BITS      (20)
#define CVTCOLOR_DESCALE(x,n)   (((x) + (1 << ((n)-1))) >> (n))

namespace aura
{

AURA_INLINE DT_BOOL SwapBlue(CvtColorType type)
{
    DT_BOOL is_b = DT_TRUE;
    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        {
            is_b = DT_FALSE;
            break;
        }

        default:
        {
            is_b = DT_TRUE;
        }
    }

    return is_b;
}

AURA_INLINE DT_BOOL SwapGreen(CvtColorType type)
{
    DT_BOOL is_g = DT_TRUE;
    switch (type)
    {
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            is_g = DT_FALSE;
            break;
        }

        default:
        {
            is_g = DT_TRUE;
        }
    }

    return is_g;
}

AURA_INLINE DT_BOOL SwapUv(CvtColorType type)
{
    DT_BOOL is_uv = DT_TRUE;
    switch (type)
    {
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        {
            is_uv = DT_FALSE;
            break;
        }

        default:
        {
            is_uv = DT_TRUE;
        }
    }

    return is_uv;
}

/**
 * @brief the formula of BGR -> GRAY
 * Gray = R * 0.299 + G * 0.587 + B * 0.114
 */
struct Bgr2GrayParam
{
    static constexpr DT_S32 BC = 3735;  // Round(0.114f  * (1 << 15));
    static constexpr DT_S32 GC = 19235; // Round(0.587f  * (1 << 15));
    static constexpr DT_S32 RC = 9798;  // Round(0.299f  * (1 << 15));
};

template <DT_U32 MODE> struct Yuv2RgbParamTraits;

/**
 * @brief the formula of YUV -> RGB
 * R = 1.164(Y - 16) + 1.596(V - 128)
 * G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
 * B = 1.164(Y - 16)                  + 2.018(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * (Y - 16)                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 */
template <>
struct Yuv2RgbParamTraits<0>
{
    static constexpr DT_S32 Y2RGB = 1220542; // Round(1.164f  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 V2R   = 1673527; // Round(1.596f  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 V2G   = -852492; // Round(-0.813f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 U2G   = -409993; // Round(-0.391f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 U2B   = 2116026; // Round(2.018f  * (1 << CVTCOLOR_COEF_BITS));
};

/**
 * @brief the formula of BT.601 YUV -> RGB
 * R = Y + 1.403(V - 128)
 * G = Y - 0.714(V - 128) - 0.343(U - 128)
 * B = Y                  + 1.770(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * Y + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * Y + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * Y                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 */
template <>
struct Yuv2RgbParamTraits<1>
{
    static constexpr DT_S32 Y2RGB = 1048576; // Round(1.000f  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 V2R   = 1471152; // Round(1.403f  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 V2G   = -748683; // Round(-0.714f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 U2G   = -359661; // Round(-0.343f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 U2B   = 1855979; // Round(1.770f  * (1 << CVTCOLOR_COEF_BITS));
};

template <DT_U32 MODE> struct Rgb2YuvParamTraits;

/**
 * @brief the formula of RGB -> YUV
 *  Y = 16  + 0.257 * r + 0.504 * g + 0.098 * b
 * Cb = 128 - 0.148 * r - 0.291 * g + 0.439 * b
 * Cr = 128 + 0.439 * r - 0.368 * g - 0.071 * b
 */
template <>
struct Rgb2YuvParamTraits<0>
{
    static constexpr DT_S32 R2Y =  269484;    // Round( 0.257f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2Y =  528482;    // Round( 0.504f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2Y =  102760;    // Round( 0.098f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 R2U = -155188;    // Round(-0.148f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 G2U = -305135;    // Round(-0.291f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2U =  460324;    // Round( 0.439f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2V = -385875;    // Round(-0.368f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2V = -74448;     // Round(-0.071f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 YC  =  16777216;  // Round( 16     * (1 << CVTCOLOR_COEF_BITS));
};

/**
 * @brief the formula of BT601 RGB -> YUV
 *  Y =  0.299  * r + 0.587  * g + 0.114  * b
 * Cb = -0.1687 * r - 0.3313 * g + 0.5    * b + 128
 * Cr =  0.5    * r - 0.4187 * g - 0.0813 * b + 128
 * 
 * @brief the formula of P010 RGB -> YUV
 *  Y = ( 0.299  * r + 0.587  * g + 0.114  * b)       * (1 << 6)
 * Cb = (-0.1687 * r - 0.3313 * g + 0.5    * b + 512) * (1 << 6)
 * Cr = ( 0.5    * r - 0.4187 * g - 0.0813 * b + 512) * (1 << 6)
 */
template <>
struct Rgb2YuvParamTraits<1>
{
    static constexpr DT_S32 R2Y =  313524;  // Round( 0.299  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2Y =  615514;  // Round( 0.587  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2Y =  119538;  // Round( 0.114  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 R2U = -176895;  // Round(-0.1687 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2U = -347393;  // Round(-0.3313 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2U =  524288;  // Round( 0.5    * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2V = -439039;  // Round(-0.4187 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2V = -85249;   // Round(-0.0813 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 YC  =  0;
};

class CvtColorImpl : public OpImpl
{
public:
    CvtColorImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    CvtColorType m_type;

    std::vector<const Array*> m_src;
    std::vector<Array*> m_dst;
};

class CvtColorNone : public CvtColorImpl
{
public:
    CvtColorNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type) override;

    Status Run() override;
};

// RGB <-> BGRA
Status CvtBgr2BgraNone(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, const OpTarget &target);
Status CvtBgr2GrayNone(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, const OpTarget &target);
Status CvtGray2BgrNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

// YUV -> RGB
Status CvtNv2RgbNone(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtY4202RgbNone(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtY4222RgbNone(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapuv, DT_BOOL swapy, CvtColorType type, const OpTarget &target);
Status CvtY4442RgbNone(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type, const OpTarget &target);

// RGB -> YUV
Status CvtRgb2NvNone(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtRgb2Y420None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtRgb2Y444None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type, const OpTarget &target);
Status CvtRgb2NvP010None(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, const OpTarget &target);

// BAYER -> BGR
Status CvtBayer2BgrNone(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, const OpTarget &target);

#if defined(AURA_ENABLE_NEON)
class CvtColorNeon : public CvtColorImpl
{
public:
    CvtColorNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type) override;

    Status Run() override;
};

// RGB <-> BGRA
Status CvtBgr2BgraNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status CvtBgra2BgrNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status CvtBgr2RgbNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status CvtBgr2GrayNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, const OpTarget &target);
Status CvtGray2BgrNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

// YUV -> RGB
Status CvtNv2RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtY4202RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtY4222RgbNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapuv, DT_BOOL swapy, CvtColorType type, const OpTarget &target);
Status CvtY4442RgbNeon(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type, const OpTarget &target);

// RGB -> YUV
Status CvtRgb2NvNeon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtRgb2Y420Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_BOOL swapuv, CvtColorType type, const OpTarget &target);
Status CvtRgb2Y444Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type, const OpTarget &target);
Status CvtRgb2NvP010Neon(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, const OpTarget &target);

// BAYER -> BGR
Status CvtBayer2BgrNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg, const OpTarget &target);
#endif

#if defined(AURA_ENABLE_OPENCL)
class CvtColorCL : public CvtColorImpl
{
public:
    CvtColorCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, CvtColorType cvtcolor_type);

private:
    // RGB <-> BGRA
    Status CvtBgr2GrayCLImpl();

    // YUV -> RGB
    Status CvtNv2RgbCLImpl();
    Status CvtY4202RgbCLImpl();
    Status CvtY4222RgbCLImpl();
    Status CvtY4442RgbCLImpl();

    // RGB -> YUV
    Status CvtRgb2NvCLImpl();
    Status CvtRgb2Y420CLImpl();
    Status CvtRgb2Y444CLImpl();

    // BAYER -> BGR
    Status CvtBayer2BgrCLImpl();

private:
    std::vector<CLKernel> m_cl_kernels;
    std::vector<CLMem> m_cl_src;
    std::vector<CLMem> m_cl_dst;
    std::string m_profiling_string;
};

#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
class CvtColorHvx : public CvtColorImpl
{
public:
    CvtColorHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
// RGB <-> BGRA
Status CvtBgr2BgraHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb);
Status CvtBgr2GrayHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb);
Status CvtGray2BgrHvx(Context *ctx, const Mat &src, Mat &dst);

// YUV -> RGB
Status CvtNv2RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, DT_BOOL swapuv, CvtColorType type);
Status CvtY4202RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, DT_BOOL swapuv, CvtColorType type);
Status CvtY4222RgbHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapuv, DT_BOOL swapy, CvtColorType type);
Status CvtY4442RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type);

// RGB -> YUV
Status CvtRgb2NvHvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv, CvtColorType type);
Status CvtRgb2Y420Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, DT_BOOL swapuv, CvtColorType type);
Status CvtRgb2Y444Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type);
Status CvtRgb2NvP010Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, DT_BOOL swapuv);

// BAYER -> BGR
Status CvtBayer2BgrHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL swapb, DT_BOOL swapg);
#  endif // AURA_BUILD_HEXAGON

using CvtColorInParamHvx = HexagonRpcParamType<std::vector<Mat>, std::vector<Mat>, CvtColorType>;

#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

#if defined(AURA_ENABLE_XTENSA)
class CvtColorVdsp : public CvtColorImpl
{
public:
    CvtColorVdsp(Context *ctx, const OpTarget &target);

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type) override;

    Status Initialize() override;

    Status DeInitialize() override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::vector<XtensaMat> m_xtensa_src;
    std::vector<XtensaMat> m_xtensa_dst;
};

using CvtColorInParamVdsp = XtensaRpcParamType<std::vector<XtensaMat>, std::vector<XtensaMat>, CvtColorType>;

#endif // defined(AURA_ENABLE_XTENSA)

} // namespace aura
#endif // AURA_OPS_CVTCOLOR_CVTCOLOR_IMPL_HPP__