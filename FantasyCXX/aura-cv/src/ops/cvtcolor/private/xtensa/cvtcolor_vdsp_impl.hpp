#ifndef AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_IMPL_HPP__
#define AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_IMPL_HPP__

#include "aura/ops/cvtcolor/xtensa/cvtcolor_vdsp.hpp"

#include "tileManager.h"
#include "tileManager_FIK_api.h"
#include "tileManager_api.h"

#include <new>

namespace aura
{
namespace xtensa
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
    static constexpr DT_S32 R2Y =  269484; // Round( 0.257f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2Y =  528482; // Round( 0.504f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2Y =  102760; // Round( 0.098f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 R2U = -155188; // Round(-0.148f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 G2U = -305135; // Round(-0.291f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2U =  460324; // Round( 0.439f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2V = -385875; // Round(-0.368f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2V = -74448;  // Round(-0.071f * (1 << CVTCOLOR_COEF_BITS)) + 1;

    static constexpr DT_S32 YC = 16777216;  // Round(16.f  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 UC = 134217728; // Round(128.f * (1 << CVTCOLOR_COEF_BITS));
};

/**
 * @brief the formula of BT601 RGB -> YUV
 *  Y = 0.5   + 0.299 * r + 0.587 * g + 0.144 * b
 * Cb = 128.5 - 0.169 * r - 0.331 * g + 0.500 * b
 * Cr = 128.5 + 0.500 * r - 0.419 * g - 0.081 * b
 */
template <>
struct Rgb2YuvParamTraits<1>
{
    static constexpr DT_S32 R2Y =  313524; // Round( 0.299f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2Y =  615514; // Round( 0.587f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2Y =  150994; // Round( 0.144f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 R2U = -177208; // Round(-0.169f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 G2U = -347077; // Round(-0.331f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2U =  524288; // Round( 0.500f * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2V = -439352; // Round(-0.419f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    static constexpr DT_S32 B2V = -84933;  // Round(-0.081f * (1 << CVTCOLOR_COEF_BITS)) + 1;

    static constexpr DT_S32 YC = 524288;    // Round(0.5f   * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 UC = 134742016; // Round(128.5f * (1 << CVTCOLOR_COEF_BITS));
};

/**
 * @brief the formula of P010 RGB -> YUV
 *  Y = ( 0.299  * r + 0.587  * g + 0.114  * b)       * (1 << 6)
 * Cb = (-0.1687 * r - 0.3313 * g + 0.5    * b + 512) * (1 << 6)
 * Cr = ( 0.5    * r - 0.4187 * g - 0.0813 * b + 512) * (1 << 6)
 */
template <>
struct Rgb2YuvParamTraits<2>
{
    static constexpr DT_S32 R2Y = 313524;    // Round( 0.299  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2Y = 615514;    // Round( 0.587  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2Y = 119538;    // Round( 0.114  * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 R2U = -176895;   // Round(-0.1687 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2U = -347393;   // Round(-0.3313 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2U = 524288;    // Round( 0.5    * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 G2V = -439039;   // Round(-0.4187 * (1 << CVTCOLOR_COEF_BITS));
    static constexpr DT_S32 B2V = -85249;    // Round(-0.0813 * (1 << CVTCOLOR_COEF_BITS));

    static constexpr DT_S32 UC  = 536870912; // Round( 512    * (1 << CVTCOLOR_COEF_BITS));
};

class CvtColorTile : public VdspOpTile
{
public:
    CvtColorTile(TileManager tm);

    Status SetArgs(const vector<TileWrapper> &src, vector<TileWrapper> &dst, CvtColorType type);

    Status DeInitialize();

    Status Run();

private:
    CvtColorType     m_type;
    vector<ElemType> m_src_elem_types;
    vector<ElemType> m_dst_elem_types;
    vector<DT_S32>   m_src_channels;
    vector<DT_S32>   m_dst_channels;
    DT_S32           m_src_sizes;
    DT_S32           m_dst_sizes;

    vector<const TileWrapper*> m_xv_src_tiles;
    vector<TileWrapper*>       m_xv_dst_tiles;
};

class CvtColorFrame : public VdspOpFrame
{
public:
    CvtColorFrame(TileManager tm);

    Status SetArgs(const vector<const Mat*> &src, const vector<Mat*> &dst, CvtColorType type);

    Status DeInitialize();

    Status Run();

private:
    static DT_VOID Prepare(xvTileManager *xv_tm, RefTile *xv_ref_tile, DT_VOID *obj, DT_VOID *tiles, DT_S32 flag);

    static DT_S32 Execute(DT_VOID *obj, DT_VOID *tiles);

private:
    CvtColorType m_type;
    DT_S32       m_src_sizes;
    DT_S32       m_dst_sizes;
    CvtColorTile *m_cvtcolor_tile;
};

DT_S32 CvtBgr2GrayVdsp(const xvTile *src, xvTile *dst, DT_BOOL swapb);
Status CvtColorRpc(TileManager xv_tm, XtensaRpcParam &rpc_param);

using CvtColorInParamVdsp = XtensaRpcParamType<vector<Mat>, vector<Mat>, CvtColorType>;

} // namespace xtensa
} // namespace aura
#endif //AURA_OPS_CVTCOLOR_XTENSA_CVTCOLOR_VDSP_IMPL_HPP__