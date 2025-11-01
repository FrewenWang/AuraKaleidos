#include "xtensa/cvtcolor_vdsp_impl.hpp"

namespace aura
{
namespace xtensa
{

template <MI_S32 IC>
MI_S32 CvtBgr2GrayVdspImpl(const xvTile *src, xvTile *dst, MI_BOOL swapb)
{
    using u8x128xC_t = typename xtensa::MQVector<MI_U8, IC>::MVType;

    MI_U16 b_coeff = Bgr2GrayParam::BC;
    MI_U16 g_coeff = Bgr2GrayParam::GC;
    MI_U16 r_coeff = Bgr2GrayParam::RC;
    if (swapb)
    {
        Swap(b_coeff, r_coeff);
    }

    valign va_store = IVP_ZALIGN();
    xb_vec2Nx8U *__restrict vqu8_src, *__restrict vqu8_dst;
    u8x128xC_t mvqu8_src;

    for (MI_S32 y = 0; y < dst->height; y++)
    {
        MI_S32 src_len = src->pitch * sizeof(MI_U8);
        MI_S32 dst_len = dst->pitch * sizeof(MI_U8);

        MI_U8 *src_row = (MI_U8*)(src->pData) + y * src->pitch;
        MI_U8 *dst_row = (MI_U8*)(dst->pData) + y * dst->pitch;

        vqu8_src = (xb_vec2Nx8U*)(src_row);
        vqu8_dst = (xb_vec2Nx8U*)(dst_row);
        for (MI_S32 x = 0; x < dst->width; x += AURA_VDSP_VLEN)
        {
            vload(vqu8_src, mvqu8_src, src_len);

            xb_vec2Nx32w ws32_sum = IVP_MULUSP2N8XR16(mvqu8_src.val[1], mvqu8_src.val[2], r_coeff | ((g_coeff) << 16));
            IVP_MULUSAI2NX8X16(ws32_sum, mvqu8_src.val[0], b_coeff, b_coeff);
            xb_vec2Nx8U vqu8_result = IVP_PACKVRU2NX32W(ws32_sum, 15);

            vstore(vqu8_dst, va_store, vqu8_result, dst_len);
            vflush(va_store, vqu8_dst);

            src_len -= AURA_VDSP_VLEN * IC;
            dst_len -= AURA_VDSP_VLEN;
        }
    }

    return AURA_XTENSA_OK;
}

MI_S32 CvtBgr2GrayVdsp(const xvTile *src, xvTile *dst, MI_BOOL swapb)
{
    if (src->height != dst->height || src->width != dst->width)
    {
        AURA_XTENSA_LOG("src and dst must have the same height and width");
        return AURA_XTENSA_ERROR;
    }

    if (XV_TYPE_CHANNELS(src->type) != 3 || XV_TYPE_CHANNELS(dst->type) != 1)
    {
        AURA_XTENSA_LOG("src channel must be 3 and dst channel must be 1");
        return AURA_XTENSA_ERROR;
    }

    MI_S32 ret = AURA_XTENSA_ERROR;

    switch (XV_TYPE_CHANNELS(src->type))
    {
        case 3:
        {
            ret = CvtBgr2GrayVdspImpl<3>(src, dst, swapb);
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("unsupported channel");
            break;
        }
    }

    if (ret != AURA_XTENSA_OK)
    {
        AURA_XTENSA_LOG("CvtBgr2GrayVdspImpl failed");
    }

    return ret;
}

} // namespace xtensa
} // namespace aura
