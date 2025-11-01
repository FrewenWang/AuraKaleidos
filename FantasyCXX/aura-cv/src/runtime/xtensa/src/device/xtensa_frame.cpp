#include "aura/runtime/xtensa/device/xtensa_frame.hpp"

#include "tileManager.h"
#include "tileManager_api.h"
#include "tileManager_FIK_api.h"

namespace aura
{
namespace xtensa
{

FrameWrapper::FrameWrapper() : buffer(0), buffer_size(0), data(0), width(0), height(0), pitch(0), pixel_res(0), num_channels(0), left_edge_pad_width(0),
                               top_edge_pad_height(0), right_edge_pad_width(0), bottom_edge_pad_height(0), padding_type(0), padding_val(0)
{}

FrameWrapper::FrameWrapper(MI_U64 buffer, MI_U32 buffer_size, MI_U64 data, MI_S32 width, MI_S32 height, MI_S32 pitch, MI_U8 pixel_res, MI_U8 num_channels,
                           MI_U8 left_edge_pad_width, MI_U8 top_edge_pad_height, MI_U8 right_edge_pad_width, MI_U8 bottom_edge_pad_height,
                           MI_U8 padding_type, MI_U32 padding_val)
                           : buffer(buffer), buffer_size(buffer_size), data(data), width(width), height(height),pitch(pitch), pixel_res(pixel_res),
                           num_channels(num_channels), left_edge_pad_width(left_edge_pad_width), top_edge_pad_height(top_edge_pad_height),
                           right_edge_pad_width(right_edge_pad_width), bottom_edge_pad_height(bottom_edge_pad_height), padding_type(padding_type),
                           padding_val(padding_val)
{}

MI_BOOL FrameWrapper::IsValid() const
{
    return (buffer != 0) && (data != 0) && (width >= 0) && (height >= 0) && (pitch >= 0) && (pixel_res != 0) && (num_channels != 0);
}

FrameWrapper::FrameWrapper(TileManager tm, const Mat *mat, BorderType border_type, const Scalar &border_value)
                           : buffer(0), buffer_size(0), data(0), width(0), height(0), pitch(0), pixel_res(0), num_channels(0), left_edge_pad_width(0),
                             top_edge_pad_height(0), right_edge_pad_width(0), bottom_edge_pad_height(0), padding_type(0), padding_val(0)
{
    do
    {
        xvTileManager *xv_tm = static_cast<xvTileManager*>(tm);
        if (NULL == xv_tm)
        {
            AURA_XTENSA_LOG("xv_tm is null");
            break;
        }

        if (MI_NULL == mat)
        {
            AURA_XTENSA_LOG("mat is null");
            break;
        }

        if (sizeof(xvFrame) != sizeof(FrameWrapper))
        {
            AURA_XTENSA_LOG("sizeof(xvFrame) != sizeof(FrameWrapper)");
            break;
        }

        Sizes3 src_sizes   = mat->GetSizes();
        Sizes3 src_strides = mat->GetStrides();
        ElemType elem_type = mat->GetElemType();
        uint64_t data      = (MI_U64)((MI_UPTR_T)(mat->GetData()));

        MI_S32 ret = AURA_XTENSA_ERROR;
        ret = xvSetupFrame(xv_tm, reinterpret_cast<xvFrame*>(this), data, src_sizes.m_width, src_sizes.m_height, src_strides.m_width / ElemTypeSize(elem_type),
                           ElemTypeSize(elem_type), src_sizes.m_channel, (MI_U8)(border_type), border_value.m_val[0]);
        if (ret != AURA_XTENSA_OK)
        {
            AURA_XTENSA_LOG("xvSetupFrame failed");
            break;
        }
    } while (0);
}

} // namespace xtensa
} // namespace aura