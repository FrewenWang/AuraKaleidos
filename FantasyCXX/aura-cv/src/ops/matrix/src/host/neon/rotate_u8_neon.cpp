#include "rotate_impl.hpp"

namespace aura
{

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_90, 1>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                       DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_width);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = src_x * dst_step + src_height - src_y - 8;

    uint8x8_t vdu8_line0 = neon::vload1(&src_data[src_addr]);
    uint8x8_t vdu8_line1 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line2 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line3 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line4 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line5 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line6 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line7 = neon::vload1(&src_data[src_addr += src_step]);

    uint8x8x2_t v2du8_trn10 = neon::vtrn(vdu8_line1, vdu8_line0);
    uint8x8x2_t v2du8_trn32 = neon::vtrn(vdu8_line3, vdu8_line2);
    uint8x8x2_t v2du8_trn54 = neon::vtrn(vdu8_line5, vdu8_line4);
    uint8x8x2_t v2du8_trn76 = neon::vtrn(vdu8_line7, vdu8_line6);

    uint16x4x2_t v2du16_trn20 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn32.val[1]), neon::vreinterpret_u16(v2du8_trn10.val[1]));
    uint16x4x2_t v2du16_trn31 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn32.val[0]), neon::vreinterpret_u16(v2du8_trn10.val[0]));
    uint16x4x2_t v2du16_trn64 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn76.val[1]), neon::vreinterpret_u16(v2du8_trn54.val[1]));
    uint16x4x2_t v2du16_trn75 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn76.val[0]), neon::vreinterpret_u16(v2du8_trn54.val[0]));

    uint32x2x2_t v2du32_trn40 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn64.val[1]), neon::vreinterpret_u32(v2du16_trn20.val[1]));
    uint32x2x2_t v2du32_trn51 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn75.val[1]), neon::vreinterpret_u32(v2du16_trn31.val[1]));
    uint32x2x2_t v2du32_trn62 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn64.val[0]), neon::vreinterpret_u32(v2du16_trn20.val[0]));
    uint32x2x2_t v2du32_trn73 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn75.val[0]), neon::vreinterpret_u32(v2du16_trn31.val[0]));

    uint8x8_t vdu8_out0 = neon::vreinterpret_u8(v2du32_trn73.val[0]);
    uint8x8_t vdu8_out1 = neon::vreinterpret_u8(v2du32_trn62.val[0]);
    uint8x8_t vdu8_out2 = neon::vreinterpret_u8(v2du32_trn51.val[0]);
    uint8x8_t vdu8_out3 = neon::vreinterpret_u8(v2du32_trn40.val[0]);
    uint8x8_t vdu8_out4 = neon::vreinterpret_u8(v2du32_trn73.val[1]);
    uint8x8_t vdu8_out5 = neon::vreinterpret_u8(v2du32_trn62.val[1]);
    uint8x8_t vdu8_out6 = neon::vreinterpret_u8(v2du32_trn51.val[1]);
    uint8x8_t vdu8_out7 = neon::vreinterpret_u8(v2du32_trn40.val[1]);

    neon::vstore(&dst_data[dst_addr], vdu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_90, 2>::operator()(DT_U16 *src_data, DT_U16 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                       DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_width);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = src_x * dst_step + src_height - src_y - 8;

    uint16x8_t vqu16_line0 = neon::vload1q(&src_data[src_addr]);
    uint16x8_t vqu16_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line3 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line4 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line5 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line6 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line7 = neon::vload1q(&src_data[src_addr += src_step]);

    uint16x8x2_t v2qu16_trn10 = neon::vtrn(vqu16_line1, vqu16_line0);
    uint16x8x2_t v2qu16_trn32 = neon::vtrn(vqu16_line3, vqu16_line2);
    uint16x8x2_t v2qu16_trn54 = neon::vtrn(vqu16_line5, vqu16_line4);
    uint16x8x2_t v2qu16_trn76 = neon::vtrn(vqu16_line7, vqu16_line6);

    uint32x4x2_t v2qu32_trn20 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn32.val[0]), neon::vreinterpret_u32(v2qu16_trn10.val[0]));
    uint32x4x2_t v2qu32_trn31 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn32.val[1]), neon::vreinterpret_u32(v2qu16_trn10.val[1]));
    uint32x4x2_t v2qu32_trn64 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn76.val[0]), neon::vreinterpret_u32(v2qu16_trn54.val[0]));
    uint32x4x2_t v2qu32_trn75 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn76.val[1]), neon::vreinterpret_u32(v2qu16_trn54.val[1]));

    uint16x8_t vqu16_out0 = neon::vcombine(neon::vgetlow(v2qu32_trn64.val[0]), neon::vgetlow(v2qu32_trn20.val[0]));
    uint16x8_t vqu16_out1 = neon::vcombine(neon::vgetlow(v2qu32_trn75.val[0]), neon::vgetlow(v2qu32_trn31.val[0]));
    uint16x8_t vqu16_out2 = neon::vcombine(neon::vgetlow(v2qu32_trn64.val[1]), neon::vgetlow(v2qu32_trn20.val[1]));
    uint16x8_t vqu16_out3 = neon::vcombine(neon::vgetlow(v2qu32_trn75.val[1]), neon::vgetlow(v2qu32_trn31.val[1]));
    uint16x8_t vqu16_out4 = neon::vcombine(neon::vgethigh(v2qu32_trn64.val[0]), neon::vgethigh(v2qu32_trn20.val[0]));
    uint16x8_t vqu16_out5 = neon::vcombine(neon::vgethigh(v2qu32_trn75.val[0]), neon::vgethigh(v2qu32_trn31.val[0]));
    uint16x8_t vqu16_out6 = neon::vcombine(neon::vgethigh(v2qu32_trn64.val[1]), neon::vgethigh(v2qu32_trn20.val[1]));
    uint16x8_t vqu16_out7 = neon::vcombine(neon::vgethigh(v2qu32_trn75.val[1]), neon::vgethigh(v2qu32_trn31.val[1]));

    neon::vstore(&dst_data[dst_addr], vqu16_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_90, 3>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                       DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_width);
    DT_U32 src_addr = src_y * src_step + src_x * 3;
    DT_U32 dst_addr = src_x * dst_step + (src_height - src_y - 8) * 3;

    uint8x8x3_t v3du8_line0 = neon::vload3(&src_data[src_addr]);
    uint8x8x3_t v3du8_line1 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line2 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line3 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line4 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line5 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line6 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line7 = neon::vload3(&src_data[src_addr += src_step]);

    uint8x8x2_t v2du8_0_trn10 = neon::vtrn(v3du8_line1.val[0], v3du8_line0.val[0]);
    uint8x8x2_t v2du8_0_trn32 = neon::vtrn(v3du8_line3.val[0], v3du8_line2.val[0]);
    uint8x8x2_t v2du8_0_trn54 = neon::vtrn(v3du8_line5.val[0], v3du8_line4.val[0]);
    uint8x8x2_t v2du8_0_trn76 = neon::vtrn(v3du8_line7.val[0], v3du8_line6.val[0]);

    uint8x8x2_t v2du8_1_trn10 = neon::vtrn(v3du8_line1.val[1], v3du8_line0.val[1]);
    uint8x8x2_t v2du8_1_trn32 = neon::vtrn(v3du8_line3.val[1], v3du8_line2.val[1]);
    uint8x8x2_t v2du8_1_trn54 = neon::vtrn(v3du8_line5.val[1], v3du8_line4.val[1]);
    uint8x8x2_t v2du8_1_trn76 = neon::vtrn(v3du8_line7.val[1], v3du8_line6.val[1]);

    uint8x8x2_t v2du8_2_trn10 = neon::vtrn(v3du8_line1.val[2], v3du8_line0.val[2]);
    uint8x8x2_t v2du8_2_trn32 = neon::vtrn(v3du8_line3.val[2], v3du8_line2.val[2]);
    uint8x8x2_t v2du8_2_trn54 = neon::vtrn(v3du8_line5.val[2], v3du8_line4.val[2]);
    uint8x8x2_t v2du8_2_trn76 = neon::vtrn(v3du8_line7.val[2], v3du8_line6.val[2]);

    uint16x4x2_t v2du16_0_trn20 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn32.val[1]), neon::vreinterpret_u16(v2du8_0_trn10.val[1]));
    uint16x4x2_t v2du16_0_trn31 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn32.val[0]), neon::vreinterpret_u16(v2du8_0_trn10.val[0]));
    uint16x4x2_t v2du16_0_trn64 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn76.val[1]), neon::vreinterpret_u16(v2du8_0_trn54.val[1]));
    uint16x4x2_t v2du16_0_trn75 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn76.val[0]), neon::vreinterpret_u16(v2du8_0_trn54.val[0]));

    uint16x4x2_t v2du16_1_trn20 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn32.val[1]), neon::vreinterpret_u16(v2du8_1_trn10.val[1]));
    uint16x4x2_t v2du16_1_trn31 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn32.val[0]), neon::vreinterpret_u16(v2du8_1_trn10.val[0]));
    uint16x4x2_t v2du16_1_trn64 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn76.val[1]), neon::vreinterpret_u16(v2du8_1_trn54.val[1]));
    uint16x4x2_t v2du16_1_trn75 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn76.val[0]), neon::vreinterpret_u16(v2du8_1_trn54.val[0]));

    uint16x4x2_t v2du16_2_trn20 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn32.val[1]), neon::vreinterpret_u16(v2du8_2_trn10.val[1]));
    uint16x4x2_t v2du16_2_trn31 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn32.val[0]), neon::vreinterpret_u16(v2du8_2_trn10.val[0]));
    uint16x4x2_t v2du16_2_trn64 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn76.val[1]), neon::vreinterpret_u16(v2du8_2_trn54.val[1]));
    uint16x4x2_t v2du16_2_trn75 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn76.val[0]), neon::vreinterpret_u16(v2du8_2_trn54.val[0]));

    uint32x2x2_t v2du32_0_trn40 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn64.val[1]), neon::vreinterpret_u32(v2du16_0_trn20.val[1]));
    uint32x2x2_t v2du32_0_trn51 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn75.val[1]), neon::vreinterpret_u32(v2du16_0_trn31.val[1]));
    uint32x2x2_t v2du32_0_trn62 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn64.val[0]), neon::vreinterpret_u32(v2du16_0_trn20.val[0]));
    uint32x2x2_t v2du32_0_trn73 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn75.val[0]), neon::vreinterpret_u32(v2du16_0_trn31.val[0]));

    uint32x2x2_t v2du32_1_trn40 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn64.val[1]), neon::vreinterpret_u32(v2du16_1_trn20.val[1]));
    uint32x2x2_t v2du32_1_trn51 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn75.val[1]), neon::vreinterpret_u32(v2du16_1_trn31.val[1]));
    uint32x2x2_t v2du32_1_trn62 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn64.val[0]), neon::vreinterpret_u32(v2du16_1_trn20.val[0]));
    uint32x2x2_t v2du32_1_trn73 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn75.val[0]), neon::vreinterpret_u32(v2du16_1_trn31.val[0]));

    uint32x2x2_t v2du32_2_trn40 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn64.val[1]), neon::vreinterpret_u32(v2du16_2_trn20.val[1]));
    uint32x2x2_t v2du32_2_trn51 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn75.val[1]), neon::vreinterpret_u32(v2du16_2_trn31.val[1]));
    uint32x2x2_t v2du32_2_trn62 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn64.val[0]), neon::vreinterpret_u32(v2du16_2_trn20.val[0]));
    uint32x2x2_t v2du32_2_trn73 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn75.val[0]), neon::vreinterpret_u32(v2du16_2_trn31.val[0]));

    uint8x8x3_t v3du8_out0, v3du8_out1, v3du8_out2, v3du8_out3, v3du8_out4, v3du8_out5, v3du8_out6, v3du8_out7;
    v3du8_out0.val[0] = neon::vreinterpret_u8(v2du32_0_trn73.val[0]);
    v3du8_out1.val[0] = neon::vreinterpret_u8(v2du32_0_trn62.val[0]);
    v3du8_out2.val[0] = neon::vreinterpret_u8(v2du32_0_trn51.val[0]);
    v3du8_out3.val[0] = neon::vreinterpret_u8(v2du32_0_trn40.val[0]);
    v3du8_out4.val[0] = neon::vreinterpret_u8(v2du32_0_trn73.val[1]);
    v3du8_out5.val[0] = neon::vreinterpret_u8(v2du32_0_trn62.val[1]);
    v3du8_out6.val[0] = neon::vreinterpret_u8(v2du32_0_trn51.val[1]);
    v3du8_out7.val[0] = neon::vreinterpret_u8(v2du32_0_trn40.val[1]);

    v3du8_out0.val[1] = neon::vreinterpret_u8(v2du32_1_trn73.val[0]);
    v3du8_out1.val[1] = neon::vreinterpret_u8(v2du32_1_trn62.val[0]);
    v3du8_out2.val[1] = neon::vreinterpret_u8(v2du32_1_trn51.val[0]);
    v3du8_out3.val[1] = neon::vreinterpret_u8(v2du32_1_trn40.val[0]);
    v3du8_out4.val[1] = neon::vreinterpret_u8(v2du32_1_trn73.val[1]);
    v3du8_out5.val[1] = neon::vreinterpret_u8(v2du32_1_trn62.val[1]);
    v3du8_out6.val[1] = neon::vreinterpret_u8(v2du32_1_trn51.val[1]);
    v3du8_out7.val[1] = neon::vreinterpret_u8(v2du32_1_trn40.val[1]);

    v3du8_out0.val[2] = neon::vreinterpret_u8(v2du32_2_trn73.val[0]);
    v3du8_out1.val[2] = neon::vreinterpret_u8(v2du32_2_trn62.val[0]);
    v3du8_out2.val[2] = neon::vreinterpret_u8(v2du32_2_trn51.val[0]);
    v3du8_out3.val[2] = neon::vreinterpret_u8(v2du32_2_trn40.val[0]);
    v3du8_out4.val[2] = neon::vreinterpret_u8(v2du32_2_trn73.val[1]);
    v3du8_out5.val[2] = neon::vreinterpret_u8(v2du32_2_trn62.val[1]);
    v3du8_out6.val[2] = neon::vreinterpret_u8(v2du32_2_trn51.val[1]);
    v3du8_out7.val[2] = neon::vreinterpret_u8(v2du32_2_trn40.val[1]);

    neon::vstore(&dst_data[dst_addr], v3du8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_90, 4>::operator()(DT_U32 *src_data, DT_U32 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                       DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_width);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = src_x * dst_step + src_height - src_y - 4;

    uint32x4_t vqu32_line0 = neon::vload1q(&src_data[src_addr]);
    uint32x4_t vqu32_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line3 = neon::vload1q(&src_data[src_addr += src_step]);

    uint32x4x2_t v2qu32_trn10 = neon::vtrn(vqu32_line1, vqu32_line0);
    uint32x4x2_t v2qu32_trn32 = neon::vtrn(vqu32_line3, vqu32_line2);

    uint32x4_t vqu32_out0 = neon::vcombine(neon::vgetlow(v2qu32_trn32.val[0]), neon::vgetlow(v2qu32_trn10.val[0]));
    uint32x4_t vqu32_out1 = neon::vcombine(neon::vgetlow(v2qu32_trn32.val[1]), neon::vgetlow(v2qu32_trn10.val[1]));
    uint32x4_t vqu32_out2 = neon::vcombine(neon::vgethigh(v2qu32_trn32.val[0]), neon::vgethigh(v2qu32_trn10.val[0]));
    uint32x4_t vqu32_out3 = neon::vcombine(neon::vgethigh(v2qu32_trn32.val[1]), neon::vgethigh(v2qu32_trn10.val[1]));

    neon::vstore(&dst_data[dst_addr], vqu32_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out3);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_180, 1>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_height - 8 - src_y) * dst_step + src_width - 8 - src_x;

    uint8x8_t vdu8_line0 = neon::vload1(&src_data[src_addr]);
    uint8x8_t vdu8_line1 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line2 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line3 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line4 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line5 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line6 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line7 = neon::vload1(&src_data[src_addr += src_step]);

    uint8x8_t vdu8_rev0 = neon::vrev64(vdu8_line0);
    uint8x8_t vdu8_rev1 = neon::vrev64(vdu8_line1);
    uint8x8_t vdu8_rev2 = neon::vrev64(vdu8_line2);
    uint8x8_t vdu8_rev3 = neon::vrev64(vdu8_line3);
    uint8x8_t vdu8_rev4 = neon::vrev64(vdu8_line4);
    uint8x8_t vdu8_rev5 = neon::vrev64(vdu8_line5);
    uint8x8_t vdu8_rev6 = neon::vrev64(vdu8_line6);
    uint8x8_t vdu8_rev7 = neon::vrev64(vdu8_line7);

    uint8x8_t vdu8_out0 = vdu8_rev7;
    uint8x8_t vdu8_out1 = vdu8_rev6;
    uint8x8_t vdu8_out2 = vdu8_rev5;
    uint8x8_t vdu8_out3 = vdu8_rev4;
    uint8x8_t vdu8_out4 = vdu8_rev3;
    uint8x8_t vdu8_out5 = vdu8_rev2;
    uint8x8_t vdu8_out6 = vdu8_rev1;
    uint8x8_t vdu8_out7 = vdu8_rev0;

    neon::vstore(&dst_data[dst_addr], vdu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_180, 2>::operator()(DT_U16 *src_data, DT_U16 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_height - 8 - src_y) * dst_step + src_width - 8 - src_x;

    uint16x8_t vqu16_line0 = neon::vload1q(&src_data[src_addr]);
    uint16x8_t vqu16_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line3 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line4 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line5 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line6 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line7 = neon::vload1q(&src_data[src_addr += src_step]);

    uint16x8_t vqu16_rev0 = neon::vrev64(vqu16_line0);
    uint16x8_t vqu16_rev1 = neon::vrev64(vqu16_line1);
    uint16x8_t vqu16_rev2 = neon::vrev64(vqu16_line2);
    uint16x8_t vqu16_rev3 = neon::vrev64(vqu16_line3);
    uint16x8_t vqu16_rev4 = neon::vrev64(vqu16_line4);
    uint16x8_t vqu16_rev5 = neon::vrev64(vqu16_line5);
    uint16x8_t vqu16_rev6 = neon::vrev64(vqu16_line6);
    uint16x8_t vqu16_rev7 = neon::vrev64(vqu16_line7);

    uint16x8_t vqu16_out0 = neon::vcombine(neon::vgethigh(vqu16_rev7), neon::vgetlow(vqu16_rev7));
    uint16x8_t vqu16_out1 = neon::vcombine(neon::vgethigh(vqu16_rev6), neon::vgetlow(vqu16_rev6));
    uint16x8_t vqu16_out2 = neon::vcombine(neon::vgethigh(vqu16_rev5), neon::vgetlow(vqu16_rev5));
    uint16x8_t vqu16_out3 = neon::vcombine(neon::vgethigh(vqu16_rev4), neon::vgetlow(vqu16_rev4));
    uint16x8_t vqu16_out4 = neon::vcombine(neon::vgethigh(vqu16_rev3), neon::vgetlow(vqu16_rev3));
    uint16x8_t vqu16_out5 = neon::vcombine(neon::vgethigh(vqu16_rev2), neon::vgetlow(vqu16_rev2));
    uint16x8_t vqu16_out6 = neon::vcombine(neon::vgethigh(vqu16_rev1), neon::vgetlow(vqu16_rev1));
    uint16x8_t vqu16_out7 = neon::vcombine(neon::vgethigh(vqu16_rev0), neon::vgetlow(vqu16_rev0));

    neon::vstore(&dst_data[dst_addr], vqu16_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_180, 3>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_y * src_step + (src_x * 3);
    DT_U32 dst_addr = (src_height - 8 - src_y) * dst_step + (src_width - 8 - src_x) * 3;

    uint8x8x3_t v3du8_line0 = neon::vload3(&src_data[src_addr]);
    uint8x8x3_t v3du8_line1 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line2 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line3 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line4 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line5 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line6 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line7 = neon::vload3(&src_data[src_addr += src_step]);

    uint8x8_t vdu8_0_rev0 = neon::vrev64(v3du8_line0.val[0]);
    uint8x8_t vdu8_0_rev1 = neon::vrev64(v3du8_line1.val[0]);
    uint8x8_t vdu8_0_rev2 = neon::vrev64(v3du8_line2.val[0]);
    uint8x8_t vdu8_0_rev3 = neon::vrev64(v3du8_line3.val[0]);
    uint8x8_t vdu8_0_rev4 = neon::vrev64(v3du8_line4.val[0]);
    uint8x8_t vdu8_0_rev5 = neon::vrev64(v3du8_line5.val[0]);
    uint8x8_t vdu8_0_rev6 = neon::vrev64(v3du8_line6.val[0]);
    uint8x8_t vdu8_0_rev7 = neon::vrev64(v3du8_line7.val[0]);

    uint8x8_t vdu8_1_rev0 = neon::vrev64(v3du8_line0.val[1]);
    uint8x8_t vdu8_1_rev1 = neon::vrev64(v3du8_line1.val[1]);
    uint8x8_t vdu8_1_rev2 = neon::vrev64(v3du8_line2.val[1]);
    uint8x8_t vdu8_1_rev3 = neon::vrev64(v3du8_line3.val[1]);
    uint8x8_t vdu8_1_rev4 = neon::vrev64(v3du8_line4.val[1]);
    uint8x8_t vdu8_1_rev5 = neon::vrev64(v3du8_line5.val[1]);
    uint8x8_t vdu8_1_rev6 = neon::vrev64(v3du8_line6.val[1]);
    uint8x8_t vdu8_1_rev7 = neon::vrev64(v3du8_line7.val[1]);

    uint8x8_t vdu8_2_rev0 = neon::vrev64(v3du8_line0.val[2]);
    uint8x8_t vdu8_2_rev1 = neon::vrev64(v3du8_line1.val[2]);
    uint8x8_t vdu8_2_rev2 = neon::vrev64(v3du8_line2.val[2]);
    uint8x8_t vdu8_2_rev3 = neon::vrev64(v3du8_line3.val[2]);
    uint8x8_t vdu8_2_rev4 = neon::vrev64(v3du8_line4.val[2]);
    uint8x8_t vdu8_2_rev5 = neon::vrev64(v3du8_line5.val[2]);
    uint8x8_t vdu8_2_rev6 = neon::vrev64(v3du8_line6.val[2]);
    uint8x8_t vdu8_2_rev7 = neon::vrev64(v3du8_line7.val[2]);

    uint8x8x3_t v3du8_out0, v3du8_out1, v3du8_out2, v3du8_out3, v3du8_out4, v3du8_out5, v3du8_out6, v3du8_out7;
    v3du8_out0.val[0] = vdu8_0_rev7;
    v3du8_out1.val[0] = vdu8_0_rev6;
    v3du8_out2.val[0] = vdu8_0_rev5;
    v3du8_out3.val[0] = vdu8_0_rev4;
    v3du8_out4.val[0] = vdu8_0_rev3;
    v3du8_out5.val[0] = vdu8_0_rev2;
    v3du8_out6.val[0] = vdu8_0_rev1;
    v3du8_out7.val[0] = vdu8_0_rev0;

    v3du8_out0.val[1] = vdu8_1_rev7;
    v3du8_out1.val[1] = vdu8_1_rev6;
    v3du8_out2.val[1] = vdu8_1_rev5;
    v3du8_out3.val[1] = vdu8_1_rev4;
    v3du8_out4.val[1] = vdu8_1_rev3;
    v3du8_out5.val[1] = vdu8_1_rev2;
    v3du8_out6.val[1] = vdu8_1_rev1;
    v3du8_out7.val[1] = vdu8_1_rev0;

    v3du8_out0.val[2] = vdu8_2_rev7;
    v3du8_out1.val[2] = vdu8_2_rev6;
    v3du8_out2.val[2] = vdu8_2_rev5;
    v3du8_out3.val[2] = vdu8_2_rev4;
    v3du8_out4.val[2] = vdu8_2_rev3;
    v3du8_out5.val[2] = vdu8_2_rev2;
    v3du8_out6.val[2] = vdu8_2_rev1;
    v3du8_out7.val[2] = vdu8_2_rev0;

    neon::vstore(&dst_data[dst_addr], v3du8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_180, 4>::operator()(DT_U32 *src_data, DT_U32 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_height - 4 - src_y) * dst_step + src_width - 4 - src_x;

    uint32x4_t vqu32_line0 = neon::vload1q(&src_data[src_addr]);
    uint32x4_t vqu32_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line3 = neon::vload1q(&src_data[src_addr += src_step]);

    uint32x4_t vqu32_rev0 = neon::vrev64(vqu32_line0);
    uint32x4_t vqu32_rev1 = neon::vrev64(vqu32_line1);
    uint32x4_t vqu32_rev2 = neon::vrev64(vqu32_line2);
    uint32x4_t vqu32_rev3 = neon::vrev64(vqu32_line3);

    uint32x4_t vqu32_out0 = neon::vcombine(neon::vgethigh(vqu32_rev3), neon::vgetlow(vqu32_rev3));
    uint32x4_t vqu32_out1 = neon::vcombine(neon::vgethigh(vqu32_rev2), neon::vgetlow(vqu32_rev2));
    uint32x4_t vqu32_out2 = neon::vcombine(neon::vgethigh(vqu32_rev1), neon::vgetlow(vqu32_rev1));
    uint32x4_t vqu32_out3 = neon::vcombine(neon::vgethigh(vqu32_rev0), neon::vgetlow(vqu32_rev0));

    neon::vstore(&dst_data[dst_addr], vqu32_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out3);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_270, 1>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_height);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_width - 8 - src_x) * dst_step + src_y;

    uint8x8_t vdu8_line0 = neon::vload1(&src_data[src_addr]);
    uint8x8_t vdu8_line1 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line2 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line3 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line4 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line5 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line6 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line7 = neon::vload1(&src_data[src_addr += src_step]);

    uint8x8x2_t v2du8_trn01 = neon::vtrn(vdu8_line0, vdu8_line1);
    uint8x8x2_t v2du8_trn23 = neon::vtrn(vdu8_line2, vdu8_line3);
    uint8x8x2_t v2du8_trn45 = neon::vtrn(vdu8_line4, vdu8_line5);
    uint8x8x2_t v2du8_trn67 = neon::vtrn(vdu8_line6, vdu8_line7);

    uint16x4x2_t v2du16_trn02 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn01.val[0]), neon::vreinterpret_u16(v2du8_trn23.val[0]));
    uint16x4x2_t v2du16_trn13 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn01.val[1]), neon::vreinterpret_u16(v2du8_trn23.val[1]));
    uint16x4x2_t v2du16_trn46 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn45.val[0]), neon::vreinterpret_u16(v2du8_trn67.val[0]));
    uint16x4x2_t v2du16_trn57 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn45.val[1]), neon::vreinterpret_u16(v2du8_trn67.val[1]));

    uint32x2x2_t v2du32_trn04 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn02.val[0]), neon::vreinterpret_u32(v2du16_trn46.val[0]));
    uint32x2x2_t v2du32_trn15 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn13.val[0]), neon::vreinterpret_u32(v2du16_trn57.val[0]));
    uint32x2x2_t v2du32_trn26 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn02.val[1]), neon::vreinterpret_u32(v2du16_trn46.val[1]));
    uint32x2x2_t v2du32_trn37 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn13.val[1]), neon::vreinterpret_u32(v2du16_trn57.val[1]));

    uint8x8_t vdu8_out0 = neon::vreinterpret_u8(v2du32_trn37.val[1]);
    uint8x8_t vdu8_out1 = neon::vreinterpret_u8(v2du32_trn26.val[1]);
    uint8x8_t vdu8_out2 = neon::vreinterpret_u8(v2du32_trn15.val[1]);
    uint8x8_t vdu8_out3 = neon::vreinterpret_u8(v2du32_trn04.val[1]);
    uint8x8_t vdu8_out4 = neon::vreinterpret_u8(v2du32_trn37.val[0]);
    uint8x8_t vdu8_out5 = neon::vreinterpret_u8(v2du32_trn26.val[0]);
    uint8x8_t vdu8_out6 = neon::vreinterpret_u8(v2du32_trn15.val[0]);
    uint8x8_t vdu8_out7 = neon::vreinterpret_u8(v2du32_trn04.val[0]);

    neon::vstore(&dst_data[dst_addr], vdu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_270, 2>::operator()(DT_U16 *src_data, DT_U16 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_height);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_width - 8 - src_x) * dst_step + src_y;

    uint16x8_t vqu16_line0 = neon::vload1q(&src_data[src_addr]);
    uint16x8_t vqu16_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line3 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line4 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line5 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line6 = neon::vload1q(&src_data[src_addr += src_step]);
    uint16x8_t vqu16_line7 = neon::vload1q(&src_data[src_addr += src_step]);

    uint16x8x2_t v2qu16_trn01 = neon::vtrn(vqu16_line0, vqu16_line1);
    uint16x8x2_t v2qu16_trn23 = neon::vtrn(vqu16_line2, vqu16_line3);
    uint16x8x2_t v2qu16_trn45 = neon::vtrn(vqu16_line4, vqu16_line5);
    uint16x8x2_t v2qu16_trn67 = neon::vtrn(vqu16_line6, vqu16_line7);

    uint32x4x2_t v2qu32_trn02 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn01.val[0]), neon::vreinterpret_u32(v2qu16_trn23.val[0]));
    uint32x4x2_t v2qu32_trn13 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn01.val[1]), neon::vreinterpret_u32(v2qu16_trn23.val[1]));
    uint32x4x2_t v2qu32_trn46 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn45.val[0]), neon::vreinterpret_u32(v2qu16_trn67.val[0]));
    uint32x4x2_t v2qu32_trn57 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn45.val[1]), neon::vreinterpret_u32(v2qu16_trn67.val[1]));

    uint16x8_t vqu16_out0 = neon::vcombine(neon::vgethigh(v2qu32_trn13.val[1]), neon::vgethigh(v2qu32_trn57.val[1]));
    uint16x8_t vqu16_out1 = neon::vcombine(neon::vgethigh(v2qu32_trn02.val[1]), neon::vgethigh(v2qu32_trn46.val[1]));
    uint16x8_t vqu16_out2 = neon::vcombine(neon::vgethigh(v2qu32_trn13.val[0]), neon::vgethigh(v2qu32_trn57.val[0]));
    uint16x8_t vqu16_out3 = neon::vcombine(neon::vgethigh(v2qu32_trn02.val[0]), neon::vgethigh(v2qu32_trn46.val[0]));
    uint16x8_t vqu16_out4 = neon::vcombine(neon::vgetlow(v2qu32_trn13.val[1]), neon::vgetlow(v2qu32_trn57.val[1]));
    uint16x8_t vqu16_out5 = neon::vcombine(neon::vgetlow(v2qu32_trn02.val[1]), neon::vgetlow(v2qu32_trn46.val[1]));
    uint16x8_t vqu16_out6 = neon::vcombine(neon::vgetlow(v2qu32_trn13.val[0]), neon::vgetlow(v2qu32_trn57.val[0]));
    uint16x8_t vqu16_out7 = neon::vcombine(neon::vgetlow(v2qu32_trn02.val[0]), neon::vgetlow(v2qu32_trn46.val[0]));

    neon::vstore(&dst_data[dst_addr], vqu16_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu16_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_270, 3>::operator()(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_height);
    DT_U32 src_addr = src_y * src_step + (src_x * 3);
    DT_U32 dst_addr = (src_width - 8 - src_x) * dst_step + (src_y * 3);

    uint8x8x3_t v3du8_line0 = neon::vload3(&src_data[src_addr]);
    uint8x8x3_t v3du8_line1 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line2 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line3 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line4 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line5 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line6 = neon::vload3(&src_data[src_addr += src_step]);
    uint8x8x3_t v3du8_line7 = neon::vload3(&src_data[src_addr += src_step]);

    uint8x8x2_t v2du8_0_trn01 = neon::vtrn(v3du8_line0.val[0], v3du8_line1.val[0]);
    uint8x8x2_t v2du8_0_trn23 = neon::vtrn(v3du8_line2.val[0], v3du8_line3.val[0]);
    uint8x8x2_t v2du8_0_trn45 = neon::vtrn(v3du8_line4.val[0], v3du8_line5.val[0]);
    uint8x8x2_t v2du8_0_trn67 = neon::vtrn(v3du8_line6.val[0], v3du8_line7.val[0]);

    uint8x8x2_t v2du8_1_trn01 = neon::vtrn(v3du8_line0.val[1], v3du8_line1.val[1]);
    uint8x8x2_t v2du8_1_trn23 = neon::vtrn(v3du8_line2.val[1], v3du8_line3.val[1]);
    uint8x8x2_t v2du8_1_trn45 = neon::vtrn(v3du8_line4.val[1], v3du8_line5.val[1]);
    uint8x8x2_t v2du8_1_trn67 = neon::vtrn(v3du8_line6.val[1], v3du8_line7.val[1]);

    uint8x8x2_t v2du8_2_trn01 = neon::vtrn(v3du8_line0.val[2], v3du8_line1.val[2]);
    uint8x8x2_t v2du8_2_trn23 = neon::vtrn(v3du8_line2.val[2], v3du8_line3.val[2]);
    uint8x8x2_t v2du8_2_trn45 = neon::vtrn(v3du8_line4.val[2], v3du8_line5.val[2]);
    uint8x8x2_t v2du8_2_trn67 = neon::vtrn(v3du8_line6.val[2], v3du8_line7.val[2]);

    uint16x4x2_t v2du16_0_trn02 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn01.val[0]), neon::vreinterpret_u16(v2du8_0_trn23.val[0]));
    uint16x4x2_t v2du16_0_trn13 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn01.val[1]), neon::vreinterpret_u16(v2du8_0_trn23.val[1]));
    uint16x4x2_t v2du16_0_trn46 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn45.val[0]), neon::vreinterpret_u16(v2du8_0_trn67.val[0]));
    uint16x4x2_t v2du16_0_trn57 = neon::vtrn(neon::vreinterpret_u16(v2du8_0_trn45.val[1]), neon::vreinterpret_u16(v2du8_0_trn67.val[1]));

    uint16x4x2_t v2du16_1_trn02 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn01.val[0]), neon::vreinterpret_u16(v2du8_1_trn23.val[0]));
    uint16x4x2_t v2du16_1_trn13 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn01.val[1]), neon::vreinterpret_u16(v2du8_1_trn23.val[1]));
    uint16x4x2_t v2du16_1_trn46 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn45.val[0]), neon::vreinterpret_u16(v2du8_1_trn67.val[0]));
    uint16x4x2_t v2du16_1_trn57 = neon::vtrn(neon::vreinterpret_u16(v2du8_1_trn45.val[1]), neon::vreinterpret_u16(v2du8_1_trn67.val[1]));

    uint16x4x2_t v2du16_2_trn02 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn01.val[0]), neon::vreinterpret_u16(v2du8_2_trn23.val[0]));
    uint16x4x2_t v2du16_2_trn13 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn01.val[1]), neon::vreinterpret_u16(v2du8_2_trn23.val[1]));
    uint16x4x2_t v2du16_2_trn46 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn45.val[0]), neon::vreinterpret_u16(v2du8_2_trn67.val[0]));
    uint16x4x2_t v2du16_2_trn57 = neon::vtrn(neon::vreinterpret_u16(v2du8_2_trn45.val[1]), neon::vreinterpret_u16(v2du8_2_trn67.val[1]));

    uint32x2x2_t v2du32_0_trn04 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn02.val[0]), neon::vreinterpret_u32(v2du16_0_trn46.val[0]));
    uint32x2x2_t v2du32_0_trn15 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn13.val[0]), neon::vreinterpret_u32(v2du16_0_trn57.val[0]));
    uint32x2x2_t v2du32_0_trn26 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn02.val[1]), neon::vreinterpret_u32(v2du16_0_trn46.val[1]));
    uint32x2x2_t v2du32_0_trn37 = neon::vtrn(neon::vreinterpret_u32(v2du16_0_trn13.val[1]), neon::vreinterpret_u32(v2du16_0_trn57.val[1]));

    uint32x2x2_t v2du32_1_trn04 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn02.val[0]), neon::vreinterpret_u32(v2du16_1_trn46.val[0]));
    uint32x2x2_t v2du32_1_trn15 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn13.val[0]), neon::vreinterpret_u32(v2du16_1_trn57.val[0]));
    uint32x2x2_t v2du32_1_trn26 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn02.val[1]), neon::vreinterpret_u32(v2du16_1_trn46.val[1]));
    uint32x2x2_t v2du32_1_trn37 = neon::vtrn(neon::vreinterpret_u32(v2du16_1_trn13.val[1]), neon::vreinterpret_u32(v2du16_1_trn57.val[1]));

    uint32x2x2_t v2du32_2_trn04 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn02.val[0]), neon::vreinterpret_u32(v2du16_2_trn46.val[0]));
    uint32x2x2_t v2du32_2_trn15 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn13.val[0]), neon::vreinterpret_u32(v2du16_2_trn57.val[0]));
    uint32x2x2_t v2du32_2_trn26 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn02.val[1]), neon::vreinterpret_u32(v2du16_2_trn46.val[1]));
    uint32x2x2_t v2du32_2_trn37 = neon::vtrn(neon::vreinterpret_u32(v2du16_2_trn13.val[1]), neon::vreinterpret_u32(v2du16_2_trn57.val[1]));

    uint8x8x3_t v3du8_out0, v3du8_out1, v3du8_out2, v3du8_out3, v3du8_out4, v3du8_out5, v3du8_out6, v3du8_out7;
    v3du8_out0.val[0] = neon::vreinterpret_u8(v2du32_0_trn37.val[1]);
    v3du8_out1.val[0] = neon::vreinterpret_u8(v2du32_0_trn26.val[1]);
    v3du8_out2.val[0] = neon::vreinterpret_u8(v2du32_0_trn15.val[1]);
    v3du8_out3.val[0] = neon::vreinterpret_u8(v2du32_0_trn04.val[1]);
    v3du8_out4.val[0] = neon::vreinterpret_u8(v2du32_0_trn37.val[0]);
    v3du8_out5.val[0] = neon::vreinterpret_u8(v2du32_0_trn26.val[0]);
    v3du8_out6.val[0] = neon::vreinterpret_u8(v2du32_0_trn15.val[0]);
    v3du8_out7.val[0] = neon::vreinterpret_u8(v2du32_0_trn04.val[0]);

    v3du8_out0.val[1] = neon::vreinterpret_u8(v2du32_1_trn37.val[1]);
    v3du8_out1.val[1] = neon::vreinterpret_u8(v2du32_1_trn26.val[1]);
    v3du8_out2.val[1] = neon::vreinterpret_u8(v2du32_1_trn15.val[1]);
    v3du8_out3.val[1] = neon::vreinterpret_u8(v2du32_1_trn04.val[1]);
    v3du8_out4.val[1] = neon::vreinterpret_u8(v2du32_1_trn37.val[0]);
    v3du8_out5.val[1] = neon::vreinterpret_u8(v2du32_1_trn26.val[0]);
    v3du8_out6.val[1] = neon::vreinterpret_u8(v2du32_1_trn15.val[0]);
    v3du8_out7.val[1] = neon::vreinterpret_u8(v2du32_1_trn04.val[0]);

    v3du8_out0.val[2] = neon::vreinterpret_u8(v2du32_2_trn37.val[1]);
    v3du8_out1.val[2] = neon::vreinterpret_u8(v2du32_2_trn26.val[1]);
    v3du8_out2.val[2] = neon::vreinterpret_u8(v2du32_2_trn15.val[1]);
    v3du8_out3.val[2] = neon::vreinterpret_u8(v2du32_2_trn04.val[1]);
    v3du8_out4.val[2] = neon::vreinterpret_u8(v2du32_2_trn37.val[0]);
    v3du8_out5.val[2] = neon::vreinterpret_u8(v2du32_2_trn26.val[0]);
    v3du8_out6.val[2] = neon::vreinterpret_u8(v2du32_2_trn15.val[0]);
    v3du8_out7.val[2] = neon::vreinterpret_u8(v2du32_2_trn04.val[0]);

    neon::vstore(&dst_data[dst_addr], v3du8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], v3du8_out7);
}

DT_VOID RotateNeonFunctor<DT_U8, RotateType::ROTATE_270, 4>::operator()(DT_U32 *src_data, DT_U32 *dst_data, DT_U32 src_step, DT_U32 dst_step,
                                                                        DT_U32 src_width, DT_U32 src_height, DT_U32 src_x, DT_U32 src_y)
{
    AURA_UNUSED(src_height);
    DT_U32 src_addr = src_y * src_step + src_x;
    DT_U32 dst_addr = (src_width - 4 - src_x) * dst_step + src_y;

    uint32x4_t vqu32_line0 = neon::vload1q(&src_data[src_addr]);
    uint32x4_t vqu32_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint32x4_t vqu32_line3 = neon::vload1q(&src_data[src_addr += src_step]);

    uint32x4x2_t v2qu32_trn01 = neon::vtrn(vqu32_line0, vqu32_line1);
    uint32x4x2_t v2qu32_trn23 = neon::vtrn(vqu32_line2, vqu32_line3);

    uint32x4_t vqu32_out0 = neon::vcombine(neon::vgethigh(v2qu32_trn01.val[1]), neon::vgethigh(v2qu32_trn23.val[1]));
    uint32x4_t vqu32_out1 = neon::vcombine(neon::vgethigh(v2qu32_trn01.val[0]), neon::vgethigh(v2qu32_trn23.val[0]));
    uint32x4_t vqu32_out2 = neon::vcombine(neon::vgetlow(v2qu32_trn01.val[1]), neon::vgetlow(v2qu32_trn23.val[1]));
    uint32x4_t vqu32_out3 = neon::vcombine(neon::vgetlow(v2qu32_trn01.val[0]), neon::vgetlow(v2qu32_trn23.val[0]));

    neon::vstore(&dst_data[dst_addr], vqu32_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu32_out3);
}

} // namespace aura