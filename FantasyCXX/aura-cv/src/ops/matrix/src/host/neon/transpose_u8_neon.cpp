#include "transpose_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

AURA_ALWAYS_INLINE DT_VOID Transpose16x16U8Neon(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_x * src_step + src_y;
    DT_U32 dst_addr = src_y * dst_step + src_x;

    uint8x16_t vqu8_line0 = neon::vload1q(&src_data[src_addr]);
    uint8x16_t vqu8_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line3 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line4 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line5 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line6 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line7 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line8 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line9 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_linea = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_lineb = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_linec = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_lined = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_linee = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_linef = neon::vload1q(&src_data[src_addr += src_step]);

    uint8x16x2_t v2qu8_trn01 = neon::vtrn(vqu8_line0, vqu8_line1);
    uint8x16x2_t v2qu8_trn23 = neon::vtrn(vqu8_line2, vqu8_line3);
    uint8x16x2_t v2qu8_trn45 = neon::vtrn(vqu8_line4, vqu8_line5);
    uint8x16x2_t v2qu8_trn67 = neon::vtrn(vqu8_line6, vqu8_line7);
    uint8x16x2_t v2qu8_trn89 = neon::vtrn(vqu8_line8, vqu8_line9);
    uint8x16x2_t v2qu8_trnab = neon::vtrn(vqu8_linea, vqu8_lineb);
    uint8x16x2_t v2qu8_trncd = neon::vtrn(vqu8_linec, vqu8_lined);
    uint8x16x2_t v2qu8_trnef = neon::vtrn(vqu8_linee, vqu8_linef);

    uint16x8x2_t v2qu16_trn02 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn01.val[0]), neon::vreinterpret_u16(v2qu8_trn23.val[0]));
    uint16x8x2_t v2qu16_trn13 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn01.val[1]), neon::vreinterpret_u16(v2qu8_trn23.val[1]));
    uint16x8x2_t v2qu16_trn46 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn45.val[0]), neon::vreinterpret_u16(v2qu8_trn67.val[0]));
    uint16x8x2_t v2qu16_trn57 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn45.val[1]), neon::vreinterpret_u16(v2qu8_trn67.val[1]));
    uint16x8x2_t v2qu16_trn8a = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn89.val[0]), neon::vreinterpret_u16(v2qu8_trnab.val[0]));
    uint16x8x2_t v2qu16_trn9b = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn89.val[1]), neon::vreinterpret_u16(v2qu8_trnab.val[1]));
    uint16x8x2_t v2qu16_trnce = neon::vtrn(neon::vreinterpret_u16(v2qu8_trncd.val[0]), neon::vreinterpret_u16(v2qu8_trnef.val[0]));
    uint16x8x2_t v2qu16_trndf = neon::vtrn(neon::vreinterpret_u16(v2qu8_trncd.val[1]), neon::vreinterpret_u16(v2qu8_trnef.val[1]));

    uint32x4x2_t v2qu32_trn04 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn02.val[0]), neon::vreinterpret_u32(v2qu16_trn46.val[0]));
    uint32x4x2_t v2qu32_trn15 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn13.val[0]), neon::vreinterpret_u32(v2qu16_trn57.val[0]));
    uint32x4x2_t v2qu32_trn26 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn02.val[1]), neon::vreinterpret_u32(v2qu16_trn46.val[1]));
    uint32x4x2_t v2qu32_trn37 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn13.val[1]), neon::vreinterpret_u32(v2qu16_trn57.val[1]));
    uint32x4x2_t v2qu32_trn8c = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn8a.val[0]), neon::vreinterpret_u32(v2qu16_trnce.val[0]));
    uint32x4x2_t v2qu32_trn9d = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn9b.val[0]), neon::vreinterpret_u32(v2qu16_trndf.val[0]));
    uint32x4x2_t v2qu32_trnae = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn8a.val[1]), neon::vreinterpret_u32(v2qu16_trnce.val[1]));
    uint32x4x2_t v2qu32_trnbf = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn9b.val[1]), neon::vreinterpret_u32(v2qu16_trndf.val[1]));

    uint8x16_t vqu8_out0 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn04.val[0]), neon::vgetlow(v2qu32_trn8c.val[0])));
    uint8x16_t vqu8_out1 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn15.val[0]), neon::vgetlow(v2qu32_trn9d.val[0])));
    uint8x16_t vqu8_out2 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn26.val[0]), neon::vgetlow(v2qu32_trnae.val[0])));
    uint8x16_t vqu8_out3 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn37.val[0]), neon::vgetlow(v2qu32_trnbf.val[0])));
    uint8x16_t vqu8_out4 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn04.val[1]), neon::vgetlow(v2qu32_trn8c.val[1])));
    uint8x16_t vqu8_out5 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn15.val[1]), neon::vgetlow(v2qu32_trn9d.val[1])));
    uint8x16_t vqu8_out6 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn26.val[1]), neon::vgetlow(v2qu32_trnae.val[1])));
    uint8x16_t vqu8_out7 = neon::vreinterpret_u8(neon::vcombine(neon::vgetlow(v2qu32_trn37.val[1]), neon::vgetlow(v2qu32_trnbf.val[1])));
    uint8x16_t vqu8_out8 = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn04.val[0]), neon::vgethigh(v2qu32_trn8c.val[0])));
    uint8x16_t vqu8_out9 = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn15.val[0]), neon::vgethigh(v2qu32_trn9d.val[0])));
    uint8x16_t vqu8_outa = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn26.val[0]), neon::vgethigh(v2qu32_trnae.val[0])));
    uint8x16_t vqu8_outb = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn37.val[0]), neon::vgethigh(v2qu32_trnbf.val[0])));
    uint8x16_t vqu8_outc = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn04.val[1]), neon::vgethigh(v2qu32_trn8c.val[1])));
    uint8x16_t vqu8_outd = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn15.val[1]), neon::vgethigh(v2qu32_trn9d.val[1])));
    uint8x16_t vqu8_oute = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn26.val[1]), neon::vgethigh(v2qu32_trnae.val[1])));
    uint8x16_t vqu8_outf = neon::vreinterpret_u8(neon::vcombine(neon::vgethigh(v2qu32_trn37.val[1]), neon::vgethigh(v2qu32_trnbf.val[1])));

    neon::vstore(&dst_data[dst_addr], vqu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out7);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out8);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out9);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_outa);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_outb);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_outc);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_outd);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_oute);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_outf);
}

AURA_ALWAYS_INLINE DT_VOID Transpose16x8U8Neon(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_x * src_step + src_y;
    DT_U32 dst_addr = src_y * dst_step + src_x;

    uint8x8_t vdu8_line0 = neon::vload1(&src_data[src_addr]);
    uint8x8_t vdu8_line1 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line2 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line3 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line4 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line5 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line6 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line7 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line8 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_line9 = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_linea = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_lineb = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_linec = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_lined = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_linee = neon::vload1(&src_data[src_addr += src_step]);
    uint8x8_t vdu8_linef = neon::vload1(&src_data[src_addr += src_step]);

    uint8x8x2_t v2du8_trn01 = neon::vtrn(vdu8_line0, vdu8_line1);
    uint8x8x2_t v2du8_trn23 = neon::vtrn(vdu8_line2, vdu8_line3);
    uint8x8x2_t v2du8_trn45 = neon::vtrn(vdu8_line4, vdu8_line5);
    uint8x8x2_t v2du8_trn67 = neon::vtrn(vdu8_line6, vdu8_line7);
    uint8x8x2_t v2du8_trn89 = neon::vtrn(vdu8_line8, vdu8_line9);
    uint8x8x2_t v2du8_trnab = neon::vtrn(vdu8_linea, vdu8_lineb);
    uint8x8x2_t v2du8_trncd = neon::vtrn(vdu8_linec, vdu8_lined);
    uint8x8x2_t v2du8_trnef = neon::vtrn(vdu8_linee, vdu8_linef);

    uint16x4x2_t v2du16_trn02 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn01.val[0]), neon::vreinterpret_u16(v2du8_trn23.val[0]));
    uint16x4x2_t v2du16_trn13 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn01.val[1]), neon::vreinterpret_u16(v2du8_trn23.val[1]));
    uint16x4x2_t v2du16_trn46 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn45.val[0]), neon::vreinterpret_u16(v2du8_trn67.val[0]));
    uint16x4x2_t v2du16_trn57 = neon::vtrn(neon::vreinterpret_u16(v2du8_trn45.val[1]), neon::vreinterpret_u16(v2du8_trn67.val[1]));
    uint16x4x2_t v2du16_trn8a = neon::vtrn(neon::vreinterpret_u16(v2du8_trn89.val[0]), neon::vreinterpret_u16(v2du8_trnab.val[0]));
    uint16x4x2_t v2du16_trn9b = neon::vtrn(neon::vreinterpret_u16(v2du8_trn89.val[1]), neon::vreinterpret_u16(v2du8_trnab.val[1]));
    uint16x4x2_t v2du16_trnce = neon::vtrn(neon::vreinterpret_u16(v2du8_trncd.val[0]), neon::vreinterpret_u16(v2du8_trnef.val[0]));
    uint16x4x2_t v2du16_trndf = neon::vtrn(neon::vreinterpret_u16(v2du8_trncd.val[1]), neon::vreinterpret_u16(v2du8_trnef.val[1]));

    uint32x2x2_t v2du32_trn04 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn02.val[0]), neon::vreinterpret_u32(v2du16_trn46.val[0]));
    uint32x2x2_t v2du32_trn15 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn13.val[0]), neon::vreinterpret_u32(v2du16_trn57.val[0]));
    uint32x2x2_t v2du32_trn26 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn02.val[1]), neon::vreinterpret_u32(v2du16_trn46.val[1]));
    uint32x2x2_t v2du32_trn37 = neon::vtrn(neon::vreinterpret_u32(v2du16_trn13.val[1]), neon::vreinterpret_u32(v2du16_trn57.val[1]));
    uint32x2x2_t v2du32_trn8c = neon::vtrn(neon::vreinterpret_u32(v2du16_trn8a.val[0]), neon::vreinterpret_u32(v2du16_trnce.val[0]));
    uint32x2x2_t v2du32_trn9d = neon::vtrn(neon::vreinterpret_u32(v2du16_trn9b.val[0]), neon::vreinterpret_u32(v2du16_trndf.val[0]));
    uint32x2x2_t v2du32_trnae = neon::vtrn(neon::vreinterpret_u32(v2du16_trn8a.val[1]), neon::vreinterpret_u32(v2du16_trnce.val[1]));
    uint32x2x2_t v2du32_trnbf = neon::vtrn(neon::vreinterpret_u32(v2du16_trn9b.val[1]), neon::vreinterpret_u32(v2du16_trndf.val[1]));

    uint8x16_t vqu8_out0 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn04.val[0], v2du32_trn8c.val[0]));
    uint8x16_t vqu8_out1 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn15.val[0], v2du32_trn9d.val[0]));
    uint8x16_t vqu8_out2 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn26.val[0], v2du32_trnae.val[0]));
    uint8x16_t vqu8_out3 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn37.val[0], v2du32_trnbf.val[0]));
    uint8x16_t vqu8_out4 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn04.val[1], v2du32_trn8c.val[1]));
    uint8x16_t vqu8_out5 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn15.val[1], v2du32_trn9d.val[1]));
    uint8x16_t vqu8_out6 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn26.val[1], v2du32_trnae.val[1]));
    uint8x16_t vqu8_out7 = neon::vreinterpret_u8(neon::vcombine(v2du32_trn37.val[1], v2du32_trnbf.val[1]));

    neon::vstore(&dst_data[dst_addr], vqu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vqu8_out7);
}

AURA_ALWAYS_INLINE DT_VOID Transpose8x16U8Neon(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_x * src_step + src_y;
    DT_U32 dst_addr = src_y * dst_step + src_x;

    uint8x16_t vqu8_line0 = neon::vload1q(&src_data[src_addr]);
    uint8x16_t vqu8_line1 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line2 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line3 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line4 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line5 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line6 = neon::vload1q(&src_data[src_addr += src_step]);
    uint8x16_t vqu8_line7 = neon::vload1q(&src_data[src_addr += src_step]);

    uint8x16x2_t v2qu8_trn01 = neon::vtrn(vqu8_line0, vqu8_line1);
    uint8x16x2_t v2qu8_trn23 = neon::vtrn(vqu8_line2, vqu8_line3);
    uint8x16x2_t v2qu8_trn45 = neon::vtrn(vqu8_line4, vqu8_line5);
    uint8x16x2_t v2qu8_trn67 = neon::vtrn(vqu8_line6, vqu8_line7);

    uint16x8x2_t v2qu16_trn02 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn01.val[0]), neon::vreinterpret_u16(v2qu8_trn23.val[0]));
    uint16x8x2_t v2qu16_trn13 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn01.val[1]), neon::vreinterpret_u16(v2qu8_trn23.val[1]));
    uint16x8x2_t v2qu16_trn46 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn45.val[0]), neon::vreinterpret_u16(v2qu8_trn67.val[0]));
    uint16x8x2_t v2qu16_trn57 = neon::vtrn(neon::vreinterpret_u16(v2qu8_trn45.val[1]), neon::vreinterpret_u16(v2qu8_trn67.val[1]));

    uint32x4x2_t v2qu32_trn04 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn02.val[0]), neon::vreinterpret_u32(v2qu16_trn46.val[0]));
    uint32x4x2_t v2qu32_trn15 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn13.val[0]), neon::vreinterpret_u32(v2qu16_trn57.val[0]));
    uint32x4x2_t v2qu32_trn26 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn02.val[1]), neon::vreinterpret_u32(v2qu16_trn46.val[1]));
    uint32x4x2_t v2qu32_trn37 = neon::vtrn(neon::vreinterpret_u32(v2qu16_trn13.val[1]), neon::vreinterpret_u32(v2qu16_trn57.val[1]));

    uint8x8_t vdu8_out0 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn04.val[0]));
    uint8x8_t vdu8_out1 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn15.val[0]));
    uint8x8_t vdu8_out2 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn26.val[0]));
    uint8x8_t vdu8_out3 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn37.val[0]));
    uint8x8_t vdu8_out4 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn04.val[1]));
    uint8x8_t vdu8_out5 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn15.val[1]));
    uint8x8_t vdu8_out6 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn26.val[1]));
    uint8x8_t vdu8_out7 = neon::vreinterpret_u8(neon::vgetlow(v2qu32_trn37.val[1]));
    uint8x8_t vdu8_out8 = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn04.val[0]));
    uint8x8_t vdu8_out9 = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn15.val[0]));
    uint8x8_t vdu8_outa = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn26.val[0]));
    uint8x8_t vdu8_outb = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn37.val[0]));
    uint8x8_t vdu8_outc = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn04.val[1]));
    uint8x8_t vdu8_outd = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn15.val[1]));
    uint8x8_t vdu8_oute = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn26.val[1]));
    uint8x8_t vdu8_outf = neon::vreinterpret_u8(neon::vgethigh(v2qu32_trn37.val[1]));

    neon::vstore(&dst_data[dst_addr], vdu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out7);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out8);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out9);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_outa);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_outb);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_outc);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_outd);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_oute);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_outf);
}

AURA_ALWAYS_INLINE DT_VOID Transpose8x8U8Neon(DT_U8 *src_data, DT_U8 *dst_data, DT_U32 src_step, DT_U32 dst_step, DT_U32 src_x, DT_U32 src_y)
{
    DT_U32 src_addr = src_x * src_step + src_y;
    DT_U32 dst_addr = src_y * dst_step + src_x;

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

    uint8x8_t vdu8_out0 = neon::vreinterpret_u8(v2du32_trn04.val[0]);
    uint8x8_t vdu8_out1 = neon::vreinterpret_u8(v2du32_trn15.val[0]);
    uint8x8_t vdu8_out2 = neon::vreinterpret_u8(v2du32_trn26.val[0]);
    uint8x8_t vdu8_out3 = neon::vreinterpret_u8(v2du32_trn37.val[0]);
    uint8x8_t vdu8_out4 = neon::vreinterpret_u8(v2du32_trn04.val[1]);
    uint8x8_t vdu8_out5 = neon::vreinterpret_u8(v2du32_trn15.val[1]);
    uint8x8_t vdu8_out6 = neon::vreinterpret_u8(v2du32_trn26.val[1]);
    uint8x8_t vdu8_out7 = neon::vreinterpret_u8(v2du32_trn37.val[1]);

    neon::vstore(&dst_data[dst_addr], vdu8_out0);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out1);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out2);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out3);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out4);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out5);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out6);
    neon::vstore(&dst_data[dst_addr += dst_step], vdu8_out7);
}

template <DT_S32 C> struct TransposeU8NeonFunctor;

template <> struct TransposeU8NeonFunctor<1>
{
    constexpr static DT_S32 BLOCK_SIZE = 16;
    Status operator()(const Mat &src, Mat &dst, DT_S32 start_blk, DT_S32 end_blk)
    {
        DT_S32 start_row = start_blk * 16;
        DT_S32 end_row   = Min(end_blk * 16, dst.GetSizes().m_height);
        DT_U8 *src_data = (DT_U8 *)src.GetData();
        DT_U8 *dst_data = (DT_U8 *)dst.GetData();
        DT_S32 src_step = src.GetRowPitch() / sizeof(DT_U8);
        DT_S32 dst_step = dst.GetRowPitch() / sizeof(DT_U8);

        DT_S32 w = dst.GetSizes().m_width;
        DT_S32 h = Min(dst.GetSizes().m_height, end_row);

        DT_S32 w_align8  = (w & (-8));
        DT_S32 w_align16 = (w & (-16));
        DT_S32 h_align8  = (h & (-8));
        DT_S32 h_align16 = (h & (-16));

        DT_S32 x = 0;
        DT_S32 y = Max(0, start_row);

        for (; y < h_align16; y += 16)
        {
            x = 0;
            for (; x < w_align16; x += 16)
            {
                Transpose16x16U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }

            for (; x < w_align8; x += 8)
            {
                Transpose8x16U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }

            if (x < w)
            {
                x = w - 8;
                Transpose8x16U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }
        }

        for (; y < h_align8; y += 8)
        {
            x = 0;
            for (; x < w_align16; x += 16)
            {
                Transpose16x8U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }

            for (; x < w_align8; x += 8)
            {
                Transpose8x8U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }

            if (x < w)
            {
                x = w - 8;
                Transpose8x8U8Neon(src_data, dst_data, src_step, dst_step, x, y);
            }
        }

        if (y < end_row)
        {
            y /= 16;
            TransposeNoneFunctor<DT_U8, 1>()(src, dst, y, end_row);
        }

        return Status::OK;
    }
};

Status TransposeU8Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            TransposeU8NeonFunctor<1> op;
            ret = wp->ParallelFor(0, AURA_ALIGN(dst.GetSizes().m_height, 16) / 16, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU8C1Neon failed.");
            }
            break;
        }
        case 2:
        {
            TransposeNoneFunctor<DT_U8, 2> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU8C2 failed.");
            }
            break;
        }
        case 3:
        {
            TransposeNoneFunctor<DT_U8, 3> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU8C3 failed.");
            }
            break;
        }
        case 4:
        {
            TransposeNoneFunctor<DT_U8, 4> op;
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, op, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "TransposeU8C4 failed.");
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "channel should be <= 4");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
