#include "device/include/hexagon_instructions_unit_test.hpp"

NEW_TESTCASE(runtime_hexagon_instructions_lut_test)
{
    Status ret = Status::OK;
    Context *ctx = UnitTest::GetInstance()->GetContext();

    // Q6_Vb_vlut128_VbVb
    {
        MI_U8 table[128];
        MI_U8 idx[128];
        MI_U8 dst[128];
        MI_U8 ref[128];

        for (MI_S32 i = 0; i < 128; i++)
        {
            table[i] = i;
            idx[i] = 127 - i;
            ref[i] = 127 - i;
        }

        HVX_Vector vu8_table = vmemu(table);
        HVX_Vector vu8_idx = vmemu(idx);
        HVX_Vector vu8_shuffe_table = Q6_Vb_vshuff_Vb(vu8_table);
        vmemu(dst) = Q6_Vb_vlut128_VbVb(vu8_idx, vu8_shuffe_table);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 128);
    }

    // // Q6_Vb_vlut256_VbVbX2
    {
        MI_U8 table[256];
        MI_U8 idx[128];
        MI_U8 dst[128];
        MI_U8 ref[128];

        for (MI_S32 i = 0; i < 256; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 2;
            ref[i] = table[idx[i]];
        }

        HVX_Vector vu8_idx = vmemu(idx);

        HVX_VectorX2 mvd8_shuff_table;
        mvd8_shuff_table.val[0] = Q6_Vb_vshuff_Vb(vmemu(table));
        mvd8_shuff_table.val[1] = Q6_Vb_vshuff_Vb(vmemu(table + AURA_HVLEN));

        vmemu(dst) = Q6_Vb_vlut256_VbVbX2(vu8_idx, mvd8_shuff_table);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 128);
    }

    // Q6_Vb_vlut512_WhVbX4
    {
        MI_U8 table[512];
        MI_U16 idx[128];
        MI_U8 dst[128];
        MI_U8 ref[128];

        for (MI_S32 i = 0; i < 512; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 4;
            ref[i] = table[idx[i]];
        }

        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX4 mvd8_shuff_table;
        mvd8_shuff_table.val[0] =  Q6_Vb_vshuff_Vb(vmemu(table));
        mvd8_shuff_table.val[1] =  Q6_Vb_vshuff_Vb(vmemu(table + AURA_HVLEN));
        mvd8_shuff_table.val[2] =  Q6_Vb_vshuff_Vb(vmemu(table + 2 * AURA_HVLEN));
        mvd8_shuff_table.val[3] =  Q6_Vb_vshuff_Vb(vmemu(table + 3 * AURA_HVLEN));

        vmemu(dst) = Q6_Vb_vlut512_WhVbX4(mvu16_idx, mvd8_shuff_table);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 128);
    }

    // Q6_Vb_vlut1024_WhVbX8
    {
        MI_U8 table[1024];
        MI_U16 idx[128];
        MI_U8 dst[128];
        MI_U8 ref[128];

        for (MI_S32 i = 0; i < 1024; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 8;
            ref[i] = table[idx[i]];
        }

        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX8 mvd8_shuff_table;
        for (MI_S32 i = 0; i < 8; i++)
        {
            mvd8_shuff_table.val[i] =  Q6_Vb_vshuff_Vb(vmemu(table + i * AURA_HVLEN));
        }
        vmemu(dst) = Q6_Vb_vlut1024_WhVbX8(mvu16_idx, mvd8_shuff_table);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 128);
    }

    // Q6_Wh_vlut128_VbVhX2
    {
        MI_U16 table[128];
        MI_U8 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 128; i++)
        {
            table[i] = i;
            idx[i] = 127 - i;
            ref[i] = 127 - i;
        }

        HVX_Vector vu8_idx = vmemu(idx);

        HVX_VectorX2 mvd16_shuff_table;
        mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(vmemu(table));
        mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2));

        HVX_VectorPair wd16_dst = Q6_Wh_vlut128_VbVhX2(vu8_idx, mvd16_shuff_table);
        wd16_dst = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_dst), Q6_V_lo_W(wd16_dst), -2);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Wh_vlutshuff128_VbVhX2
    {
        MI_U16 table[128];
        MI_U8 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 128; i++)
        {
            table[i] = i;
            idx[i] = 127 - i;
            ref[i] = 127 - i;
        }

        HVX_Vector vu8_idx = vmemu(idx);

        HVX_VectorX2 mvd16_shuff_table;
        mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(vmemu(table));
        mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2));

        HVX_VectorPair wd16_dst = Q6_Wh_vlutshuff128_VbVhX2(vu8_idx, mvd16_shuff_table);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Wh_vlut256_VbVhX4
    {
        MI_U16 table[256];
        MI_U8 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 256; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 2;
            ref[i] = table[idx[i]];
        }

        HVX_Vector vu8_idx = vmemu(idx);

        HVX_VectorX4 mvd16_shuff_table;
        mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(vmemu(table));
        mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2));
        mvd16_shuff_table.val[2] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN));
        mvd16_shuff_table.val[3] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * 3));

        HVX_VectorPair wd16_dst = Q6_Wh_vlut256_VbVhX4(vu8_idx, mvd16_shuff_table);
        wd16_dst = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_dst), Q6_V_lo_W(wd16_dst), -2);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Wh_vlutshuff256_VbVhX4
    {
        MI_U16 table[256];
        MI_U8 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 256; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 2;
            ref[i] = table[idx[i]];
        }

        HVX_Vector vu8_idx = vmemu(idx);

        HVX_VectorX4 mvd16_shuff_table;
        mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(vmemu(table));
        mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2));
        mvd16_shuff_table.val[2] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN));
        mvd16_shuff_table.val[3] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * 3));

        HVX_VectorPair wd16_dst = Q6_Wh_vlutshuff256_VbVhX4(vu8_idx, mvd16_shuff_table);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Wh_vlut512_WhVhX8
    {
        MI_U16 table[512];
        MI_U16 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 512; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 4;
            ref[i] = table[idx[i]];
        }

        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX8 mvd16_shuff_table;
        for (MI_S32 i = 0; i < 8; i++)
        {
            mvd16_shuff_table.val[i] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * i));
        }

        HVX_VectorPair wd16_dst = Q6_Wh_vlut512_WhVhX8(mvu16_idx, mvd16_shuff_table);
        wd16_dst = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_dst), Q6_V_lo_W(wd16_dst), -2);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Wh_vlutshuff512_WhVhX8
    {
        MI_U16 table[512];
        MI_U16 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 512; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 4;
            ref[i] = table[idx[i]];
        }


        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX8 mvd16_shuff_table;
        for (MI_S32 i = 0; i < 8; i++)
        {
            mvd16_shuff_table.val[i] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * i));
        }

        HVX_VectorPair wd16_dst = Q6_Wh_vlutshuff512_WhVhX8(mvu16_idx, mvd16_shuff_table);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Vh_vlut1024_WhVhX16
    {
        MI_U16 table[1024];
        MI_U16 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 1024; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 8;
            ref[i] = table[idx[i]];
        }

        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX16 mvd16_shuff_table;
        for (MI_S32 i = 0; i < 16; i++)
        {
            mvd16_shuff_table.val[i] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * i));
        }

        HVX_VectorPair wd16_dst = Q6_Vh_vlut1024_WhVhX16(mvu16_idx, mvd16_shuff_table);
        wd16_dst = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_dst), Q6_V_lo_W(wd16_dst), -2);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    // Q6_Vh_vlutshuff1024_WhVhX16
    {
        MI_U16 table[1024];
        MI_U16 idx[128];
        MI_U16 dst[128];
        MI_U16 ref[128];

        for (MI_S32 i = 0; i < 1024; i++)
        {
            table[i] = i;
        }
        for (MI_S32 i = 0; i < 128; i++)
        {
            idx[i] = i * 8;
            ref[i] = table[idx[i]];
        }

        HVX_VectorX2 mvu16_idx;
        mvu16_idx.val[0] = vmemu(idx);
        mvu16_idx.val[1] = vmemu(idx + AURA_HVLEN / 2);

        HVX_VectorX16 mvd16_shuff_table;
        for (MI_S32 i = 0; i < 16; i++)
        {
            mvd16_shuff_table.val[i] = Q6_Vh_vshuff_Vh(vmemu(table + AURA_HVLEN / 2 * i));
        }

        HVX_VectorPair wd16_dst = Q6_Vh_vlutshuff1024_WhVhX16(mvu16_idx, mvd16_shuff_table);
        vmemu(dst) = Q6_V_lo_W(wd16_dst);
        vmemu(dst + AURA_HVLEN / 2) = Q6_V_hi_W(wd16_dst);

        ret |= CHECK_CMP_VECTOR(ctx, ref, dst, 256);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}