#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_LUT_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_LUT_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

// lut 128 8bit table, index u8
// before calling function, we need shuff the table
// HVX_Vector vd8_shuff_table = Q6_Vb_vshuff_Vb(*table++);
AURA_INLINE HVX_Vector Q6_Vb_vlut128_VbVb(HVX_Vector vu8_idx, HVX_Vector vd8_shuff_table)
{
    HVX_Vector vd8_result;
    vd8_result = Q6_Vb_vlut32_VbVbR(vu8_idx, vd8_shuff_table, 0);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, vd8_shuff_table, 1);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, vd8_shuff_table, 2);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, vd8_shuff_table, 3);
    return vd8_result;
}

// lut 256 8bit table, index u8
// before calling function, we need shuff the table
// HVX_VectorX2 mvd8_shuff_table;
// mvd8_shuff_table.val[0] =  Q6_Vb_vshuff_Vb(*table++);
// mvd8_shuff_table.val[1] =  Q6_Vb_vshuff_Vb(*table++);
AURA_INLINE HVX_Vector Q6_Vb_vlut256_VbVbX2(HVX_Vector vu8_idx, HVX_VectorX2 mvd8_shuff_table)
{
    HVX_Vector vd8_result;
    vd8_result = Q6_Vb_vlut32_VbVbR(vu8_idx, mvd8_shuff_table.val[0], 0);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[0], 1);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[0], 2);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[0], 3);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[1], 4);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[1], 5);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[1], 6);
    vd8_result = Q6_Vb_vlut32or_VbVbVbR(vd8_result, vu8_idx, mvd8_shuff_table.val[1], 7);
    return vd8_result;
}

// lut 512 8bit table, index u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX4 mvd8_shuff_table;
// mvd8_shuff_table.val[0] =  Q6_Vb_vshuff_Vb(*table++);
// mvd8_shuff_table.val[1] =  Q6_Vb_vshuff_Vb(*table++);
// mvd8_shuff_table.val[2] =  Q6_Vb_vshuff_Vb(*table++);
// mvd8_shuff_table.val[3] =  Q6_Vb_vshuff_Vb(*table++);
AURA_INLINE HVX_Vector Q6_Vb_vlut512_WhVbX4(HVX_VectorX2 mvu16_idx, HVX_VectorX4 mvd8_shuff_table)
{
    HVX_Vector vd8_result0, vd8_result1;
    HVX_Vector vd8_vcont1 = Q6_Vb_vsplat_R(1);
    HVX_VectorPair wu16_vtmp0 = Q6_W_vdeal_VVR(mvu16_idx.val[1], mvu16_idx.val[0], -1);
    HVX_VectorPred q_0 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), Q6_V_vzero()); 
    HVX_VectorX2 mvd8_shuff_tablex2 = {mvd8_shuff_table.val[0], mvd8_shuff_table.val[1]};
    vd8_result0 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result0 = Q6_V_vmux_QVV(q_0, vd8_result0, Q6_V_vzero());
    HVX_VectorPred q_1 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), vd8_vcont1);
    mvd8_shuff_tablex2.val[0] = mvd8_shuff_table.val[2];
    mvd8_shuff_tablex2.val[1] = mvd8_shuff_table.val[3];
    vd8_result1 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result1 = Q6_V_vmux_QVV(q_1, vd8_result1, Q6_V_vzero());
    return Q6_V_vor_VV(vd8_result0, vd8_result1);
}

// lut 1024 8bit table, index u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX8 v_shuff_table;
// v_shuff_table.val[0] =  Q6_Vb_vshuff_Vb(*table++);
// v_shuff_table.val[1] =  Q6_Vb_vshuff_Vb(*table++);
// ...
// v_shuff_table.val[7] =  Q6_Vb_vshuff_Vb(*table++);
AURA_INLINE HVX_Vector Q6_Vb_vlut1024_WhVbX8(HVX_VectorX2 mvu16_idx, HVX_VectorX8 mvd8_shuff_table)
{
    HVX_Vector vd8_result0, vd8_result1, vd8_vcontn;
    HVX_Vector vd8_vcont1 = Q6_Vb_vsplat_R(1);
    HVX_VectorPair wu16_vtmp0 = Q6_W_vdeal_VVR(mvu16_idx.val[1], mvu16_idx.val[0], -1);
    HVX_VectorPred q_0 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), Q6_V_vzero()); 
    HVX_VectorX2 mvd8_shuff_tablex2 = {mvd8_shuff_table.val[0], mvd8_shuff_table.val[1]};
    vd8_result0 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result0 = Q6_V_vmux_QVV(q_0, vd8_result0, Q6_V_vzero());
    HVX_VectorPred q_1 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), vd8_vcont1);
    vd8_vcontn = Q6_Vub_vadd_VubVub_sat(vd8_vcont1, vd8_vcont1);
    mvd8_shuff_tablex2.val[0] = mvd8_shuff_table.val[2];
    mvd8_shuff_tablex2.val[1] = mvd8_shuff_table.val[3];
    vd8_result1 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result1 = Q6_V_vmux_QVV(q_1, vd8_result1, Q6_V_vzero());
    vd8_result1 = Q6_V_vor_VV(vd8_result0, vd8_result1);
    HVX_VectorPred q_2 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), vd8_vcontn);
    mvd8_shuff_tablex2.val[0] = mvd8_shuff_table.val[4];
    mvd8_shuff_tablex2.val[1] = mvd8_shuff_table.val[5];
    vd8_result0 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result0 = Q6_V_vmux_QVV(q_2, vd8_result0, Q6_V_vzero());
    vd8_result1 = Q6_V_vor_VV(vd8_result0, vd8_result1);

    vd8_vcontn = Q6_Vub_vadd_VubVub_sat(vd8_vcontn, vd8_vcont1);
    HVX_VectorPred q_3 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(wu16_vtmp0), vd8_vcontn);
    mvd8_shuff_tablex2.val[0] = mvd8_shuff_table.val[6];
    mvd8_shuff_tablex2.val[1] = mvd8_shuff_table.val[7];
    vd8_result0 = Q6_Vb_vlut256_VbVbX2(Q6_V_lo_W(wu16_vtmp0), mvd8_shuff_tablex2);
    vd8_result0 = Q6_V_vmux_QVV(q_3, vd8_result0, Q6_V_vzero());
    vd8_result1 = Q6_V_vor_VV(vd8_result0, vd8_result1);

    return vd8_result1;
}

// lut 128 16 bit table, indx u8
// before calling function, we need shuff the table
// HVX_VectorX2 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// after calling function, we need to shuff the output
// Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
AURA_INLINE HVX_VectorPair Q6_Wh_vlut128_VbVhX2(HVX_Vector vu8_idx, HVX_VectorX2 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result;
    wd16_result = Q6_Wh_vlut16_VbVhR(vu8_idx, mvd16_shuff_table.val[0], 0);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 1);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 2);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 3);

    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 4);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 5);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 6);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 7);
    return wd16_result;
}

// lut 128 16 bit table, indx u8
// before calling function, we need shuff the table
// HVX_VectorX2 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
AURA_INLINE HVX_VectorPair Q6_Wh_vlutshuff128_VbVhX2(HVX_Vector vu8_idx, HVX_VectorX2 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result = Q6_Wh_vlut128_VbVhX2(vu8_idx, mvd16_shuff_table);
    wd16_result = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
    return wd16_result;
}

// lut 256 16 bit table, indx u8
// before calling function, we need shuff the table
// HVX_VectorX4 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[2] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[3] = Q6_Vh_vshuff_Vh(*table++);
// after calling function, we need to shuff the output
// Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
AURA_INLINE HVX_VectorPair Q6_Wh_vlut256_VbVhX4(HVX_Vector vu8_idx, HVX_VectorX4 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result;
    wd16_result = Q6_Wh_vlut16_VbVhR(vu8_idx, mvd16_shuff_table.val[0], 0);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 1);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 2);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[0], 3);

    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 4);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 5);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 6);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[1], 7);

    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[2], 8);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[2], 9);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[2], 10);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[2], 11);

    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[3], 12);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[3], 13);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[3], 14);
    wd16_result = Q6_Wh_vlut16or_WhVbVhR(wd16_result, vu8_idx, mvd16_shuff_table.val[3], 15);

    return wd16_result;
}

// lut 256 16 bit table, indx u8
// before calling function, we need shuff the table
// HVX_VectorX4 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[2] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[3] = Q6_Vh_vshuff_Vh(*table++);
AURA_INLINE HVX_VectorPair Q6_Wh_vlutshuff256_VbVhX4(HVX_Vector vu8_idx, HVX_VectorX4 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result = Q6_Wh_vlut256_VbVhX4(vu8_idx, mvd16_shuff_table);
    wd16_result = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
    return wd16_result;
}

// lut 512 16 bit table, indx u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX8 v_shuffe_table;
// v_shuffe_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// v_shuffe_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// ...
// v_shuffe_table.val[7] = Q6_Vh_vshuff_Vh(*table++);
// after calling function, we need to shuff the output
// Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
AURA_INLINE HVX_VectorPair Q6_Wh_vlut512_WhVhX8(HVX_VectorX2 mvu16_idx, HVX_VectorX8 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result0, wd16_result1;
    HVX_Vector vu16_vcont1 = Q6_Vh_vsplat_R(1);
    HVX_VectorPair wu16_vtmp0 = Q6_W_vdeal_VVR(mvu16_idx.val[1], mvu16_idx.val[0], -1);
    HVX_VectorPair wu16_index_hi = Q6_Wuh_vzxt_Vub(Q6_V_hi_W(wu16_vtmp0));
    HVX_VectorPred q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), Q6_V_vzero()); 
    HVX_VectorPred q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), Q6_V_vzero()); 

    HVX_VectorX4 mvd16_shuff_table0 = {mvd16_shuff_table.val[0], mvd16_shuff_table.val[1], mvd16_shuff_table.val[2], mvd16_shuff_table.val[3]};
    wd16_result0 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table0);
    HVX_Vector wd16_vtmp1 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result0), Q6_V_vzero());
    HVX_Vector wd16_vtmp2 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result0), Q6_V_vzero());

    q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), vu16_vcont1); 
    q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), vu16_vcont1); 

    HVX_VectorX4 mvd16_shuff_table1 = {mvd16_shuff_table.val[4], mvd16_shuff_table.val[5], mvd16_shuff_table.val[6], mvd16_shuff_table.val[7]};
    wd16_result1 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table1);
    HVX_Vector wd16_vtmp3 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result1), Q6_V_vzero());
    HVX_Vector wd16_vtmp4 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result1), Q6_V_vzero());
    
    return Q6_W_vcombine_VV(Q6_V_vor_VV(wd16_vtmp2, wd16_vtmp4), Q6_V_vor_VV(wd16_vtmp1, wd16_vtmp3));
}

// lut 512 16 bit table, indx u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX8 v_shuffe_table;
// v_shuffe_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// v_shuffe_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// ...
// v_shuffe_table.val[7] = Q6_Vh_vshuff_Vh(*table++);
AURA_INLINE HVX_VectorPair Q6_Wh_vlutshuff512_WhVhX8(HVX_VectorX2 mvu16_idx, HVX_VectorX8 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result = Q6_Wh_vlut512_WhVhX8(mvu16_idx, mvd16_shuff_table);
    wd16_result = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
    return wd16_result;
}

// lut 1024 16 bit table, indx u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX16 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// ...
// mvd16_shuff_table.val[15] = Q6_Vh_vshuff_Vh(*table++);
// after calling function, we need to shuff the output
// Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
AURA_INLINE HVX_VectorPair Q6_Vh_vlut1024_WhVhX16(HVX_VectorX2 mvu16_idx, HVX_VectorX16 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result0, wd16_result1;
    HVX_Vector vu16_vcontn;
    HVX_Vector vu16_vcont1 = Q6_Vh_vsplat_R(1);
    HVX_VectorPair wu16_vtmp0 = Q6_W_vdeal_VVR(mvu16_idx.val[1], mvu16_idx.val[0], -1);

    HVX_VectorPair wu16_index_hi = Q6_Wuh_vzxt_Vub(Q6_V_hi_W(wu16_vtmp0));
    HVX_VectorPred q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), Q6_V_vzero()); 
    HVX_VectorPred q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), Q6_V_vzero()); 

    HVX_VectorX4 mvd16_shuff_table0 = {mvd16_shuff_table.val[0], mvd16_shuff_table.val[1], mvd16_shuff_table.val[2], mvd16_shuff_table.val[3]};
    wd16_result0 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table0);
    HVX_Vector vd16_vtmp1 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result0), Q6_V_vzero());
    HVX_Vector vd16_vtmp2 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result0), Q6_V_vzero());

    q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), vu16_vcont1); 
    q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), vu16_vcont1); 

    vu16_vcontn = Q6_Vub_vadd_VubVub_sat(vu16_vcont1, vu16_vcont1);
    HVX_VectorX4 mvd16_shuff_table1 = {mvd16_shuff_table.val[4], mvd16_shuff_table.val[5], mvd16_shuff_table.val[6], mvd16_shuff_table.val[7]};   
    wd16_result1 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table1);
    HVX_Vector vd16_vtmp3 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result1), Q6_V_vzero());
    HVX_Vector vd16_vtmp4 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result1), Q6_V_vzero());
    vd16_vtmp3 = Q6_V_vor_VV(vd16_vtmp1, vd16_vtmp3);
    vd16_vtmp4 = Q6_V_vor_VV(vd16_vtmp2, vd16_vtmp4);

    q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), vu16_vcontn); 
    q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), vu16_vcontn); 

    HVX_VectorX4 mvd16_shuff_table2 = {mvd16_shuff_table.val[8], mvd16_shuff_table.val[9], mvd16_shuff_table.val[10], mvd16_shuff_table.val[11]};
    wd16_result0 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table2);
    vd16_vtmp1 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result0), Q6_V_vzero());
    vd16_vtmp2 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result0), Q6_V_vzero());
    vd16_vtmp3 = Q6_V_vor_VV(vd16_vtmp1, vd16_vtmp3);
    vd16_vtmp4 = Q6_V_vor_VV(vd16_vtmp2, vd16_vtmp4);

    vu16_vcontn = Q6_Vub_vadd_VubVub_sat(vu16_vcontn, vu16_vcont1);
    q_0 = Q6_Q_vcmp_eq_VhVh(Q6_V_lo_W(wu16_index_hi), vu16_vcontn); 
    q_1 = Q6_Q_vcmp_eq_VhVh(Q6_V_hi_W(wu16_index_hi), vu16_vcontn); 
    HVX_VectorX4 mvd16_shuff_table3 = {mvd16_shuff_table.val[12], mvd16_shuff_table.val[13], mvd16_shuff_table.val[14], mvd16_shuff_table.val[15]};
    wd16_result0 = Q6_Wh_vlut256_VbVhX4(Q6_V_lo_W(wu16_vtmp0), mvd16_shuff_table3);
    vd16_vtmp1 = Q6_V_vmux_QVV(q_0, Q6_V_lo_W(wd16_result0), Q6_V_vzero());
    vd16_vtmp2 = Q6_V_vmux_QVV(q_1, Q6_V_hi_W(wd16_result0), Q6_V_vzero());

    return Q6_W_vcombine_VV(Q6_V_vor_VV(vd16_vtmp2, vd16_vtmp4), Q6_V_vor_VV(vd16_vtmp1, vd16_vtmp3));
}

// lut 1024 16 bit table, indx u16
// HVX_VectorX2 mvu16_idx;
// mvu16_idx.val[0] = *idx++;
// mvu16_idx.val[1] = *idx++;
// before calling function, we need shuff the table
// HVX_VectorX16 mvd16_shuff_table;
// mvd16_shuff_table.val[0] = Q6_Vh_vshuff_Vh(*table++);
// mvd16_shuff_table.val[1] = Q6_Vh_vshuff_Vh(*table++);
// ...
// mvd16_shuff_table.val[15] = Q6_Vh_vshuff_Vh(*table++);
AURA_INLINE HVX_VectorPair Q6_Vh_vlutshuff1024_WhVhX16(HVX_VectorX2 mvu16_idx, HVX_VectorX16 mvd16_shuff_table)
{
    HVX_VectorPair wd16_result = Q6_Vh_vlut1024_WhVhX16(mvu16_idx, mvd16_shuff_table);
    wd16_result = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_result), Q6_V_lo_W(wd16_result), -2);
    return wd16_result;
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_LUT_HPP__