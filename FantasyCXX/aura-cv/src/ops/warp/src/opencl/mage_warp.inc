#include "cl_helper.inc"

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define REMAP_SCALE_BITS_HALF                               (5)
#define REMAP_INTER_TAB_SIZE                                (32)

/*==========================================        ELEM_LENGTH        =============================================================*/
// possible ELEM_COUNTS values: 1, 2, 4 for C1
#define ELEM_COUNTS_C1                                      ELEM_COUNTS

// possible ELEM_COUNTS values: 1, 2, 4, 8 for C2
#define elem_counts_1C2                                     2
#define elem_counts_2C2                                     4
#define elem_counts_4C2                                     8
#define elem_counts_8C2                                     16

#define ELEM_COUNTS_C2_STR_STR(size)                        elem_counts_##size##C2
#define ELEM_COUNTS_C2_STR(size)                            ELEM_COUNTS_C2_STR_STR(size)
#define ELEM_COUNTS_C2                                      ELEM_COUNTS_C2_STR(ELEM_COUNTS)

// possible ELEM_COUNTS values: 1, 2, 4 for C3
#define elem_counts_1C3                                     3
#define elem_counts_2C3                                     8
#define elem_counts_4C3                                     16

#define ELEM_COUNTS_C3_STR_STR(size)                        elem_counts_##size##C3
#define ELEM_COUNTS_C3_STR(size)                            ELEM_COUNTS_C3_STR_STR(size)
#define ELEM_COUNTS_C3                                      ELEM_COUNTS_C3_STR(ELEM_COUNTS)

// possible ELEM_COUNTS values: 1, 2, 4 for C4
#define elem_counts_1C4                                     4
#define elem_counts_2C4                                     8
#define elem_counts_4C4                                     16

#define ELEM_COUNTS_C4_STR_STR(size)                        elem_counts_##size##C4
#define ELEM_COUNTS_C4_STR(size)                            ELEM_COUNTS_C4_STR_STR(size)
#define ELEM_COUNTS_C4                                      ELEM_COUNTS_C4_STR(ELEM_COUNTS)

#define ELEM_LENGTH_STR_STR(channel)                        ELEM_COUNTS_C##channel
#define ELEM_LENGTH_STR(channel)                            ELEM_LENGTH_STR_STR(channel)
#define ELEM_LENGTH                                         ELEM_LENGTH_STR(CHANNEL)

/*==========================================        ELEM_LENGTH        =============================================================*/

#define VS8                                                 VTYPE(char,  ELEM_LENGTH)
#define VS32                                                VTYPE(int,   ELEM_LENGTH)
#define VF32                                                VTYPE(float, ELEM_LENGTH)
#define VTp                                                 VTYPE(Tp,    ELEM_LENGTH)
#define V2Tp                                                VTYPE(Tp,    2)
#define V3Tp                                                VTYPE(Tp,    3)

#define VS8C1                                               VTYPE(char,  ELEM_COUNTS)
#define VS8C2                                               VTYPE(char,  ELEM_COUNTS_C2)
#define VS32C2                                              VTYPE(int,   ELEM_COUNTS_C2)
#define VF32C2                                              VTYPE(float, ELEM_COUNTS_C2)
#define VTpC2                                               VTYPE(Tp,    ELEM_COUNTS_C2)
#define VF32C4                                              VTYPE(float, ELEM_COUNTS_C4)

/*==========================================        WARP_CONVERT       =============================================================*/

#if IS_FLOAT(MapType)
#  define REMAP_CONVERT_MAP(data, type, round)              CONVERT_SAT_ROUND_STR(data, type, round)
#else
#  define REMAP_CONVERT_MAP(data, type, round)              CONVERT(data, type)
#endif

#if IS_INTEGRAL(Tp)
#  define WARP_CONVERT_SRC(data, type, round)              CONVERT_SAT_ROUND_STR(data, type, round)
#else
#  define WARP_CONVERT_SRC(data, type, round)              CONVERT(data, type)
#endif

/*==========================================        WARP_CONVERT       =============================================================*/

/*==========================================        WARP_OFFSET        =============================================================*/

#define WARP_AFFINE_OFFSET(vf32_offset, map_x_row, map_y_row, x_idx, y_idx)                                                                               \
    {                                                                                                                                                     \
        VF32C2 vf32_map_x  = VLOAD(map_x_row + (x_idx), ELEM_COUNTS_C2);                                                                                  \
        float2 v2f32_map_y = VLOAD(map_y_row + (y_idx), 2);                                                                                               \
        vf32_offset.even   = vf32_map_x.even + v2f32_map_y.even;                                                                                          \
        vf32_offset.odd    = vf32_map_x.odd  + v2f32_map_y.odd;                                                                                           \
    }

#define warp_perspective_offset_v1(vf32_offset, map_x_row, map_y_row, x_idx, y_idx)                                                                       \
    {                                                                                                                                                     \
        float3 v3f32_map_x = VLOAD(map_x_row + (x_idx) * 3, 3);                                                                                           \
        float3 v3f32_map_y = VLOAD(map_y_row + (y_idx) * 3, 3);                                                                                           \
        float  w           = v3f32_map_x.s2 + v3f32_map_y.s2;                                                                                             \
        w                  = select(1.f / w, 0.f, isequal(w, 0.f));                                                                                       \
        vf32_offset        = (v3f32_map_x.s01 + v3f32_map_y.s01) * w;                                                                                     \
    }

#define warp_perspective_offset_v2(vf32_offset, map_x_row, map_y_row, x_idx, y_idx)                                                                       \
    {                                                                                                                                                     \
        float3 v3f32_map_x0 = VLOAD(map_x_row + ((x_idx) + 0) * 3, 3);                                                                                    \
        float3 v3f32_map_x1 = VLOAD(map_x_row + ((x_idx) + 1) * 3, 3);                                                                                    \
        float3 v3f32_map_y  = VLOAD(map_y_row + (y_idx) * 3, 3);                                                                                          \
        float2 v2f32_w      = (float2)(v3f32_map_x0.s2, v3f32_map_x1.s2) + v3f32_map_y.s2;                                                                \
        v2f32_w             = select(1.f / v2f32_w, (float2)(0.f), isequal(v2f32_w, 0.f));                                                                \
        vf32_offset.s01     = (v3f32_map_x0.s01 + v3f32_map_y.s01) * v2f32_w.s0;                                                                          \
        vf32_offset.s23     = (v3f32_map_x1.s01 + v3f32_map_y.s01) * v2f32_w.s1;                                                                          \
    }

#define warp_perspective_offset_v4(vf32_offset, map_x_row, map_y_row, x_idx, y_idx)                                                                       \
    {                                                                                                                                                     \
        float3 v3f32_map_x0 = VLOAD(map_x_row + ((x_idx) + 0) * 3, 3);                                                                                    \
        float3 v3f32_map_x1 = VLOAD(map_x_row + ((x_idx) + 1) * 3, 3);                                                                                    \
        float3 v3f32_map_x2 = VLOAD(map_x_row + ((x_idx) + 2) * 3, 3);                                                                                    \
        float3 v3f32_map_x3 = VLOAD(map_x_row + ((x_idx) + 3) * 3, 3);                                                                                    \
        float3 v3f32_map_y  = VLOAD(map_y_row + (y_idx) * 3, 3);                                                                                          \
        float4 v4f32_w      = (float4)(v3f32_map_x0.s2, v3f32_map_x1.s2, v3f32_map_x2.s2, v3f32_map_x3.s2) + v3f32_map_y.s2;                              \
        v4f32_w             = select(1.f / v4f32_w, (float4)(0.f), isequal(v4f32_w, 0.f));                                                                \
        vf32_offset.s01     = (v3f32_map_x0.s01 + v3f32_map_y.s01) * v4f32_w.s0;                                                                          \
        vf32_offset.s23     = (v3f32_map_x1.s01 + v3f32_map_y.s01) * v4f32_w.s1;                                                                          \
        vf32_offset.s45     = (v3f32_map_x2.s01 + v3f32_map_y.s01) * v4f32_w.s2;                                                                          \
        vf32_offset.s67     = (v3f32_map_x3.s01 + v3f32_map_y.s01) * v4f32_w.s3;                                                                          \
    }

#define WARP_PERSPECTIVE_OFFSET_STR(vf32_offset, map_x_row, map_y_row, x_idx, y_idx, size)    warp_perspective_offset_v##size(vf32_offset, map_x_row, map_y_row, x_idx, y_idx)
#define WARP_PERSPECTIVE_OFFSET(vf32_offset, map_x_row, map_y_row, x_idx, y_idx, size)        WARP_PERSPECTIVE_OFFSET_STR(vf32_offset, map_x_row, map_y_row, x_idx, y_idx, size)

/*==========================================        WARP_OFFSET        =============================================================*/

/*==========================================        CUBIC WEIGHT       =============================================================*/

#define calc_warp_cubic_weight_v1(vf32_weight, vf32_xy)                                                                                                   \
        vf32_weight[0].s1  = vf32_xy.s0;                                                                                                                  \
        vf32_weight[1].s1  = vf32_xy.s1;                                                                                                                  \
        vf32_weight[0].s2  = 1.f - vf32_weight[0].s1;                                                                                                     \
        vf32_weight[1].s2  = 1.f - vf32_weight[1].s1;                                                                                                     \
        vf32_weight[0].s12 = 1.25f * native_powr(vf32_weight[0].s12, 3) -                                                                                 \
                             2.25f * native_powr(vf32_weight[0].s12, 2) + 1.f;                                                                            \
        vf32_weight[1].s12 = 1.25f * native_powr(vf32_weight[1].s12, 3) -                                                                                 \
                             2.25f * native_powr(vf32_weight[1].s12, 2) + 1.f;                                                                            \
        vf32_xy            = vf32_xy + 1.f;                                                                                                               \
        vf32_xy            = (-0.75f) * (native_powr(vf32_xy, 3) - 5.f * native_powr(vf32_xy, 2) + 8.f * vf32_xy - 4.f);                                  \
        vf32_weight[0].s0  = vf32_xy.s0;                                                                                                                  \
        vf32_weight[1].s0  = vf32_xy.s1;                                                                                                                  \
        vf32_weight[0].s3  = 1.f - vf32_weight[0].s0 - vf32_weight[0].s1 - vf32_weight[0].s2;                                                             \
        vf32_weight[1].s3  = 1.f - vf32_weight[1].s0 - vf32_weight[1].s1 - vf32_weight[1].s2;

#define calc_warp_cubic_weight_v2(vf32_weight, vf32_xy)                                                                                                   \
        vf32_weight[0].s15   = vf32_xy.even;                                                                                                              \
        vf32_weight[1].s15   = vf32_xy.odd;                                                                                                               \
        vf32_weight[0].s26   = 1.f - vf32_weight[0].s15;                                                                                                  \
        vf32_weight[1].s26   = 1.f - vf32_weight[1].s15;                                                                                                  \
        vf32_weight[0].s1256 = 1.25f * native_powr(vf32_weight[0].s1256, 3) -                                                                             \
                               2.25f * native_powr(vf32_weight[0].s1256, 2) + 1.f;                                                                        \
        vf32_weight[1].s1256 = 1.25f * native_powr(vf32_weight[1].s1256, 3) -                                                                             \
                               2.25f * native_powr(vf32_weight[1].s1256, 2) + 1.f;                                                                        \
        vf32_xy              = vf32_xy + 1.f;                                                                                                             \
        vf32_xy              = (-0.75f) * (native_powr(vf32_xy, 3) - 5.f * native_powr(vf32_xy, 2) + 8.f * vf32_xy - 4.f);                                \
        vf32_weight[0].s04   = vf32_xy.even;                                                                                                              \
        vf32_weight[1].s04   = vf32_xy.odd;                                                                                                               \
        vf32_weight[0].s37   = 1.f - vf32_weight[0].s04 - vf32_weight[0].s15 - vf32_weight[0].s26;                                                        \
        vf32_weight[1].s37   = 1.f - vf32_weight[1].s04 - vf32_weight[1].s15 - vf32_weight[1].s26;

#define CALC_WARP_CUBIC_WEIGHT_STR(vf32_weight, vf32_xy, elem_counts)    calc_warp_cubic_weight_v##elem_counts(vf32_weight, vf32_xy)
#define CALC_WARP_CUBIC_WEIGHT(vf32_weight, vf32_xy, elem_counts)        CALC_WARP_CUBIC_WEIGHT_STR(vf32_weight, vf32_xy, elem_counts)

/*==========================================        CUBIC WEIGHT       =============================================================*/