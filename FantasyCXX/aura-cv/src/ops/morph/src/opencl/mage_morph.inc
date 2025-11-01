#include "cl_helper.inc"

#define VTp                                                 VTYPE(Tp, ELEM_COUNTS)
#define V2Tp                                                VTYPE(Tp, 2)
#define V3Tp                                                VTYPE(Tp, 3)

#define morph_replace1_vec1(data, val)                      (data.s0 = val)
#define morph_replace1_vec2(data, val)                      (data.s1 = val)
#define morph_replace1_vec3(data, val)                      (data.s2 = val)
#define morph_replace1_vec4(data, val)                      (data.s3 = val)
#define morph_replace1_vec8(data, val)                      (data.s7 = val)
#define morph_replace1_vec16(data, val)                     (data.sf = val)

#define MORPH_REPLACE_END_STR(data, length, num, val)       morph_replace##num##_vec##length(data, val)
#define MORPH_REPLACE_END(data, length, num, val)           MORPH_REPLACE_END_STR(data, length, num, val)

#define MORPH_STR_ERODE(x, y)                               min(x, y)
#define MORPH_STR_DILATE(x, y)                              max(x, y)
#define MORPH_STR(type, x, y)                               MORPH_STR_##type(x, y)
#define MORPH_OP(type, x, y)                                MORPH_STR(type, x, y)
