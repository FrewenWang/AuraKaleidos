// ====================================
//           CL compiler option
// ====================================
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ====================================
//           CL vector decl
// ====================================
#define char1                                       char
#define uchar1                                      uchar
#define short1                                      short
#define ushort1                                     ushort
#define int1                                        int
#define uint1                                       uint
#define long1                                       long
#define ulong1                                      ulong
#define half1                                       half
#define float1                                      float

#define VTYPE_STR(type, size)                       type##size
#define VTYPE(type, size)                           VTYPE_STR(type, size)

// ====================================
//            Types
// ====================================
struct Scalar
{
    float val[4];
};

// ====================================
//           Load & Store
// ====================================
// 用于从内存地址 addr 的 offset 偏移处加载 单个标量元素
#define vload1(offset, addr)                        (addr)[offset]
// 通过宏拼接（## 运算符）动态生成不同向量宽度的加载函数名（如 vload4、vload8），并从内存地址 addr的0 偏移 加载 size 个元素。
#define VLOAD_STR(addr, size)                       vload##size(0, addr)
// 进一步封装 VLOAD_STR，提供更统一的宏接口，使调用者只需关注 addr（内存地址）和 size（向量宽度）参数。
#define VLOAD(addr, size)                           VLOAD_STR(addr, size)

#define vstore1(data, offset, addr)                 (addr)[offset] = data
#define VSTORE_STR(data, addr, size)                vstore##size(data, 0, addr)
#define VSTORE(data, addr, size)                    VSTORE_STR(data, addr, size)

#define read_iaura_char                             read_iaurai
#define read_iaura_short                            read_iaurai
#define read_iaura_int                              read_iaurai
#define read_iaura_uchar                            read_iauraui
#define read_iaura_ushort                           read_iauraui
#define read_iaura_uint                             read_iauraui
#define read_iaura_half                             read_iaurah
#define read_iaura_float                            read_iauraf

#define READ_IAURA_STR(type)                        read_iaura_##type
#define READ_IAURA(type)                            READ_IAURA_STR(type)

#define write_iaura_char                            write_iaurai
#define write_iaura_short                           write_iaurai
#define write_iaura_int                             write_iaurai
#define write_iaura_uchar                           write_iauraui
#define write_iaura_ushort                          write_iauraui
#define write_iaura_uint                            write_iauraui
#define write_iaura_half                            write_iaurah
#define write_iaura_float                           write_iauraf

#define WRITE_IAURA_STR(type)                       write_iaura_##type
#define WRITE_IAURA(type)                           WRITE_IAURA_STR(type)

// ====================================
//            Convert
// ====================================
#define convert_to_promote_type_from_uchar(size)    ushort##size
#define convert_to_promote_type_from_char(size)     short##size
#define convert_to_promote_type_from_ushort(size)   uint##size
#define convert_to_promote_type_from_short(size)    int##size
#define convert_to_promote_type_from_uint(size)     ulong##size
#define convert_to_promote_type_from_int(size)      long##size
#define convert_to_promote_type_from_ulong(size)    ulong##size
#define convert_to_promote_type_from_long(size)     long##size
#define convert_to_promote_type_from_half(size)     float##size
#define convert_to_promote_type_from_float(size)    float##size

#define PROMOTE_VTYPE_STR(type, size)               convert_to_promote_type_from_##type(size)
#define PROMOTE_VTYPE(type, size)                   PROMOTE_VTYPE_STR(type, size)
#define PROMOTE_STYPE(type)                         PROMOTE_VTYPE_STR(type, 1)

#define convert_float_sat                           convert_float
#define convert_float1_sat                          convert_float
#define convert_float2_sat                          convert_float2
#define convert_float3_sat                          convert_float3
#define convert_float4_sat                          convert_float4
#define convert_float8_sat                          convert_float8
#define convert_float16_sat                         convert_float16
#define convert_half_sat                            convert_float
#define convert_half1_sat                           convert_half
#define convert_half2_sat                           convert_half2
#define convert_half3_sat                           convert_half3
#define convert_half4_sat                           convert_half4
#define convert_half8_sat                           convert_half8
#define convert_half16_sat                          convert_half16

#define convert_float1                              convert_float
#define convert_half1                               convert_half
#define convert_char1                               convert_char
#define convert_uchar1                              convert_uchar
#define convert_short1                              convert_short
#define convert_ushort1                             convert_ushort
#define convert_int1                                convert_int
#define convert_uint1                               convert_uint
#define convert_long1                               convert_long
#define convert_ulong1                              convert_ulong

#define convert_char1_sat                           convert_char_sat
#define convert_uchar1_sat                          convert_uchar_sat
#define convert_short1_sat                          convert_short_sat
#define convert_ushort1_sat                         convert_ushort_sat
#define convert_int1_sat                            convert_int_sat
#define convert_uint1_sat                           convert_uint_sat
#define convert_long1_sat                           convert_long_sat
#define convert_ulong1_sat                          convert_ulong_sat

#define as_float1                                   as_float
#define as_half1                                    as_half
#define as_char1                                    as_char
#define as_uchar1                                   as_uchar
#define as_short1                                   as_short
#define as_ushort1                                  as_ushort
#define as_int1                                     as_int
#define as_uint1                                    as_uint
#define as_long1                                    as_long
#define as_ulong1                                   as_ulong

#define CONVERT_STR(data, type)                     (convert_##type((data)))
#define CONVERT(data, type)                         CONVERT_STR(data, type)

#define CONVERT_SAT_STR(data, type)                 (convert_##type##_sat((data)))
#define CONVERT_SAT(data, type)                     CONVERT_SAT_STR(data, type)

#define CONVERT_SAT_ROUND_STR(data, type, round)    (convert_##type##_sat_##round((data)))
#define CONVERT_SAT_ROUND(data, type, round)        CONVERT_SAT_ROUND_STR(data, type, round)  // rounding mode here is rte/rtz/rtp/rtn

#define REINTERPRET_STR(data, type, size)           as_##type##size(data)
#define REINTERPRET(data, type, size)               REINTERPRET_STR(data, type, size)

#define convert_to_iaura_type_from_char4(data)      CONVERT(data, int4)
#define convert_to_iaura_type_from_short4(data)     CONVERT(data, int4)
#define convert_to_iaura_type_from_int4(data)       (data)
#define convert_to_iaura_type_from_uchar4(data)     CONVERT(data, uint4)
#define convert_to_iaura_type_from_ushort4(data)    CONVERT(data, uint4)
#define convert_to_iaura_type_from_uint4(data)      (data)
#define convert_to_iaura_type_from_half4(data)      (data)
#define convert_to_iaura_type_from_float4(data)     (data)

#define CONVERT_WI_STR(data, type)                  convert_to_iaura_type_from_##type(data)
#define CONVERT_WI(data, type)                      CONVERT_WI_STR(data, type)
// ====================================
//            Traits
// ====================================
#define float_is_float                              true
#define half_is_float                               false
#define uchar_is_float                              false
#define char_is_float                               false
#define ushort_is_float                             false
#define short_is_float                              false
#define uint_is_float                               false
#define int_is_float                                false
#define ulong_is_float                              false
#define long_is_float                               false

#define IS_FLOAT_STR(type)                          type##_is_float
#define IS_FLOAT(type)                              IS_FLOAT_STR(type)

#define float_is_half                               false
#define half_is_half                                true
#define uchar_is_half                               false
#define char_is_half                                false
#define ushort_is_half                              false
#define short_is_half                               false
#define uint_is_half                                false
#define int_is_half                                 false
#define ulong_is_half                               false
#define long_is_half                                false

#define IS_HALF_STR(type)                           type##_is_half
#define IS_HALF(type)                               IS_HALF_STR(type)

#define IS_INTEGRAL(type)                           (!(IS_FLOAT_STR(type) || IS_HALF_STR(type)))

#define float_is_unsigned                           false
#define half_is_unsigned                            false
#define uchar_is_unsigned                           true
#define char_is_unsigned                            false
#define ushort_is_unsigned                          true
#define short_is_unsigned                           false
#define uint_is_unsigned                            true
#define int_is_unsigned                             false
#define ulong_is_unsigned                           true
#define long_is_unsigned                            false

#define IS_UNSIGNED_STR(type)                       type##_is_unsigned
#define IS_UNSIGNED(type)                           IS_UNSIGNED_STR(type)

#define IS_SIGNED_STR(type)                         (!(type##_is_unsigned))
#define IS_SIGNED(type)                             IS_SIGNED_STR(type)

#define uchar_is_uchar                              true
#define half_is_uchar                               false
#define float_is_uchar                              false
#define char_is_uchar                               false
#define ushort_is_uchar                             false
#define short_is_uchar                              false
#define uint_is_uchar                               false
#define int_is_uchar                                false
#define ulong_is_uchar                              false
#define long_is_uchar                               false

#define IS_UCHAR_STR(type)                          type##_is_uchar
#define IS_UCHAR(type)                              IS_UCHAR_STR(type)

#define char_is_char                                true
#define half_is_char                                false
#define float_is_char                               false
#define uchar_is_char                               false
#define ushort_is_char                              false
#define short_is_char                               false
#define uint_is_char                                false
#define int_is_char                                 false
#define ulong_is_char                               false
#define long_is_char                                false

#define IS_CHAR_STR(type)                           type##_is_char
#define IS_CHAR(type)                               IS_CHAR_STR(type)

#define ushort_is_ushort                            true
#define half_is_ushort                              false
#define float_is_ushort                             false
#define uchar_is_ushort                             false
#define char_is_ushort                              false
#define short_is_ushort                             false
#define uint_is_ushort                              false
#define int_is_ushort                               false
#define ulong_is_ushort                             false
#define long_is_ushort                              false

#define IS_USHORT_STR(type)                         type##_is_ushort
#define IS_USHORT(type)                             IS_USHORT_STR(type)

#define short_is_short                              true
#define half_is_short                               false
#define float_is_short                              false
#define uchar_is_short                              false
#define char_is_short                               false
#define ushort_is_short                             false
#define uint_is_short                               false
#define int_is_short                                false
#define ulong_is_short                              false
#define long_is_short                               false

#define IS_SHORT_STR(type)                          type##_is_short
#define IS_SHORT(type)                              IS_SHORT_STR(type)

#define uint_is_uint                                true
#define half_is_uint                                false
#define float_is_uint                               false
#define uchar_is_uint                               false
#define char_is_uint                                false
#define ushort_is_uint                              false
#define short_is_uint                               false
#define int_is_uint                                 false
#define ulong_is_uint                               false
#define long_is_uint                                false

#define IS_UINT_STR(type)                           type##_is_uint
#define IS_UINT(type)                               IS_UINT_STR(type)

#define int_is_int                                  true
#define half_is_int                                 false
#define float_is_int                                false
#define uchar_is_int                                false
#define char_is_int                                 false
#define ushort_is_int                               false
#define short_is_int                                false
#define uint_is_int                                 false
#define ulong_is_int                                false
#define long_is_int                                 false

#define IS_INT_STR(type)                            type##_is_int
#define IS_INT(type)                                IS_INT_STR(type)

// ====================================
//         Vector Operation
// ====================================
#define rev_1(data)                                 ((data))
#define rev_2(data)                                 ((data).s10)
#define rev_3(data)                                 ((data).s210)
#define rev_4(data)                                 ((data).s3210)
#define rev_8(data)                                 ((data).s76543210)
#define rev_16(data)                                ((data).sFEDCBA9876543210)

#define REVERSE_STR(data, s)                        rev_##s((data))
#define REVERSE(data, s)                            REVERSE_STR(data, s)

#define rot_r1_0(data)                              ((data))
#define rot_r1_1(data)                              ((data))

#define rot_r2_0(data)                              ((data))
#define rot_r2_1(data)                              ((data).s10)
#define rot_r2_2(data)                              ((data))

#define rot_r3_0(data)                              ((data))
#define rot_r3_1(data)                              ((data).s201)
#define rot_r3_2(data)                              ((data).s120)
#define rot_r3_3(data)                              ((data))

#define rot_r4_0(data)                              ((data))
#define rot_r4_1(data)                              ((data).s3012)
#define rot_r4_2(data)                              ((data).s2301)
#define rot_r4_3(data)                              ((data).s1230)
#define rot_r4_4(data)                              ((data))

#define rot_r8_0(data)                              ((data))
#define rot_r8_1(data)                              ((data).s70123456)
#define rot_r8_2(data)                              ((data).s67012345)
#define rot_r8_3(data)                              ((data).s56701234)
#define rot_r8_4(data)                              ((data).s45670123)
#define rot_r8_5(data)                              ((data).s34567012)
#define rot_r8_6(data)                              ((data).s23456701)
#define rot_r8_7(data)                              ((data).s12345670)
#define rot_r8_8(data)                              ((data))

#define rot_r16_0(data)                             ((data))
#define rot_r16_1(data)                             ((data).sF0123456789ABCDE)
#define rot_r16_2(data)                             ((data).sEF0123456789ABCD)
#define rot_r16_3(data)                             ((data).sDEF0123456789ABC)
#define rot_r16_4(data)                             ((data).sCDEF0123456789AB)
#define rot_r16_5(data)                             ((data).sBCDEF0123456789A)
#define rot_r16_6(data)                             ((data).sABCDEF0123456789)
#define rot_r16_7(data)                             ((data).s9ABCDEF012345678)
#define rot_r16_8(data)                             ((data).s89ABCDEF01234567)
#define rot_r16_9(data)                             ((data).s789ABCDEF0123456)
#define rot_r16_10(data)                            ((data).s6789ABCDEF012345)
#define rot_r16_11(data)                            ((data).s56789ABCDEF01234)
#define rot_r16_12(data)                            ((data).s456789ABCDEF0123)
#define rot_r16_13(data)                            ((data).s3456789ABCDEF012)
#define rot_r16_14(data)                            ((data).s23456789ABCDEF01)
#define rot_r16_15(data)                            ((data).s123456789ABCDEF0)
#define rot_r16_16(data)                            ((data))

#define ROT_R_STR(data, s, n)                       rot_r##s##_##n(data)
#define ROT_R(data, s, n)                           ROT_R_STR(data, s, n)

#define rot_l1_0(data)                              ((data))
#define rot_l1_1(data)                              ((data))

#define rot_l2_0(data)                              ((data))
#define rot_l2_1(data)                              ((data).s10)
#define rot_l2_2(data)                              ((data))

#define rot_l3_0(data)                              ((data))
#define rot_l3_1(data)                              ((data).s120)
#define rot_l3_2(data)                              ((data).s201)
#define rot_l3_3(data)                              ((data))

#define rot_l4_0(data)                              ((data))
#define rot_l4_1(data)                              ((data).s1230)
#define rot_l4_2(data)                              ((data).s2301)
#define rot_l4_3(data)                              ((data).s3012)
#define rot_l4_4(data)                              ((data))

#define rot_l8_0(data)                              ((data))
#define rot_l8_1(data)                              ((data).s12345670)
#define rot_l8_2(data)                              ((data).s23456701)
#define rot_l8_3(data)                              ((data).s34567012)
#define rot_l8_4(data)                              ((data).s45670123)
#define rot_l8_5(data)                              ((data).s56701234)
#define rot_l8_6(data)                              ((data).s67012345)
#define rot_l8_7(data)                              ((data).s70123456)
#define rot_l8_8(data)                              ((data))

#define rot_l16_0(data)                             ((data))
#define rot_l16_1(data)                             ((data).s123456789ABCDEF0)
#define rot_l16_2(data)                             ((data).s23456789ABCDEF01)
#define rot_l16_3(data)                             ((data).s3456789ABCDEF012)
#define rot_l16_4(data)                             ((data).s456789ABCDEF0123)
#define rot_l16_5(data)                             ((data).s56789ABCDEF01234)
#define rot_l16_6(data)                             ((data).s6789ABCDEF012345)
#define rot_l16_7(data)                             ((data).s789ABCDEF0123456)
#define rot_l16_8(data)                             ((data).s89ABCDEF01234567)
#define rot_l16_9(data)                             ((data).s9ABCDEF012345678)
#define rot_l16_10(data)                            ((data).sABCDEF0123456789)
#define rot_l16_11(data)                            ((data).sBCDEF0123456789A)
#define rot_l16_12(data)                            ((data).sCDEF0123456789AB)
#define rot_l16_13(data)                            ((data).sDEF0123456789ABC)
#define rot_l16_14(data)                            ((data).sEF0123456789ABCD)
#define rot_l16_15(data)                            ((data).sF0123456789ABCDE)
#define rot_l16_16(data)                            ((data))

#define ROT_L_STR(data, s, n)                       rot_l##s##_##n(data)
#define ROT_L(data, s, n)                           ROT_L_STR(data, s, n)

#define v_offs_1(dt)                                (dt##1)(0)
#define v_offs_2(dt)                                (dt##2)(0, 1)
#define v_offs_3(dt)                                (dt##3)(0, 1, 2)
#define v_offs_4(dt)                                (dt##4)(0, 1, 2, 3)
#define v_offs_8(dt)                                (dt##8)(0, 1, 2, 3, 4, 5, 6, 7)
#define v_offs_16(dt)                               (dt##16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

#define VEC_OFFS_STR(dt, s)                         v_offs_##s(dt)
#define VEC_OFFS(dt, s)                             VEC_OFFS_STR(dt, s)

// ====================================
//               Reduce
// ====================================
#define sum_reduce_1(data)                          (data)
#define sum_reduce_2(data)                          ((data).s0) + ((data).s1)
#define sum_reduce_3(data)                          (sum_reduce_2((data).s01) + ((data).s2))
#define sum_reduce_4(data)                          (sum_reduce_2((data).s01)       + sum_reduce_2((data).s23))
#define sum_reduce_8(data)                          (sum_reduce_4((data).s0123)     + sum_reduce_4((data).s4567))
#define sum_reduce_16(data)                         (sum_reduce_8((data).s01234567) + sum_reduce_8((data).s89ABCDEF))

#define SUM_REDUCE_STR(data, size)                  sum_reduce_##size(data)
#define SUM_REDUCE(data, size)                      SUM_REDUCE_STR(data, size)

#define prod_reduce_1(data)                         (data)
#define prod_reduce_2(data)                         ((data).s0) * ((data).s1)
#define prod_reduce_3(data)                         (prod_reduce_2((data).s01) * ((data).s2))
#define prod_reduce_4(data)                         (prod_reduce_2((data).s01)       * prod_reduce_2((data).s23))
#define prod_reduce_8(data)                         (prod_reduce_4((data).s0123)     * prod_reduce_4((data).s4567))
#define prod_reduce_16(data)                        (prod_reduce_8((data).s01234567) * prod_reduce_8((data).s89ABCDEF))

#define PROD_REDUCE_STR(data, size)                 prod_reduce_##size(data)
#define PROD_REDUCE(data, size)                     PROD_REDUCE_STR(data, size)

#define max_reduce_1(data)                          (data)
#define max_reduce_2(data)                          max(((data).s0), ((data).s1))
#define max_reduce_3(data)                          max(max_reduce_2((data).s01), ((data).s2))
#define max_reduce_4(data)                          max(max_reduce_2((data).s01), max_reduce_2((data).s23))
#define max_reduce_8(data)                          max(max_reduce_4((data).s0123), max_reduce_4((data).s4567))
#define max_reduce_16(data)                         max(max_reduce_8((data).s01234567), max_reduce_8((data).s89ABCDEF))

#define MAX_REDUCE_STR(data, size)                  max_reduce_##size(data)
#define MAX_REDUCE(data, size)                      MAX_REDUCE_STR(data, size)

#define min_reduce_1(data)                          (data)
#define min_reduce_2(data)                          min(((data).s0), ((data).s1))
#define min_reduce_3(data)                          min(min_reduce_2((data).s01), ((data).s2))
#define min_reduce_4(data)                          min(min_reduce_2((data).s01), min_reduce_2((data).s23))
#define min_reduce_8(data)                          min(min_reduce_4((data).s0123), min_reduce_4((data).s4567))
#define min_reduce_16(data)                         min(min_reduce_8((data).s01234567), min_reduce_8((data).s89ABCDEF))

#define MIN_REDUCE_STR(data, size)                  min_reduce_##size(data)
#define MIN_REDUCE(data, size)                      MIN_REDUCE_STR(data, size)

#define DOT_INT_STR(a, b, size, DST_TYPE)           SUM_REDUCE(CONVERT(a, VECTOR_TYPE(DST_TYPE, size)) * CONVERT(b, VECTOR_TYPE(DST_TYPE, size)), size)
#define DOT_INT(a, b, size, DST_TYPE)               DOT_INT_STR(a, b, size, DST_TYPE)

// ====================
//       Border index
// ====================
#if BORDER_CONSTANT
#  define LEFT_BORDER_IDX(idx)                      ((idx) < 0 ? (-1) : (idx))
#  define RIGHT_BORDER_IDX(idx, len)                ((idx) > ((len) - 1) ? (-1) : (idx))
#  define BORDER_IDX(idx, len)                      ((idx) < 0 ? (-1) : ((idx) > ((len) - 1) ? (-1) : (idx)))
#elif BORDER_REPLICATE
#  define LEFT_BORDER_IDX(idx)                      ((idx) < 0 ? 0 : (idx))
#  define RIGHT_BORDER_IDX(idx, len)                ((idx) > ((len) - 1) ? (len) - 1 : (idx))
#  define BORDER_IDX(idx, len)                      ((idx) < 0 ? 0 : ((idx) > ((len) - 1) ? (len) - 1 : (idx)))
#elif BORDER_REFLECT_101
#  define LEFT_BORDER_IDX(idx)                      ((idx) < 0 ? 0 - (idx) : (idx))
#  define RIGHT_BORDER_IDX(idx, len)                ((idx) > ((len) - 1) ? (((len) - 1) << 1) - (idx) : (idx))
#  define BORDER_IDX(idx, len)                      ((idx) < 0 ? 0 - (idx) : ((idx) > ((len) - 1) ? (((len) - 1) << 1) - (idx) : (idx)))
#else
#  define LEFT_BORDER_IDX(idx)                      "INVALID"
#  define RIGHT_BORDER_IDX(idx, len)                "INVALID"
#  define BORDER_IDX(idx, len)                      "INVALID"
#endif

#  define TOP_BORDER_IDX(idx)                       LEFT_BORDER_IDX(idx)
#  define BOTTOM_BORDER_IDX(idx, len)               RIGHT_BORDER_IDX(idx, len)

// ====================
//       Printf
// ====================
#define printf_signed1(data)                        printf("%d ", data)
#define printf_signed2(data)                        {printf_signed1(data.hi); printf_signed1(data.lo);}
#define printf_signed3(data)                        {printf_signed1(data.s2); printf_signed2(data.s01);}
#define printf_signed4(data)                        {printf_signed2(data.hi); printf_signed2(data.lo);}
#define printf_signed8(data)                        {printf_signed4(data.hi); printf_signed4(data.lo);}
#define printf_signed16(data)                       {printf_signed8(data.hi); printf_signed8(data.lo);}
#define printf_signed(data, size)                   printf_signed##size(data)

#define printf_unsigned1(data)                      printf("%u ", data)
#define printf_unsigned2(data)                      {printf_unsigned1(data.hi); printf_unsigned1(data.lo);}
#define printf_unsigned3(data)                      {printf_unsigned1(data.s2); printf_unsigned2(data.s01);}
#define printf_unsigned4(data)                      {printf_unsigned2(data.hi); printf_unsigned2(data.lo);}
#define printf_unsigned8(data)                      {printf_unsigned4(data.hi); printf_unsigned4(data.lo);}
#define printf_unsigned16(data)                     {printf_unsigned8(data.hi); printf_unsigned8(data.lo);}
#define printf_unsigned(data, size)                 printf_unsigned##size(data)

#define printf_float1(data)                         printf("%f ", data)
#define printf_float2(data)                         {printf_float1(data.hi); printf_float1(data.lo);}
#define printf_float3(data)                         {printf_float1(data.s2); printf_float2(data.s01);}
#define printf_float4(data)                         {printf_float2(data.hi); printf_float2(data.lo);}
#define printf_float8(data)                         {printf_float4(data.hi); printf_float4(data.lo);}
#define printf_float16(data)                        {printf_float8(data.hi); printf_float8(data.lo);}
#define printf_float(data, size)                    printf_float##size(data)

#define printf_char(data, size)                     printf_signed(data, size)
#define printf_short(data, size)                    printf_signed(data, size)
#define printf_int(data, size)                      printf_signed(data, size)
#define printf_uchar(data, size)                    printf_unsigned(data, size)
#define printf_ushort(data, size)                   printf_unsigned(data, size)
#define printf_uint(data, size)                     printf_unsigned(data, size)
#define printf_half(data, size)                     printf_float(data, size)

#define PRINTF_VEC_STR(data, type, size)            printf_##type(data, size)
#define PRINTF_VEC(tag, data, type, size)           {printf("[%s]", tag); PRINTF_VEC_STR(data, type, size); printf("\n");}
#define PRINTF(tag, data, type)                     PRINTF_VEC(tag, data, type, 1)
