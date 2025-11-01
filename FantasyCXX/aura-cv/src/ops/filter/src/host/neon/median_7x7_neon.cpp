#include "median_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VqType>
AURA_ALWAYS_INLINE AURA_VOID Median7x7Core(VqType *vqs)
{
    #define OP(r0, r1) MinMaxOp<VqType>(vqs[r0], vqs[r1])
    OP(1, 2);   OP(0, 2);   OP(0, 1);   OP(4, 5);   OP(3, 5);   OP(3, 4);
    OP(0, 3);   OP(1, 4);   OP(2, 5);   OP(2, 4);   OP(1, 3);   OP(2, 3);
    OP(7, 8);   OP(6, 8);   OP(6, 7);   OP(10, 11); OP(9, 11);  OP(9, 10);
    OP(6, 9);   OP(7, 10);  OP(8, 11);  OP(8, 10);  OP(7, 9);   OP(8, 9);
    OP(0, 6);   OP(1, 7);   OP(2, 8);   OP(2, 7);   OP(1, 6);   OP(2, 6);
    OP(3, 9);   OP(4, 10);  OP(5, 11);  OP(5, 10);  OP(4, 9);   OP(5, 9);
    OP(3, 6);   OP(4, 7);   OP(5, 8);   OP(5, 7);   OP(4, 6);   OP(5, 6);

    OP(13, 14); OP(12, 14); OP(12, 13); OP(16, 17); OP(15, 17); OP(15, 16);
    OP(12, 15); OP(13, 16); OP(14, 17); OP(14, 16); OP(13, 15); OP(14, 15);
    OP(19, 20); OP(18, 20); OP(18, 19); OP(22, 23); OP(21, 23); OP(21, 22);
    OP(18, 21); OP(19, 22); OP(20, 23); OP(20, 22); OP(19, 21); OP(20, 21);
    OP(12, 18); OP(13, 19); OP(14, 20); OP(14, 19); OP(13, 18); OP(14, 18);
    OP(15, 21); OP(16, 22); OP(17, 23); OP(17, 22); OP(16, 21); OP(17, 21);
    OP(15, 18); OP(16, 19); OP(17, 20); OP(17, 19); OP(16, 18); OP(17, 18);

    OP(0, 12);  OP(1, 13);  OP(2, 14);  OP(2, 13);  OP(1, 12);  OP(2, 12);
    OP(3, 15);  OP(4, 16);  OP(5, 17);  OP(5, 16);  OP(4, 15);  OP(5, 15);
    OP(3, 12);  OP(4, 13);  OP(5, 14);  OP(5, 13);  OP(4, 12);  OP(5, 12);
    OP(6, 18);  OP(7, 19);  OP(8, 20);  OP(8, 19);  OP(7, 18);  OP(8, 18);
    OP(9, 21);  OP(10, 22); OP(11, 23); OP(11, 22); OP(10, 21); OP(11, 21);
    OP(9, 18);  OP(10, 19); OP(11, 20); OP(11, 19); OP(10, 18); OP(11, 18);
    OP(6, 12);  OP(7, 13);  OP(8, 14);  OP(8, 13);  OP(7, 12);  OP(8, 12);

    OP(9, 15);  OP(10, 16); OP(11, 17); OP(11, 16); OP(10, 15); OP(11, 15);
    OP(9, 12);  OP(10, 13); OP(11, 14); OP(11, 13); OP(10, 12); OP(11, 12);
    OP(25, 26); OP(24, 26); OP(24, 25); OP(28, 29); OP(27, 29); OP(27, 28);
    OP(24, 27); OP(25, 28); OP(26, 29); OP(26, 28); OP(25, 27); OP(26, 27);
    OP(31, 32); OP(30, 32); OP(30, 31); OP(34, 35); OP(33, 35); OP(33, 34);
    OP(30, 33); OP(31, 34); OP(32, 35); OP(32, 34); OP(31, 33); OP(32, 33);
    OP(24, 30); OP(25, 31); OP(26, 32); OP(26, 31); OP(25, 30); OP(26, 30);

    OP(27, 33); OP(28, 34); OP(29, 35); OP(29, 34); OP(28, 33); OP(29, 33);
    OP(27, 30); OP(28, 31); OP(29, 32); OP(29, 31); OP(28, 30); OP(29, 30);
    OP(37, 38); OP(36, 38); OP(36, 37); OP(40, 41); OP(39, 41); OP(39, 40);
    OP(36, 39); OP(37, 40); OP(38, 41); OP(38, 40); OP(37, 39); OP(38, 39);
    OP(43, 44); OP(42, 44); OP(42, 43); OP(45, 46); OP(47, 48); OP(45, 47);
    OP(46, 48); OP(46, 47); OP(42, 46); OP(42, 45); OP(43, 47); OP(44, 48);

    OP(44, 47); OP(43, 45); OP(44, 46); OP(44, 45); OP(36, 43); OP(36, 42);
    OP(37, 44); OP(38, 45); OP(38, 44); OP(37, 42); OP(38, 43); OP(38, 42);
    OP(39, 46); OP(40, 47); OP(41, 48); OP(41, 47); OP(40, 46); OP(41, 46);
    OP(39, 43); OP(39, 42); OP(40, 44); OP(41, 45); OP(41, 44); OP(40, 42);
    OP(41, 43); OP(41, 42); OP(24, 37); OP(24, 36); OP(25, 38); OP(26, 39);
    OP(26, 38); OP(25, 36); OP(26, 37); OP(26, 36); OP(27, 40); OP(28, 41);

    OP(29, 42); OP(29, 41); OP(28, 40); OP(29, 40); OP(27, 37); OP(27, 36);
    OP(28, 38); OP(29, 39); OP(29, 38); OP(28, 36); OP(29, 37); OP(29, 36);
    OP(30, 43); OP(31, 44); OP(32, 45); OP(32, 44); OP(31, 43); OP(32, 43);
    OP(33, 46); OP(34, 47); OP(35, 48); OP(35, 47); OP(34, 46); OP(35, 46);
    OP(33, 43); OP(34, 44); OP(35, 45); OP(35, 44); OP(34, 43); OP(35, 43);
    OP(30, 37); OP(30, 36); OP(31, 38); OP(32, 39); OP(32, 38); OP(31, 36);
    OP(32, 37); OP(32, 36); OP(33, 40); OP(34, 41); OP(35, 42); OP(35, 41);
    OP(34, 40); OP(35, 40); OP(33, 37); OP(33, 36); OP(34, 38); OP(35, 39);
    OP(35, 38); OP(34, 36); OP(35, 37); OP(35, 36);

    OP(0, 25);  vqs[24] = neon::vmax(vqs[0], vqs[24]);
    OP(1, 26);  OP(2, 27);
    OP(2, 26);  vqs[24] = neon::vmax(vqs[1], vqs[24]);
    OP(2, 25);  vqs[24] = neon::vmax(vqs[2], vqs[24]);
    OP(3, 28);  OP(4, 29);  OP(5, 30);  OP(5, 29);  OP(4, 28);  OP(5, 28);
    OP(3, 25);  vqs[24] = neon::vmax(vqs[3], vqs[24]);
    OP(4, 26);  OP(5, 27);
    OP(5, 26);  vqs[24] = neon::vmax(vqs[4], vqs[24]);
    OP(5, 25);  vqs[24] = neon::vmax(vqs[5], vqs[24]);
    OP(6, 31);  OP(7, 32);  OP(8, 33);  OP(8, 32);
    OP(7, 31);  OP(8, 31);  OP(9, 34);  OP(10, 35); OP(11, 36); OP(11, 35);
    OP(10, 34); OP(11, 34); OP(9, 31);  OP(10, 32); OP(11, 33); OP(11, 32);

    OP(10, 31); OP(11, 31);
    OP(6, 25);  vqs[24] = neon::vmax(vqs[6], vqs[24]);
    OP(7, 26);  OP(8, 27);
    OP(8, 26);  vqs[24] = neon::vmax(vqs[7], vqs[24]);
    OP(8, 25);  vqs[24] = neon::vmax(vqs[8], vqs[24]);
    OP(9, 28);  OP(10, 29); OP(11, 30); OP(11, 29); OP(10, 28); OP(11, 28);
    OP(9, 25);  vqs[24] = neon::vmax(vqs[9], vqs[24]);
    OP(10, 26); OP(11, 27);
    OP(11, 26); vqs[24] = neon::vmax(vqs[10], vqs[24]);
    OP(11, 25); vqs[24] = neon::vmax(vqs[11], vqs[24]);
    OP(12, 37); OP(13, 38); OP(14, 39); OP(14, 38); OP(13, 37); OP(14, 37);
    OP(15, 40); OP(16, 41); OP(17, 42); OP(17, 41); OP(16, 40); OP(17, 40);
    OP(15, 37); OP(16, 38); OP(17, 39); OP(17, 38); OP(16, 37); OP(17, 37);
    OP(18, 43); OP(19, 44); OP(20, 45); OP(20, 44); OP(19, 43); OP(20, 43);

    OP(21, 46); OP(22, 47);
    vqs[23] = neon::vmin(vqs[23], vqs[48]);
    vqs[23] = neon::vmin(vqs[23], vqs[47]);
    OP(22, 46); vqs[23] = neon::vmin(vqs[23], vqs[46]);
    OP(21, 43); OP(22, 44);
    vqs[23] = neon::vmin(vqs[23], vqs[45]);
    vqs[23] = neon::vmin(vqs[23], vqs[44]);
    OP(22, 43); vqs[23] = neon::vmin(vqs[23], vqs[43]);
    OP(18, 37); OP(19, 38); OP(20, 39); OP(20, 38); OP(19, 37); OP(20, 37);

    OP(21, 40); OP(22, 41);
    vqs[23] = neon::vmin(vqs[23], vqs[42]);
    vqs[23] = neon::vmin(vqs[23], vqs[41]);
    OP(22, 40); vqs[23] = neon::vmin(vqs[23], vqs[40]);
    OP(21, 37); OP(22, 38);
    vqs[23] = neon::vmin(vqs[23], vqs[39]);
    vqs[23] = neon::vmin(vqs[23], vqs[38]);
    OP(22, 37); vqs[23] = neon::vmin(vqs[23], vqs[37]);
    OP(12, 25); vqs[24] = neon::vmax(vqs[12], vqs[24]);
    OP(13, 26); OP(14, 27);
    OP(14, 26); vqs[24] = neon::vmax(vqs[13], vqs[24]);
    OP(14, 25); vqs[24] = neon::vmax(vqs[14], vqs[24]);
    OP(15, 28); OP(16, 29); OP(17, 30); OP(17, 29); OP(16, 28); OP(17, 28);

    OP(15, 25); vqs[24] = neon::vmax(vqs[15], vqs[24]);
    OP(16, 26); OP(17, 27);
    OP(17, 26); vqs[24] = neon::vmax(vqs[16], vqs[24]);
    OP(17, 25); vqs[24] = neon::vmax(vqs[17], vqs[24]);
    OP(18, 31); OP(19, 32);
    OP(20, 33); OP(20, 32); OP(19, 31); OP(20, 31); OP(21, 34); OP(22, 35);

    vqs[23] = neon::vmin(vqs[23], vqs[36]);
    vqs[23] = neon::vmin(vqs[23], vqs[35]);
    OP(22, 34); vqs[23] = neon::vmin(vqs[23], vqs[34]);
    OP(21, 31); OP(22, 32);
    vqs[23] = neon::vmin(vqs[23], vqs[33]);
    vqs[23] = neon::vmin(vqs[23], vqs[32]);
    OP(22, 31); vqs[23] = neon::vmin(vqs[23], vqs[31]);
    OP(18, 25); vqs[18] = neon::vmin(vqs[18], vqs[24]);
    OP(19, 26); OP(20, 27);
    OP(20, 26); vqs[24] = neon::vmax(vqs[19], vqs[24]);
    OP(20, 25); vqs[24] = neon::vmax(vqs[20], vqs[24]);

    OP(21, 28); OP(22, 29);
    vqs[23] = neon::vmin(vqs[23], vqs[30]);
    vqs[23] = neon::vmin(vqs[23], vqs[29]);
    OP(22, 28); vqs[23] = neon::vmin(vqs[23], vqs[28]);
    OP(21, 25); vqs[24] = neon::vmax(vqs[21], vqs[24]);
    OP(22, 26); vqs[23] = neon::vmin(vqs[23], vqs[27]);
    vqs[23] = neon::vmin(vqs[23], vqs[26]);
    vqs[24] = neon::vmax(vqs[22], vqs[24]);
    vqs[23] = neon::vmin(vqs[23], vqs[25]);
    vqs[24] = neon::vmax(vqs[23], vqs[24]);
    #undef OP
}

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VqType>
AURA_ALWAYS_INLINE AURA_VOID Median7x7Vector(VqType &vq_src_p2x0, VqType &vq_src_p2x1, VqType &vq_src_p2x2,
                                           VqType &vq_src_p1x0, VqType &vq_src_p1x1, VqType &vq_src_p1x2,
                                           VqType &vq_src_p0x0, VqType &vq_src_p0x1, VqType &vq_src_p0x2,
                                           VqType &vq_src_cx0,  VqType &vq_src_cx1,  VqType &vq_src_cx2,
                                           VqType &vq_src_n0x0, VqType &vq_src_n0x1, VqType &vq_src_n0x2,
                                           VqType &vq_src_n1x0, VqType &vq_src_n1x1, VqType &vq_src_n1x2,
                                           VqType &vq_src_n2x0, VqType &vq_src_n2x1, VqType &vq_src_n2x2,
                                           VqType &vq_result)
{
    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(16 / sizeof(Tp));

    VqType vqs[49];

    vqs[0] = neon::vext<ELEM_COUNTS - 3>(vq_src_p2x0, vq_src_p2x1);
    vqs[1] = neon::vext<ELEM_COUNTS - 2>(vq_src_p2x0, vq_src_p2x1);
    vqs[2] = neon::vext<ELEM_COUNTS - 1>(vq_src_p2x0, vq_src_p2x1);
    vqs[3] = vq_src_p2x1;
    vqs[4] = neon::vext<1>(vq_src_p2x1, vq_src_p2x2);
    vqs[5] = neon::vext<2>(vq_src_p2x1, vq_src_p2x2);
    vqs[6] = neon::vext<3>(vq_src_p2x1, vq_src_p2x2);

    vqs[0 + 7] = neon::vext<ELEM_COUNTS - 3>(vq_src_p1x0, vq_src_p1x1);
    vqs[1 + 7] = neon::vext<ELEM_COUNTS - 2>(vq_src_p1x0, vq_src_p1x1);
    vqs[2 + 7] = neon::vext<ELEM_COUNTS - 1>(vq_src_p1x0, vq_src_p1x1);
    vqs[3 + 7] = vq_src_p1x1;
    vqs[4 + 7] = neon::vext<1>(vq_src_p1x1, vq_src_p1x2);
    vqs[5 + 7] = neon::vext<2>(vq_src_p1x1, vq_src_p1x2);
    vqs[6 + 7] = neon::vext<3>(vq_src_p1x1, vq_src_p1x2);

    vqs[0 + 14] = neon::vext<ELEM_COUNTS - 3>(vq_src_p0x0, vq_src_p0x1);
    vqs[1 + 14] = neon::vext<ELEM_COUNTS - 2>(vq_src_p0x0, vq_src_p0x1);
    vqs[2 + 14] = neon::vext<ELEM_COUNTS - 1>(vq_src_p0x0, vq_src_p0x1);
    vqs[3 + 14] = vq_src_p0x1;
    vqs[4 + 14] = neon::vext<1>(vq_src_p0x1, vq_src_p0x2);
    vqs[5 + 14] = neon::vext<2>(vq_src_p0x1, vq_src_p0x2);
    vqs[6 + 14] = neon::vext<3>(vq_src_p0x1, vq_src_p0x2);

    vqs[0 + 21] = neon::vext<ELEM_COUNTS - 3>(vq_src_cx0, vq_src_cx1);
    vqs[1 + 21] = neon::vext<ELEM_COUNTS - 2>(vq_src_cx0, vq_src_cx1);
    vqs[2 + 21] = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    vqs[3 + 21] = vq_src_cx1;
    vqs[4 + 21] = neon::vext<1>(vq_src_cx1, vq_src_cx2);
    vqs[5 + 21] = neon::vext<2>(vq_src_cx1, vq_src_cx2);
    vqs[6 + 21] = neon::vext<3>(vq_src_cx1, vq_src_cx2);

    vqs[0 + 28] = neon::vext<ELEM_COUNTS - 3>(vq_src_n0x0, vq_src_n0x1);
    vqs[1 + 28] = neon::vext<ELEM_COUNTS - 2>(vq_src_n0x0, vq_src_n0x1);
    vqs[2 + 28] = neon::vext<ELEM_COUNTS - 1>(vq_src_n0x0, vq_src_n0x1);
    vqs[3 + 28] = vq_src_n0x1;
    vqs[4 + 28] = neon::vext<1>(vq_src_n0x1, vq_src_n0x2);
    vqs[5 + 28] = neon::vext<2>(vq_src_n0x1, vq_src_n0x2);
    vqs[6 + 28] = neon::vext<3>(vq_src_n0x1, vq_src_n0x2);

    vqs[0 + 35] = neon::vext<ELEM_COUNTS - 3>(vq_src_n1x0, vq_src_n1x1);
    vqs[1 + 35] = neon::vext<ELEM_COUNTS - 2>(vq_src_n1x0, vq_src_n1x1);
    vqs[2 + 35] = neon::vext<ELEM_COUNTS - 1>(vq_src_n1x0, vq_src_n1x1);
    vqs[3 + 35] = vq_src_n1x1;
    vqs[4 + 35] = neon::vext<1>(vq_src_n1x1, vq_src_n1x2);
    vqs[5 + 35] = neon::vext<2>(vq_src_n1x1, vq_src_n1x2);
    vqs[6 + 35] = neon::vext<3>(vq_src_n1x1, vq_src_n1x2);

    vqs[0 + 42] = neon::vext<ELEM_COUNTS - 3>(vq_src_n2x0, vq_src_n2x1);
    vqs[1 + 42] = neon::vext<ELEM_COUNTS - 2>(vq_src_n2x0, vq_src_n2x1);
    vqs[2 + 42] = neon::vext<ELEM_COUNTS - 1>(vq_src_n2x0, vq_src_n2x1);
    vqs[3 + 42] = vq_src_n2x1;
    vqs[4 + 42] = neon::vext<1>(vq_src_n2x1, vq_src_n2x2);
    vqs[5 + 42] = neon::vext<2>(vq_src_n2x1, vq_src_n2x2);
    vqs[6 + 42] = neon::vext<3>(vq_src_n2x1, vq_src_n2x2);

    Median7x7Core<VqType>(vqs);

    vq_result = vqs[24];

    vq_src_p2x0 = vq_src_p2x1;
    vq_src_p1x0 = vq_src_p1x1;
    vq_src_p0x0 = vq_src_p0x1;
    vq_src_cx0  = vq_src_cx1;
    vq_src_n0x0 = vq_src_n0x1;
    vq_src_n1x0 = vq_src_n1x1;
    vq_src_n2x0 = vq_src_n2x1;

    vq_src_p2x1 = vq_src_p2x2;
    vq_src_p1x1 = vq_src_p1x2;
    vq_src_p0x1 = vq_src_p0x2;
    vq_src_cx1  = vq_src_cx2;
    vq_src_n0x1 = vq_src_n0x2;
    vq_src_n1x1 = vq_src_n1x2;
    vq_src_n2x1 = vq_src_n2x2;
}

template<typename Tp, MI_S32 C>
static AURA_VOID Median7x7Row(const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                            const Tp *src_n0, const Tp *src_n1, const Tp *src_n2,
                            Tp *dst_c, MI_S32 width)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(16 / sizeof(Tp));
    constexpr MI_S32 VOFFSET = ELEM_COUNTS * C;
    const MI_S32 width_align = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p2[3], mvq_src_p1[3], mvq_src_p0[3], mvq_src_c[3], mvq_src_n0[3], mvq_src_n1[3], mvq_src_n2[3];
    MVqType mvq_result;

    // left border
    {
        neon::vload(src_p2,           mvq_src_p2[1]);
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_c,            mvq_src_c[1]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_n2,           mvq_src_n2[1]);
        neon::vload(src_p2 + VOFFSET, mvq_src_p2[2]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c  + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);
        neon::vload(src_n2 + VOFFSET, mvq_src_n2[2]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p2[1].val[ch], src_p2[ch], src_p2[ch]);
            mvq_src_p1[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p1[1].val[ch], src_p1[ch], src_p1[ch]);
            mvq_src_p0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], src_p0[ch]);
            mvq_src_c[0].val[ch]  = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_c[1].val[ch],  src_c[ch],  src_c[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], src_n0[ch]);
            mvq_src_n1[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n1[1].val[ch], src_n1[ch], src_n1[ch]);
            mvq_src_n2[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n2[1].val[ch], src_n2[ch], src_n2[ch]);

            Median7x7Vector<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                mvq_result.val[ch]);
        }
        neon::vstore(dst_c, mvq_result);
    }

    //middle
    {
        for (MI_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Median7x7Vector<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                    mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                    mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                    mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                    mvq_result.val[ch]);
            }
            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p2 + x - VOFFSET, mvq_src_p2[0]);
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_c  + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_n2 + x - VOFFSET, mvq_src_n2[0]);
            neon::vload(src_p2 + x,           mvq_src_p2[1]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_c  + x,           mvq_src_c[1]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_n2 + x,           mvq_src_n2[1]);
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Median7x7Vector<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                    mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                    mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                    mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                    mvq_result.val[ch]);
            }
            neon::vstore(dst_c + x, mvq_result);
        }
    }

    // right border
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p2[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p2[1].val[ch], src_p2[last + ch], src_p2[last + ch]);
            mvq_src_p1[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p1[1].val[ch], src_p1[last + ch], src_p1[last + ch]);
            mvq_src_p0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last + ch], src_p0[last + ch]);
            mvq_src_c[2].val[ch]  = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_c[1].val[ch],  src_c[last + ch],  src_c[last + ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last + ch], src_n0[last + ch]);
            mvq_src_n1[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n1[1].val[ch], src_n1[last + ch], src_n1[last + ch]);
            mvq_src_n2[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n2[1].val[ch], src_n2[last + ch], src_n2[last + ch]);

            Median7x7Vector<Tp>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                mvq_result.val[ch]);
        }
        neon::vstore(dst_c + x, mvq_result);
    }
}

template <typename Tp, MI_S32 C>
static Status Median7x7NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 y = start_row;

    const Tp *src_p2 = src.Ptr<Tp, BorderType::REPLICATE>(y - 3);
    const Tp *src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(y - 2);
    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(y - 1);
    const Tp *src_c  = src.Ptr<Tp>(y);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 1);
    const Tp *src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 2);
    const Tp *src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(y + 3);

    for (; y < end_row; y++)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        Median7x7Row<Tp, C>(src_p2, src_p1, src_p0, src_c, src_n0, src_n1, src_n2, dst_c, width);

        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src_n2;
        src_n2 = src.Ptr<Tp>(y + 4, BorderType::REPLICATE);
    }

    return Status::OK;
}

template <typename Tp>
static Status Median7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    MI_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Median7x7NeonImpl<Tp, 1>, std::cref(src), std::ref(dst));
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Median7x7NeonImpl<Tp, 2>, std::cref(src), std::ref(dst));
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Median7x7NeonImpl<Tp, 3>, std::cref(src), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Median7x7Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median7x7NeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = Median7x7NeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_S8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = Median7x7NeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = Median7x7NeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_S16> failed");
            }
            break;
        }

        case ElemType::U32:
        {
            ret = Median7x7NeonHelper<MI_U32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_U32> failed");
            }
            break;
        }

        case ElemType::S32:
        {
            ret = Median7x7NeonHelper<MI_S32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_S32> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Median7x7NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = Median7x7NeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median7x7NeonHelper<MI_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura