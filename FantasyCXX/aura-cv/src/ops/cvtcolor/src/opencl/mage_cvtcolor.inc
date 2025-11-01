
// src/runtime/include/aura/runtime/core/opencl/cl_helper.inc
// 文件位于rumtime里面的core核心类里面的cl_helper.inc
// VLOAD的函数定义肯定就在里面
#include "cl_helper.inc"

#define CVTCOLOR_COEF_BITS      (20)
#define CVTCOLOR_DESCALE(x,n)   (((x) + (1 << ((n)-1))) >> (n))

/**
 * @brief the formula of RGB -> YUV
 *  Y = 16  + 0.257 * R + 0.504 * G + 0.098 * B
 * Cb = 128 - 0.148 * R - 0.291 * G + 0.439 * B
 * Cr = 128 + 0.439 * R - 0.368 * G - 0.071 * B
 *
 * @brief the formula of BT601 RGB -> YUV
 *  Y = 0.5   + 0.299 * r + 0.587 * g + 0.144 * b
 * Cb = 128.5 - 0.169 * r - 0.331 * g + 0.500 * b
 * Cr = 128.5 + 0.500 * r - 0.419 * g - 0.081 * b
 */
#if CVTCOLOR_RGB2YUV_601
    int r2y =  313524;   // round(0.299f  * (1 << CVTCOLOR_COEF_BITS));
    int g2y =  615514;   // round(0.587f  * (1 << CVTCOLOR_COEF_BITS));
    int b2y =  119538;   // round(0.144f  * (1 << CVTCOLOR_COEF_BITS));
    int r2u = -176895;   // round(-0.169f * (1 << CVTCOLOR_COEF_BITS));
    int g2u = -347393;   // round(-0.331f * (1 << CVTCOLOR_COEF_BITS));
    int b2u =  524288;   // round(0.500f  * (1 << CVTCOLOR_COEF_BITS));
    int r2v =  524288;   // round(0.500f  * (1 << CVTCOLOR_COEF_BITS));
    int g2v = -439039;   // round(-0.419f * (1 << CVTCOLOR_COEF_BITS));
    int b2v = -85249;    // round(-0.081f * (1 << CVTCOLOR_COEF_BITS));
    int yc  =  0;
#else
    int r2y =  269484;   // round(0.257f  * (1 << CVTCOLOR_COEF_BITS));
    int g2y =  528482;   // round(0.504f  * (1 << CVTCOLOR_COEF_BITS));
    int b2y =  102760;   // round(0.098f  * (1 << CVTCOLOR_COEF_BITS));
    int r2u = -155188;   // round(-0.148f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    int g2u = -305135;   // round(-0.291f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    int b2u =  460324;   // round(0.439f  * (1 << CVTCOLOR_COEF_BITS));
    int r2v =  460324;   // round(0.439f  * (1 << CVTCOLOR_COEF_BITS));
    int g2v = -385875;   // round(-0.368f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    int b2v = -74448;    // round(-0.071f * (1 << CVTCOLOR_COEF_BITS)) + 1;
    int yc  =  16777216; // round( 16     * (1 << CVTCOLOR_COEF_BITS));
#endif // CVTCOLOR_RGB2YUV_601

/**
 * @brief the formula of YUV -> RGB
 * R = 1.164(Y - 16) + 1.596(V - 128)
 * G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
 * B = 1.164(Y - 16)                  + 2.018(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * (Y - 16)                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 *
 * @brief the formula of Bt.601 YUV -> RGB
 * R = Y + 1.403(V - 128)
 * G = Y - 0.714(V - 128) - 0.343(U - 128)
 * B = Y                  + 1.770(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * Y + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * Y + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * Y                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 */
#if CVTCOLOR_YUV2RGB_601
    int y2rgb = 1048576; // round(1.000f  * (1 << CVTCOLOR_COEF_BITS));
    int v2r   = 1471152; // round(1.403f  * (1 << CVTCOLOR_COEF_BITS));
    int v2g   = -748683; // round(-0.714f * (1 << CVTCOLOR_COEF_BITS));
    int u2g   = -359661; // round(-0.343f * (1 << CVTCOLOR_COEF_BITS));
    int u2b   = 1855979; // round(1.770f  * (1 << CVTCOLOR_COEF_BITS));
#else
    int y2rgb = 1220542; // round(1.164f  * (1 << CVTCOLOR_COEF_BITS));
    int v2r   = 1673527; // round(1.596f  * (1 << CVTCOLOR_COEF_BITS));
    int v2g   = -852492; // round(-0.813f * (1 << CVTCOLOR_COEF_BITS));
    int u2g   = -409993; // round(-0.391f * (1 << CVTCOLOR_COEF_BITS));
    int u2b   = 2116026; // round(2.018f  * (1 << CVTCOLOR_COEF_BITS));
#endif // CVTCOLOR_YUV2RGB_601
