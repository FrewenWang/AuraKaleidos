#include "cl_helper.inc"

#define V2St                                       VTYPE(St, 2)
#define V3St                                       VTYPE(St, 3)
#define V4St                                       VTYPE(St, 4)
#define V8St                                       VTYPE(St, 8)
#define V16St                                      VTYPE(St, 16)

#define V2InterType                                VTYPE(InterType, 2)
#define V3InterType                                VTYPE(InterType, 3)
#define V4InterType                                VTYPE(InterType, 4)
#define V8InterType                                VTYPE(InterType, 8)
#define V16InterType                               VTYPE(InterType, 16)

#define V2Dt                                       VTYPE(Dt, 2)
#define V3Dt                                       VTYPE(Dt, 3)
#define V4Dt                                       VTYPE(Dt, 4)
#define V8Dt                                       VTYPE(Dt, 8)
#define V16Dt                                      VTYPE(Dt, 16)

#if IS_FLOAT(St)
#  define SOBEL_CONVERT(data, type)                data
#else
#  define SOBEL_CONVERT                            CONVERT
#endif // IS_FLOAT(St)

#define TBL_K5(D, K0, K1, K2, K3, K4)              ((D == 0)? K0 : ((D == 1) ? K1 : ((D == 2) ? K2 : ((D == 3) ? K3 : K4))))
#define K0(D)                                      TBL_K5(D, 1, -1,  1, -1,  1)
#define K1(D)                                      TBL_K5(D, 4, -2,  0,  2, -4)
#define K2(D)                                      TBL_K5(D, 6,  0, -2,  0,  6)
#define K3(D)                                      TBL_K5(D, 4,  2,  0, -2, -4)
#define K4(D)                                      TBL_K5(D, 1,  1,  1,  1,  1)

#define INIT_KERNEL_STR(dir, D)                    static constant InterType dir##0 = K0(D); \
                                                   static constant InterType dir##1 = K1(D); \
                                                   static constant InterType dir##2 = K2(D); \
                                                   static constant InterType dir##3 = K3(D); \
                                                   static constant InterType dir##4 = K4(D);

#define INIT_KERNEL(dir, DY)                       INIT_KERNEL_STR(dir, DY)

INIT_KERNEL(v, DY)
INIT_KERNEL(h, DX)
