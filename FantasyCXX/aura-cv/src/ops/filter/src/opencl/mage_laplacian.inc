#include "cl_helper.inc"

#define V2St            VTYPE(St, 2)
#define V3St            VTYPE(St, 3)
#define V4St            VTYPE(St, 4)
#define V8St            VTYPE(St, 8)
#define V16St           VTYPE(St, 16)

#define V2InterType     VTYPE(InterType, 2)
#define V3InterType     VTYPE(InterType, 3)
#define V4InterType     VTYPE(InterType, 4)
#define V8InterType     VTYPE(InterType, 8)
#define V16InterType    VTYPE(InterType, 16)

#define V2Dt            VTYPE(Dt, 2)
#define V3Dt            VTYPE(Dt, 3)
#define V4Dt            VTYPE(Dt, 4)
#define V8Dt            VTYPE(Dt, 8)
#define V16Dt           VTYPE(Dt, 16)

#if IS_FLOAT(St)
#  define LAPLACIAN_CONVERT(data, type) data
#else
#  define LAPLACIAN_CONVERT             CONVERT
#endif
