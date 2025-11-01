#include "cl_helper.inc"

#define VTp             VTYPE(Tp, 1)
#define V2Tp            VTYPE(Tp, 2)
#define V3Tp            VTYPE(Tp, 3)
#define V4Tp            VTYPE(Tp, 4)
#define V8Tp            VTYPE(Tp, 8)
#define V16Tp           VTYPE(Tp, 16)

#if IS_FLOAT(Tp)
#  define FILTER2D_CONVERT(data, type) (data)
#else
#  define FILTER2D_CONVERT             CONVERT
#endif