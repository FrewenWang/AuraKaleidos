#include "cl_helper.inc"

#if IS_FLOAT(InterType)
#  define PYRAMID_UP_CONVERT(data0, data1, q, dtype)      (CONVERT(native_divide(data0 + data1, (1 << q)), dtype))
#else
#  define PYRAMID_UP_CONVERT(data0, data1, q, dtype)      (CONVERT_SAT((data0 + data1) >> q, dtype))
#endif

#define V2Tp   VTYPE(Tp, 2)
#define V3Tp   VTYPE(Tp, 3)
#define V4Tp   VTYPE(Tp, 4)
#define V8Tp   VTYPE(Tp, 8)
#define V2It   VTYPE(InterType, 2)
#define V3It   VTYPE(InterType, 3)
#define V4It   VTYPE(InterType, 4)
#define V8It   VTYPE(InterType, 8)
#define V16It  VTYPE(InterType, 16)
