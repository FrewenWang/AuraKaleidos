#ifndef SSE_FILL_MAT_H
#define SSE_FILL_MAT_H

#include <xmmintrin.h>
#include "../../../../../../third_party/ncnn/20190320/src/mat.h"

namespace iov {

inline void fill_mat_sse(Mat &blob, const float &_v) {
    int size = blob.total();
    int nn = size >> 2;
    int remain = size - (nn << 2);
    float *ptr = (float *) blob.data;

    __m128 xmm0;
    xmm0 = _mm_load_ss(&_v);
    xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0);

    for (int i = 0; i < nn; ++i) {
        _mm_storeu_ps(ptr, xmm0);
        ptr += 4;
    }

    for (; remain > 0; remain--) {
        *ptr++ = _v;
    }
}
} // namespace iov

#endif // SSE_FILL_MAT_H