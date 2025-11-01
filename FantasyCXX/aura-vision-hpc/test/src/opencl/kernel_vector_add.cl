// kernel.cl
__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C) {
    int id = get_global_id(0);  // 获取当前工作项ID
    C[id] = A[id] + B[id];      // 对应位置相加
}