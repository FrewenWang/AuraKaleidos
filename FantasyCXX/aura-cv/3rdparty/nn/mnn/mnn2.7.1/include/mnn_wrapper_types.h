#ifndef AURA_NN_MNN_WRAPPER_TYPES_H__
#define AURA_NN_MNN_WRAPPER_TYPES_H__

namespace MNN
{

typedef void* Context;

enum NNBackend
{
    CPU = 0,
    GPU,
};

enum Precision
{
    PRECISION_NORMAL = 0,
    PRECISION_HIGH,
    PRECISION_LOW,
};

enum Memory
{
    MEMORY_NORMAL = 0,
    MEMORY_HIGH,
    MEMORY_LOW,
};

enum Tuning
{
    // choose one tuning mode Only
    GPU_TUNING_NONE   = 1 << 0, /**< Forbidden tuning, performance not good */
    GPU_TUNING_HEAVY  = 1 << 1, /**< heavily tuning, usually not suggested */
    GPU_TUNING_WIDE   = 1 << 2, /**< widely tuning, performance good. Default */
    GPU_TUNING_NORMAL = 1 << 3, /**< normal tuning, performance may be ok */
    GPU_TUNING_FAST   = 1 << 4, /**< fast tuning, performance may not good */
};

enum CLMem
{
    GPU_MEMORY_NONE = 0,
    // choose one opencl memory mode Only
    // User can try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IAURA both,
    // then choose the better one according to performance
    GPU_MEMORY_BUFFER = 1 << 6, /**< User assign mode */
    GPU_MEMORY_IAURA  = 1 << 7, /**< User assign mode */
};

enum MnnDataType
{
    DATA_TYPE_INVALID = 0,
    DATA_TYPE_UINT8,
    DATA_TYPE_INT8,
    DATA_TYPE_UINT16,
    DATA_TYPE_INT16,
    DATA_TYPE_UINT32,
    DATA_TYPE_INT32,
    DATA_TYPE_UINT64,
    DATA_TYPE_INT64,
    DATA_TYPE_HALF,
    DATA_TYPE_FLOAT,
    DATA_TYPE_DOUBLE,
};

struct TensorDesc
{
    char *name;
    int zero_point;
    float scale;
    int rank;
    int *dims;
    MnnDataType elem_type;
};

} // namespace MNN

#endif // AURA_NN_MNN_WRAPPER_TYPES_H__