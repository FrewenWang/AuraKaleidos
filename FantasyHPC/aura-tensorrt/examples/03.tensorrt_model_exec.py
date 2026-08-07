from time import time
import numpy as np
import tensorrt as trt
from cuda import cudart  # 安装 pip install cuda-python

np.random.seed(31193)
nWarmUp = 10
nTest = 30

nB, nC, nH, nW = 1, 3, 224, 224

data = dummy_input.cpu().numpy()


def run1(engine):
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    output_type = engine.get_tensor_dtype(output_name)
    output_shape = engine.get_tensor_shape(output_name)

    context = engine.create_execution_context()
    context.set_input_shape(input_name, [nB, nC, nH, nW])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(output_shape, dtype=trt.nptype(output_type))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    # do a complete inference
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                           stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    # Count time of memory copy from host to device
    for i in range(nWarmUp):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopyHtoD" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of inference
    for i in range(nWarmUp):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    trtTimeStart = time()
    for i in range(nTest):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, Inference" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of memory copy from device to host
    for i in range(nWarmUp):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopyDtoH" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of end to end
    for i in range(nWarmUp):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopy + Inference" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

    print("tensorrt result:", np.argmax(outputH0))


if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    f = open("./model.trt", "rb")  # 读取trt模型
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))  # 创建一个Runtime(传入记录器Logger)
    engine = runtime.deserialize_cuda_engine(f.read())  # 从文件中加载trt引擎
    run1(engine)  # do inference with single stream
    print(dummy_input.shape, dummy_input.dtype)