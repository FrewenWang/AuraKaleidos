import numpy as np
from aimet_common import libpymo

x = np.random.randn(100)

quant_scheme = libpymo.QuantizationMode.QUANTIZATION_TF
analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)


bitwidth = 8
is_symmetric, strict_symmetric, unsigned_symmetric = True, False, True
use_cuda = False
analyzer.updateStats(x, use_cuda)
encoding, _ = analyzer.computeEncoding(
    bitwidth, is_symmetric, strict_symmetric, unsigned_symmetric
)

print(
    f"Min: {encoding.min}, Max: {encoding.max}, Scale(delta): {encoding.delta}, Offset: {encoding.offset}"
)


quantizer = libpymo.TensorQuantizationSimForPython()
out = quantizer.quantizeDequantize(
    x, encoding, libpymo.RoundingMode.ROUND_NEAREST, bitwidth, use_cuda
)
print(out)
