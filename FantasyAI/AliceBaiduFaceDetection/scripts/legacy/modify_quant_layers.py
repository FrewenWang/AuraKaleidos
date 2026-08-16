
"""for paddleCloud quantum train. modify scale_attr default value"""

# file_path = '/usr/local/lib/python3.7/site-packages/paddle/nn/quant/quant_layers.py'  # for v100
file_path = '/usr/local/lib/python3.7/dist-packages/paddle/nn/quant/quant_layers.py'  # for a100
with open(file_path,  'r') as f:
    lines = f.readlines()
lines[69] = lines[69].replace('0.0', '0.001')  # initializer=Constant(0.0), modify 0.0 to 0.001
with open(file_path, 'w') as fout:
    for line in lines:
        fout.write(line)
