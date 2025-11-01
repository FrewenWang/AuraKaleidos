# 我们来学习Tensorflow2.4

import tensorflow as tf

# 创建一个Tensor字符串常量
hello = tf.constant("hello world!!")
print(hello)

# 访问hello中的字符串，我们需要用到numpy（）

# 调用numpy()，来访问一个Tensor值
print(hello.numpy())

# 本程序原期望的运行结果为 hello world ，而实际运行结果为 b'hello world'

# 在网上查阅后，博主的理解：

# b前缀的字符串为bytes类型的字符串
# python语言中有两种不同的字符串，一个用于存储文本（unicode类型文本字符串 u'hello world' ），
# 一个用于存储原始字节（byte类型字节字符串 b'hello world' ）
# 在python3中，str变量默认采用unicode类型，因而省略了u前缀
# 字节型字符串和文本型字符串之间可以通过编码encode()和解码decode()相互转换。


# 那么既然hello.numpy()返回的是字节型字符串
# 我们对其进行decode()解码操作
# 调用decode()解码，默认为utf-8解码
print(hello.numpy().decode())
