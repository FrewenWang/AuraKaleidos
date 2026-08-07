# 引入类库
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# 定义一个tensorflow的常量值
data1 = tf.constant(2.5)

# 定义一个Tensorflow的变量值
data2 = tf.Variable(10, name='var')

# print(data1)
# print(data2)
# 输出结果：
# tf.Tensor(2.5, shape=(), dtype=float32)
# <tf.Variable 'var:0' shape=() dtype=int32, numpy=10>

# 我们尝试解析一下这个结果：
# Tensor表明一个张量 shape表明维度 数据类型是一个float32
sess = tf.compat.v1.Session()

init = tf.compat.v1.global_variables_initializer()

with sess:
    sess.run(init)
    print(sess.run(data1))
    print(sess.run(data2))
