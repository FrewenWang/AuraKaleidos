# 引入类库
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# 定义一个tensorflow的常量值
data1 = tf.constant(2.5)

# 定义一个Tensorflow的变量值
data2 = tf.Variable(10, name='var')

print(data1)
print(data2)
# 输出结果：
# tf.Tensor(2.5, shape=(), dtype=float32)
# <tf.Variable 'var:0' shape=() dtype=int32, numpy=10>

# 我们尝试解析一下这个结果：
# Tensor表明一个张量 shape表明维度 数据类型是一个float32
# Tensor表明一个张量 shape表明维度 数据类型是一个float32

sess = tf.compat.v1.Session()

# 我们使用Tensorflow2.0的时候，提示：
# RuntimeError: The Session graph is empty.  Add operations to the graph before calling run().
# 我们需要在导入tf的时候：tf.compat.v1.disable_eager_execution()
print(sess.run(data1))

# 我们尝试打印一个变量
# 变量打印之前需要初始化
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(data2))
# 上述代码在Tensorflow2中会报错
# AttributeError: module 'tensorflow' has no attribute 'global_variables_initializer'

init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print("打印变量：{}".format(sess.run(data2)))
print("打印变量：%d" % sess.run(data2))
print("打印变量:", sess.run(data2))

# 定点类型的参数：
data3 = tf.constant(2, dtype=tf.int32)
print(sess.run(data3))

sess.close()



