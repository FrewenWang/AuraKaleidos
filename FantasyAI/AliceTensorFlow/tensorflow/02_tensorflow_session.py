# 引入类库
import tensorflow as tf
# 更新：找到了一个更简单的方法，在引用tensorflow时，直接用：
# import tensorflow.compat.v1 as tf
import os

# 查阅资料发现，原因是2.0与1.0版本不兼容，在程序开始部分添加以下代码：
# tensorflow的官网对disable_eager_execution()方法是这样解释的：
# 翻译过来为：此函数只能在创建任何图、运算或张量之前调用。它可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头。

tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

a = tf.constant(2)
b = tf.constant(3)

# 安装好tensorflow2.0之后，当使用Session时，
# 报错AttributeError: module 'tensorflow' has no attribute 'Session'：
sess = tf.compat.v1.Session()

# 我们使用Tensorflow2.0的时候，提示：
# RuntimeError: The Session graph is empty.  Add operations to the graph before calling run().
# 我们需要在导入tf的时候：tf.compat.v1.disable_eager_execution()

with sess:
    print("a:%i" % sess.run(a), "b:%i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constant:%i" % sess.run(a * b))
