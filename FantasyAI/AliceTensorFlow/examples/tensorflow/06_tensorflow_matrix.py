import tensorflow as tf

# 查阅资料发现，原因是2.0与1.0版本不兼容，在程序开始部分添加以下代码：
# tensorflow的官网对disable_eager_execution()方法是这样解释的：
# 翻译过来为：此函数只能在创建任何图、运算或张量之前调用。它可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头。
tf.compat.v1.disable_eager_execution()
