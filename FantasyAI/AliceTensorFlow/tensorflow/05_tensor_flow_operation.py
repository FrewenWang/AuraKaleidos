import tensorflow as tf

# 查阅资料发现，原因是2.0与1.0版本不兼容，在程序开始部分添加以下代码：
# tensorflow的官网对disable_eager_execution()方法是这样解释的：
# 翻译过来为：此函数只能在创建任何图、运算或张量之前调用。它可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头。
tf.compat.v1.disable_eager_execution()

# 我们来学一下变量与常量的四则运算：
# 主要是加减乘除
# 基础数据类型 运算符 流程 字典 数组

# 定义一个tensorflow的Int32类型的常量值,我们动图dtype的声明他的类型
data1 = tf.constant(2, dtype=tf.int32)
# 定义一个tensorflow的变量.我们使用一个变量
data2 = tf.Variable(10)

# tensorflow2.0版本中Session模块的兼容处理
# https://blog.csdn.net/yuan_xiangjun/article/details/105469721
sess = tf.compat.v1.Session()
# 因为Data2是个变量，所以我们要进行初始化
init = tf.compat.v1.global_variables_initializer()

# 计算常量的相加
dataAdd = tf.add(data1, data2)
# 计算常量的相乘
dataMul = tf.multiply(data1, data2)
# 计算常量的相减
dataSub = tf.subtract(data1, data2)
# 计算常量的相除
dataDiv = tf.divide(data1, data2)

# 使用tf.assign进行数据的拷贝
# tensorFlow2中AttributeError: module 'tensorflow' has no attribute 'assign'
# 完成当前的数据拷贝只有变量才可以被重新拷贝复制，
dataCopy = tf.compat.v1.assign(data2, dataAdd)  # dataAdd ->data2

with sess:
    # 我们执行初始化
    sess.run(init)
    print(sess.run(dataAdd))
    print(sess.run(dataMul))
    print(sess.run(dataSub))
    print(sess.run(dataDiv))
    # 这个地方我们的执行：将dataAdd赋值给data2 也就是12 ->data2
    print('sess.run(dataCopy)', sess.run(dataCopy))
    # 这个地方我们的执行：我们发现data2已经是12了。所以dataCopy = tf.compat.v1.assign(data2, dataAdd)
    # 那么dataAdd就等于data1+data2 = 2+12 = 14
    # 然后在赋值给dataCopy
    print('sess.run(dataCopy)', sess.run(dataCopy))
    # 下面两个是一样的道理：有点很难理解！！！！
    # 运算图必须要在Session中执行，所以dataCopy.eval()相当于get_default_session().run(dataCopy)
    print('dataCopy.eval()', dataCopy.eval())  # 8+6->14->data = 14
    # 获取默认的Session
    print('tf.get_default_session()', tf.compat.v1.get_default_session().run(dataCopy))
print('end!')
