//
// Created by Frewen.Wang on 2022/11/20.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <unistd.h>
// #include <Eigen/Dense>
//  c++使用eigen提示"fatal error: Eigen/Dense: No such file or
//  directory"的解决办法
//  解决方法：https://blog.csdn.net/chengde6896383/article/details/88339643
//  解决方法一：
//  1. 将/usr/include/eigen3/Eigen递归复制到/usr/include/Eigen
//  cp -rf /usr/include/eigen3/Eigen /usr/include/Eigen -R
//  解决方法二：
//  #include <eigen3/Eigen/Dense>
#include <Eigen/Dense>

const static char *TAG = "TestEigen3";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;

/**
 * 文章参考：
 * https://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html#1%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85
 */
class TestEigen3 : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGD(TAG, "SetUpTestSuite");
    }
    
    static void TearDownTestSuite() {
        ALOGD(TAG, "TearDownTestSuite");
    }
};

/**
 * 文章参考：https://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html#1%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85
 * MatrixXd这个类型的意思是一个动态大小的矩阵，X代表大小不确定，d表示数据类型是double，注意d并不是dimension(维度)的缩写。
 * 所以Matrix3d的意思并不是说是一个3维的矩阵，正确的意思是表示一个3X3的double类型的矩阵。
 * 与之对应的还有Matrix3f、Matrix3i、Matrix3cd等，这些都大同小异，表示数据类型分别是float、int以及double型的复数。
 * 在Eigen中为了使用方便，已经预定义好了很多这种类型，我们直接调用即可。
 * 根据文档的表述，对于小于等于4的矩阵建议用固定大小矩阵，大于4的推荐动态矩阵。
 * （A rule of thumb is to use fixed-size matrices for size 4-by-4 and smaller.）更详细的内容后续介绍。
 */
TEST_F(TestEigen3, testMatrixXd) {
    ALOGD(TAG, "============== testMatrixXd ==============");
    // 以Xd方式声明一个3x3的矩阵
    Eigen::MatrixXd mat(3, 3);
    // 将矩阵(0,0)位置元素赋为1.3
    mat(0, 0) = 0.0;
    mat(0, 1) = 0.1;
    mat(0, 2) = 0.2;
    mat(1, 0) = 1.0;
    mat(1, 1) = 1.1;
    mat(1, 2) = 1.2;
    mat(2, 0) = 2.0;
    mat(2, 1) = 2.1;
    mat(2, 2) = 2.2;
    std::cout << mat << std::endl;
}


/**
 * 文章参考：https://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html#1%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85
 * 本部分主要介绍Eigen中矩阵与向量的定义及基本使用。在Eigen中，所有的矩阵与向量都是Matrix模板类的对象，向量可以理解为是矩阵的行或列为1时的特殊情况。
 * 一般而言，创建一个矩阵需要调用Matrix<>构造，其共包含6个参数，前三个为必须参数，后三个为可选参数，有默认值。
 * 三个必须参数如下：
 * Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
 * 简而言之第一个参数指定矩阵的数据类型：double、float、int…，第二和第三个参数分别表示矩阵的行数与列数。
 * 因此一个简单的矩阵声明如下：
 */
TEST_F(TestEigen3, testEigen3Matrix) {
    ALOGD(TAG, "============== testEigen3Matrix ==============");
    Matrix<double, 5, 3> m1;
    m1.setZero();
    std::cout << m1 << std::endl;
    /**
     *            0 6.94132e-310 6.95159e-310
        6.94132e-310 6.94132e-310 2.23153e-314
        6.95159e-310 7.90505e-323 6.94132e-310
        2.23157e-314 7.90505e-323 6.94132e-310
        6.94132e-310 6.94132e-310 6.95159e-310
     */
    /**
     * 可以看到新建一个矩阵的时候，如果不赋初值，默认全为0。
     *  上述代码可以构建出任意形状的矩阵，好用是好用，就是有些麻烦。
     *  Eigen也考虑到了这一点，因此对于一些常用大小的矩阵Eigen已经帮我们提前定义好了，我们只需要用就可以了。
     *  这个在上一部分已经说过了，这里就不再赘述。
     */
}


/**
 * 上面介绍的都是大小在编译前就确定的矩阵，这种矩阵可以称为固定矩阵(fixed matrix)，
 * 另一种是编译前大小不确定的矩阵，称为动态矩阵(dynamic matrix)。
 * 对于这种情况，可以使用Dynamic关键字。
 */
TEST_F(TestEigen3, testDynamicMatrix) {
    ALOGD(TAG, "============== testDynamicMatrix ==============");
    // 需要注意的是尖括号里只能是常量，不能传入变量，像下面这样写是会报错的，编译都过不了。
    // int a = 3;
    // int b = 7;
    // Matrix<double, a, b> m;
    
    // 到这里你可能会好奇，既然不能以传入变量的方式动态创建矩阵，那怎么控制矩阵大小呢？答案是利用矩阵的resize()函数。
    // 从字面意思就知道这个函数是用来改变矩阵大小的。因此上面这段代码可以这样改写：
    int a = 3;
    int b = 7;
    Matrix<double, Dynamic, Dynamic> m;
    m.resize(a, b);
    m.setZero();
    std::cout << m << std::endl;
    // 但需要注意的是resize()这个函数是”毁灭性”(destructive)的，改变大小以后矩阵元素会改变。所以它一般只用来对动态矩阵做初始化。
    // 而我们认为的改变大小的功能对应conservativeResize()函数。顺带一提，如果矩阵变大了，多余的元素为0。
}


/**
 * 在Eigen中可以方便的使用逗号初始化语法(comma-initializer syntax)，如下。
 */
TEST_F(TestEigen3, testCommaInitializerSyntax) {
    ALOGD(TAG, "============== testCommaInitializerSyntax ==============");
    Matrix3d m;
    m << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    std::cout << m << std::endl;
}


/**
 * 测试矩阵的转置
 */
TEST_F(TestEigen3, testMatrix3dTranspose) {
    ALOGD(TAG, "============== testMatrix3dTranspose ==============");
    ALOGD(TAG, "所谓矩阵的转置，其实就是矩阵的行元素变列元素，列元素变行元素");
    Matrix3d m;
    m << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    std::cout << m << std::endl;
    std::cout << "矩阵转置后：" << std::endl;
    std::cout << m.transpose() << std::endl;
}


TEST_F(TestEigen3, testIsometry3d) {
    // ALOGD(TAG, "============== testIsometry3d ==============");
    //设置旋转向量V
    Eigen::AngleAxisd V(3.1415926 / 4, Eigen::Vector3d(1, 0, 1).normalized());
    //设置平移向量
    Eigen::Vector3d translation(1, 3, 4);
    
    
    //将T初始化为单位阵，再做其他操作。
    //虽然称为3D，实质上为4*4矩阵。
    Eigen::Isometry3d T = Isometry3d::Identity();
    //设置欧式变换矩阵——方式1
    //此种方式和下一种方式输出相同。
    //a.translate(b)等价于aXb，描述的是在世界坐标系下的平移(虽然b在设置时为只含有3个元素列向量，猜想内部可能会有操作时其为下面这种形式)。
    //形式为 0	0	0	1
    //      0	0	0	3
    //      0	0	0	4
    //      0	0	0	0
    //此操作相当于将translation中的数值放入欧式变换矩阵translation位置(注意！！！此种效果的前提T为单位矩阵)
    //结果为 1	0	0	1
    //      0	1	0	3
    //      0	0	1	4
    //      0	0	0	1
    T.translate(translation);
    //a.rotate(b)等价于aXb，描述的是在世界坐标系下的旋转。
    T.rotate(V);
    //结果为 0.853553	 -0.5	0.146447		1
    //           0.5	0.707	    -0.5		3
    //      0.146447	  0.5	0.853553		4
    //      	   0	    0	       0		1
    //matrix()返回变换对应的矩阵，T输出时用此函数。
    std::cout << T.matrix() << std::endl;
}

