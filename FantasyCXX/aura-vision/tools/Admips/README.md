## Admips
Admips 是一个运行在 Android 系统的 DMIPS 测试工具，可以测试算法的计算量（DMIPS），以及 CPU 占用和内存占用；

### 编译
编译 arm64-v8a 平台
```shell script
./build.sh -r -t 2
```
编译 armeabi-v7a 平台
```shell script
./build.sh -r -t 1
```
编译后的 apk 将会拷贝至 output 文件夹；

### 使用
安装并运行（编译后首次运行）
测试 dmips
```shell script
./run.sh -i -m dmips
```
安装并测试 cpumem
```shell script
./run.sh -i -m cpumem
```
不安装直接运行
```shell script
./run.sh -m dmips
```
### 获取测试结果
```shell script
./pull_result.sh
```
将会在 Admips 根目录下创建 result 文件夹，并将结果文件拉取到该文件夹下；

### 定制化测试过程
#### 如何配置测试参数
在源码文件src/main/cpp/admips_main.cpp 中，可以修改测试次数、帧率、硬件的 dmips 系数等参数；
```c++
static constexpr int TEST_CNT = 500; // 测试次数
static constexpr float DMIPS_COEFF = 75.52f; // dmips 系数，不同 cpu 需要查询确定
static constexpr int CORE_NUM = 6; // 核数
static constexpr int TEST_FPS = 10; // 除 FaceID 功能外，其他功能 cpumem 的测试帧率
static constexpr int TEST_FACEID_FPS = 3; // FaceID 功能cpumem 的测试帧
```
#### 如何打开或关闭某个/某项组合的能力测试
```c++
// 配置要测试的能力（组合）
std::unordered_map<std::string, bool> _test_switch {
        {"Single", true},   // 所有单项能力
        {"FaceID", true},   // FaceID组合能力
        {"DMS", true},      // DMS 组合能力
        {"MMI", true},      // MMI 组合能力
        {"DMS_MMI", true}   // DMS + MMI 组合能力
};
```
#### 如何修改测试能力或组合能力
在源码文件src/main/cpp/admips_main.cpp 中，预定义了各种单项能力，可以进行删减；也预定义了 FaceID、DMS、MMI、DMS_MMI 四种组合能力；
```c++
// 单项能力
std::vector<std::pair<AbilityId, std::string>> _ability_list;
// 组合能力
std::vector<AbilityId> _faceid_ability_list;
std::vector<AbilityId> _dms_ability_list;
std::vector<AbilityId> _mmi_ability_list;
std::vector<AbilityId> _dms_mmi_ability_list;
```