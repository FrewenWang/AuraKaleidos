## 根据项目定制 detector
### how-to-do
* 在项目根目录下的 CMakeLists.txt 中配置 PRODUCT 为项目名
* 在 detector/customize 目录下创建以项目名命名的文件夹
* 在项目文件夹内添加需要定制化的 xxx_detector.cpp，并实现相应逻辑，注意xxx_detector.cpp的名称要与 detector 根目录下的文件名相同
* 注意，项目文件夹内只需要新增要定制的 detector，不需要增加所有，编译时找不到的会自动采用主线 detector

### example
为福特项目定制一个人脸特征检测器，建议的操作如下：
* 将 detector 下的 face_feature_detector.cpp 拷贝至 detector/customize/ford/目录下
* 修改 detector/customize/ford/face_feature_detector.cpp 内部实现逻辑
* 需要注意的是，由于文件层级结构发生变化，注意头文件的包含是否正确，为了实现统一，可以都以 src 为跟路径进行包含，如
```$cpp
#include "detector/abs_detector.h"
```

