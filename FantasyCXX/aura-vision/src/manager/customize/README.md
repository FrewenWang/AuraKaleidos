## 根据项目定制 manager
### how-to-do
* 在项目根目录下的 CMakeLists.txt 中配置 PRODUCT 为项目名
* 在 manager/customize 目录下创建以项目名命名的文件夹
* 在项目文件夹内添加需要定制化的 xxx_manager.cpp，必要时也可重写头文件，并实现相应逻辑，注意xxx_manager.cpp的名称要与 manager 根目录下的文件名相同
* 注意，项目文件夹内只需要新增要定制的 manager，不需要增加所有，编译时找不到的会自动采用主线 manager

### example
为福特项目定制一个疲劳检测管理器，建议的操作如下：
* 将 manager 下的 face_fatigue_manager.cpp 拷贝至 manager/customize/ford/目录下
* 修改 manager/customize/ford/face_fatigue_manager.cpp 内部实现逻辑
* 需要注意的是，由于文件层级结构发生变化，注意头文件的包含是否正确