#新创建的项目，默认使用 CMake 进行构建。 最低版本3.10.2
#CMake最低版本，该行在CMakeLists.txt文件中是必需的，它使我们能够使用在特定版本中定义的CMake功能。
cmake_minimum_required(VERSION 3.10.2)
# 定义项目的名称。这个名称保存在名为PROJECT_NAME的变量中。
project(OpenCVDemo)

set(CMAKE_CXX_STANDARD 14)

# 管理依赖项
# CMake具备搜索依赖项和外部库的能力，这使我们能够根据项目中的外部组件构建复杂的项目，并添加一些要求。
# find_library(OpenCV)
# 其中，需要注意的是:
# find_package 是用于查找依赖包。
# include_directories 是用于添加 Headers 搜索路径，在这里将 OpenCV 的头文件添加到 include 路径。
#set(OpenCV_DIR 这里填入OpenCV的build目录)
# TODO 这里是使用的我们编译OpenCV的本地源码，后续考虑怎么设置到常规依赖
# 文章参考：https://.zhihu.com/p/106473031
set(OpenCV_DIR /Users/frewen/03.ProgramStudy/20.AI/04.Resource/OpenCV/opencv-4.5.3/build)
# 添加依赖项，OpenCV
find_package(OpenCV REQUIRED)

message("OpenCV version: ${OpenCV_VERSION}")
message("OpenCV version: ${OpenCV_VERSION}")


# add_executable 是为工程生成一个可执行文件，并指定所使用的源文件。
# add_executable 常用的语法如下：
# add_executable(<name> [WIN32] [MACOSX_BUNDLE]  [EXCLUDE_FROM_ALL]  source1 source2 … sourceN)
# <name> 是生成的可执行目标文件，会根据本地平台创建出来。
# 设置参加编译的工程或者依赖库的源码
set(src src/main.cpp src/utils/image_util.cpp src/utils/image_util.h)


add_executable(OpenCVDemo ${src})
add_executable(OpenCVDemo1 01_demo_channel_exchange.cpp)
add_executable(OpenCVDemo2 02_demo_image_gray.cpp)
add_executable(OpenCVDemo3 03_demo_image_dajin_2value.cpp)
# 多个应用程序之间分解共享代码是软件开发中常见且有用的做法。


# target_link_libraries 是为了将目标文件与库文件进行链接。
# target_link_libraries 常用的语法如下：
# target_link_libraries(<target> [item1] [item2] [...] [[debug|optimized|general] <item>] ...)
# <target>  是 add_executable 生成的目标文件。OpenCV_LIBS 是一个可执行文件的变量，表示
# 可以通过下面的方式链接OpenCV 库。
target_link_libraries(OpenCVDemo ${OpenCV_LIBS})
target_link_libraries(OpenCVDemo1 ${OpenCV_LIBS})
target_link_libraries(OpenCVDemo2 ${OpenCV_LIBS})
target_link_libraries(OpenCVDemo3 ${OpenCV_LIBS})