<h1 align="center">可视化使用说明</h1>

## Description
*基于Aura 2.0的可视化需求，采用更为直观的方式来观察各个算子的优化程度，并对比OpenCV的性能，生成动态的优化报告。*

## Install
### 环境依赖：
>	*python V3.8*

>	*pyecharts 1.9.1*

安装好python3后通过pip安装pyecharts
```
pip3 install pyecharts
```
## Usage
python代码实现在aura/tools/aura2.0/visual.py

解析本地auto_test.json：
```
python3 ./visual.py -p filepath
```
报告会生成在当前路径下以传入的json文件名命名的文件夹中

解析本地文件夹中所有json文件：
```
python3 ./visual.py -p folderpath
```
报告会生成在当前路径下visual_report文件夹中

解析手机上auto_test.json：
```
python3 ./visual.py -s -p mobile_filepath
```
报告会生成在当前路径下以传入的json文件名命名的文件夹中

解析手机上文件夹下所有json文件：
```
python3 ./visual.py -s -p mobile_folderpath
```
报告会生成在当前路径下visual_report文件夹中

-s 后可以加设备序列号来指定（通过adb devices来获取）

增加filter选项，通过-f来指定要过滤的输入输出数据宽度大小：
```
python3 ./visual.py -p folderpath/file_path -f 1000
```
类型为int，默认值为0，输入和输出数据均小于指定宽度大小将被过滤掉，不会显示在图表中。
### 关于如何生成auto_test.json:
首先生成config.json
```
./aura_test_main -c -d config.json
```
将config.json中report_type改为json:
```
{
    "async_affinity": "LITTLE",
    "cache_bin_path": "",
    "cache_bin_prefix": "",
    "compute_affinity": "BIG",
    "data_path": "./data/",
    "device_info": "unknown",
    "is_stress_test": false,
    "log_file_name": "log",
    "log_level": "DEBUG",
    "log_output": "STDOUT",
    "ndk_info": "ndk_r19c",
    "report_name": "auto_test",
    "report_type": "json",
    "stress_count": 1
}
```
运行aura_test_main即可生成对应的auto_test.json
```
./aura_test_main -r config.json case_name
```

## Charts
在当前文件夹下生成如下目录：
```
visual_report
├── interfaces
│   ├── interface1.html
│   └── interface2.html
│   └── ......
├── Aura 2.0.html
```
Aura2.0.html显示树图结构，通过点击叶节点跳转到对应interface的报告：
<p align="center">
<img  src="./iauras/tree.png" width="80%">
</p>

可视化界面：
<p align="center">
<img  src="./iauras/3.png" width="100%">
</p>

- 标题为interface的名称

- 副标题为interface对应的参数设置

- 图例配置项可以选择显示对应实现的用时，以及不同实现之间的加速比（加速比图中会虚线标识当前图表中的平均加速比）

- 工具栏有3个工具，分别为数据视图，保存图片和还原

- 区域缩放配置项可以放大局部来观察具体的用例，通过拖动区域组件或鼠标滚轮来选择

- 性能柱状图显示具体性能信息

- 精度测试结果通过饼状图来显示

<p align="center">
<img  src="./iauras/data_view.png" width="80%">
</p>

- 数据视图主要通过表格来显示具体的数据值，为每个输入输出加上序号便于在柱状图上查找

<h1 align="center">json解析脚本使用说明</h1>

## Description

主要用于解析aura_test_main生成的auto_test.json文件

Aura2.0可视化脚本可以通过图像直观展示各个算子性能，但是在数据显示上可读性不足，该脚本将json文件中的数据统计到EXCEL表格中，便于阅读比较算子性能数据。

有两个python文件，其中：

1. json_parser_complete.py 将json文件中所有的数据全部统计到EXCEL中，会生成多个表格（每个模块单独生成一个表格），如果解析多个json文件会生成多个文件夹，存放路径是：设置的输出路径/机型名/模块名.xlsx；
2. json_parser_simplified.py 会将整个AURA2.0工程中所有算子性能全部统计到一个aura2.xlsx表格中（注意：并没有统计完整的数据，筛选了每个算子常用的尺寸数据进入到EXCEL表格)

## Usage

python代码实现在aura/tools/aura2.0/json_parser_complete.py; aura/tools/aura2.0/json_parser_simplified.py, json_parser_simplified.py和json_parser_complete.py对外接口完全一致，下面使用介绍以json_parser_complete.py为例

python json_parser_complete.py -d 机型名 -p  json路径 -o 输出EXCEL表格存放路径

eg: python json_parser_complete.py 

-d **MTK_DX1,QCOM-8550,QCOM-8650** 

-p *"../../../visual_report/auto_test_all_mtkdx1.json","../../../visual_report/auto_test_all_8550.json","../../../visual_report/auto_test_all_8650.json"* 

-o ../../../build

**注意：**

脚本支持同时解析多个json文件

-d后面接的设备列表和 -p后面接的json文件路径列表，列表元素之间以英语逗号","隔开，元素之间不能存在空格，而且设备列表数目和json文件路径数目必须相等。

-o 是可选项，如果不设置，输出EXCEL文件就存放在当前python文件(json_parser_complete.py)存放的文件夹

表格记录是以none测试为标准，所以算子性能记录必须有none实现，否则会异常退出
