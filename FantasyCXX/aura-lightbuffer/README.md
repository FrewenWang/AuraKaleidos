# Aura-LightBuffer

## 概述

`LightBuffer` 是轻量级的接口定义和二进制序列化工具，用于替代原工程依赖的 Protocol Buffers。

核心功能包括接口定义支持（IDL）、二进制序列化和反序列化、文本配置解析。

LightBuffer 的 schema 是 protocol buffer 的子集，目前实现了一些常用的核心特性，兼容 proto2 语法，使用时基本可以无缝切换。

LightBuffer 无第三方依赖，库体积小（< 600KB），便于跨平台编译和集成。

支持的一些特性如下:

* 基于 Message 定义消息（或配置）的格式；
* 支持文档注释；
* 基本类型支持 string, int64, int32, int16, int8, float, double, bool, bytes；
* 自动代码生成（目前支持 cpp 和 python）；
* 配置文件读取和解析，配置文本的结构兼容 protocol buffer 的文本格式；
* 序列化采用平铺格式，无索引结构，相对比较紧凑；
* 接口形式兼容 protobuf，切换使用代码变更较小；
* 对于 Message 的嵌套定义目前尚不支持；

## 编译

### 编译 LightBufferC

进入 LightBuffer 目录下,执行编译脚本:



```shell script
./build.sh
```
