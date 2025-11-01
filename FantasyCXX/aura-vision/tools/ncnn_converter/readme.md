## ncnn_converter【caffe2ncnn】
    caffe使用protobuf序列化协议定义模型结构及序列化模型权重。
    因此解析caffe模型需要protobuf环境。
    
### 1 caffe2ncnn工具
编译使用脚本`build.sh`  
osx-x86_64已预编译该工具：prebuilt/osx-x86_64/caffe2ncnn
* 在ncnn开源的caffe2ncnn converter基础上支持模型版本记录功能，以方便版本回溯
* 使用方式同原生caffe2ncnn
```shell
./caffe2ncnn [caffeproto] [caffemodel] [ncnnproto] [ncnnbin] [quantizelevel] [int8scaletable]
``` 

### 2 protobuf-3.13.0 mac osx-x86_64编译安装
#### 2.1 下载安装(安装至默认目录)
```shell script
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-all-3.13.0.tar.gz
tar -xvf protobuf-all-3.13.0.tar.gz
cd protobuf-all-3.13.0
./configure
make && make install
```
#### 2.2 环境变量设置
```shell script
# which protoc 查询实际的安装路径后（不一定是以下固定路径），按以下方式添加
# 在 ~/.bash_profile 中加入：
export PROTOBUF=/usr/local/protobuf (或者是export PROTOBUF=/usr/local)
export PATH=$PROTOBUF/bin:$PATH
# 保存后更新环境变量
source ~/.bash_profile
```
#### 2.3 测试protobuf环境
```shell script
protoc --version 
# 输出如下则安装成功：
libprotoc 3.13.0
```

### caffe.proto 说明
使用protoc，根据caffe.proto，在当前目录下生成cpp的序列化结构类文件caffe.pb.h和caffe.pb.cc：  
`protoc --cpp_out=./ caffe.proto`  
当前目录下已存在的caffe.pb.h和caffe.pb.cc为3.13.0版本的protoc生成，
因此编译caffe2ncnn工具时，保持当前环境下的protoc同样为3.13.0版本。