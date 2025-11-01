if [ "$(uname)" == "Darwin" ]; then
  TARGET="osx"
  ARCH="x86_64"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  TARGET="linux"
  ARCH="x86_64"
fi

# 设置编译类型为release版本
BUILD_TYPE=release
# 当前目录下创建build目录
if [ ! -d build ]; then
  mkdir -p build
fi
cd build

# auralib-cmake [option] <path-to-source> 指向含有顶级CMakeLists.txt的那个目录
# auralib-cmake [option] <path-to-existing-build> 指向含有CMakeCache.txt的那个目录
# 指向含有CMakeCache.txt的那个目录执行cmake指令
cmake -DBUILD_PLAINC=ON ..
# 在当前目录下执行构建一个工程。 --target是plainc
cmake --build . --target plainc

# 创建prebuilt。删除目录中原有数据
cd ..
if [ ! -d prebuilt/${TARGET}-${ARCH} ]; then
  mkdir -p prebuilt/${TARGET}-${ARCH}
else
  rm -rf prebuilt/${TARGET}-${ARCH}/*
fi

# 拷贝生成物到prebuilt目录
cp ./build/plainc ./prebuilt/${TARGET}-${ARCH}/
