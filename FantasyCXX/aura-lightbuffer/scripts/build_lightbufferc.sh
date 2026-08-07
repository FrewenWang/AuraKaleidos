if [ "$(uname)" == "Darwin" ]; then
  TARGET="osx"
  ARCH="x86_64"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  TARGET="linux"
  ARCH="x86_64"
fi

BUILD_TYPE=release

if [ ! -d build ]; then
  mkdir -p build
fi
cd build

cmake -DBUILD_PLAINC=ON ..
cmake --build . --target plainc

cd ..
if [ ! -d prebuilt/${TARGET}-${ARCH} ]; then
  mkdir -p prebuilt/${TARGET}-${ARCH}
fi
cp ./build/plainc ./prebuilt/${TARGET}-${ARCH}/
