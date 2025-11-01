#!/usr/bin/env bash

mkdir build
cd build
cmake ..
make

cd ..
cp build/caffe2ncnn prebuilt/osx-x86_64