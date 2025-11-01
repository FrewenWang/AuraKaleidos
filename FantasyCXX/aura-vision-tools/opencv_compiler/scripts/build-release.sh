#!/usr/bin/env bash


# I_BUILD_VRESION="2.4.13.4"
# I_BUILD_VRESION="4.7.0"
# I_BUILD_VRESION="4.7.0"
I_BUILD_VRESION="4.9.0"

URL_OPENCV="https://github.com/opencv/opencv/archive/refs/tags/$I_BUILD_VRESION.zip"
URL_OPENCV_CONTRIB="https://github.com/opencv/opencv_contrib/archive/refs/tags/$I_BUILD_VRESION.zip"


ZIP_OPENCV="opencv-$I_BUILD_VRESION.zip"
ZIP_OPENCV_CONTRIB="opencv_contrib-$I_BUILD_VRESION.zip"

rm -rf "$ZIP_OPENCV"
rm -rf "$ZIP_OPENCV_CONTRIB"

rm -rf "opencv-$I_BUILD_VRESION"
rm -rf "opencv_contrib-$I_BUILD_VRESION"

wget -O "$ZIP_OPENCV" "$URL_OPENCV"
wget -O "$ZIP_OPENCV_CONTRIB" "$URL_OPENCV_CONTRIB"

unzip -q "$ZIP_OPENCV"
unzip -q "$ZIP_OPENCV_CONTRIB"

DIR_INSTALL="../../install"
DIR_CONTRIB="../../opencv_contrib-$I_BUILD_VRESION/modules"

# shellcheck disable=SC2046
./build.sh   -r   -v "$I_BUILD_VRESION" -a $(cat options.txt)  -i  "$DIR_INSTALL"  -e "$DIR_CONTRIB"
