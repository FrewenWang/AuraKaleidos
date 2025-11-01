#!/usr/bin/env bash

show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -t, --target            Set platform target 1-android-armv7a, 2-android-armv8a, 3-android-x86_64, 4-android-x86"
    echo "   -s, --speed             Set test operation to speed_test"
    echo "   -a, --accuracy          Set test operation to accuracy_test"
    echo "   -f, --feature           Set accuracy test feature，supported features is below：
                                         dms | face_call | face_liveness_ir | face_liveness_rgb
                                         face_recognize | gesture_type | face_landmark_eye_close
                                         face_attribute_rgb | face_attribute_ir | face_eye_center
                                         face_cover | face_emotion
                                      "
    echo "   -l, --local              Use local detect result do feature eval"
    echo "   -h, --help              show help message"
    echo
}

BUILD_TYPE="Release"
TARGET="android-arm64-v8a"

USE_LOCAL_DETECT_RESULT="False"
# parse arguments
while [ $# != 0 ]
do
  case "$1" in
    -t)
        TARGET_INDEX=$2
        shift
        ;;
    --target)
        TARGET_INDEX=$2
        shift
        ;;
    -s)
        TEST_OPERATION="speed"
        ;;
    --speed)
        TEST_OPERATION="speed"
        ;;
    -a)
        TEST_OPERATION="accuracy"
        ;;
    --accuracy)
        TEST_OPERATION="accuracy"
        ;;
    -f)
        TEST_FEATURE=$2
        ;;
    --feature)
        TEST_FEATURE=$2
        ;;
    -l)
        USE_LOCAL_DETECT_RESULT="True"
        ;;
    --local)
        USE_LOCAL_DETECT_RESULT="True"
        ;;
    -h)
        show_help
        exit 1
        ;;
    --help)
        show_help
        exit 1
        ;;
    *)
        ;;
  esac
  shift
done

case "$TARGET_INDEX" in
1)
    TARGET="android"
    ARCH="armeabi-v7a"
    ;;
2)
    TARGET="android"
    ARCH="arm64-v8a"
    ;;
3)
    TARGET="android"
    ARCH="x86_64"
    ;;
4)
    TARGET="android"
    ARCH="x86"
    ;;
*)
    echo "Not supported target!"
    exit 1
    ;;
esac

XPERF_BIN=build/${TARGET}-${ARCH}-${BUILD_TYPE}/xperf
if [ ! -d xperf-run ]; then
  mkdir xperf-run
fi
cp ${XPERF_BIN} xperf-run/
cp ../../../build/${TARGET}-${ARCH}-${BUILD_TYPE}/install/lib/${ARCH}/*.so xperf-run/
cp res/*.jpg xperf-run/

if [ -d xperf-run ]; then
  adb push xperf-run /data/local/tmp/
  bin_path="/data/local/tmp/xperf-run"
  adb shell "chmod +x ${bin_path}/xperf"

  if [ $TEST_OPERATION == "speed" ]; then
    adb shell "cd ${bin_path} \
           && export LD_LIBRARY_PATH=${bin_path}:${LD_LIBRARY_PATH} \
           && ./xperf speed test_face.jpg"
  elif [ $TEST_OPERATION == "accuracy" ]; then
    python py_eval/eval.py \
        --bin_path=${bin_path} \
        --eval_feature=$TEST_FEATURE \
        --use_local_detect_result=$USE_LOCAL_DETECT_RESULT
  else
    echo "Test operatioon not supported!"
  fi

else
  echo "No executable found, please compile vision_demo first!"
fi