


rm -rf build && ./build.sh -r -t 2 -p qcom

adb push /home/wangzhijiang/01.WorkSpace/wangzhijiang/build/android-arm64-v8a-release/test/test_image /data/local/frewen/

adb shell " cd /data/local/frewen/   &&    \
           ls -l  && \
           export LD_LIBRARY_PATH=${PWD}:/vendor/lib64   && \
           ./test_image
           "