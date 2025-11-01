


rm -rf build && ./build.sh -r -t 2 -p qcom -s qcom

adb push build/android-arm64-v8a-release/install/gaussian_cl/bin/test_image /data/local/frewen/

adb shell " cd /data/local/frewen/   &&    \
           ls -l  && \
           export LD_LIBRARY_PATH=${PWD}:/vendor/lib64   && \
           ./test_image
           "