#!/bin/bash

adb wait-for-device root
adb wait-for-device remount

adb shell setprop persist.vendor.camera.mivi.loglevel 0          # 控制log等级，越小log越多
adb shell setprop persist.vendor.camera.mivi.groupsEnable 0xFFFF # 控制log组（预览、拍照等），这个值默认全部的组log开启
adb shell setprop persist.vendor.camera.offlinedebug.mask 0      # 开启offlinelog

adb shell setprop persist.vendor.mialgo.sd.log 0

adb shell killall cameraserver com.android.camera vendor.qti.camera.provider-service_64
