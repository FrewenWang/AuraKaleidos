@echo off
set PATH=%~dp0\.platform-tools;%PATH%
set TARGET_PREFIX=/data/local/tmp/.qdb
set TARGET_QNX=198.18.32.11

adb wait-for-device root >_qdb.log
adb wait-for-device >>_qdb.log

adb shell mkdir -p %TARGET_PREFIX% >>_qdb.log
adb push %~dp0\..\..\..\target\bin\ssh %TARGET_PREFIX% >>_qdb.log
adb push %~dp0\..\..\..\target\lib\libssh.so %TARGET_PREFIX% >>_qdb.log
adb push %~dp0\..\..\..\target\bin\sftp %TARGET_PREFIX% >>_qdb.log
adb shell chmod u+x %TARGET_PREFIX%/ssh >>_qdb.log
adb shell chmod u+x %TARGET_PREFIX%/sftp >>_qdb.log

if "%1"=="" (
    adb forward tcp:8022 tcp:8022 >>_qdb.log
    adb shell -tt LD_LIBRARY_PATH=%TARGET_PREFIX% %TARGET_PREFIX%/ssh -E /dev/null -o StrictHostKeyChecking=no -o HostKeyAlgorithms=ssh-rsa -L 8022:%TARGET_QNX%:22 root@%TARGET_QNX%
) ^
else if "%1"=="push" (
    adb push "%2" %TARGET_PREFIX%
    adb shell "echo put %TARGET_PREFIX%/%2 %3 >%TARGET_PREFIX%/sftp.bat" >>_qdb.log
    adb shell LD_LIBRARY_PATH=%TARGET_PREFIX% %TARGET_PREFIX%/ssh -E /dev/null -o StrictHostKeyChecking=no -o HostKeyAlgorithms=ssh-rsa root@%TARGET_QNX% /mnt/bin/mount -uw /mnt >>_qdb.log
    adb shell LD_LIBRARY_PATH=%TARGET_PREFIX% %TARGET_PREFIX%/sftp -o StrictHostKeyChecking=no -o HostKeyAlgorithms=ssh-rsa -S %TARGET_PREFIX%/ssh -b %TARGET_PREFIX%/sftp.bat root@%TARGET_QNX%
) ^
else if "%1" == "pull" (
    adb shell "echo get %2 %TARGET_PREFIX%/%~n2%~x2 >%TARGET_PREFIX%/sftp.bat" >>_qdb.log
    adb shell LD_LIBRARY_PATH=%TARGET_PREFIX% %TARGET_PREFIX%/sftp -o StrictHostKeyChecking=no -o HostKeyAlgorithms=ssh-rsa -S %TARGET_PREFIX%/ssh -b %TARGET_PREFIX%/sftp.bat root@%TARGET_QNX%
    adb pull %TARGET_PREFIX%/%~n2%~x2
)
