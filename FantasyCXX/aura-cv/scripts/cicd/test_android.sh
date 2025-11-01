#!/bin/bash

# helper function
Error()
{
    if [ $# -eq 2 ] && [ $1 != 0 ]; then
        echo -e "\033[41;37m[aura2.0 test]: $2\033[0m"
        exit 1
    fi
}

SearchOps()
{
    serarch_path=$1
    # echo "search_path: $serarch_path"

    cd "$serarch_path"

    ops_searched=$(ls -d */ | sed 's#/##')
    # echo "ops_searched: $ops_searched"

    ops_searched_unroll=$(echo ${ops_searched[@]} | tr '\n' ' ')
    # echo "ops_searched_unroll: $ops_searched_unroll"

    echo "$ops_searched_unroll"
}

GetAndroidDevice()
{
    [ $# -lt 2 ] && Error 1 "param < 2"

    local find_hardware=$1
    local vendor_id=$2

    local device_info_vec=()

    local search_file="data/misc/platform_aura2_cicd.txt"

    for sno in $(adb devices | grep ".device$" | awk '{print $1}')
    do
        adb -s $sno wait-for-device root 1>/dev/null 2>&1
        #hardware=$(adb -s $sno shell cat /proc/cpuinfo | grep "Hardware")
        vendor=$(adb -s $sno shell cat "$search_file" | grep 'Vendor' | awk -F ':' '{print $NF}')
        hardware=$(adb -s $sno shell cat "$search_file" | grep 'HardWare' | awk -F ':' '{print $NF}')

        if [[ $vendor == *"$vendor_id"* ]]; then
            if [ ${hardware} == "$find_hardware" ]; then
                device_info=$sno
                device_info_vec+=("$device_info")
            fi
        fi
    done

    echo "${device_info_vec[@]}"
}

InitDevice()
{
    [ $# -lt 2 ] && Error 1 "param < 2"

    local device_id=$1
    local platform_info=$2

    echo "init device"

    timeout 20 adb -s ${device_id} wait-for-device root
    Error $? "exec adb root failed"
    timeout 20 adb -s ${device_id} wait-for-device remount
    Error $? "exec adb remount failed"
    timeout 20 adb -s ${device_id} wait-for-device reboot
    Error $? "exec adb reboot failed"
    timeout 120 adb -s ${device_id} wait-for-device root
    Error $? "exec adb root failed"
    timeout 20 adb -s ${device_id} wait-for-device remount
    Error $? "exec adb remount failed"

    if [ ${platform_info} == "MTK" ]; then
        adb -s ${device_id} shell 'echo 0 0 > /proc/ppm/policy/ut_fix_freq_idx'
        adb -s ${device_id} shell 'echo 0 0 0 > /proc/ppm/policy/ut_fix_freq_idx'
    fi

    #wait qcom turn off the screen
    echo "wait for 1min for wait qcom turn off the screen"
    sleep 1m
    echo "wait end"

    # turn on the screen
    echo "turn on the screen"
    adb -s ${device_id} shell input keyevent 224
    sleep 2s
    adb -s ${device_id} shell input swipe 300 1000 300 500
    adb -s ${device_id} shell settings put system screen_off_timeout 2147483647
    adb -s ${device_id} shell settings put system def_lockscreen_disabled true
}

PushData()
{
    [ $# -lt 3 ] && Error 1 "param < 3"

    local device_id=$1
    local host_data_path=$2
    local device_data_path=$3

    echo "push data to device"

    # adb -s ${device_id} shell rm -rf ${device_data_path}
    # adb -s ${device_id} shell mkdir -p ${device_data_path}

    adb -s ${device_id} push ${host_data_path}/ ${device_data_path}/ > /dev/null
    Error $? "push data to device failed, $device_id, $host_data_path, $device_data_path"
}

PushAndroidExe()
{
    # find: locate qcom/arm64-v8a/static/release/non_asan/install/**/aura_test_main

    [ $# -lt 4 ] && Error 1 "param < 4"

    local device_id=$1
    local root_search_path=$2
    local device_path=$3
    local run_test_type=$4

    echo "push android exe to device"

    # 1.1 find the path of aura_test_main
    if [ -d ${root_search_path} ]; then
        local exe_path=$(find ${root_search_path} -path "*/install/*" -type f -name "aura_test_main")

        if [ ! -f ${exe_path} ]; then
            Error 1 "aura_test_main not found in path: $root_search_path"
        fi
    else
        Error 1 "root_search_path: $root_search_path not found"
    fi

    # 1.1 push aura_test_main to device
    adb -s ${device_id} push ${exe_path} ${device_path} > /dev/null
    Error $? "push aura_test_main to device failed: ${device_id}, ${exe_path}, ${device_path}"

    # 2. push hwasan's lib
    if [ "$run_test_type" == "hwasan" ]; then
        local lib_path=$(find ${NDK_PATH} -type f -name "libclang_rt.hwasan-aarch64-android.so")

        if [ ! -f ${lib_path} ]; then
            Error 1 "libclang_rt.hwasan-aarch64-android.so not found in path: $NDK_PATH"
        fi

        adb -s ${device_id} push ${lib_path} ${device_path} > /dev/null
        Error $? "push aura_test_main to device failed: ${device_id}, ${lib_path}, ${device_path}"
    fi
}

PushHexagonLibs()
{
    # find: locate build_hexagon/v68/share/release/install/**/libaura_hexagon_skel.so
    [ $# -lt 3 ] && Error 1 "param < 3"

    local device_id=$1
    local root_search_path=$2
    local device_path=$3

    echo "push hexagon libs to device"

    # find the path of aura_test_main
    if [ -d ${root_search_path} ]; then
        local exe_path=$(find ${root_search_path} -path "*/install/*" -type f -name "libaura_hexagon_skel.so")

        if [ ! -f ${exe_path} ]; then
            Error 1 "aura_test_main not found in path: $root_search_path"
        fi
    else
        Error 1 "root_search_path: $root_search_path not found"
    fi

    # push aura_test_main to device
    adb -s ${device_id} push ${exe_path} ${device_path} > /dev/null
    Error $? "push libaura_hexagon_skel.so to device failed, $deivce_id, $exe_path, $device_path"
}

# Terminate the background processes.
CleanBackgroudPid()
{
    [ $# -lt 1 ] && Error 1 "param < 1"

    pids=$1

    for pid in ${pids[@]}; do
        if [ -n "$pid" ]; then
            if ps -p $pid > /dev/null 2>&1; then
                echo "start kill cur pid: $pid"

                kill -9 $pid
                Error $? "kill cur pid failed: $pid"

                echo "kill cur pid: $pid sucess"
            else
                echo "cur pid:$pid not exist"
            fi
        fi
    done
}

CleanlogcatPid()
{
    echo "start kill logcat pid: $logcat_pid"

    CleanBackgroudPid "$logcat_pid"

    echo "kill logcat pid: $logcat_pid sucess"
}

task_group_pids=()
CleanTaskGroupPids()
{
    echo "start kill task group pids: ${task_group_pids[*]}"

    CleanBackgroudPid "${task_group_pids[*]}"

    echo "kill task group pids: ${task_group_pids[*]} sucess"

    # 下面是针对CI/CD卡住的临时解决方法: 由于当前脚本退出时CI/CD会卡住，怀疑手机和CI/CD存在某个adb通信进程阻塞了CI/CD进程，导致CI/CD卡住不能进行后续操作, 故重启手机切断可能的进程
    echo "start adb reboot: ${device_info[*]}"

    for device_id in ${device_info[@]}; do
        adb -s $device_id reboot
    done

    echo "adb reboot: ${device_info[*]} sucess"
}

# Capture SIGTERM and SIGINT, then call CleanlogcatPid
trap CleanlogcatPid SIGTERM SIGINT

# Capture SIGTERM, SIGINT and EXIT, then call CleanTaskGroupPids
trap CleanTaskGroupPids SIGTERM SIGINT EXIT

RunTest()
{
    [ $# -lt 7 ] && Error 1 "param < 7"

    local device_id=$1
    local platform_info=$2
    local run_test_type=$3

    local work_path=$4
    local build_android_path=$5

    local device_test_path=$6
    local op_cases=$7

    echo "run test"

    # 1.1 generate auto_test.json
(
cat << EOF
{
    "async_affinity": "LITTLE",
    "cache_bin_path": "",
    "cache_bin_prefix": "",
    "compute_affinity": "BIG",
    "data_path": "./../data/",
    "device_info": "$platform_info",
    "log_file_name": "log",
    "log_level": "DEBUG",
    "log_output": "STDOUT",
    "ndk_info": "ndk_r26c",
    "report_name": "auto_test",
    "report_type":"txt",
    "stress_count": 0
}
EOF
) > ${work_path}/${build_android_path}/config.json
    Error $? "create config.json failed"

    # push config.json to device
    adb -s ${device_id} push ${work_path}/${build_android_path}/config.json ${device_test_path}
    Error $? "push config.json to device failed"

    # 1.2 generate auto_test.sh
(
cat << EOF
#!/bin/sh
TEST_CASE_PATH=${device_test_path}
TEST_CASE_LIST="${op_cases}"
RUN_TEST_TYPE="${run_test_type}"

export LD_LIBRARY_PATH="\${TEST_CASE_PATH}:/system/lib64:/vendor/lib64:/vendor/lib64/egl"

PWD=\$(pwd)
cd \${TEST_CASE_PATH}/

echo "[aura2.0 test]: begin test"

# serarch cases by '-l'
run_cases=()
for case in \${TEST_CASE_LIST[@]}; do
    rets_l=\$(\${TEST_CASE_PATH}/aura_test_main -l \${case})

    echo "rets_l: \${rets_l}"

    sub_run_cases=\$(awk '/^======================================================================================/{n++} \
                n==2{start=1} start==1&&/^======================================================================================/{start=0;next} start==1{print}' <<< "\$rets_l")

    run_cases+=("\${sub_run_cases[@]}")
done

echo "run_cases_num: "\${#run_cases[@]}""

run_cases_unroll=\$(echo \${run_cases[@]})
echo "run_cases_unroll: \${run_cases_unroll}"

stress_count=5

# filter the none test for asan and hwasan
if [ "\$RUN_TEST_TYPE" == "asan" ] || [ "\$RUN_TEST_TYPE" == "hwasan" ]; then
    # Use `tr` to split the string into multiple lines, then use `grep` to remove lines ending with 'none',
    # and finally combine the lines back into a single string
    run_cases_unroll_filtered=\$(echo \$run_cases_unroll | tr ' ' '\n' | grep -v '_none\$' | tr '\n' ' ')

    # Remove the trailing space
    run_cases_unroll=\$(echo \$run_cases_unroll_filtered | sed 's/ \$//')

    echo ""
    echo "run_cases_unroll filered none: \${run_cases_unroll}"

    stress_count=2
fi

# filter the hvx test for MTK
if [ "$platform_info" == "MTK" ]; then
    run_cases_unroll_filtered=\$(echo \$run_cases_unroll | tr ' ' '\n' | grep -v '_hvx\$' | tr '\n' ' ')

    # Remove the trailing space
    run_cases_unroll=\$(echo \$run_cases_unroll_filtered | sed 's/ \$//')

    echo ""
    echo "run_cases_unroll filered hvx: \${run_cases_unroll}"
fi

echo "get test cases over"
echo "stress_count: \${stress_count}"

# run cases by '-r'
\${TEST_CASE_PATH}/aura_test_main -r \${TEST_CASE_PATH}/config.json \${run_cases_unroll} -s \${stress_count}

if [ \$? != 0 ];then
    echo "[aura2.0 test]: test failed"
    cd \${PWD}
    exit 1
fi

echo "[aura2.0 test]: end test"
cd \${PWD}

EOF
) > ${work_path}/${build_android_path}/auto_test.sh
    # push auto_test.sh to device
    chmod a+x ${work_path}/${build_android_path}/auto_test.sh
    adb -s ${device_id} push ${work_path}/${build_android_path}/auto_test.sh ${device_test_path}
    Error $? "push auto_test.sh to device failed"

    adb -s ${device_id} shell "mkdir -p ${device_test_path}/log"
    Error $? "create log dir failed"

    mkdir -p "${work_path}/${build_android_path}/log"
    Error $? "create build log dir failed: ${work_path}/${build_android_path}"

    # 2.1 run auto_test.sh
    date
    new_name_run_log="${work_path}/${build_android_path}/log/log_${platform_info}_${run_test_type}_$(date +%F_%T).txt"
    new_name_auto_test="${work_path}/${build_android_path}/log/auto_test_${platform_info}_${run_test_type}_$(date +%F_%T).txt"
    name_logcat="${work_path}/${build_android_path}/log/logcat_${platform_info}_${run_test_type}_$(date +%F_%T).txt"

    echo "new_name_run_log: $new_name_run_log"
    echo "new_name_auto_test: $new_name_auto_test"
    echo "name_logcat: $name_logcat"

    # 2.1.1 logcat
    # Start adb logcat and run it in the background, redirecting the output to a log file
    {
        adb -s ${device_id} logcat -c
        adb -s ${device_id} logcat | grep -E "kgsl-3d0|FAULT|fail|DEBUG|backtrace|crash|error|leak|build with source" > "${name_logcat}"
    } &

    # get pid of logcat
    logcat_pid=$!
    echo "logcat_pid: $logcat_pid, main_pid: $$"

    # 2.1.2 run auto_test.sh
    echo "execute auto_test.sh"

    CMD="${device_test_path}/auto_test.sh &> ${device_test_path}/log/auto_test_log.txt; echo \$?"
    ret=$(adb -s ${device_id} shell "$CMD")

    # Terminate the background logcat process.
    CleanlogcatPid

    # if test fail, firstly pulling test log
    if [ "$ret" != 0 ]; then
        adb -s ${device_id} pull ${device_test_path}/log/auto_test_log.txt "${new_name_run_log}" > /dev/null
        adb -s ${device_id} pull ${device_test_path}/auto_test.txt "${new_name_auto_test}" > /dev/null
        Error $ret "exec auto_test.sh failed, device_test_path: ${device_test_path}"
    fi

    Error $ret "exec auto_test.sh failed"
    date

    adb -s ${device_id} pull ${device_test_path}/log/auto_test_log.txt "${new_name_run_log}" > /dev/null
    adb -s ${device_id} pull ${device_test_path}/auto_test.txt "${new_name_auto_test}" > /dev/null
    Error $? "pull test log failed, device_test_path: ${device_test_path}"

    # 2.2 check test result
    parsed_file="${new_name_auto_test}"
    # Parse the value of 'Failed:'
    failed=$(grep 'Failed:' "$parsed_file" | awk '{print $2}')
    Error $failed "auto_test failed, please check $parsed_file"

    # 2.3 copy auto_test.txt to artifact path
    log_saved_path="${work_path}/log/android"

    mkdir -p "$log_saved_path"
    [ ! -d "$log_saved_path" ] && Error 1 "Failed to create directory ${log_saved_path}"

    cp "$new_name_run_log" "$log_saved_path"
    cp "$new_name_auto_test" "$log_saved_path"
    cp "$name_logcat" "$log_saved_path"
    Error $? "Failed to copy ${new_name_auto_test} to ${log_saved_path}"

    echo "****** cur test sucess: platform_info: $platform_info, run_test_type: $run_test_type, device_id: $device_id **********"
}

# check num of param
[ $# -lt 4 ] && Error 1 "param < 4"

work_path=$1
build_android_path=$2
build_hexagon_path=$3
platform_info=$4

# set build env
source /home/mi-aura/workspace/aura2.0_build_env/set_env.sh

# get device info
device_info=()
hvx_install_prefix=()
platform_dir=""
hvx_libs_device_path=""

if [ "$platform_info" == "SM8550" ]; then
    device_info=$(GetAndroidDevice "$platform_info" "Qualcomm")
    hvx_install_prefix="$work_path/$build_hexagon_path/v73/share/release"
    platform_dir="android"
    hvx_libs_device_path="/odm/lib/rfsa/adsp"
elif [ "$platform_info" == "SM8650" ]; then
    device_info=$(GetAndroidDevice "$platform_info" "Qualcomm")
    hvx_install_prefix="$work_path/$build_hexagon_path/v75/share/release"
    platform_dir="android"
    hvx_libs_device_path="/odm/lib/rfsa/adsp"
elif [ "$platform_info" == "MTK" ]; then
    device_info=$(GetAndroidDevice "MT6897" "MTK")
    platform_dir="android"
else
    Error 1 "don't support platform: $platform_info"
fi

echo "device_info: ${device_info[*]}"
IFS=' ' read -r -a device_info <<< "${device_info[*]}"

if [ -z "$device_info" ]; then
    Error 1 "can't find device"
fi

# run_test_num=("non_asan" "hwasan" "asan")
run_test_num=("non_asan" "hwasan")

real_device_num=${#device_info[@]}
real_test_num=${#run_test_num[@]}
echo "real_device_num: $real_device_num, real_test_num: $real_test_num"

# op cases
run_op_cases_str="$(SearchOps "$work_path/src/ops")"
echo "run_op_cases_str: ${run_op_cases_str}"
IFS=' ' read -r -a run_op_cases <<< "$run_op_cases_str"

# run_op_cases=("cvtcolor" "feature2d")
echo "run_op_cases: ${run_op_cases[*]}"

cd "$work_path"

device_work_path="/data/local/tmp/aura2.0_test"

TaskGroup()
{
    [ $# -lt 2 ] && Error 1 "param < 2"

    local run_test=$1
    local device_id=$2

    echo "**** start TaskGroup run_test: $run_test, device_id: $device_id ****"

    adb -s ${device_id} shell mkdir -p $device_work_path
    Error $? "create $device_work_path failed"

    # init
    InitDevice "$device_id" "$platform_info"

    # push datasets
    PushData "$device_id"  "$work_path/data" "$device_work_path"

    # test 

    # push hexagon libs
    if [ ${platform_info} != "MTK" ]; then
        PushHexagonLibs "$device_id" "$hvx_install_prefix" "$hvx_libs_device_path"
    fi

    device_test_path=$device_work_path/${platform_info}_${run_test}
    adb -s ${device_id} shell rm -rf $device_test_path
    adb -s ${device_id} shell mkdir -p $device_test_path

    if [ "$run_test" == "non_asan" ]; then
        build_type="release"
    elif [ "$run_test" == "asan" ] || [ "$run_test" == "hwasan" ]; then
        build_type="debug"
    fi

    # push aura_test_main to device
    PushAndroidExe "$device_id" "$work_path/$build_android_path/$platform_dir/arm64-v8a/static/$build_type/$run_test/" "$device_test_path" "$run_test"

    # run test
    RunTest "$device_id" "$platform_info" "$run_test" "$work_path" "$build_android_path" "$device_test_path" "${run_op_cases[*]}"

    # turn off the screen
    adb -s ${device_id} shell settings put system screen_off_timeout 300000
    adb -s ${device_id} shell settings put system def_lockscreen_disabled false

    echo "**** end TaskGroup run_test: $run_test, device_id: $device_id ****"
}

# Async run test
task_per_device=$((real_test_num / real_device_num))
remainder_num=$((real_test_num % real_device_num))

for ((i = 0; i < real_device_num; i++)); do
    device_id=${device_info[$i]}

    start_index=$((i * task_per_device))
    end_index=$((start_index + task_per_device - 1))

    # for remainder
    if [[ $i -eq $((real_device_num - 1)) && $remainder_num -gt 0 ]]; then
        end_index=$((end_index + remainder_num))
    fi

    echo "Assign task to device: $device_id,: start_index: $start_index, end_index: $end_index"

    if [[ $real_device_num -eq 1 ]]; then
        # Run it in the foreground if only one device
        echo " Run it in the foreground"
        for ((j = start_index; j <= end_index; j++)); do
            run_test_type=${run_test_num[$j]}

            if [[ $j -lt $real_test_num ]]; then
                    TaskGroup "$run_test_type" "$device_id"
            fi
        done
    else
        # Run it in the background
        echo " Run it in the background"
        {
            for ((j = start_index; j <= end_index; j++)); do
                run_test_type=${run_test_num[$j]}

                if [[ $j -lt $real_test_num ]]; then
                        TaskGroup "$run_test_type" "$device_id"
                fi
            done
        } &

        task_group_pids+=($!)
    fi
done

# wait for all TaskGroup finished
wait

echo "**** all test finished ****"

# Automatically call trap function to clean up the task_group_pids when exit

exit 0
