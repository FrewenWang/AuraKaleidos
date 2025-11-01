#!/bin/bash

function check_platform()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    # adb -s ${ADB_SERIAL} shell getprop | grep mtk | wc -l
    num_qti_str=$(adb -s ${ADB_SERIAL} shell "ls /vendor/lib/ | grep qti | wc -l")
    num_mtk_str=$(adb -s ${ADB_SERIAL} shell "ls /vendor/lib/ | grep mediatek | wc -l")

    if [ $num_mtk_str -gt 0 ];then
        mtk_str=$(adb -s ${ADB_SERIAL} shell "ls /proc/ | grep gpufreqv2")
        if [ -z $mtk_str ]; then
            echo "MTKV1"
        else
            echo "MTKV2"
        fi
    elif [ $num_qti_str -gt 0 ];then
        echo "QCOM"
    else
        echo "UNKNOWN"
    fi
}

function boost_qcom_cpu()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    end_count=$((NUM_CPU_CORES - 1))
    for cpu_id in $(seq 0 ${end_count});do
        adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpu$cpu_id/cpufreq/scaling_governor"
    done
}

function boost_qcom_gpu()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    adb -s ${ADB_SERIAL} shell "echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel"
    adb -s ${ADB_SERIAL} shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_rail_on"
    adb -s ${ADB_SERIAL} shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_clk_on"
    adb -s ${ADB_SERIAL} shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_bus_on"
    adb -s ${ADB_SERIAL} shell "echo 10000000 > /sys/class/kgsl/kgsl-3d0/idle_timer"
    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/class/kgsl/kgsl-3d0/devfreq/governor"
}

function boost_mtk_cpu_v1()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    adb -s ${ADB_SERIAL} shell "echo 0 0 > /proc/ppm/policy/ut_fix_freq_idx"

    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"

    end_count=$((NUM_CPU_CORES - 1))

    for cpu_id in $(seq 0 ${end_count});do
        adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpu$cpu_id/cpufreq/scaling_governor"
    done
}

function boost_mtk_cpu_v2()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor"
    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
    adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"

    end_count=$((NUM_CPU_CORES - 1))

    for cpu_id in $(seq 0 ${end_count});do
        adb -s ${ADB_SERIAL} shell "echo 'performance' > /sys/devices/system/cpu/cpu$cpu_id/cpufreq/scaling_governor"
    done
}

function boost_mtk_gpu_v1()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    gpu_max_freq_str=$(adb -s ${ADB_SERIAL} shell "cat /proc/gpufreq/gpufreq_opp_dump" | grep "\[00\]" | awk -F ',' '{print $1}')
    gpu_max_freq=${gpu_max_freq_str##*=}

    adb -s ${ADB_SERIAL} shell "echo 0 0 > /proc/ppm/policy/ut_fix_freq_idx"
    adb -s ${ADB_SERIAL} shell "echo  ${gpu_max_freq} > /proc/gpufreq/gpufreq_opp_freq"
    adb -s ${ADB_SERIAL} shell "echo 0 > /sys/devices/platform/10012000.dvfsrc/helio-dvfsrc/dvfsrc_force_vcore_dvfs_opp"
    adb -s ${ADB_SERIAL} shell "echo always_on > /sys/class/misc/mali0/device/power_policy"
}

function boost_mtk_gpu_v2()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    adb -s ${ADB_SERIAL} shell "echo 0 > /proc/gpufreqv2/fix_target_opp_index"
    adb -s ${ADB_SERIAL} shell "echo always_on > /sys/class/misc/mali0/device/power_policy"
    # adb -s ${ADB_SERIAL} shell "echo 0 > 1c00f000.dvfsrc/helio-dvfsrc/dvfsrc_force_vcore_dvfs_opp"
}


function get_current_info_qcom()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    end_count=$((NUM_CPU_CORES - 1))

    for idx in $(seq 0 ${end_count});do
        cur_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/scaling_cur_freq")
        max_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/cpuinfo_max_freq")
        echo "CPU Core[$idx]: Current Freq: ${cur_freq}kHz Max Freq: ${max_freq}kHz"
    done

    # adb -s ${ADB_SERIAL} shell cat /sys/class/kgsl/kgsl-3d0/gpuclk
    # adb -s ${ADB_SERIAL} shell cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq
    gpu_cur_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")
    gpu_max_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/max_freq")

    echo "GPU Governor  : $(adb -s ${ADB_SERIAL} shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/governor")"
    echo "Adreno GPU    : Current Freq: ${gpu_cur_freq}kHz Max Freq: ${gpu_max_freq}kHz"
}

function get_current_info_mtk_v1()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    end_count=$((NUM_CPU_CORES - 1))

    for idx in $(seq 0 ${end_count});do
        cur_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/scaling_cur_freq")
        max_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/cpuinfo_max_freq")
        echo "CPU Core[$idx]: Current Freq: ${cur_freq}kHz, Max Freq: ${max_freq}kHz"
    done

    gpu_cur_freq_str=$(adb -s ${ADB_SERIAL} shell "cat /proc/gpufreq/gpufreq_var_dump" | grep real|awk -F ',' '{print $1}')
    gpu_cur_freq=${gpu_cur_freq_str##*:}

    gpu_max_freq_str=$(adb -s ${ADB_SERIAL} shell "cat /proc/gpufreq/gpufreq_opp_dump" | grep "\[00\]" | awk -F ',' '{print $1}')
    gpu_max_freq=${gpu_max_freq_str##*=}
    echo "Mali GPU : Current Freq:${gpu_cur_freq}kHz, Max Freq:${gpu_max_freq}kHz"
}

function get_current_info_mtk_v2()
{
    if [ -z ${ADB_SERIAL} ];then
        echo "ADB_SERIAL is empty, ${FUNCNAME} failed"
        return
    fi

    end_count=$((NUM_CPU_CORES - 1))

    for idx in $(seq 0 ${end_count});do
        cur_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/scaling_cur_freq")
        max_freq=$(adb -s ${ADB_SERIAL} shell "cat /sys/devices/system/cpu/cpu${idx}/cpufreq/cpuinfo_max_freq")
        echo "CPU Core[$idx]: Current Freq: ${cur_freq}kHz, Max Freq: ${max_freq}kHz"
    done

    gpu_cur_freq_str=$(adb -s ${ADB_SERIAL} shell "cat /proc/gpufreqv2/gpufreq_status | grep STACK-OPP" | awk -F',' '{print $2}')
    gpu_cur_freq=${gpu_cur_freq_str##*:}

    gpu_max_freq_str=$(adb -s ${ADB_SERIAL} shell "cat /proc/gpufreqv2/stack_working_opp_table" | grep "\[00\]" | awk -F',' '{print $1}')
    gpu_max_freq=${gpu_max_freq_str##*:}
    echo "Mali GPU : Current Freq:${gpu_cur_freq}kHz, Max Freq:${gpu_max_freq}kHz"
}

# check_boost
function main()
{
    # check current platform
    PLATFORM=$(check_platform)
    if [ $PLATFORM == "QCOM" ];then
        boost_qcom_cpu
        boost_qcom_gpu
        get_current_info_qcom
    elif [ $PLATFORM == "MTKV1" ];then
        boost_mtk_cpu_v1
        boost_mtk_gpu_v1
        get_current_info_mtk_v1
    elif [ $PLATFORM == "MTKV2" ];then
        boost_mtk_cpu_v2
        boost_mtk_gpu_v2
        get_current_info_mtk_v2
    fi
}

function get_adb_serial()
{
    ADB_SERIAL=$1
    NUM_DEVICES=$(adb devices | grep "device$" | wc -l)

    if [ ${NUM_DEVICES} -lt 1 ];then
        echo "No adb devices found."
        exit
    elif [ $NUM_DEVICES -eq 1 ];then
        if [ -n ${ADB_SERIAL} ];then
            ADB_SERIAL=$(adb devices |sed -n '2p' | awk '{print $1}')
            if [ -z ${ADB_SERIAL} ];then
                echo "No adb devices found."
                exit
            else
                echo "ADB use default device: ${ADB_SERIAL}"
            fi
        fi
    else
        if [ -z ${ADB_SERIAL} ];then
            echo "Multiple adb devices found, must input an serial."
            adb devices
            exit
        fi
    fi

    if [[ ! "$(adb devices)" =~ "${ADB_SERIAL}" ]];then
        echo "Serial num has error, please check."
        adb devices | grep "device$"
        exit
    fi
}

echo "======================= Aura Boost Script =========================="
ADB_SERIAL=""
get_adb_serial $1

NUM_CPU_CORES=$(adb -s ${ADB_SERIAL} shell "ls -d /sys/devices/system/cpu/cpu[0-9]*" | wc -l)
GPU_INFO=$(adb -s ${ADB_SERIAL} shell dumpsys SurfaceFlinger | grep GLE)

echo "===================================================================="
echo "Current Platform: $(check_platform)"
echo "CPU Cores:        ${NUM_CPU_CORES}"
echo "GPU Info :        $GPU_INFO"
echo "===================================================================="
# enable root
adb -s ${ADB_SERIAL} root
adb -s ${ADB_SERIAL} remount
adb -s ${ADB_SERIAL} shell setenforce 0
# execute boost script
main
