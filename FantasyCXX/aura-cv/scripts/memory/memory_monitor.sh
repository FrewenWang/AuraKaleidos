#!/bin/sh
###############################################################
###            Common functions for qcom or mtk             ###
###############################################################
function common_query_platform()
{
    num_qti_str=$(ls /vendor/lib64/ | grep qti | wc -l)
    num_mtk_str=$(ls /vendor/lib64/ | grep mediatek | wc -l)

    if [ ${num_qti_str} -eq 0 ] && [ ${num_mtk_str} -eq 0 ];then
        echo "common_query_platform failed"
        return -1;
    fi

    if [ ${num_qti_str} -gt ${num_mtk_str} ];then
        echo "QCOM"
    else
        echo "MTK"
    fi
}

function common_query_proc_dma_buf()
{
    if [ ${#} -lt 1 ];then
        echo "common_query_proc_dma_buf argc < 1"
        return -1;
    else
        pid_num=$1
        if [ ! -d /proc/${pid_num}/ ]; then
            echo "pid ${pid_num} not exist"
            return -1;
        else
            mem_str=$(dmabuf_dump ${pid_num} | (grep "dmabuf total" || true) | awk '{print $3}')
            if [ -z "${mem_str}" ];then
                echo "0"
            else
                echo ${mem_str}
            fi
        fi
    fi
}

function common_check_pid_valid()
{
    if [ ! -d /proc/${1}/ ]; then
        return -1;
    fi
}

function common_wait_for_proc()
{
    if [ ${#} -lt 1 ];then
        echo "common_query_proc_dma_buf argc < 1"
        return -1;
    else
        while true;do
            pid=`pgrep ${1}`
            if [ -z ${pid} ];then
                echo "Wait for process: ${1}"
                continue
            else
                sleep 1
                break;
            fi
        done
    fi
}
###############################################################
###            Functions for qcom platform only             ###
###############################################################
function qcom_query_soc_info()
{
    # /proc/device-tree/model or ro.soc.model or ro.vendor.qti.soc_model
    soc_info=`getprop ro.vendor.qti.soc_model`        # eg: SM8550/SM8450/SM8350
    gpu_info=`cat /sys/class/kgsl/kgsl-3d0/gpu_model` # eg: Adreno730v1
    printf "Soc: %s Gpu: %s\n" "${soc_info}" "${gpu_info}"
}

function qcom_query_system_gpu_total_mem()
{
    kgsl_mem=$(cat /sys/class/kgsl/kgsl/page_alloc)
    kgsl_mem=`expr ${kgsl_mem} / 1024`
    echo ${kgsl_mem}
}

function qcom_query_proc_gpu_total_mem()
{
    if [ ${#} -lt 1 ];then
        echo "qcom_query_proc_gpu_total_mem  argc < 1"
        return -1;
    else
        if [ ! -d "/sys/class/kgsl/kgsl/proc/${1}" ];then
            echo "0"
        else
            # private gpu memory, KB
            gpumem_mapped=$(cat /sys/class/kgsl/kgsl/proc/${1}/gpumem_mapped)
            gpumem_unmapped=$(cat /sys/class/kgsl/kgsl/proc/${1}/gpumem_unmapped)
            echo `expr ${gpumem_mapped} / 1024 + ${gpumem_unmapped} / 1024`
        fi
    fi
}

function qcom_query_proc_gpu_mem_stat()
{
    if [ ${#} -lt 1 ];then
        echo "Must has PID id"
        return -1
    else
        mem_types=("cl" "cl_kernel_stack" "cl_buffer_map" "cl_iaura_map" "cl_iaura_nomap" "kernel" "any(0)")
        tmpfile_path="/data/local/tmp/temp_cl_mem.txt"


        printf "%15s %5s %10s %10s\n" "type" "count" "max(KB)" "sum(KB)"
        cat "/d/kgsl/proc/${1}/mem" > ${tmpfile_path} || true
        # echo ${mem_info} | awk 'NR > 1 {s[$7] += $3} END{ for(i in s){  print i, s[i] }}'
        for type in "${mem_types[@]}";do
            type_meminfo="$(grep -w -s ${type} ${tmpfile_path} || true)"
            if [ -z "${type_meminfo}" ];then
                continue
            fi
            echo "${type_meminfo}" | awk 'BEGIN{max=0;count=0;sum=0}{if($3>max){max=$3};sum+=$3; count+=1;}END{printf("%15s %5d %10d %10d\n"), $7, count, max/1024, sum/1024}'
        done
    fi
}
###############################################################
###             Functions for mtk platform only             ###
###############################################################
function mtk_query_soc_info()
{
    # /proc/device-tree/model or ro.vendor.soc.model or ro.soc.model
    soc_info="$(getprop ro.soc.model)"                   # eg: MT6983
    gpu_info=""
    if [ -e /sys/class/misc/mali0/device/gpuinfo ];then
        gpu_info="$(cat /sys/class/misc/mali0/device/gpuinfo)" # eg: Mali-G710 10 cores r0p0 0xA862
    else
        gpu_info="$(dumpsys SurfaceFlinger | grep GLES)"        
        gpu_info=${gpu_info%,*}
    fi
    printf "Soc: %s Gpu: %s\n" "${soc_info}" "${gpu_info}"
}

function mtk_query_system_gpu_total_mem
{
    if [ ! -e /d/mali0/gpu_memory ];then
        echo "/d/mali0/gpu_memory not exist"
        return -1
    else
        page_count=$(grep "mali0" /d/mali0/gpu_memory | awk '{print $2 * 4}');
        echo "${page_count}"
    fi
}

function mtk_query_proc_gpu_ctx_dir()
{
    if [ ${#} -lt 1 ];then
        echo "mtk_query_proc_gpu_ctx_dir argc < 1"
        return -1;
    else
        if [ ! -d /d/mali0/ctx ];then
            echo "Current only support new mali driver"
            return -1;
        else
            dir_count=$(ls /d/mali0/ctx | (grep ${1} || true) | wc -l)

            if [ ${dir_count} -gt 1 ];then
                echo "Multiple dir in /d/mali0/ctx"
                return -1
            elif [ ${dir_count} -eq 0 ];then
                echo "/d/mali0/ctx/no_exist"
            else
                dir_name="$(ls /d/mali0/ctx/ | (grep $1 || true))"
                echo "/d/mali0/ctx/${dir_name}"
            fi
        fi
    fi
}

function mtk_query_proc_gpu_total_mem()
{
    if [ ${#} -lt 1 ];then
        echo "mtk_query_proc_gpu_total_mem required /d/mali0/ctx/<pid_id>_xxx dir"
        return -1;
    else
        if [ ! -e "${ctx_dir}/mem_profile" ];then
            echo "0"
        else
            mem_info=$(tail ${ctx_dir}/mem_profile | grep "Total allocated memory" | awk '{print $4 / 1024}')
            echo "${mem_info}"
        fi
    fi
}

function mtk_query_proc_gpu_mem_stat()
{
    if [ ${#} -lt 1 ];then
        echo "mtk_query_proc_gpu_mem_stat required /d/mali0/ctx/<pid_id>_xxx dir"
        return -1;
    else
        if [ ! -e "${ctx_dir}/mem_profile" ];then
            echo ""
        else
            mem_info=$(grep Channel ${ctx_dir}/mem_profile | grep -v "Total memory: 0)")
            echo "${mem_info}"
        fi
    fi
}
###############################################################
###                     Qcom Main Loop                      ###
###############################################################
function qcom_main_loop()
{
    if [ ! -d ${LOG_DIR} ];then
        echo "LOG_DIR not exist"
        return -1
    fi

    qcom_query_soc_info | tee ${LOG_DIR}/device_info.txt

    dma_buf_mem=0; sys_gpu_mem=0; pid_gpu_mem=0

    while [ ! ${STOP_FLAG} -eq 1 ];do
        common_check_pid_valid ${PID_ID}
        current_time=$(date +%s.%N)
        time_elapsed=$(echo "$current_time - ${START_TIME}" | bc)
        dma_buf_mem=$(common_query_proc_dma_buf ${PID_ID})
        sys_gpu_mem=$(qcom_query_system_gpu_total_mem)
        proc_gpu_mem=$(qcom_query_proc_gpu_total_mem ${PID_ID})

        printf "Time Stamp: %012.6fs dma_buf:%8d KB   sys_gpu:%8d KB   proc_gpu_mem: %8d KB\n" ${time_elapsed} ${dma_buf_mem} ${sys_gpu_mem} ${proc_gpu_mem} | tee -a ${LOG_DIR}/dma_buf_gpu.log
        sleep ${PERIOD}
    done
}

function qcom_sub_loop()
{
    if [ ! -d ${LOG_DIR} ];then
        echo "LOG_DIR not exist"
        return -1
    fi

    while [ ! ${STOP_FLAG} -eq 1 ];do
        common_check_pid_valid ${PID_ID}
        current_time=$(date +%s.%N)
        time_elapsed=$(echo "$current_time - ${START_TIME}" | bc)
        gpu_mem_stat="$(qcom_query_proc_gpu_mem_stat ${PID_ID})"
        printf "Time Stamp: %012.6fs\n%s\n" ${time_elapsed} "${gpu_mem_stat}" | tee -a ${LOG_DIR}/gpu_mem_stat.log

        proc_mem_info="$(dumpsys meminfo ${PID_ID})"
        printf "---------------------------------------------------------------\n" | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        printf "Time Stamp: %012.6fs\n %s\n" ${time_elapsed} "${proc_mem_info}"    | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        printf "===============================================================\n" | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        sleep ${SUB_PERIOD}
    done
}
###############################################################
###                     MTK Main Loop                       ###
###############################################################
function mtk_main_loop()
{
    if [ ! -d ${LOG_DIR} ];then
        echo "LOG_DIR not exist"
        return -1
    fi

    mtk_query_soc_info | tee ${LOG_DIR}/device_info.txt

    ctx_dir=$(mtk_query_proc_gpu_ctx_dir ${PID_ID})
    dma_buf_mem=0; sys_gpu_mem=0; pid_gpu_mem=0

    while [ ! ${STOP_FLAG} -eq 1 ];do
        common_check_pid_valid ${PID_ID}
        current_time=$(date +%s.%N)
        time_elapsed=$(echo "$current_time - ${START_TIME}" | bc)
        dma_buf_mem=$(common_query_proc_dma_buf ${PID_ID})
        sys_gpu_mem=$(mtk_query_system_gpu_total_mem ${ctx_dir})
        proc_gpu_mem=$(mtk_query_proc_gpu_total_mem ${ctx_dir})

        printf "Time Stamp: %012.6fs dma_buf:%8d KB   sys_gpu:%8d KB   proc_gpu_mem: %9.2f KB\n" ${time_elapsed} ${dma_buf_mem} ${sys_gpu_mem} ${proc_gpu_mem} | tee -a ${LOG_DIR}/dma_buf_gpu.log
        sleep ${PERIOD}
    done
}

function mtk_sub_loop()
{
    if [ ! -d ${LOG_DIR} ];then
        echo "LOG_DIR not exist"
        return -1
    fi

    ctx_dir=$(mtk_query_proc_gpu_ctx_dir ${PID_ID})

    while [ ! ${STOP_FLAG} -eq 1 ];do
        common_check_pid_valid ${PID_ID}
        current_time=$(date +%s.%N)
        time_elapsed=$(echo "$current_time - ${START_TIME}" | bc)
        gpu_mem_stat="$(mtk_query_proc_gpu_mem_stat ${ctx_dir})"
        printf "Time Stamp: %012.6fs\n%s\n" ${time_elapsed} "${gpu_mem_stat}" | tee -a ${LOG_DIR}/gpu_mem_stat.log

        proc_mem_info="$(dumpsys meminfo ${PID_ID})"
        printf "---------------------------------------------------------------\n" | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        printf "Time Stamp: %012.6fs\n %s\n" ${time_elapsed} "${proc_mem_info}"    | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        printf "===============================================================\n" | tee -a ${LOG_DIR}/dumpsys_meminfo.log
        sleep ${SUB_PERIOD}
    done
}
###############################################################
###                   Global Variables                      ###
###############################################################
LOG_DIR="/data/local/tmp/$(date '+%Y-%m-%d-%H-%M')"
PLATFORM=""
PERIOD=0.25
SUB_PERIOD=0.5
PID_ID=""
START_TIME=$(date +%s.%N)
STOP_FLAG=0
###############################################################
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --pid       <proc_id>    PID of process to be sampled"
    echo "  -n, --name      <proc_name>  Name of process to be sampled"
    echo "  -i, --interval  <interval>   Period of sampling (default: 1)"
    echo "  -d, --dir                    Report store dir, default use date"
    echo "  -h, --help                   Show this help"
}

while getopts "p:n:i:d:h" opt; do
    case ${opt} in
        p)
            PID_ID="${OPTARG}"
            ;;
        n)
            common_wait_for_proc "${OPTARG}"
            PID_ID=$(pgrep "${OPTARG}")
            ;;
        i)
            PERIOD="${OPTARG}"
            SUB_PERIOD="$(echo "${PERIOD} * 2" | bc)"
            ;;
        d)  LOG_DIR="${OPTARG}"
            ;;
        h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [ -z "${PID_ID}" ];then
    echo "PID_ID is empty"
    return -1
fi

# Trap functions for Ctrl+C
function common_trap_func()
{
    echo "common_trap_func called"
    STOP_FLAG=1;
}

set -e
trap common_trap_func INT
PLATFORM="$(common_query_platform)"
if [ ! -d ${LOG_DIR} ];then
    mkdir -p ${LOG_DIR}
fi

if [ "${PLATFORM}" = "QCOM" ];then
    qcom_sub_loop &
    qcom_main_loop
    wait
elif [ "${PLATFORM}" = "MTK" ];then
    mtk_sub_loop &
    mtk_main_loop
fi
set +e

