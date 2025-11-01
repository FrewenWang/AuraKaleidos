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

RunTest()
{
    [ $# -lt 3 ] && Error 1 "param < 3"

    local work_path=$1
    local build_exe_path=$2

    local op_cases=$3

    echo "run test"
    echo "build_exe_path: ${build_exe_path}"

    cd "${build_exe_path}"

    # 1.1 generate auto_test.json
(
cat << EOF
{
    "async_affinity": "LITTLE",
    "cache_bin_path": "",
    "cache_bin_prefix": "",
    "compute_affinity": "BIG",
    "data_path": "${work_path}/data/",
    "device_info": "linux x64",
    "log_file_name": "log",
    "log_level": "DEBUG",
    "log_output": "STDOUT",
    "ndk_info": "none",
    "report_name": "auto_test",
    "report_type":"txt",
    "stress_count": 0
}
EOF
) > ./config.json
    Error $? "create config.json failed"

    echo "[aura2.0 test]: begin test"

    # 2.1 run test cases
    # serarch cases by '-l'
    run_cases=()
    for case in ${op_cases[@]}; do
        rets_l=$(./aura_test_main -l ${case})

        echo "rets_l: ${rets_l}"

        sub_run_cases=$(awk '/^======================================================================================/{n++} \
                    n==2{start=1} start==1&&/^======================================================================================/{start=0;next} start==1{print}' <<< "$rets_l")

        run_cases+=("${sub_run_cases[@]}")
    done

    echo "run_cases_num: "${#run_cases[@]}""

    run_cases_unroll=$(echo ${run_cases[@]})
    echo "run_cases_unroll: ${run_cases_unroll}"

    echo "get test cases over"

    # run cases by '-r'
    ./aura_test_main -r ./config.json ${run_cases_unroll} -s 2

    if [ $? != 0 ];then
        echo "[aura2.0 test]: test failed"
        cd ${work_path}
        exit 1
    fi

    echo "[aura2.0 test]: end test"

    # 2.2 parse log
    parsed_file="./auto_test.txt"
    if [ ! -f "$parsed_file" ]; then
        Error 1 "auto_test.txt not found, path: ${PWD}"
    fi

    # Parse the value of 'Failed:'
    failed=$(grep 'Failed:' "$parsed_file" | awk '{print $2}')
    Error $failed "auto_test failed, please check $parsed_file"

    # 2.3 copy auto_test.txt to artifact path
    log_saved_path="${work_path}/log/linux"

    mkdir -p "$log_saved_path"
    [ ! -d "$log_saved_path" ] && Error 1 "Failed to create directory ${log_saved_path}"

    cp "$parsed_file" "$log_saved_path"
    Error $? "Failed to copy ${parsed_file} to ${log_saved_path}"

    cd ${work_path}
}

# check num of param
[ $# -lt 2 ] && Error 1 "param < 2"

work_path=$1
build_linux_path=$2

# find the path of aura_test_main
build_exe_path=$(find "${work_path}/${build_linux_path}/x64/static/release/install" -type f -name "aura_test_main" -exec dirname {} \;)
Error $? "aura_test_main not found, path: ${build_exe_path}, ${work_path}, ${build_linux_path}"

echo "build_exe_path: ${build_exe_path}"

# op cases
run_op_cases_str="$(SearchOps "$work_path/src/ops")"
echo "run_op_cases_str: ${run_op_cases_str}"
IFS=' ' read -r -a run_op_cases <<< "$run_op_cases_str"

echo "run_op_cases: ${run_op_cases[*]}"

#------------
# delete test:resize/
for i in "${!run_op_cases[@]}"; do
    if [ "${run_op_cases[$i]}" == "resize" ]; then
        unset 'run_op_cases[$i]'
    fi
done

run_op_cases=("${run_op_cases[@]}")
echo "run_op_cases: ${run_op_cases[*]}"

#------------

# run 
RunTest "$work_path" "$build_exe_path" "${run_op_cases[*]}"

exit 0