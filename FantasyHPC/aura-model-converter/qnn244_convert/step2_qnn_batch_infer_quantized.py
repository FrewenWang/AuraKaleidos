#!/usr/bin/env python3
# =============================================================================
#
#  Author: zhoulongfei@xiaomi.com
#
# =============================================================================

import argparse
import os
import shutil
import sys

from printk import print_colored_box

from adb_utils import (
    adb_push,
    qnn_infer_one_batch,
    run_cmd,
    send_qnn,
)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
END_COLOR = "\033[0m"


def send_context_bin(
    board_sdk_dir, model_dir, target_dir, enable_optrace, suffix=""
):
    # sysMonApp getstate --q6 cdsp

    # export LD_PRELOAD={board_sdk_dir}/memory_profiler/libmemory_profiler.so
    if enable_optrace:
        profiling_cmd = "--profiling_level detailed --profiling_option optrace"
    else:
        profiling_cmd = "--profiling_level basic"

    run_str = f"""#!/bin/sh
export PATH={board_sdk_dir}/bin
export LD_LIBRARY_PATH={board_sdk_dir}/lib
export ADSP_LIBRARY_PATH={board_sdk_dir}/dsp

qnn-net-run --backend libQnnHtp.so --retrieve_context model_htp{suffix}.bin \
--input_list input/input_file_local.list --perf_profile burst --synchronous --shared_buffer \
--config_file config/config.json                          \
--log_level error {profiling_cmd}"""

    if not os.path.exists("./tmp/"):
        os.makedirs("./tmp/")

    with open("./tmp/run.sh", "w") as f:
        f.write(run_str)

    # Copy files to board.
    manifest = ["./tmp/run.sh", f"{model_dir}/model_htp{suffix}.bin", "config"]
    adb_push(manifest, target_dir)


def run_profile_viewer(qnn_sdk_dir, outdir, model_dir, enable_optrace):
    toolkit_dir = os.path.join(qnn_sdk_dir, "bin/x86_64-linux-clang")
    command = (
        f"{toolkit_dir}/qnn-profile-viewer --input_log "
        f"{outdir}/qnn-profiling-data_0.log"
        + f" > {outdir}/profiling_summary.txt 2>&1"
    )
    ok = os.system(command)
    if ok != 0:
        print(RED + "qnn-profile-viewer command failed." + END_COLOR)
        print("The command is:\n" + BLUE + command + "\n" + END_COLOR)
        sys.exit(1)

    if enable_optrace:
        command = (
            f"{toolkit_dir}/qnn-profile-viewer --input_log "
            f"{outdir}/qnn-profiling-data.log "
            + f"--schematic {model_dir}/model_schematic.bin "
            + f"--reader {qnn_sdk_dir}/lib/x86_64-linux-clang/"
            "libQnnHtpOptraceProfilingReader.so --output model_optrace.json"
        )
        ok = os.system(command)
        if ok != 0:
            print(RED + "qnn-profile-viewer command failed." + END_COLOR)
            print("The command is:\n" + BLUE + command + "\n" + END_COLOR)
            sys.exit(1)
        os.system(f"mv model_optrace.json {outdir}")
        os.system(f"mv model_optrace_qnn_htp_analysis_summary.html {outdir}")

    command = f"""
echo ==========================================================================================
echo
echo Execute Time[ave, min, max]: $(cat {outdir}/profiling_summary.txt|grep "Backend (RPC (execute) time): "|grep -E -o "[0-9]+ us")
echo
echo Following profiling files are stored in : {outdir}
echo
echo $(ls {outdir})
echo
echo ==========================================================================================
"""
    os.system(command)


def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"目录 {directory} 及其子目录已删除")
    else:
        print(f"目录 {directory} 不存在")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", type=int, required=False, default=1)
    args = parser.parse_args()
    print(args)

    board_dir = "/data/local/tmp/qnn"
    version = 81

    send_qnn_flag = True
    root_dir = os.environ.get("ENV_PROJECT_ROOT_DIR")
    env_model_dir = os.environ.get("ENV_MODEL_DIR")
    model_dir = f"{root_dir}/convert_output/{env_model_dir}"
    infer_outdir = f"{root_dir}/infer_output/{env_model_dir}"
    input_list = os.environ.get("ENV_TEST_LIST")

    detailed = args.detailed == 1

    board_sdk_dir = board_dir
    # board_sdk_dir = f"{board_dir}/qnn_sdk"
    if detailed:
        board_tmp_dir = f"{board_dir}/detailed"
        suffix = "_detailed"
    else:
        board_tmp_dir = f"{board_dir}/tmp"
        suffix = ""

    run_cmd("adb devices")
    run_cmd("adb root")

    # Get the qnn-sdk-dir from env
    qnn_sdk_dir = os.environ.get("QNN_SDK_ROOT")
    # qnn_sdk_dir = "/home/zlf/Documents/projects/mganv3_animal_1536_2048_o1_models/qnn-sdk-2.24.5.240809"
    if not qnn_sdk_dir:
        raise RuntimeError(
            RED + "QNN_SDK_ROOT variable is not set." + END_COLOR
        )
    print_colored_box(
        "QNN_SDK_ROOT: " + qnn_sdk_dir,
        55,
        text_color="green",
        box_color="green",
    )

    # step 1: send_qnn
    if send_qnn_flag:
        print_colored_box(
            "Send QNN to targets", 55, text_color="blue", box_color="yellow"
        )
        send_qnn(qnn_sdk_dir, board_sdk_dir, version)
        print_colored_box(
            "Send QNN to Target successfully",
            55,
            text_color="green",
            box_color="green",
        )

    # step 2: send_context_bin
    print_colored_box(
        "Send model to targets", 55, text_color="blue", box_color="yellow"
    )
    send_context_bin(board_sdk_dir, model_dir, board_tmp_dir, detailed, suffix)
    # send_model_lib(board_sdk_dir, model_dir, board_tmp_dir)

    # step 3: infer
    if not os.path.exists(infer_outdir):
        os.makedirs(infer_outdir)
    if not os.path.exists(infer_outdir):
        raise RuntimeError(RED + "fail to mkdir: " + infer_outdir + END_COLOR)

    with open(input_list) as f:
        lines = f.readlines()
    qnn_infer_one_batch(lines, infer_outdir, board_tmp_dir)
    run_profile_viewer(
        qnn_sdk_dir, f"{infer_outdir}/output", model_dir, detailed
    )
    if detailed:
        # remove old result
        delete_directory(f"{infer_outdir}/detailed")
        os.rename(f"{infer_outdir}/output", f"{infer_outdir}/detailed")
    else:
        # remove old result
        delete_directory(f"{infer_outdir}/basic")
        os.rename(f"{infer_outdir}/output", f"{infer_outdir}/basic")


if __name__ == "__main__":
    main()
