#!/usr/bin/env python3
# =============================================================================
#
#  Author: zhoulongfei@xiaomi.com
#
# =============================================================================
import os
import shlex
import subprocess
import sys


def run_cmd(cmd):
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
        sys.exit(1)


def handle_error(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def adb_devices():
    returncode, stdout, stderr = run_command(["adb", "devices"])
    if returncode != 0:
        handle_error(f"Failed to list devices: {stderr}")

    devices = stdout.splitlines()
    if len(devices) <= 1:  # 第一行是标题 "List of devices attached"
        print("No devices attached.")
    else:
        print("Connected devices:")
        for device in devices[1:]:  # 跳过标题行
            print(device)


def adb_shell(cmd):
    run_cmd(f"adb shell {cmd}")


def adb_pull(src, dst):
    run_cmd(f"adb pull {src} {dst}")


def adb_push(src, dst):
    adb_shell(f"mkdir -p {dst}")
    for file in src:
        run_cmd(f"adb push {file} {dst}")


def send_qnn(qnn_sdk_root, board_sdk_dir, version=79):
    def add_path(prefix, filenames):
        return [os.path.join(prefix, filename) for filename in filenames]

    bin_dir = qnn_sdk_root + "/bin/aarch64-android/"
    lib_dir = qnn_sdk_root + "/lib/aarch64-android/"
    dsp_dir = qnn_sdk_root + f"/lib/hexagon-v{version}/unsigned/"
    bins = ["qnn-net-run"]
    libs = [
        "libQnnHtp.so",
        f"libQnnHtpV{version}Stub.so",
        "libQnnHtpNetRunExtensions.so",
    ]
    dsps = [f"libQnnHtpV{version}.so", f"libQnnHtpV{version}Skel.so"]

    adb_shell(f"rm -rf {board_sdk_dir}")
    adb_push(add_path(bin_dir, bins), board_sdk_dir + "/bin/")
    adb_push(add_path(lib_dir, libs), board_sdk_dir + "/lib/")
    adb_push(add_path(dsp_dir, dsps), board_sdk_dir + "/dsp/")
    # adb_push(["/home/zlf/codes/mi-camera-algorithm/memory_profiler/device/lib/libmemory_profiler.so"], board_sdk_dir+"/memory_profiler/")


def send_snpe(qnn_sdk_root, board_sdk_dir, version=79):
    def add_path(prefix, filenames):
        return [os.path.join(prefix, filename) for filename in filenames]

    bin_dir = qnn_sdk_root + "/bin/aarch64-android/"
    lib_dir = qnn_sdk_root + "/lib/aarch64-android/"
    dsp_dir = qnn_sdk_root + f"/lib/hexagon-v{version}/unsigned/"
    bins = ["snpe-net-run"]
    libs = ["libSNPE.so", f"libSnpeHtpV{version}Stub.so"]
    dsps = [f"libSnpeHtpV{version}Skel.so"]

    adb_shell(f"rm -rf {board_sdk_dir}")
    adb_push(add_path(bin_dir, bins), board_sdk_dir + "/bin/")
    adb_push(add_path(lib_dir, libs), board_sdk_dir + "/lib/")
    adb_push(add_path(dsp_dir, dsps), board_sdk_dir + "/dsp/")


def qnn_infer_one_sample(sample, outdir, target_dir):
    # Copy files to board.
    input_local_filenames = []
    for name in sample:
        modelname = os.path.basename(name)
        input_local_filenames.append(f"input/{modelname}")
    with open("./tmp/input_file_local.list", "w") as f:
        f.write(" ".join(input_local_filenames))

    sample.append("./tmp/input_file_local.list")

    adb_shell(f"rm -rf {target_dir}/input")
    adb_shell(f"rm -rf {target_dir}/output")
    adb_push(sample, f"{target_dir}/input")

    # Run inference on device
    adb_shell(f"cd {target_dir} && sh run.sh")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Transfer results to host
    adb_pull(f"{target_dir}/output/qnn-profiling-data_0.log", outdir)


def qnn_infer_one_batch(lines, outdir, target_dir):
    manifest = set()
    with open("./tmp/input_file_local.list", "w") as f:
        for line in lines:
            sample = line.split()
            input_local_filenames = []
            for name in sample:
                manifest.add(name)
                modelname = os.path.basename(name)
                input_local_filenames.append(f"input/{modelname}")
            f.writelines(" ".join(input_local_filenames) + "\n")

    manifest.add("./tmp/input_file_local.list")

    print(manifest)

    adb_shell(f"rm -rf {target_dir}/input")
    adb_shell(f"rm -rf {target_dir}/output")
    print("========================manifest==================================")
    print("manifest===============", manifest)
    adb_push(manifest, f"{target_dir}/input")

    # Run inference on device
    adb_shell(f"cd {target_dir} && sh run.sh")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Transfer results to host
    adb_pull(f"{target_dir}/output/", outdir)


def snpe_infer_one_sample(sample, outdir, target_dir):
    # Copy files to board.
    input_local_filenames = []
    for name in sample:
        modelname = os.path.basename(name)
        input_local_filenames.append(f"input/{modelname}")
    with open("./tmp/input_file_local.list", "w") as f:
        f.write(" ".join(input_local_filenames))

    sample.append("./tmp/input_file_local.list")

    adb_shell(f"rm -rf {target_dir}/input")
    adb_shell(f"rm -rf {target_dir}/output")
    adb_push(sample, f"{target_dir}/input")

    # Run inference on device
    adb_shell(f"cd {target_dir} && sh run.sh")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Transfer results to host
    adb_pull(f"{target_dir}/output/SNPEDiag_0.log", outdir)


def snpe_infer_one_batch(lines, outdir, target_dir):
    manifest = []
    with open("./tmp/input_file_local.list", "w") as f:
        for line in lines:
            sample = line.split()
            input_local_filenames = []
            for name in sample:
                manifest.append(name)
                modelname = os.path.basename(name)
                input_local_filenames.append(f"input/{modelname}")
            f.writelines(" ".join(input_local_filenames) + "\n")

    manifest.append("./tmp/input_file_local.list")

    adb_shell(f"rm -rf {target_dir}/input")
    adb_shell(f"rm -rf {target_dir}/output")
    adb_push(manifest, f"{target_dir}/input")

    # Run inference on device
    adb_shell(f"cd {target_dir} && sh run.sh")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Transfer results to host
    adb_pull(f"{target_dir}/output/", outdir)
