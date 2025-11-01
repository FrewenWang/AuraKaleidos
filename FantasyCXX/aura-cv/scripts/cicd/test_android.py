import subprocess
import time
import os
import yaml
import json
import datetime
import multiprocessing
import math
import argparse
import logging
import signal

class RunTest:
    def __init__(self, work_path, build_android_path, build_hexagon_path, stress_config_path, vendor_id, hardware_name, logger) -> None:
        self._work_path          = work_path
        self._build_android_path = build_android_path
        self._build_hexagon_path = build_hexagon_path
        self._stress_config_path = stress_config_path
        self._vendor_id          = vendor_id
        self._hardware_name      = hardware_name
        self._logger             = logger
        self._device_ids         = []
        self._device_search_file = 'data/misc/platform_aura2_cicd.txt'
        self._device_work_path   = '/data/local/tmp/aura2.0_test'

        self._stress_config      = None

        # release mode for aura2.0 release
        self._test_types_for_release_mode = ['non_asan', 'hwasan']
        # stree test mode for user's own stress test
        self._test_types_for_stress_mode  = ['non_asan_perf', 'non_asan_mem', 'hwasan']

        # HVX arch version and devices path
        self._hvx_arch_version = ''
        self._hvx_devices_path = ''
        if self._hardware_name == 'SM8550':
            self._hvx_arch_version = 'v73'
            self._hvx_devices_path = '/odm/lib/rfsa/adsp'
        elif self._hardware_name == 'SM8650':
            self._hvx_arch_version = 'v75'
            self._hvx_devices_path = '/odm/lib/rfsa/adsp'

        logger.info(f"Running stress test on {self._hardware_name} with vendor id: {self._vendor_id}, config: {self._stress_config_path}")

        # Parse envs
        envs_res = subprocess.run(f"bash -c 'source /home/mi-aura/workspace/aura2.0_build_env/set_env.sh && env'", shell=True,
                                  capture_output = True, text = True)

        if envs_res.returncode != 0:
            self._raise_error(f"Failed to source script: {envs_res.stderr}")
        else:
            for env in envs_res.stdout.splitlines():
                if '=' not in env:
                    continue
                key = env.split('=')[0]
                value = env.split('=')[1]
                os.environ[key] = value

        # Parse stress config
        if os.path.exists(self._stress_config_path):
            self._parse_stress_config()
        else:
            self._logger.info("Stress config file not found, and use release test mode")

    def _parse_stress_config(self):
        try:
            with open(self._stress_config_path, "r") as f:
                self._stress_config = yaml.safe_load(f)
                self._logger.info(f"Stress config parsed successfully: {self._stress_config}")
        except FileNotFoundError:
            self._raise_error(f"Stress config file not found at {self._stress_config_path}")
        except yaml.YAMLError as e:
            self._raise_error(f"Invalid stress config format: {e}")

    def _raise_error(self, msg):
        # set all devices to not busy
        self._set_all_avialable_devices_busy_state(set_busy = False)

        raise RuntimeError(msg)

    def _signal_handler(self, signum, frame):
        # Signal handler
        # set all devices to not busy
        self._logger.info(f"Signal {signum} received, set all devices to not busy")
        self._set_all_avialable_devices_busy_state(set_busy = False)
        exit(0)

    def _set_all_avialable_devices_busy_state(self, set_busy = False):
        for device_id in self._device_ids:
            self._set_device_busy_state(device_id, set_busy = set_busy)

    def _set_device_busy_state(self, device_id, set_busy = False):
        sed_command = None

        if set_busy:
            sed_command = f"sed -i 's/Busy: Not/Busy: Yes/g' {self._device_search_file}"
        else:
            sed_command = f"sed -i 's/Busy: Yes/Busy: Not/g' {self._device_search_file}"

        # 由于shell中可能会干扰引号，这里我们使用双引号包裹整个命令
        adb_command = f"adb -s {device_id} shell \"{sed_command}\""
        ret_shell = self._run_command(adb_command)
        if ret_shell is None:
            raise RuntimeError(f"adb -s {device_id} shell '{adb_command}' failed")

    # find a usable device and set it as busy
    def _get_android_device(self):
        # get connected devices
        adb_device_res = subprocess.run('adb devices', shell=True, capture_output=True, text=True)
        all_devices    = [line.split()[0] for line in adb_device_res.stdout.splitlines() if line.endswith('device')]

        for device_id in all_devices:
            self._logger.info(f"Check device {device_id}")
            # root
            subprocess.run(f'adb -s {device_id} wait-for-device root', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # check if the device search file exists
            check_file_cmd = f"adb -s {device_id} shell 'test -f {self._device_search_file} && echo File_found || echo File_not_found'"
            check_file_cmd_res = self._run_command(check_file_cmd)
            if check_file_cmd_res is None or 'File_not_found' in check_file_cmd_res.stdout:
                self._logger.info(f"Device({device_id}) search file({self._device_search_file}) not found at, then skip this device")
                continue

            # Get Vendor
            find_vendor = subprocess.run(
                f'adb -s {device_id} shell "cat {self._device_search_file} | grep Vendor"',
                shell=True, capture_output=True, text=True
            ).stdout.strip().split(':')[1].strip()

            # Get Hardware
            find_hardware = subprocess.run(
                f'adb -s {device_id} shell "cat {self._device_search_file} | grep Hardware"',
                shell=True, capture_output=True, text=True
            ).stdout.strip().split(':')[1].strip()

            # find if the device is busy or not
            find_state = subprocess.run(
                f'adb -s {device_id} shell "cat {self._device_search_file} | grep Busy"',
                shell=True, capture_output=True, text=True
            ).stdout.strip().split(':')[1].strip()

            self._logger.info(f"Device info: device_id: {device_id}, Vendor: {find_vendor}, Hardware: {find_hardware}, Busy: {find_state}")

            # check if the device matches vendor, hardware, and is not busy
            if find_vendor == self._vendor_id and find_hardware == self._hardware_name and find_state == 'Not':
                # check again to make sure the device is not busy
                find_state = subprocess.run(
                    f'adb -s {device_id} shell "cat {self._device_search_file} | grep Busy"',
                    shell=True, capture_output=True, text=True
                ).stdout.strip().split(':')[1].strip()

                if find_state == 'Not':
                    # set the device as busy
                    sed_command = f"adb -s {device_id} shell \"sed -i 's/Busy: Not/Busy: Yes/g' {self._device_search_file}\""
                    subprocess.run(sed_command, shell=True, check=True, text=True)

                    self._device_ids.append(device_id)
                    self._logger.info(f"Device {device_id} is available")
            else:
                self._logger.info(f"Device {device_id} is not available, {find_vendor}, {find_hardware}, {find_state}")

    # wait untill one device is available
    def _wait_for_device(self):
        wait_time = 0
        while len(self._device_ids) == 0:
            self._get_android_device()

            if len(self._device_ids) > 0:
                break

            minutes = 10
            self._logger.info(f"Waiting for {minutes} minutes to get a available device")

            time.sleep(minutes * 60)

            wait_time += minutes

            if wait_time > 120:
                self._logger.error("No device found")
                exit(1)


    def _run_command(self, command, timeout=0):
        """Run a shell command with a timeout and handle errors."""
        try:
            if timeout == 0:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            else:
                result = subprocess.run(command, shell=True, timeout=timeout, check=True, capture_output=True, text=True)
            return result
        except subprocess.TimeoutExpired:
            self._logger.error(f"Command timed out: {command}")
            return None
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Command failed with error: {e}, command: {command}")
            return None

    def _init_device(self, device_id):
        """Initialize the device with adb commands."""
        if not device_id:
            self._raise_error("Device ID info are required")

        self._logger.info(f"Initializing device {device_id}")

        if self._stress_config is None:
            # Run adb commands with timeout and handle errors
            self._run_command(f"adb -s {device_id} wait-for-device root", timeout=20)
            self._run_command(f"adb -s {device_id} wait-for-device remount", timeout=20)
            self._run_command(f"adb -s {device_id} wait-for-device reboot", timeout=20)
            self._run_command(f"adb -s {device_id} wait-for-device root", timeout=120)
            self._run_command(f"adb -s {device_id} wait-for-device remount", timeout=20)

            # Wait for device to turn off the screen
            self._logger.info("wait for 1min for QCOM to turn off the screen")
            time.sleep(60)  # Sleep for 1 minute
            self._logger.info("wait end")

            # Turn on the screen
            self._logger.info("turn on the screen")

            self._run_command(f"adb -s {device_id} shell input keyevent 224")
            time.sleep(2)  # Sleep for 2 seconds
            self._run_command(f"adb -s {device_id} shell input swipe 300 1000 300 500")
            self._run_command(f"adb -s {device_id} shell settings put system screen_off_timeout 2147483647")
            self._run_command(f"adb -s {device_id} shell settings put system def_lockscreen_disabled true")
        else:
            # boost the device
            self._logger.info(f"boost the device {device_id} begin")
            subprocess.run(f"bash -c 'bash {self._work_path}/scripts/cicd/boost.sh {device_id}'", shell=True)
            self._logger.info(f"boost the device {device_id} over")

    def _push_data(self, device_id, host_data_path, device_data_path):
        """Push data from host to device using adb."""
        if not device_id or not host_data_path or not device_data_path:
            self._raise_error("Device ID, host data path, and device data path are required")

        self._logger.info(f"Push data from {host_data_path} to device {device_id}")

        # Optional: remove and create directories (uncomment if needed)
        # run_command(f"adb -s {device_id} shell rm -rf {device_data_path}")
        # run_command(f"adb -s {device_id} shell mkdir -p {device_data_path}")

        # Push data to device
        push_command = f"adb -s {device_id} push {host_data_path} {device_data_path}"
        result = self._run_command(push_command)

        if result is None:
            self._logger.error(f"Push data to device failed, {device_id}, {host_data_path}, {device_data_path}")
        else:
            self._logger.info("Data pushed successfully")

    def _pull_file(self, device_id, device_path, local_path):
        """Pull a file from the device using adb."""
        if not device_id or not device_path or not local_path:
            self._raise_error("Device ID, device path, and local path are required")

        self._logger.info(f"Pull file from device {device_id}:{device_path} to {local_path}")

        command = f"adb -s {device_id} pull {device_path} {local_path}"
        result = self._run_command(command)

        if result is None:
            self._logger.error(f"Pull file from device failed, {device_id}, {device_path}, {local_path}")
        else:
            self._logger.info("File pulled successfully")

    def _find_file(self, root_search_path, filename):
        """Find a file in the given directory."""
        for root, dirs, files in os.walk(root_search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def _push_hexagon_libs(self, device_id, root_search_path, device_path):
        """Push hexagon libs to the device."""
        if not device_id or not root_search_path or not device_path:
            self._raise_error("Device ID, root search path, and device path are required")

        self._logger.info(f"Push hexagon libs from {root_search_path} to device {device_id}")

        # Check if the root search path is a directory
        if not os.path.isdir(root_search_path):
            self._raise_error(f"root_search_path: {root_search_path} not found")

        # Find the path of libaura_hexagon_skel.so
        exe_path = self._find_file(root_search_path, "libaura_hexagon_skel.so")
    
        if not exe_path:
            self._raise_error(f"libaura_hexagon_skel.so not found in path: {root_search_path}")

        # Push libaura_hexagon_skel.so to the device
        push_command = f"adb -s {device_id} push {exe_path} {device_path}"
        result = self._run_command(push_command)

        if result is None:
            self._raise_error("Push libaura_hexagon_skel.so to device failed, {device_id}, {exe_path}, {device_path}")
        else:
            self._logger.info("libaura_hexagon_skel.so pushed successfully")

    def _push_android_exe(self, device_id, root_search_path, device_path, run_test_type):
        """Push Android executable and libraries to the device."""
        if not device_id or not root_search_path or not device_path or not run_test_type:
            self._raise_error("Device ID, root search path, device path, and run test type are required")

        self._logger.info(f"Push android exe from {root_search_path} to device {device_id}")

        # 1.1 Find the path of aura_test_main
        if not os.path.isdir(root_search_path):
            self._raise_error(f"root_search_path: {root_search_path} not found")

        exe_path = self._find_file(root_search_path, "aura_test_main")
        if not exe_path:
            self._raise_error("aura_test_main not found in path: {root_search_path}")

        # Push aura_test_main to device
        push_command = f"adb -s {device_id} push {exe_path} {device_path}"
        result = self._run_command(push_command)

        if result is None:
            self._raise_error("Push aura_test_main to device failed, {device_id}, {exe_path}, {device_path}")
        else:
            self._logger.info("aura_test_main pushed successfully")

        ndk_path = os.environ.get("NDK_PATH")
        # 2. Push hwasan's lib if required
        if run_test_type == "hwasan":
            if ndk_path is None:
                self._raise_error("NDK path is required for HWASAN test type")

            lib_path = self._find_file(ndk_path, "libclang_rt.hwasan-aarch64-android.so")
            if not lib_path:
                self._raise_error("libclang_rt.hwasan-aarch64-android.so not found in path: {ndk_path}")

            push_lib_command = f"adb -s {device_id} push {lib_path} {device_path}"
            lib_result = self._run_command(push_lib_command)

            if lib_result is None:
                self._raise_error("Push libclang_rt.hwasan-aarch64-android.so to device failed, {device_id}, {lib_path}, {device_path}")
            else:
                self._logger.info("libclang_rt.hwasan-aarch64-android.so pushed successfully")

    def _generate_config_json(self, work_path, build_android_path, hareware_name, report_type = "txt"):
        """Generate the config.json file."""
        config = {
            "async_affinity": "LITTLE",
            "cache_bin_path": "",
            "cache_bin_prefix": "",
            "compute_affinity": "BIG",
            "data_path": "./../data/",
            "device_info": hareware_name,
            "log_file_name": "log",
            "log_level": "DEBUG",
            "log_output": "STDOUT",
            "ndk_info": "ndk_r26d",
            "report_name": "auto_test",
            "report_type": report_type,
            "stress_count": 0
        }
    
        config_path = os.path.join(work_path, build_android_path, "config.json")
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
    
        return config_path

    def _run_test_for_release_mode(self, device_id, hareware_name, run_test_type, work_path, build_android_path, device_test_path):
        self._logger.info("_run_test_for_release_mode")

        # Generate config.json && Push config.json
        report_type = "json" if self._stress_config is not None and run_test_type == 'non_asan_perf' else "txt"
        config_json_path = self._generate_config_json(work_path, build_android_path, hareware_name, report_type)
        self._push_data(device_id, config_json_path, device_test_path)

        os.makedirs(f"{work_path}/{build_android_path}/log", exist_ok=True)
        log_dir_path = f"{device_test_path}/log"
        self._run_command(f"adb -s {device_id} shell mkdir -p {log_dir_path}")
    
        current_time = datetime.datetime.now().strftime("%F_%T")
        new_name_run_log = f"{work_path}/{build_android_path}/log/log_{hareware_name}_{run_test_type}_{current_time}.txt"
        new_name_auto_test = f"{work_path}/{build_android_path}/log/auto_test_{hareware_name}_{run_test_type}_{current_time}.txt"
        name_logcat = f"{work_path}/{build_android_path}/log/logcat_{hareware_name}_{run_test_type}_{current_time}.txt"

        self._logger.info(f"new_name_run_log: {new_name_run_log}")
        self._logger.info(f"new_name_auto_test: {new_name_auto_test}")
        self._logger.info(f"name_logcat: {name_logcat}")

        logcat_proc = None
        try:
            # Start logcat
            logcat_command = f"adb -s {device_id} logcat | grep -E 'kgsl-3d0|FAULT|fail|DEBUG|backtrace|crash|error|leak|build with source' > {name_logcat}"
            logcat_proc = subprocess.Popen(logcat_command, shell=True)

            # config
            run_str        = "all"
            black_list_str = "runtime_,algos_,"

            stress_count   = 5 if run_test_type == 'non_asan' else 2

            if self._vendor_id == "MTK":
                black_list_str += "_hvx,"

            if run_test_type == "asan" or run_test_type == "hwasan":
                black_list_str += "_none,"

            self._logger.info(f"Release config value: run_str: {run_str}, black_list: {black_list_str}, stress_count: {stress_count}")

            release_cmd = f"adb -s {device_id} shell 'cd {device_test_path}; export LD_LIBRARY_PATH=.\:/system/lib64:/vendor/lib64:/vendor/lib64/egl; ./aura_test_main -r {run_str} -b {black_list_str} -c ./config.json &> {device_test_path}/log/auto_test_log.txt'"

            self._logger.info(f"**** Start release test ****")
            result = self._run_command(release_cmd)
            self._logger.info(f"**** release test over ****")

            # Pull logs
            if result is not None:
                self._pull_file(device_id, f"{device_test_path}/log/auto_test_log.txt", new_name_run_log)
                self._pull_file(device_id, f"{device_test_path}/auto_test.txt", new_name_auto_test)
        finally:
            if logcat_proc is not None:
                logcat_proc.kill()

        # Check test result
        if os.path.exists(new_name_auto_test):
            with open(new_name_auto_test) as f:
                for line in f:
                    if "Failed:" in line:
                        failed = int(line.split()[1])
                        if failed > 0:
                            self._raise_error(f"auto_test failed with {failed} failed cases")
                        else:
                            self._logger.info(f"auto_test success: {hareware_name}, {run_test_type}, {device_id}")
                            break

        # Copy files to artifact path
        log_saved_path = os.path.join(work_path, "log", "android")
        os.makedirs(log_saved_path, exist_ok=True)

        # Copy log files to artifact path
        for log_file in [new_name_run_log, new_name_auto_test, name_logcat]:
            self._run_command(f"cp {log_file} {log_saved_path}")

        self._logger.info(f"****** cur test success: hareware_name: {hareware_name}, run_test_type: {run_test_type}, device_id: {device_id} **********")

    def _run_test_for_stress_mode(self, device_id, hareware_name, run_test_type, work_path, build_android_path, device_test_path):
        self._logger.info("_run_test_for_stress_mode")

        # Generate config.json && Push config.json
        report_type = "json" if self._stress_config is not None and run_test_type == 'non_asan_perf' else "txt"
        config_json_path = self._generate_config_json(work_path, build_android_path, hareware_name, report_type)
        self._push_data(device_id, config_json_path, device_test_path)

        if run_test_type == 'non_asan_mem':
            # Push memory_monitor.sh for stress mode
            memory_monitor_sh_path = os.path.join(work_path, "scripts", "memory/memory_monitor.sh")
            self._push_data(device_id, memory_monitor_sh_path, device_test_path)

        os.makedirs(f"{work_path}/{build_android_path}/log", exist_ok=True)
        log_dir_path = f"{device_test_path}/log"
        self._run_command(f"adb -s {device_id} shell mkdir -p {log_dir_path}")
    
        current_time = datetime.datetime.now().strftime("%F_%T")
        new_name_run_log   = f"{work_path}/{build_android_path}/log/log_{hareware_name}_{run_test_type}_{current_time}.txt"
        new_name_auto_test = f"{work_path}/{build_android_path}/log/auto_test_{hareware_name}_{run_test_type}_{current_time}.{report_type}"
        name_logcat        = f"{work_path}/{build_android_path}/log/logcat_{hareware_name}_{run_test_type}_{current_time}.txt"
        perf_viual_folder  = f"{work_path}/{build_android_path}/log/perf_visual_{hareware_name}_{run_test_type}_{current_time}"
        stress_report_name = f"mem_stress_report_{hareware_name}_{run_test_type}_{current_time}"
        host_mem_info_path = f"{work_path}/{build_android_path}/log/{stress_report_name}"
        mem_plot_save_path = host_mem_info_path + "/" + stress_report_name

        self._logger.info(f"new_name_run_log: {new_name_run_log}")
        self._logger.info(f"new_name_auto_test: {new_name_auto_test}")
        self._logger.info(f"name_logcat: {name_logcat}")
        self._logger.info(f"perf_viual_folder: {perf_viual_folder}")
        self._logger.info(f"stress_report_name: {stress_report_name}")

        logcat_proc = None
        try:
            # Start logcat
            logcat_command = f"adb -s {device_id} logcat | grep -E 'kgsl-3d0|FAULT|fail|DEBUG|backtrace|crash|error|leak|build with source' > {name_logcat}"
            logcat_proc = subprocess.Popen(logcat_command, shell=True)

            # Parse the stress config
            run_str      = ""
            stress_count = 10
            filter_width = 1000

            ret_find = self._find_value_of_second_depth_of_yaml('run')
            if ret_find is not None:
                run_str += " -r "
                run_str += ",".join(ret_find)

            ret_find = self._find_value_of_second_depth_of_yaml('filter')
            if ret_find is not None:
                run_str += " -f "
                run_str += ",".join(ret_find)

            ret_find = self._find_value_of_second_depth_of_yaml('blacklist')
            if ret_find is not None:
                run_str += " -b "
                run_str += ",".join(ret_find)

            if isinstance(self._stress_config['stress'], int):
                stress_count = self._stress_config['stress']

            if isinstance(self._stress_config['filter_width'], int):
                filter_width = self._stress_config['filter_width']

            self._logger.info(f"stress config value: {run_str}, {stress_count}, {filter_width}")

            # Run aura_test_main
            if run_test_type == 'non_asan_perf':
                # Preformance mode
                perf_cmd = f"adb -s {device_id} shell 'cd {device_test_path}; export LD_LIBRARY_PATH=.\:/system/lib64:/vendor/lib64:/vendor/lib64/egl; ./aura_test_main {run_str} -c ./config.json &> {device_test_path}/log/auto_test_log.txt'"

                self._logger.info(f"**** Start performance test ****")
                result = self._run_command(perf_cmd)
                self._logger.info(f"**** Performance test over ****")

                # Try to pull logs envn if result is not None(ok) or not(fail)
                self._pull_file(device_id, f"{device_test_path}/log/auto_test_log.txt", new_name_run_log)
                self._pull_file(device_id, f"{device_test_path}/auto_test.{report_type}", new_name_auto_test)

                if result is not None:
                    # Performance visualization
                    json_root_path = os.path.dirname(new_name_auto_test)
                    self._run_command(f"python3 {work_path}/scripts/visual/visual.py -p {new_name_auto_test} -f {filter_width} -o {perf_viual_folder}")

                    self._logger.info(f"Performance visualization over")
            elif run_test_type == 'non_asan_mem':
                # # Memory mode
                stress_test_perioid = 1

                mem_cmd = f"adb -s {device_id} shell 'cd {device_test_path}; export LD_LIBRARY_PATH=.\:/system/lib64:/vendor/lib64:/vendor/lib64/egl; source ./memory_monitor.sh -n aura_test_main -i {stress_test_perioid} -d {stress_report_name} &./aura_test_main {run_str} -s {stress_count} -c ./config.json &> {device_test_path}/log/auto_test_log.txt'"

                self._logger.info(f"**** Start memory test ****")
                result = self._run_command(mem_cmd)
                self._logger.info(f"**** Memory test over ****")

                # Try to pull logs envn if result is not None(ok) or not(fail)
                self._pull_file(device_id, f"{device_test_path}/log/auto_test_log.txt", new_name_run_log)
                self._pull_file(device_id, f"{device_test_path}/auto_test.{report_type}", new_name_auto_test)
                self._pull_file(device_id, f"{device_test_path}/{stress_report_name}", host_mem_info_path)

                if result is not None:
                    # plot memory info
                    self._run_command(f"python3 {work_path}/scripts/memory/plot_mem_info.py {host_mem_info_path} {mem_plot_save_path}")

                    self._logger.info(f"Memory visualization over")
            elif run_test_type == 'asan' or run_test_type == 'hwasan':
                # Preformance mode
                asan_cmd = f"adb -s {device_id} shell 'cd {device_test_path}; export LD_LIBRARY_PATH=.\:/system/lib64:/vendor/lib64:/vendor/lib64/egl; ./aura_test_main {run_str} -c ./config.json &> {device_test_path}/log/auto_test_log.txt'"

                self._logger.info(f"**** Start asan test ****")
                result = self._run_command(asan_cmd)
                self._logger.info(f"**** asan test over ****")

                # Try to pull logs envn if result is not None(ok) or not(fail)
                self._pull_file(device_id, f"{device_test_path}/log/auto_test_log.txt", new_name_run_log)
                self._pull_file(device_id, f"{device_test_path}/auto_test.{report_type}", new_name_auto_test)
        finally:
            if logcat_proc is not None:
                logcat_proc.kill()

        # Check test result
        if run_test_type != "non_asan_perf":
            with open(new_name_auto_test) as f:
                for line in f:
                    if "Failed:" in line:
                        failed = int(line.split()[1])
                        if failed > 0:
                            self._raise_error(f"auto_test failed with {failed} failed cases")
                        else:
                            self._logger.info(f"auto_test success: {hareware_name}, {run_test_type}, {device_id}")
                            break

        # Copy files to artifact path
        log_saved_path = os.path.join(work_path, "log", "android")
        os.makedirs(log_saved_path, exist_ok=True)

        # Copy log files to artifact path
        for log_file in [new_name_run_log, new_name_auto_test, name_logcat]:
            self._run_command(f"cp {log_file} {log_saved_path}")

        # Copy visual files to artifact path
        if run_test_type == "non_asan_perf":
            self._run_command(f"cp -rf {perf_viual_folder} {log_saved_path}")

        # Copy memory info to artifact path
        if run_test_type == "non_asan_mem":
            self._run_command(f"cp -rf {host_mem_info_path} {log_saved_path}")
    
        self._logger.info(f"****** cur test success: hareware_name: {hareware_name}, run_test_type: {run_test_type}, device_id: {device_id} **********")

    def _run_on_each_device(self, device_id, cur_run_test_types):
        # 1. Init Device
        self._init_device(device_id)

        # 2. Push test datasheet to device
        self._push_data(device_id, f"{self._work_path}/data/", self._device_work_path)

        for run_test_type in cur_run_test_types:
            self._logger.info(f"**********cur device test: id: {device_id}, test_type: {run_test_type} **********")

            # 3. Push exe and libs to device
            # 3.1 Push test exe to device
            device_test_path = os.path.join(self._device_work_path, f"{self._hardware_name}_{run_test_type}")

            self._run_command(f"adb -s {device_id} shell 'rm -rf {device_test_path}'")
            ret_mkdir = self._run_command(f"adb -s {device_id} shell 'mkdir -p {device_test_path}'")
            if ret_mkdir is None:
                self._raise_error(f"adb -s {device_id} shell 'mkdir -p {device_test_path}' failed")

            lib_build_type = "release" if "non_asan" in run_test_type else "debug"

            lib_run_type   = None
            if "non_asan" in run_test_type:
                # non_asan_pref/non_asan_mem/non_asan
                lib_run_type = "non_asan"
            else:
                # hwasan/asan
                lib_run_type = run_test_type

            self._push_android_exe(device_id, os.path.join(self._work_path, self._build_android_path,
                                   f"android/arm64-v8a/static/{lib_build_type}/{lib_run_type}/"), device_test_path, run_test_type)

            # 3.2 Push libs to device
            if "Qualcomm" == self._vendor_id:
                self._push_hexagon_libs(device_id, os.path.join(self._work_path, self._build_hexagon_path,
                                        f"{self._hvx_arch_version}/share/release/install"), self._hvx_devices_path)

            # 4. Run test
            if self._stress_config is not None:
                # 4.1 Run stress test
                self._run_test_for_stress_mode(device_id, self._hardware_name, run_test_type, self._work_path,
                                               self._build_android_path, device_test_path)
            else:
                # 4.2 Run release test
                self._run_test_for_release_mode(device_id, self._hardware_name, run_test_type, self._work_path,
                                               self._build_android_path, device_test_path)

        # 5. Deinit device
        # 5.1 Turn off screen
        self._run_command(f"adb -s {device_id} shell settings put system screen_off_timeout 300000")
        self._run_command(f"adb -s {device_id} shell settings put system def_lockscreen_disabled false")

        # 5.2 Set device to not busy mode
        self._set_device_busy_state(device_id, False)

        # 5.3 Reboot device
        self._run_command(f"adb -s {device_id} shell reboot")

        self._logger.info(f"****** cur device success: id: {device_id}, test_types: {cur_run_test_types} **********")

    def _find_value_of_second_depth_of_yaml(self, key):
        if (self._stress_config is not None and
            self._vendor_id in self._stress_config and isinstance(self._stress_config[self._vendor_id], dict) and
            key in self._stress_config[self._vendor_id] and isinstance(self._stress_config[self._vendor_id][key], list) and
            len(self._stress_config[self._vendor_id][key]) > 0 and self._stress_config[self._vendor_id][key][0] is not None):

            self._logger.info(f"****** Find value of second depth of yaml:{self._vendor_id}, {key} **********")
            return self._stress_config[self._vendor_id][key]
        else:
            self._logger.info(f"****** No value of second depth of yaml:{self._vendor_id}, {key} **********")
            return None

    def _run_wrapper(self):
        # 0. Set signal handler to set all devices to not busy mode
        signal.signal(signal.SIGINT,  self._signal_handler) # SIGINT: Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler) # SIGTERM: kill
        signal.signal(signal.SIGHUP,  self._signal_handler) # SIGHUP: terminal close
        signal.signal(signal.SIGPIPE, self._signal_handler) # SIGPIPE: pipe error

        # 1.1 Check steres test config
        if (self._stress_config is not None and
            self._find_value_of_second_depth_of_yaml('run') is None and
            self._find_value_of_second_depth_of_yaml('filter') is None):

            self._logger.info(f"****** No run for this vendor: {self._vendor_id}, and see below for more info **********")
            if self._vendor_id in self._stress_config:
                self._logger.info(f"******Config for this vendor: {self._stress_config[self._vendor_id]} *********")
            else:
                self._logger.info(f"******Config: no config for this vendor *********")

            exit(0)

        # 1.2 Find avialable devices
        self._logger.info("****** start to find available device **********")
        self._wait_for_device()
        self._logger.info(f"****** find available device success with num: {len(self._device_ids)}, id: {self._device_ids} **********")

        # 2. Split the tasks to each device
        test_types_for_device = self._test_types_for_stress_mode if self._stress_config is not None else self._test_types_for_release_mode
        tasks_per_device      = len(test_types_for_device) // len(self._device_ids)
        remainder_tasks       = len(test_types_for_device) % len(self._device_ids)

        # 3. Run task on each device
        processes = []
        for i in range(len(self._device_ids)): 
            device_id = self._device_ids[i]

            start_index_of_tasks = i * tasks_per_device
            end_index_of_tasks   = (i + 1) * tasks_per_device - 1

            end_index_of_tasks = end_index_of_tasks + remainder_tasks if i == len(self._device_ids) - 1 and remainder_tasks > 0 else end_index_of_tasks

            self._logger.info(f"Assign task to device: {device_id}, start_index: {start_index_of_tasks}, end_index: {end_index_of_tasks}")

            cur_test_types_for_device = test_types_for_device[start_index_of_tasks : end_index_of_tasks + 1]

            if len(self._device_ids) == 1:
                self._run_on_each_device(device_id, cur_test_types_for_device)

                self._logger.info(f"Sync run all tasks on only one device: {device_id}, test_types: {cur_test_types_for_device}")
            else:
                p = multiprocessing.Process(target=self._run_on_each_device, args=(device_id, cur_test_types_for_device))
                processes.append(p)
                p.start()

                self._logger.info(f"Async run tasks on device: {device_id}, test_types: {cur_test_types_for_device}")

        # 4. Wait all async tasks finished
        for p in processes:
            p.join()

        self._logger.info("****** _run_wrapper over **********")

def main():
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument('-wp', '--work_path',          required=True, help='Path to the work directory')
    parser.add_argument('-ba', '--build_android_path', required=True, help='Path to the Android build directory')
    parser.add_argument('-bh', '--build_hexagon_path', required=True, help='Path to the Hexagon build directory')
    parser.add_argument('-sc', '--stress_config_path', required=True, help='Path to the stress test configuration file')
    parser.add_argument('-vi', '--vendor_id',          required=True, help='Vendor identification number')
    parser.add_argument('-hn', '--hardware_name',      required=True, help='Name of the hardware')

    args = parser.parse_args()

    log_format = '%(asctime)-15s [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format)
    logger = logging.getLogger('ci/cd run test')

    run_test_impl = RunTest(args.work_path, args.build_android_path, args.build_hexagon_path, args.stress_config_path,
                            args.vendor_id, args.hardware_name, logger)

    run_test_impl ._run_wrapper()

if __name__ == "__main__":
    main()

# Usege:
# python3 test_android.py -wp /path/to/work_dir -ba /path/to/android_build_dir -bh /path/to/hexagon_build_dir -sc /path/to/stress_config_file -vi 1234567890 -hn test_device
