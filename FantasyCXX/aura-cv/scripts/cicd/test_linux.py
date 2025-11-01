import subprocess
import time
import os
import yaml
import json
import datetime
import argparse
import logging
import glob

class RunTest:
    def __init__(self, work_path, build_linux_path, stress_config_path, logger, vendor_id = 'Linux', hardware_name = 'x64') -> None:
        self._work_path          = work_path
        self._build_linux_path   = build_linux_path
        self._stress_config_path = stress_config_path
        self._logger             = logger
        self._vendor_id          = vendor_id
        self._hardware_name      = hardware_name

        self._stress_config      = None

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
        raise RuntimeError(msg)

    def _find_file(self, root_search_path, filename):
        """Find a file in the given directory."""
        for root, dirs, files in os.walk(root_search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def _fuzzy_find_file(self, search_pattern, filename):
        for search_path in glob.glob(search_pattern, recursive=True):
            file = self._find_file(search_path, filename)

            if file is not None:
                return file

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

    def _generate_config_json(self, work_path, save_path, hareware_name, report_type = "txt"):
        """Generate the config.json file."""
        config = {
            "async_affinity": "LITTLE",
            "cache_bin_path": "",
            "cache_bin_prefix": "",
            "compute_affinity": "BIG",
            "data_path": f"{work_path}/data/",
            "device_info": hareware_name,
            "log_file_name": "log",
            "log_level": "DEBUG",
            "log_output": "STDOUT",
            "ndk_info": "ndk_r26d",
            "report_name": "auto_test",
            "report_type": report_type,
            "stress_count": 0
        }
    
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
    
        return config_path

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

    def _run_test_for_release_mode(self, work_path, exe_path):
        self._logger.info("_run_test_for_release_mode")

        os.makedirs(f"{work_path}/{self._build_linux_path}/log", exist_ok=True)
        current_time = datetime.datetime.now().strftime("%F_%T")
        new_name_auto_test = f"{work_path}/{self._build_linux_path}/log/auto_test_{self._vendor_id}_{current_time}.txt"

        exe_root_path = os.path.dirname(exe_path)
        run_cmd = f"cd {exe_root_path} && export LD_LIBRARY_PATH=./../lib:$LD_LIBRARY_PATH; ./aura_test_main -r all -f none -b runtime,resize -c config.json -s 2"

        self._logger.info(f"run_cmd: {run_cmd}")

        self._logger.info(f"**** Start test for release mode ****")
        subprocess.run(run_cmd, shell=True)
        self._logger.info(f"**** End test for release mode ****")

        auto_test_file = os.path.join(exe_root_path, "auto_test.txt")
        if os.path.exists(auto_test_file):
            # Check test result
            with open(auto_test_file) as f:
                for line in f:
                    if "Failed:" in line:
                        failed = int(line.split()[1])
                        if failed > 0:
                            self._raise_error(f"auto_test failed with {failed} failed cases")
                        else:
                            self._logger.info(f"auto_test success: {self._hardware_name}")
                            break

            self._run_command(f"cp {auto_test_file} {new_name_auto_test}")

            # Copy files to artifact path
            log_saved_path = os.path.join(work_path, "log", "linux")
            os.makedirs(log_saved_path, exist_ok=True)

            # Copy log files to artifact path
            for log_file in [new_name_auto_test]:
                self._run_command(f"cp {log_file} {log_saved_path}")
        else:
            self._raise_error(f"auto_test.txt not found")

    def _run_test_for_stress_mode(self, work_path, exe_path):
        self._logger.info("_run_test_for_release_mode")

        os.makedirs(f"{work_path}/{self._build_linux_path}/log", exist_ok=True)
        current_time = datetime.datetime.now().strftime("%F_%T")
        new_name_auto_test = f"{work_path}/{self._build_linux_path}/log/auto_test_{self._vendor_id}_{current_time}.txt"

        # Parse the stress config
        run_str = ""

        ret_find = self._find_value_of_second_depth_of_yaml('run')
        if ret_find is not None:
            run_str += " -r "
            run_str += ",".join(ret_find)

        ret_find = self._find_value_of_second_depth_of_yaml('filter')
        if ret_find is not None:
            run_str += " -f none,"
            run_str += ",".join(ret_find)

        ret_find = self._find_value_of_second_depth_of_yaml('blacklist')
        if ret_find is not None:
            run_str += " -b "
            run_str += ",".join(ret_find)

        self._logger.info(f"stress config value: {run_str}")

        exe_root_path = os.path.dirname(exe_path)
        run_cmd = f"cd {exe_root_path} && export LD_LIBRARY_PATH=./../lib:$LD_LIBRARY_PATH; ./aura_test_main {run_str} -c config.json"

        self._logger.info(f"run_cmd: {run_cmd}")

        self._logger.info(f"**** Start test for stress mode ****")
        # self._run_command(run_cmd)
        subprocess.run(run_cmd, shell=True)
        self._logger.info(f"**** End test for stress mode ****")

        auto_test_file = os.path.join(exe_root_path, "auto_test.txt")
        if os.path.exists(auto_test_file):
            # Check test result
            with open(auto_test_file) as f:
                for line in f:
                    if "Failed:" in line:
                        failed = int(line.split()[1])
                        if failed > 0:
                            self._raise_error(f"auto_test failed with {failed} failed cases")
                        else:
                            self._logger.info(f"auto_test success: {self._vendor_id}")
                            break

            self._run_command(f"cp {auto_test_file} {new_name_auto_test}")

            # Copy files to artifact path
            log_saved_path = os.path.join(work_path, "log", "linux")
            os.makedirs(log_saved_path, exist_ok=True)

            # Copy log files to artifact path
            for log_file in [new_name_auto_test]:
                self._run_command(f"cp {log_file} {log_saved_path}")
        else:
            self._raise_error(f"auto_test.txt not found")

    def _run_wrapper(self):
        # 1 Check steres test config
        if (self._stress_config is not None and
            self._find_value_of_second_depth_of_yaml('run') is None and
            self._find_value_of_second_depth_of_yaml('filter') is None):

            self._logger.info(f"****** No run for this vendor: {self._vendor_id}, and see below for more info **********")
            if self._vendor_id in self._stress_config:
                self._logger.info(f"******Config for this vendor: {self._stress_config[self._vendor_id]} *********")
            else:
                self._logger.info(f"******Config: no config for this vendor *********")

            exit(0)

        # 2 find exe
        exe_path = self._fuzzy_find_file(f"{self._work_path}/{self._build_linux_path}/**/install/**/bin/", 'aura_test_main')
        if not exe_path:
            self._raise_error(f"aura_test_main not found, {self._work_path}, {self._build_linux_path}")

        self._logger.info(f"exe_path: {exe_path}")
        exe_root_path = os.path.dirname(exe_path)

        # 3. Generate json in exe root path
        self._generate_config_json(self._work_path, exe_root_path, self._hardware_name)

        # 4. Run test
        if self._stress_config is not None:
            self._run_test_for_stress_mode(self._work_path, exe_path)
        else:
            self._run_test_for_release_mode(self._work_path, exe_path)

def main():
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument('-wp', '--work_path',          required=True, help='Path to the work directory')
    parser.add_argument('-bl', '--build_linux_path',   required=True, help='Path to the Linux build directory')
    parser.add_argument('-sc', '--stress_config_path', required=True, help='Path to the stress test configuration file')

    args = parser.parse_args()

    log_format = '%(asctime)-15s [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format)
    logger = logging.getLogger('ci/cd run test')

    run_test_impl = RunTest(args.work_path, args.build_linux_path, args.stress_config_path, logger)

    run_test_impl ._run_wrapper()

if __name__ == "__main__":
    main()

# Usege:
# python3 test_android.py -wp /home/test/work_path -bl build/linux -sc config.yaml