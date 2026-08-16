"""
JSON文件配置模块
向后兼容包装器，使用新的统一基类
"""

import os

from src.config.base_config import ReadBaseConfig
from src.config.config_hardware import HardwareConfig
from src.config.platform_compat import get_platform, is_windows

from .file_config_base import JsonFileHandler


class _JsonFileConfig(JsonFileHandler):
    """JSON文件配置基类（向后兼容）"""

    def __init__(self):
        super().__init__()
        self.basedata_info = ReadBaseConfig()
        self._platform = get_platform()
        component_path = self.basedata_info.get_filepath("jsonfile_path")
        if is_windows():
            user = HardwareConfig.get_user()
            self.json_file_path = os.path.join(
                rf"C:\Users\{user}", component_path
            )
        else:
            # macOS/Linux: 使用跨平台路径
            self.json_file_path = os.path.join(
                self._platform.get_app_data_path(), component_path
            )

    def _get_read_json(self):
        """读取JSON文件"""
        return self.load_json(self.json_file_path)


class HandleJson(_JsonFileConfig):
    """JSON文件处理器（向后兼容）"""

    def __init__(self):
        super().__init__()

    def read_json(self, components_name):
        """读取指定组件名称的数据"""
        component_lists = []
        json_datas = self._get_read_json()
        if json_datas:
            for i in json_datas:
                if components_name in i:
                    component_lists.append(i[components_name])
        return component_lists

    def count_componentname(self):
        """统计组件名称"""
        registered_names = []
        registered_name_map = {}
        component_lists = self.read_json("regName")
        for component_list in component_lists:
            components = component_list.split("_", 2)
            if len(components) > 1:
                registered_names.append(components[1])
        for registered_name in set(registered_names):
            if registered_name != "break":
                registered_name_map[registered_name] = registered_names.count(
                    registered_name
                )
        return registered_name_map

    def count_list(self, listname):
        """统计列表元素"""
        result_list = {}
        for i in set(listname):
            result_list.update({i: listname.count(i)})
        return result_list

    def match_component(self, result_components):
        """匹配组件"""
        error_components = {}
        target_components = self.count_componentname()
        for key, value in target_components.items():
            if (
                key in result_components
                and self.count_list(result_components)[key] == value
            ):
                continue
            else:
                current_count = self.count_list(result_components).get(key, 0)
                error_components[key] = int(target_components[key]) - int(
                    current_count
                )
        return error_components if error_components else {}

    def match_cid(self, result_cid):
        """匹配last_cid"""
        if len(result_cid):
            cid_list = self.read_json("cid")
            if cid_list and result_cid[-1] in cid_list:
                return result_cid
            else:
                return None
        return None
