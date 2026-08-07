import numpy as np
import matplotlib.pyplot as plt
import time
import os


class SimpleDataComparator:
    def __init__(self, file1, file2, dtype, shape):
        """
        简化的数据比较器
        
        参数:
        file1, file2: 要比较的数据文件路径
        dtype: 数据类型 ('uint8', 'int16', 'float32'等)
        shape: 数据形状 (height, width)
        """
        self.shape = shape

        # 读取数据文件
        self.data1 = np.fromfile(file1, dtype=dtype).reshape(shape)
        self.data2 = np.fromfile(file2, dtype=dtype).reshape(shape)

        # 验证数据大小
        if self.data1.shape != self.data2.shape:
            raise ValueError("两个数据文件的尺寸不同")

    def find_differences(self, tolerance=0.0):
        """
        找出所有不同的位置
        tolerance: 允许的差异容限（0表示完全相等）
        """
        # 比较数据，找出差异位置
        diff_map = np.abs(self.data1 - self.data2) > tolerance
        diff_indices = np.where(diff_map)

        # 收集差异详情
        diff_list = []
        for y, x in zip(diff_indices[0], diff_indices[1]):
            diff_list.append({
                'row': y,
                'col': x,
                'value1': self.data1[y, x],
                'value2': self.data2[y, x],
                'difference': abs(self.data1[y, x] - self.data2[y, x])
            })

        # 基本统计
        total_pixels = np.prod(self.shape)
        diff_count = len(diff_list)
        diff_percent = 100 * diff_count / total_pixels

        return {
            'diff_list': diff_list,
            'total_pixels': total_pixels,
            'diff_count': diff_count,
            'diff_percent': diff_percent
        }

    def visualize_differences(self, diff_info, max_display=1000):
        """
        可视化显示差异位置
        
        参数:
        diff_info: find_differences返回的差异信息
        max_display: 最多显示的点数
        """
        # 基本统计
        print(f"发现 {diff_info['diff_count']} 个不同的位置")
        print(f"占总像素的 {diff_info['diff_percent']:.4f}%")

        # 随机采样部分差异点（如果数量太多）
        if diff_info['diff_count'] > max_display:
            print(f"差异点过多，仅显示前 {max_display} 个示例")
            display_points = np.random.choice(
                diff_info['diff_list'], max_display, replace=False)
        else:
            display_points = diff_info['diff_list']

        # 在图像上标记差异点
        plt.figure(figsize=(12, 8))

        # 创建差异掩码（红色表示差异位置）
        diff_mask = np.zeros(self.shape + (3,), dtype=np.uint8)
        diff_mask[:, :, 0] = 255  # 红色通道

        # 叠加原始图像（绿色通道）
        norm_data = (self.data1 - np.min(self.data1)) / \
            (np.max(self.data1) - np.min(self.data1) + 1e-9)
        diff_mask[:, :, 1] = (norm_data * 200).astype(np.uint8)  # 绿色通道显示源数据

        # 创建差异热图
        diff_map = np.abs(self.data1 - self.data2)
        norm_diff = (diff_map - np.min(diff_map)) / \
            (np.max(diff_map) - np.min(diff_map) + 1e-9)
        diff_mask[:, :, 2] = (norm_diff * 255).astype(np.uint8)  # 蓝色通道显示差异强度

        # 显示图像
        plt.imshow(diff_mask)
        plt.title(f"差异位置可视化: {diff_info['diff_count']} 个不同点")

        # 添加坐标网格
        plt.grid(True, alpha=0.3)
        plt.gca().set_xticks(
            np.arange(0, self.shape[1], max(1, self.shape[1]//20)))
        plt.gca().set_yticks(
            np.arange(0, self.shape[0], max(1, self.shape[0]//20)))

        # 在图像上标记差异点（用小点标记）
        for point in display_points:
            plt.scatter(point['col'], point['row'], s=5,
                        c='white', edgecolors='black')

        plt.tight_layout()
        plt.show()

        # 打印部分差异详情
        print("\n差异详情示例:")
        print("行 | 列 | 文件1值 | 文件2值 | 差异")
        print("----------------------------------")
        for i, point in enumerate(display_points[:min(10, len(display_points))]):
            print(
                f"{point['row']:3} | {point['col']:3} | {point['value1']:.4f} | {point['value2']:.4f} | {point['difference']:.4f}")


# 使用示例
if __name__ == "__main__":
    # 定义文件路径
    data_dir = "./"
    file1 = os.path.join(data_dir, "image_data_4k_1.bin")
    file2 = os.path.join(data_dir, "image_data_4k_1.bin")
    shape = (4096, 3072)  # 数据形状
    dtype = np.float32    # 数据类型

    # 创建比较器
    start_time = time.time()
    comparator = SimpleDataComparator(file1, file2, dtype, shape)
    print(f"数据加载耗时: {time.time() - start_time:.2f}秒")

    # 找出差异
    diff_info = comparator.find_differences(tolerance=1.0)  # 差异超过10才算不同

    # 可视化差异
    comparator.visualize_differences(diff_info)

    # 打印更多差异详情（最多10个）
    if diff_info['diff_count'] > 0:
        print("\n更多差异详情:")
        print("行 | 列 | 文件1值 | 文件2值 | 差异")
        print("----------------------------------")
        for point in diff_info['diff_list'][:10]:
            print(
                f"{point['row']:3} | {point['col']:3} | {point['value1']:.4f} | {point['value2']:.4f} | {point['difference']:.4f}")
