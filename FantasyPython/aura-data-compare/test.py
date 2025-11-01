
# 初始化比较器
import time
from data_comparator import LargeDataComparator


comparator = LargeDataComparator(
    file1="image_data_4k_1.bin",
    file2="image_data_4k_2.bin",
    dtype="f32",
    shape=(4096, 3072),
    endian="little",
    chunksize=4096 * 256  # 每块256行
)

# 生成热力图报告（自动下采样）
comparator.plot_heatmap_comparison(max_display_size=1024)

# 生成交互式HTML报告
comparator.save_difference_report("sensor_comparison.html")

# 对于性能分析：
start = time.time()
stats = comparator.calculate_stats()
print(f"统计分析用时: {time.time()-start:.2f}秒, 相似度: {stats['similarity']:.4f}")