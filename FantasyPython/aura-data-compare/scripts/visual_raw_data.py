import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def visualize_boundary_pixels(data, width, height, dtype='uint8', boundary_size=3):
    """
    边界像素值实时显示系统
    Args:
        data: 一维像素数据
        width: 图像宽度
        height: 图像高度
        dtype: 数据类型 ('uint8'/'float32')
        boundary_size: 边界显示区域大小（像素数）
    """
    # 重塑为二维数组
    img = data.reshape((height, width))

    # 创建交互界面
    fig, ax = plt.subplots(figsize=(16, 9))

    # 根据数据类型选择显示模式
    if dtype == 'float32':
        im = ax.imshow(img, cmap='viridis')
        cbar = plt.colorbar(im, fraction=0.03, pad=0.01)
        cbar.set_label('Raw Float Value', rotation=270, labelpad=15)
    else:
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        cbar = plt.colorbar(im, fraction=0.03, pad=0.01)
        cbar.set_label('U8 Pixel Value', rotation=270, labelpad=15)

    # 设置标题
    plt.title(f"Boundary Pixel Display | Size: {width}×{height} | Type: {dtype}", fontsize=14)

    # 使用等宽字体确保对齐
    mono_font = FontProperties(family='monospace', size=8)

    # 绘制边界像素值（上下左右）
    def draw_boundary_values():
        # 上边界
        for x in range(0, width, 2):  # 间隔显示避免重叠
            if x < width and boundary_size < height:
                val = img[boundary_size, x]
                val_str = f"{val:.2f}" if dtype == 'float32' else f"{val:3d}"
                ax.text(x, boundary_size - 0.5, val_str,
                        color='red', ha='center', fontproperties=mono_font)

        # 下边界
        for x in range(0, width, 2):
            if x < width and (height - boundary_size - 1) >= 0:
                val = img[height - boundary_size - 1, x]
                val_str = f"{val:.2f}" if dtype == 'float32' else f"{val:3d}"
                ax.text(x, height - boundary_size - 0.5, val_str,
                        color='red', ha='center', fontproperties=mono_font)

        # 左边界
        for y in range(boundary_size, height - boundary_size, 2):
            if y < height and boundary_size < width:
                val = img[y, boundary_size]
                val_str = f"{val:.2f}" if dtype == 'float32' else f"{val:3d}"
                ax.text(boundary_size - 0.5, y, val_str,
                        color='blue', ha='right', va='center', fontproperties=mono_font)

        # 右边界
        for y in range(boundary_size, height - boundary_size, 2):
            if y < height and (width - boundary_size - 1) >= 0:
                val = img[y, width - boundary_size - 1]
                val_str = f"{val:.2f}" if dtype == 'float32' else f"{val:3d}"
                ax.text(width - boundary_size - 0.5, y, val_str,
                        color='blue', ha='left', va='center', fontproperties=mono_font)

    # 初始绘制边界值
    draw_boundary_values()

    # 中心区域简化显示（只显示统计信息）
    center_x, center_y = width // 2, height // 2
    center_val = img[center_y, center_x]
    center_str = f"Center: {center_val:.4f}" if dtype == 'float32' else f"Center: {center_val}"
    ax.text(center_x, center_y, "◉", color='yellow', ha='center', va='center', fontsize=12)
    ax.text(center_x, center_y + 1, center_str, color='white', ha='center', fontproperties=mono_font)

    # 添加网格线（仅边界区域）
    ax.axhline(y=boundary_size, color='cyan', linestyle='--', alpha=0.7)
    ax.axhline(y=height - boundary_size - 1, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=boundary_size, color='cyan', linestyle='--', alpha=0.7)
    ax.axvline(x=width - boundary_size - 1, color='cyan', linestyle='--', alpha=0.7)

    # 显示说明文本
    info_text = (f"Displaying boundary {boundary_size}px\n"
                 "Top/Bottom: Red text\n"
                 "Left/Right: Blue text")
    ax.text(0.98, 0.02, info_text,
            transform=ax.transAxes, color='white',
            ha='right', va='bottom', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.show()


# ===== 使用示例 =====
if __name__ == "__main__":
    # 示例：512x512 float32图像
    data_f32 = np.random.rand(512 * 512).astype(np.float32)
    visualize_boundary_pixels(data_f32, 512, 512, 'float32', boundary_size=5)