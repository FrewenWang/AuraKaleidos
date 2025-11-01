#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(40, 24))

def call_back(event):
    axtemp=event.inaxes

    if axtemp is None:
        return

    x_min, x_max = axtemp.get_xlim()
    range_val = (x_max - x_min) / 10
    if event.button == 'up':
        axtemp.set(xlim=(x_min + range_val, x_max - range_val))
    elif event.button == 'down':
        axtemp.set(xlim=(x_min - range_val, x_max + range_val))
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', call_back)
fig.canvas.mpl_connect('button_press_event', call_back)

def parse_log(filename):
    data = []
    with open(filename) as file:
        lines = file.readlines()
        cur_data = [0, 0, 0, 0, 0, 0, 0]
        for idx in range(len(lines)):
            line = lines[idx].strip()
            if line.startswith("----------------"):
                cur_data = [0, 0, 0, 0, 0, 0, 0]
            elif line.startswith("Time Stamp"):
                cur_data[0] = float(line.split()[2].strip('s'))
            elif line.startswith("Native Heap:"):
                cur_data[1] = float(line.split()[2])
            elif line.startswith("Graphics:"):
                cur_data[2] = float(line.split()[1])
            elif line.startswith("TOTAL PSS:"):
                cur_data[3] = float(line.split()[2])
            elif line.startswith("Gfx dev"):
                cur_data[4] = float(line.split()[2])
            elif line.startswith("EGL mtrack"):
                cur_data[5] = float(line.split()[2])
            elif line.startswith("GL mtrack"):
                cur_data[6] = float(line.split()[2])
            elif line.startswith("=============="):
                data.append(cur_data)
            else:
                continue
    return data

def plot_mem(dir_name, save_file = ""):
    dma_buf_gpu_file = os.path.join(dir_name + "/dma_buf_gpu.log")
    dumpsys_mem_file = os.path.join(dir_name + "/dumpsys_meminfo.log")

    dma_buf_gpu = [] # timestamp dma_buf sys_gpu  proc_gpu
    dumpsys_mem = [] # timestamp gfx egl gl native graphic total_pss

    with open(dma_buf_gpu_file) as f:
        for line in f:
            info_str = line.split()
            time_str, dma_buf_str, sys_gpu_str, proc_gpu_str = info_str[2].strip('s'), info_str[4], info_str[7], info_str[10]

            time = float(time_str)
            dma_buf = float(dma_buf_str)
            sys_gpu = float(sys_gpu_str)
            proc_gpu = float(proc_gpu_str)
            dma_buf_gpu.append([time, dma_buf, sys_gpu, proc_gpu])

        dumpsys_mem = parse_log(dumpsys_mem_file)

        dma_buf_gpu_data = np.array(dma_buf_gpu)
        dumpsys_mem_data = np.array(dumpsys_mem)

        labels=[["Dma-buf Size", "System-gpu Size", "Process-gpu Size"], ["Native-heap Size", "Graphics Size", "Total-pss Size"], ["Gfx Dev Size", "EGL mtrack Size", "GL mtrack Size"]]
        colors=[["darkorange", "forestgreen", "royalblue"], ["fuchsia", "darkcyan", "darkviolet"], [ "limegreen", "tomato", "deeppink"]]

        for row in range(3):
            axs[row, 0].plot(dma_buf_gpu_data[:, 0], dma_buf_gpu_data[:, row + 1], marker='o', ms=1, c=colors[0][row])
            axs[row, 0].set_title(labels[0][row], color=colors[0][row], fontsize = 8)
            axs[row, 0].set_ylabel("Memory(KB)", fontsize = 8)
            axs[row, 0].set_xlabel("Time (s)", fontsize = 8)

            axs[row, 1].plot(dumpsys_mem_data[:, 0], dumpsys_mem_data[:, row + 1], marker='o', ms=1, c=colors[1][row])
            axs[row, 1].set_title(labels[1][row], color=colors[1][row], fontsize = 8)
            axs[row, 1].set_ylabel("Memory(KB)", fontsize = 8)
            axs[row, 1].set_xlabel("Time (s)", fontsize = 8)

            axs[row, 2].plot(dumpsys_mem_data[:, 0], dumpsys_mem_data[:, row + 4], marker='o', ms=1, c=colors[2][row])
            axs[row, 2].set_title(labels[2][row], color=colors[2][row], fontsize = 8)
            axs[row, 2].set_ylabel("Memory(KB)", fontsize = 8)
            axs[row, 2].set_xlabel("Time (s)", fontsize = 8)

            axs[row, 0].grid(True)
            axs[row, 1].grid(True)
            axs[row, 2].grid(True)
        if (len(save_file)):
            plt.savefig(save_file + ".jpg")
        else:
            plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        plot_mem(sys.argv[1])
    elif len(sys.argv) == 3:
        plot_mem(sys.argv[1], sys.argv[2])
    else:
        print("Usage: plot_mem.py <file_name>")