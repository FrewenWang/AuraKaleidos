//
// Created by wangzhijiang on 25-5-15.
//

#include "../ops/GaussianUtils.h"

#include <cstdio>

void GaussianUtils::Gaussian3x3SigmaU8C1(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst,
                                         size_t ostride)
{
    if ((NULL == src) || (NULL == dst))
    {
        printf("input param invalid!\n");
        return;
    }
    // 遍历第一行开始，逐行遍历
    for (size_t row = 0; row < height; row++)
    {
        // 基于101反射。计算上一行数据的偏移索引
        int last      = (row == 0) ? 1 : -1;
        int next      = (row == height - 1) ? -1 : 1;
        // 上一行的索引，如果是第一行则last等于1. 则就是使用第二行
        uint8_t *src0 = src + (row + last) * istride;
        // 设置当前行的起始索引值，src+ row*步长(也就是一行的数据的数量)
        uint8_t *src1 = src + row * istride;
        uint8_t *src2 = src + (row + next) * istride;
        // 计算输出结果的开始指针位置
        uint8_t *p_dst = dst + row * ostride;
        for (size_t col = 0; col < width; col++)
        {
            // 计算左右的索引
            int left     = (col == 0) ? 1 : ((col == width - 1) ? width - 2 : col - 1);
            int right    = (col == 0) ? 1 : ((col == width - 1) ? width - 2 : col + 1);
            uint16_t acc = 0;
            /// 然后进行加速
            acc += src0[left] + src0[right] + src0[col] * 2;
            acc += (src1[left] + src1[right]) * 2 + src1[col] * 4;
            acc += src2[left] + src2[right] + src2[col] * 2;
            /// 基于每个坐标点，进行周围8个像素进行高斯模糊的处理
            p_dst[col] = ((acc + (1 << 3)) >> 4) & 0xFF;
        }
    }
}

bool GaussianUtils::Gauss3x3Sigma0U8C1RemainData(uint8_t *src, int remain_col_index, int row, int col, int src_pitch,
                                                 int dst_pitch, uint8_t *dst)
{
    if ((NULL == src) || (NULL == dst))
    {
        printf("Gauss3x3Sigma0U8C1RemainData null ptr\n");
        return false;
    }
    // 上一行
    const uint8_t *p_prev_row = NULL;
    // 当前行
    const uint8_t *p_curr_row = NULL;
    // 下一行
    const uint8_t *p_next_row = NULL;
    uint8_t *p_out            = dst;

    int j;
    unsigned short acc = 0;
    // 处理第一列
    for (int i = 0; i < row; i++)
    {
        p_curr_row = src + i * src_pitch;
        p_prev_row = p_curr_row + ((i - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((i + 1) > (row - 1) ? -1 : 1) * src_pitch;
        // acc = ((p_prev_row[0] + p_prev_row[1]) << 1) + ((p_curr_row[0] + p_curr_row[1]) << 2) +
        //       ((p_next_row[0] + p_next_row[1]) << 1);
        // // 具体的计算结果，将卷积核的9个数据进行乘加
        // acc = p_prev_row[1] + p_prev_row[0] << 1 + p_prev_row[1]
        //       + p_curr_row[1] << 1 + p_curr_row[0] << 2 + p_curr_row[1] << 1
        //       + p_next_row[1] + p_next_row[0] << 1 + p_next_row[1];
        // 下面我们进行合并同类项
        acc = (p_prev_row[0] + p_prev_row[1]) << 1
              + (p_curr_row[0] + p_curr_row[1]) << 2
              + (p_next_row[0] + p_next_row[1]) << 1;

        p_out[i * dst_pitch] = ((acc + (1 << 3)) >> 4) & 0xFF;
    }

    // process last col
    for (int i = 0; i < row; i++)
    {
        p_curr_row = src + i * src_pitch + col - 1;
        p_prev_row = p_curr_row + ((i - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((i + 1) > (row - 1) ? -1 : 1) * src_pitch;

        acc = ((p_prev_row[0] + p_prev_row[-1]) << 1) + ((p_curr_row[0] + p_curr_row[-1]) << 2) +
              ((p_next_row[0] + p_next_row[-1]) << 1);
        p_out[i * dst_pitch + col - 1] = ((acc + (1 << 3)) >> 4) & 0xFFFF;
    }

    // 上面已经将所有的第一列和最后一列的进行处理完毕
    // 这个是进行遍历每一行
    for (int j = 0; j < row; j++) // height
    {
        p_curr_row = src + j * src_pitch;
        p_prev_row = p_curr_row + ((j - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((j + 1) > (row - 1) ? -1 : 1) * src_pitch;
        p_out      = dst + j * dst_pitch;
        // 每一行都进行处理，然后我们将每一行的剩余的索引（除去最后一列，所以我们只需要i<col-1）
        for (int i = remain_col_index; i < col - 1; i++) // width
        {
            acc = 0;
            acc += p_prev_row[i - 1] + p_prev_row[i + 1] + (p_prev_row[i + 0] << 1) +
                    ((p_curr_row[i - 1] + p_curr_row[i + 1] + (p_curr_row[i + 0] << 1)) << 1) + p_next_row[i - 1] +
                    p_next_row[i + 1] + (p_next_row[i + 0] << 1);
            p_out[i] = ((acc + (1 << 3)) >> 4) & 0xFF;
        }
    }
}
