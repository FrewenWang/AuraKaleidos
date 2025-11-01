//
// Created by wangzhijiang on 25-5-10.
//

#include "common_util.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace gaussian
{
CommonUtil::CommonUtil()
{
}

CommonUtil::~CommonUtil()
{
}

int CommonUtil::Reflect(int pos, int size)
{
    if (pos < 0)
    {
        return -pos - 1;
    }
    if (pos >= size)
    {
        return 2 * size - pos - 1;
    }
    return pos;
}

void CommonUtil::InitGray(cl_uchar *src, size_t width, size_t height)
{
    if (src == NULL)
    {
        printf("src is NULL\n");
    }
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            size_t idx = (y * width + x);
            src[idx] = idx % 256;
        }
    }
}

bool CommonUtil::ImgDataCompare(const cl_uchar *cpu, const cl_uchar *gpu, const size_t width, const size_t height)
{
    if (cpu == NULL || gpu == NULL)
    {
        printf("input param invalid! cpu or gpu is NULL \n");
        return false;
    }
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            size_t idx = i * width + j;
            if (cpu[idx] != gpu[idx])
            {
                printf("Mismatch at (%lu,%lu), cpu=%d,gpu=%d idx=[%lu] \n", i, j, cpu[idx], gpu[idx], idx);
                return false;
            } else
            {
                if (idx < 10)
                {
                    printf("match OK at (%lu,%lu), cpu=%d,gpu=%d idx=[%lu] \n", i, j, cpu[idx], gpu[idx], idx);
                }
            }
        }
    }
    return true;
}

int CommonUtil::Gaussian3x3Sigma0U8C1(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst,
                                      size_t ostride)
{
    if ((NULL == src) || (NULL == dst))
    {
        printf("input param invalid!\n");
        return -1;
    }

    for (size_t row = 0; row < height; row++)
    {
        int last = (row == 0) ? 1 : -1;
        int next = (row == height - 1) ? -1 : 1;
        uint8_t *src0 = src + (row + last) * istride;
        uint8_t *src1 = src + row * istride;
        uint8_t *src2 = src + (row + next) * istride;

        uint8_t *p_dst = dst + row * ostride;
        for (size_t col = 0; col < width; col++)
        {
            int left = (col == 0) ? 1 : ((col == width - 1) ? width - 2 : col - 1);
            int right = (col == 0) ? 1 : ((col == width - 1) ? width - 2 : col + 1);
            uint16_t acc = 0;
            acc += src0[left] + src0[right] + src0[col] * 2;
            acc += (src1[left] + src1[right]) * 2 + src1[col] * 4;
            acc += src2[left] + src2[right] + src2[col] * 2;

            p_dst[col] = ((acc + (1 << 3)) >> 4) & 0xFF;
        }
    }
    return 0;
}

bool CommonUtil::Gauss3x3Sigma0U8C1RemainData(uint8_t *src, int remain_col_index, int row, int col, int src_pitch,
                                              int dst_pitch, uint8_t *dst)
{
    if ((NULL == src) || (NULL == dst))
    {
        printf("Gauss3x3Sigma0U8C1RemainData null ptr\n");
        return -1;
    }

    const uint8_t *p_prev_row = NULL;
    const uint8_t *p_curr_row = NULL;
    const uint8_t *p_next_row = NULL;
    uint8_t *p_out = dst;

    int i, j;
    unsigned short acc = 0;
    // process first col
    for (i = 0; i < row; i++)
    {
        p_curr_row = src + i * src_pitch;
        p_prev_row = p_curr_row + ((i - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((i + 1) > (row - 1) ? -1 : 1) * src_pitch;

        acc = ((p_prev_row[0] + p_prev_row[1]) << 1) + ((p_curr_row[0] + p_curr_row[1]) << 2) +
              ((p_next_row[0] + p_next_row[1]) << 1);
        p_out[i * dst_pitch] = ((acc + (1 << 3)) >> 4) & 0xFF;
    }

    // process last col
    for (i = 0; i < row; i++)
    {
        p_curr_row = src + i * src_pitch + col - 1;
        p_prev_row = p_curr_row + ((i - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((i + 1) > (row - 1) ? -1 : 1) * src_pitch;

        acc = ((p_prev_row[0] + p_prev_row[-1]) << 1) + ((p_curr_row[0] + p_curr_row[-1]) << 2) +
              ((p_next_row[0] + p_next_row[-1]) << 1);
        p_out[i * dst_pitch + col - 1] = ((acc + (1 << 3)) >> 4) & 0xFFFF;
    }

    for (j = 0; j < row; j++) // height
    {
        p_curr_row = src + j * src_pitch;
        p_prev_row = p_curr_row + ((j - 1) < 0 ? 1 : -1) * src_pitch;
        p_next_row = p_curr_row + ((j + 1) > (row - 1) ? -1 : 1) * src_pitch;
        p_out = dst + j * dst_pitch;
        for (i = remain_col_index; i < col - 1; i++) // width
        {
            acc = 0;
            acc += p_prev_row[i - 1] + p_prev_row[i + 1] + (p_prev_row[i + 0] << 1) +
                    ((p_curr_row[i - 1] + p_curr_row[i + 1] + (p_curr_row[i + 0] << 1)) << 1) + p_next_row[i - 1] +
                    p_next_row[i + 1] + (p_next_row[i + 0] << 1);
            p_out[i] = ((acc + (1 << 3)) >> 4) & 0xFF;
        }
    }
    return 0;
}

bool CommonUtil::SaveToFile(const cl_uchar *text, size_t text_length, char *filename)
{
    FILE *fp = fopen(filename, "wt+");
    if (NULL == fp)
    {
        return false;
    }
    fwrite(text, 1, text_length, fp);
    fclose(fp);
    return true;
}

void CommonUtil::GenLocalSize(size_t local_work_size[3])
{
#if defined(QCOM)
    local_work_size[0] = 32;
    local_work_size[1] = 32;
#elif defined(MTK)
    local_work_size[0] = 16;
    local_work_size[1] = 16;
#else
    local_work_size[0] = 16;
    local_work_size[1] = 16;
#endif
    local_work_size[2] = 0;
}

std::string CommonUtil::ClReadKernelSource(const std::string &filename)
{
    std::ifstream fs(filename);
    if (!fs.is_open())
    {
        fprintf(stderr, "Failed to open file %s\n", filename.c_str());
    }
    return std::string((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
}

}
