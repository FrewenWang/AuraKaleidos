//
// Created by wangzhijiang on 25-5-15.
//
#pragma once
#include <cstdint>

class GaussianUtils
{
public:
    GaussianUtils();

    static void Gaussian3x3SigmaU8C1(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst,
                                     size_t ostride);

    /**
     * @brief Gauss3x3Sigma0U8C1RemainData
     * @param src image data
     * @param remain_col_index  image width
     * @param row image height
     * @param col input stride
     * @param src_pitch  src pitch
     * @param dst_pitch  dst pitch
     * @param dst  gaussian blur result
     * @return
     */
    static bool Gauss3x3Sigma0U8C1RemainData(uint8_t *src, int remain_col_index, int row, int col,
                                             int src_pitch, int dst_pitch, uint8_t *dst);
};
