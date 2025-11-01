//
// Created by wangzhijiang on 25-5-10.
//

#ifndef GAUSSIAN_CL_COMMON_UTIL_H
#define GAUSSIAN_CL_COMMON_UTIL_H

#include <string>
#include <CL/cl_platform.h>

namespace gaussian
{
class CommonUtil
{
public:
    /**
     * The constructor of the class
     */
    CommonUtil();

    /**
     * @brief The destructor of the class
     */
    ~CommonUtil();

    /**
     * @brief 101 reflect
     * @param pos
     * @param size
     * @return
     */
    static int Reflect(int pos, int size);

    /**
     * @brief init gray data as image data
     * @param src gray data
     * @param width data width
     * @param height data height
     */
    static void InitGray(cl_uchar *src, size_t width, size_t height);

    /**
     * @brief compare cpu result and gpu result
     * @param cpu cpu result
     * @param gpu gpu result
     * @param width image with
     * @param height image height
     * @return success or not
     */
    static bool ImgDataCompare(const cl_uchar *cpu, const cl_uchar *gpu, size_t width, size_t height);

    /**
     * @brief Gaussian3x3Sigma0U8C1
     * @param src image data
     * @param width  image width
     * @param height image height
     * @param istride input stride
     * @param dst gaussian filter result
     * @param ostride output stride
     * @return success or not
     */
    static int Gaussian3x3Sigma0U8C1(uint8_t *src, size_t width, size_t height, size_t istride, uint8_t *dst,
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

    /**
     * @brief save content to file
     * @param text
     * @param text_length
     * @param filename
     * @return
     */
    static bool SaveToFile(const cl_uchar *text, size_t text_length, char *filename);

    /**
     * @brief generate local work size
     * @param local_work_size
     */
    static void GenLocalSize(size_t local_work_size[3]);

    /**
     * @brief read kernel source file
     * @param filename kernel source file name
     * @return kernel source file content
     */
    static std::string ClReadKernelSource(const std::string &filename);
};
}


#endif //GAUSSIAN_CL_COMMON_UTIL_H
