//
//  utils.cpp
//  2DPointTo3D
//
//  Created by jinguojing on 2019/8/13.
//  Copyright © 2019 Lei,Yu(IOV). All rights reserved.
//

#ifdef BUILD_EXPERIMENTAL

#include "pdm_util.h"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include "pdm.h"
#include <dirent.h>

namespace aura::vision {

    void PdmUtil::point_temp_106_to_68(float *dst, float *src) {

        for (int i = 0; i < 106; i++) {
            if (i <= 32 && i % 2 == 0)            // 0-16
            {
                dst[i] = src[i * 2];
                dst[i + 1] = src[i * 2 + 1];
            } else if (i >= 33 && i <= 37)                // 17-21
            {
                dst[i * 2 - 32] = src[i * 2];
                dst[i * 2 - 31] = src[i * 2 + 1];
            } else if (i >= 42 && i <= 46)                // 22-26
            {
                dst[i * 2 - 40] = src[i * 2];
                dst[i * 2 - 39] = src[i * 2 + 1];
            } else if (i >= 71 && i <= 74)                // 27-30
            {
                dst[i * 2 - 88] = src[i * 2];
                dst[i * 2 - 87] = src[i * 2 + 1];
            } else if (i >= 81 && i <= 85)                // 31-35
            {
                dst[i * 2 - 100] = src[i * 2];
                dst[i * 2 - 99] = src[i * 2 + 1];
            } else if (i >= 51 && i <= 56)                // 36-41
            {
                dst[i * 2 - 30] = src[i * 2];
                dst[i * 2 - 29] = src[i * 2 + 1];
            } else if (i >= 61 && i <= 66)                // 42-47
            {
                dst[i * 2 - 38] = src[i * 2];
                dst[i * 2 - 37] = src[i * 2 + 1];
            } else if (i == 86)                        // 48
            {
                dst[i * 2 - 76] = src[i * 2];
                dst[i * 2 - 75] = src[i * 2 + 1];
            } else if (i == 92)                        // 49
            {
                dst[i * 2 - 86] = src[i * 2];
                dst[i * 2 - 85] = src[i * 2 + 1];
            } else if (i >= 87 && i <= 89)                        // 50-52
            {
                dst[i * 2 - 74] = src[i * 2];
                dst[i * 2 - 73] = src[i * 2 + 1];
            } else if (i == 93)                        // 53
            {
                dst[i * 2 - 80] = src[i * 2];
                dst[i * 2 - 79] = src[i * 2 + 1];
            } else if (i == 90)                        // 54
            {
                dst[i * 2 - 72] = src[i * 2];
                dst[i * 2 - 71] = src[i * 2 + 1];
            } else if (i == 94)                        // 55
            {
                dst[i * 2 - 78] = src[i * 2];
                dst[i * 2 - 77] = src[i * 2 + 1];
            } else if (i == 105)                        // 56
            {
                dst[i * 2 - 98] = src[i * 2];
                dst[i * 2 - 97] = src[i * 2 + 1];
            } else if (i == 91)                        // 57
            {
                dst[i * 2 - 68] = src[i * 2];
                dst[i * 2 - 67] = src[i * 2 + 1];
            } else if (i == 104)                        // 58
            {
                dst[i * 2 - 92] = src[i * 2];
                dst[i * 2 - 91] = src[i * 2 + 1];
            } else if (i == 95 || i == 96)                        // 59-60
            {
                dst[i * 2 - 72] = src[i * 2];
                dst[i * 2 - 71] = src[i * 2 + 1];
            } else if (i >= 98 && i <= 100)                        // 61-63
            {
                dst[i * 2 - 74] = src[i * 2];
                dst[i * 2 - 73] = src[i * 2 + 1];
            } else if (i == 97)                        // 64
            {
                dst[i * 2 - 66] = src[i * 2];
                dst[i * 2 - 65] = src[i * 2 + 1];
            } else if (i == 103)                        // 65
            {
                dst[i * 2 - 76] = src[i * 2];
                dst[i * 2 - 75] = src[i * 2 + 1];
            } else if (i == 102)                        // 66
            {
                dst[i * 2 - 72] = src[i * 2];
                dst[i * 2 - 71] = src[i * 2 + 1];
            } else if (i == 101)                        // 57
            {
                dst[i * 2 - 68] = src[i * 2];
                dst[i * 2 - 67] = src[i * 2 + 1];
            }
        }
    }

/*
 extract 51 landmarks from 106 landmarks
 @dst: results of 51 points
 @src: input 106 points
 */
    void PdmUtil::point_temp_106_to_51(float *dst, float *src) {
        for (int i = 33; i < 106; i++) {
            if (i >= 33 && i <= 37)                // 17-21
            {
                dst[i * 2 - 32] = src[i * 2];
                dst[i * 2 - 31] = src[i * 2 + 1];
            } else if (i >= 42 && i <= 46)                // 22-26
            {
                dst[i * 2 - 40] = src[i * 2];
                dst[i * 2 - 39] = src[i * 2 + 1];
            } else if (i >= 71 && i <= 74)                // 27-30
            {
                dst[i * 2 - 88] = src[i * 2];
                dst[i * 2 - 87] = src[i * 2 + 1];
            } else if (i >= 81 && i <= 85)                // 31-35
            {
                dst[i * 2 - 100] = src[i * 2];
                dst[i * 2 - 99] = src[i * 2 + 1];
            } else if (i >= 51 && i <= 56)                // 36-41
            {
                dst[i * 2 - 30] = src[i * 2];
                dst[i * 2 - 29] = src[i * 2 + 1];
            } else if (i >= 61 && i <= 66)                // 42-47
            {
                dst[i * 2 - 38] = src[i * 2];
                dst[i * 2 - 37] = src[i * 2 + 1];
            } else if (i >= 86)                        // 48-67
            {
                dst[i * 2 - 76] = src[i * 2];
                dst[i * 2 - 75] = src[i * 2 + 1];
            }
        }
    }

/*
 extract 68 landmarks from 72 landmarks
 @dst: results of 68 points
 @src: input 72 points
 */
    void PdmUtil::point_temp_72_to_68(float *dst, float *src) {
        {//face shape
            dst[0 * 2] = src[0 * 2];
            dst[0 * 2 + 1] = src[0 * 2 + 1];

            dst[1 * 2] = src[1 * 2];
            dst[1 * 2 + 1] = src[1 * 2 + 1];

            dst[2 * 2] = src[2 * 2];
            dst[2 * 2 + 1] = src[2 * 2 + 1];

            dst[3 * 2] = src[3 * 2];
            dst[3 * 2 + 1] = src[3 * 2 + 1];

            dst[4 * 2] = src[4 * 2];
            dst[4 * 2 + 1] = src[4 * 2 + 1];

            dst[5 * 2] = src[4 * 2];
            dst[5 * 2 + 1] = src[4 * 2 + 1];

            dst[6 * 2] = src[5 * 2];
            dst[6 * 2 + 1] = src[5 * 2 + 1];

            dst[7 * 2] = src[5 * 2];
            dst[7 * 2 + 1] = src[5 * 2 + 1];

            dst[8 * 2] = src[6 * 2];
            dst[8 * 2 + 1] = src[6 * 2 + 1];

            dst[9 * 2] = src[7 * 2];
            dst[9 * 2 + 1] = src[7 * 2 + 1];

            dst[10 * 2] = src[7 * 2];
            dst[10 * 2 + 1] = src[7 * 2 + 1];

            dst[11 * 2] = src[8 * 2];
            dst[11 * 2 + 1] = src[8 * 2 + 1];

            dst[12 * 2] = src[8 * 2];
            dst[12 * 2 + 1] = src[8 * 2 + 1];

            dst[13 * 2] = src[9 * 2];
            dst[13 * 2 + 1] = src[9 * 2 + 1];

            dst[14 * 2] = src[10 * 2];
            dst[14 * 2 + 1] = src[10 * 2 + 1];

            dst[15 * 2] = src[11 * 2];
            dst[15 * 2 + 1] = src[11 * 2 + 1];

            dst[16 * 2] = src[12 * 2];
            dst[16 * 2 + 1] = src[12 * 2 + 1];

        }
        {//left eyebrow
            dst[17 * 2] = src[22 * 2];
            dst[17 * 2 + 1] = src[22 * 2 + 1];

            dst[18 * 2] = src[23 * 2];
            dst[18 * 2 + 1] = src[23 * 2 + 1];

            dst[19 * 2] = src[24 * 2];
            dst[19 * 2 + 1] = src[24 * 2 + 1];

            dst[20 * 2] = src[25 * 2];
            dst[20 * 2 + 1] = src[25 * 2 + 1];

            dst[21 * 2] = src[26 * 2];
            dst[21 * 2 + 1] = src[26 * 2 + 1];
        }
        {//right eyebrow
            dst[22 * 2] = src[39 * 2];
            dst[22 * 2 + 1] = src[39 * 2 + 1];

            dst[23 * 2] = src[40 * 2];
            dst[23 * 2 + 1] = src[40 * 2 + 1];

            dst[24 * 2] = src[41 * 2];
            dst[24 * 2 + 1] = src[41 * 2 + 1];

            dst[25 * 2] = src[42 * 2];
            dst[25 * 2 + 1] = src[42 * 2 + 1];

            dst[26 * 2] = src[43 * 2];
            dst[26 * 2 + 1] = src[43 * 2 + 1];
        }
        {//nose
            dst[27 * 2] = (src[47 * 2] + src[56 * 2]) / 2;
            dst[27 * 2 + 1] = (src[47 * 2 + 1] + src[56 * 2 + 1]) / 2;

            dst[28 * 2] = (src[48 * 2] + src[55 * 2]) / 2;
            dst[28 * 2 + 1] = (src[48 * 2 + 1] + src[55 * 2 + 1]) / 2;

            dst[29 * 2] = (src[49 * 2] + src[54 * 2]) / 2;
            dst[29 * 2 + 1] = (src[49 * 2 + 1] + src[54 * 2 + 1]) / 2;

            dst[30 * 2] = src[57 * 2];
            dst[30 * 2 + 1] = src[57 * 2 + 1];

            dst[31 * 2] = src[50 * 2];
            dst[31 * 2 + 1] = src[50 * 2 + 1];

            dst[32 * 2] = src[51 * 2];
            dst[32 * 2 + 1] = src[51 * 2 + 1];

            dst[33 * 2] = (src[51 * 2] + src[52 * 2]) / 2;
            dst[33 * 2 + 1] = (src[51 * 2 + 1] + src[52 * 2 + 1]) / 2;

            dst[34 * 2] = src[52 * 2];
            dst[34 * 2 + 1] = src[52 * 2 + 1];

            dst[35 * 2] = src[53 * 2];
            dst[35 * 2 + 1] = src[53 * 2 + 1];
        }
        {//left eye
            dst[36 * 2] = src[13 * 2];
            dst[36 * 2 + 1] = src[13 * 2 + 1];

            dst[37 * 2] = src[14 * 2];
            dst[37 * 2 + 1] = src[14 * 2 + 1];

            dst[38 * 2] = src[16 * 2];
            dst[38 * 2 + 1] = src[16 * 2 + 1];

            dst[39 * 2] = src[17 * 2];
            dst[39 * 2 + 1] = src[17 * 2 + 1];

            dst[40 * 2] = src[18 * 2];
            dst[40 * 2 + 1] = src[18 * 2 + 1];

            dst[41 * 2] = src[20 * 2];
            dst[41 * 2 + 1] = src[20 * 2 + 1];
        }
        {//right eye
            dst[42 * 2] = src[30 * 2];
            dst[42 * 2 + 1] = src[30 * 2 + 1];

            dst[43 * 2] = src[31 * 2];
            dst[43 * 2 + 1] = src[31 * 2 + 1];

            dst[44 * 2] = src[33 * 2];
            dst[44 * 2 + 1] = src[33 * 2 + 1];

            dst[45 * 2] = src[34 * 2];
            dst[45 * 2 + 1] = src[34 * 2 + 1];

            dst[46 * 2] = src[35 * 2];
            dst[46 * 2 + 1] = src[35 * 2 + 1];

            dst[47 * 2] = src[37 * 2];
            dst[47 * 2 + 1] = src[37 * 2 + 1];
        }
        {//lips

            dst[48 * 2] = src[58 * 2];
            dst[48 * 2 + 1] = src[58 * 2 + 1];

            dst[49 * 2] = (src[58 * 2] + src[59 * 2]) / 2;
            dst[49 * 2 + 1] = (src[58 * 2 + 1] + src[59 * 2 + 1]) / 2;

            dst[50 * 2] = src[59 * 2];
            dst[50 * 2 + 1] = src[59 * 2 + 1];

            dst[51 * 2] = src[60 * 2];
            dst[51 * 2 + 1] = src[60 * 2 + 1];

            dst[52 * 2] = src[61 * 2];
            dst[52 * 2 + 1] = src[61 * 2 + 1];

            dst[53 * 2] = (src[61 * 2] + src[62 * 2]) / 2;
            dst[53 * 2 + 1] = (src[61 * 2 + 1] + src[62 * 2 + 1]) / 2;

            dst[54 * 2] = src[62 * 2];
            dst[54 * 2 + 1] = src[62 * 2 + 1];

            dst[55 * 2] = src[63 * 2];
            dst[55 * 2 + 1] = src[63 * 2 + 1];

            dst[56 * 2] = (src[63 * 2] + src[64 * 2]) / 2;
            dst[56 * 2 + 1] = (src[63 * 2 + 1] + src[64 * 2 + 1]) / 2;

            dst[57 * 2] = src[64 * 2];
            dst[57 * 2 + 1] = src[64 * 2 + 1];

            dst[58 * 2] = (src[64 * 2] + src[65 * 2]) / 2;
            dst[58 * 2 + 1] = (src[64 * 2 + 1] + src[65 * 2 + 1]) / 2;

            dst[59 * 2] = src[65 * 2];
            dst[59 * 2 + 1] = src[65 * 2 + 1];

            dst[60 * 2] = src[58 * 2];
            dst[60 * 2 + 1] = src[58 * 2 + 1];

            dst[61 * 2] = src[66 * 2];
            dst[61 * 2 + 1] = src[66 * 2 + 1];

            dst[62 * 2] = src[67 * 2];
            dst[62 * 2 + 1] = src[67 * 2 + 1];

            dst[63 * 2] = src[68 * 2];
            dst[63 * 2 + 1] = src[68 * 2 + 1];

            dst[64 * 2] = src[62 * 2];
            dst[64 * 2 + 1] = src[62 * 2 + 1];

            dst[65 * 2] = src[69 * 2];
            dst[65 * 2 + 1] = src[69 * 2 + 1];

            dst[66 * 2] = src[70 * 2];
            dst[66 * 2 + 1] = src[70 * 2 + 1];

            dst[67 * 2] = src[71 * 2];
            dst[67 * 2 + 1] = src[71 * 2 + 1];

        }

    }

/*
 extract 51 landmarks from 72 landmarks
 @dst: results of 51 points
 @src: input 106 points
 */
    void PdmUtil::point_temp_72_to_51(float *dst, float *src) {
        {//left eyebrow
            dst[0 * 2] = src[22 * 2];
            dst[0 * 2 + 1] = src[22 * 2 + 1];

            dst[1 * 2] = src[23 * 2];
            dst[1 * 2 + 1] = src[23 * 2 + 1];

            dst[2 * 2] = src[24 * 2];
            dst[2 * 2 + 1] = src[24 * 2 + 1];

            dst[3 * 2] = src[25 * 2];
            dst[3 * 2 + 1] = src[25 * 2 + 1];

            dst[4 * 2] = src[26 * 2];
            dst[4 * 2 + 1] = src[26 * 2 + 1];
        }
        {//right eyebrow
            dst[5 * 2] = src[39 * 2];
            dst[5 * 2 + 1] = src[39 * 2 + 1];

            dst[6 * 2] = src[40 * 2];
            dst[6 * 2 + 1] = src[40 * 2 + 1];

            dst[7 * 2] = src[41 * 2];
            dst[7 * 2 + 1] = src[41 * 2 + 1];

            dst[8 * 2] = src[42 * 2];
            dst[8 * 2 + 1] = src[42 * 2 + 1];

            dst[9 * 2] = src[43 * 2];
            dst[9 * 2 + 1] = src[43 * 2 + 1];
        }
        {//nose
            dst[10 * 2] = (src[47 * 2] + src[56 * 2]) / 2;
            dst[10 * 2 + 1] = (src[47 * 2 + 1] + src[56 * 2 + 1]) / 2;

            dst[11 * 2] = (src[48 * 2] + src[55 * 2]) / 2;
            dst[11 * 2 + 1] = (src[48 * 2 + 1] + src[55 * 2 + 1]) / 2;

            dst[12 * 2] = (src[49 * 2] + src[54 * 2]) / 2;
            dst[12 * 2 + 1] = (src[49 * 2 + 1] + src[54 * 2 + 1]) / 2;

            dst[13 * 2] = src[57 * 2];
            dst[13 * 2 + 1] = src[57 * 2 + 1];

            dst[14 * 2] = src[50 * 2];
            dst[14 * 2 + 1] = src[50 * 2 + 1];

            dst[15 * 2] = src[51 * 2];
            dst[15 * 2 + 1] = src[51 * 2 + 1];

            dst[16 * 2] = (src[51 * 2] + src[52 * 2]) / 2;
            dst[16 * 2 + 1] = (src[51 * 2 + 1] + src[52 * 2 + 1]) / 2;

            dst[17 * 2] = src[52 * 2];
            dst[17 * 2 + 1] = src[52 * 2 + 1];

            dst[18 * 2] = src[53 * 2];
            dst[18 * 2 + 1] = src[53 * 2 + 1];
        }
        {//left eye
            dst[19 * 2] = src[13 * 2];
            dst[19 * 2 + 1] = src[13 * 2 + 1];

            dst[20 * 2] = src[14 * 2];
            dst[20 * 2 + 1] = src[14 * 2 + 1];

            dst[21 * 2] = src[16 * 2];
            dst[21 * 2 + 1] = src[16 * 2 + 1];

            dst[22 * 2] = src[17 * 2];
            dst[22 * 2 + 1] = src[17 * 2 + 1];

            dst[23 * 2] = src[18 * 2];
            dst[23 * 2 + 1] = src[18 * 2 + 1];

            dst[24 * 2] = src[20 * 2];
            dst[24 * 2 + 1] = src[20 * 2 + 1];
        }
        {//right eye
            dst[25 * 2] = src[30 * 2];
            dst[25 * 2 + 1] = src[30 * 2 + 1];

            dst[26 * 2] = src[31 * 2];
            dst[26 * 2 + 1] = src[31 * 2 + 1];

            dst[27 * 2] = src[33 * 2];
            dst[27 * 2 + 1] = src[33 * 2 + 1];

            dst[28 * 2] = src[34 * 2];
            dst[28 * 2 + 1] = src[34 * 2 + 1];

            dst[29 * 2] = src[35 * 2];
            dst[29 * 2 + 1] = src[35 * 2 + 1];

            dst[30 * 2] = src[37 * 2];
            dst[30 * 2 + 1] = src[37 * 2 + 1];
        }
        {//lips

            dst[31 * 2] = src[58 * 2];
            dst[31 * 2 + 1] = src[58 * 2 + 1];

            dst[32 * 2] = (src[58 * 2] + src[59 * 2]) / 2;
            dst[32 * 2 + 1] = (src[58 * 2 + 1] + src[59 * 2 + 1]) / 2;

            dst[33 * 2] = src[59 * 2];
            dst[33 * 2 + 1] = src[59 * 2 + 1];

            dst[34 * 2] = src[60 * 2];
            dst[34 * 2 + 1] = src[60 * 2 + 1];

            dst[35 * 2] = src[61 * 2];
            dst[35 * 2 + 1] = src[61 * 2 + 1];

            dst[36 * 2] = (src[61 * 2] + src[62 * 2]) / 2;
            dst[36 * 2 + 1] = (src[61 * 2 + 1] + src[62 * 2 + 1]) / 2;

            dst[37 * 2] = src[62 * 2];
            dst[37 * 2 + 1] = src[62 * 2 + 1];

            dst[38 * 2] = src[63 * 2];
            dst[38 * 2 + 1] = src[63 * 2 + 1];

            dst[39 * 2] = (src[63 * 2] + src[64 * 2]) / 2;
            dst[39 * 2 + 1] = (src[63 * 2 + 1] + src[64 * 2 + 1]) / 2;

            dst[40 * 2] = src[64 * 2];
            dst[40 * 2 + 1] = src[64 * 2 + 1];

            dst[41 * 2] = (src[64 * 2] + src[65 * 2]) / 2;
            dst[41 * 2 + 1] = (src[64 * 2 + 1] + src[65 * 2 + 1]) / 2;

            dst[42 * 2] = src[65 * 2];
            dst[42 * 2 + 1] = src[65 * 2 + 1];

            dst[43 * 2] = src[58 * 2];
            dst[43 * 2 + 1] = src[58 * 2 + 1];

            dst[44 * 2] = src[66 * 2];
            dst[44 * 2 + 1] = src[66 * 2 + 1];

            dst[45 * 2] = src[67 * 2];
            dst[45 * 2 + 1] = src[67 * 2 + 1];

            dst[46 * 2] = src[68 * 2];
            dst[46 * 2 + 1] = src[68 * 2 + 1];

            dst[47 * 2] = src[62 * 2];
            dst[47 * 2 + 1] = src[62 * 2 + 1];

            dst[48 * 2] = src[69 * 2];
            dst[48 * 2 + 1] = src[69 * 2 + 1];

            dst[49 * 2] = src[70 * 2];
            dst[49 * 2 + 1] = src[70 * 2 + 1];

            dst[50 * 2] = src[71 * 2];
            dst[50 * 2 + 1] = src[71 * 2 + 1];

        }

    }

/*
 generate 106 landmarks by combing 68 and 72 landmakrs in a rough way
 @dst: results of 106 points
 @src: input 68 points and 72 points
 */
    void PdmUtil::point_temp_68and72_to_106(float *dst, float *src68, float *src72) {
        {//face shape
            for (int i = 0; i < 16; i++) {

                dst[(i * 2) * 2] = src68[i * 2];
                dst[(i * 2) * 2 + 1] = src68[i * 2 + 1];

                dst[(i * 2 + 1) * 2] = (src68[i * 2] + src68[(i + 1) * 2]) / 2.0;
                dst[(i * 2 + 1) * 2 + 1] = (src68[i * 2 + 1] + src68[(i + 1) * 2 + 1]) / 2.0;
            }

            dst[32 * 2] = src68[16 * 2];
            dst[32 * 2 + 1] = src68[16 * 2 + 1];
        }
        {//left eyebrow
            for (int i = 33, j = 17; i < 38; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }
            for (int i = 38, j = 29; i < 41; i++, j--) {
                dst[i * 2] = src72[j * 2];
                dst[i * 2 + 1] = src72[j * 2 + 1];
            }
            dst[41 * 2] = src68[21 * 2];
            dst[41 * 2 + 1] = src72[27 * 2 + 1];
        }
        {//right eyebrow
            for (int i = 42, j = 22; i < 47; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }
            for (int i = 48, j = 46; i < 51; i++, j--) {
                dst[i * 2] = src72[j * 2];
                dst[i * 2 + 1] = src72[j * 2 + 1];
            }
            dst[47 * 2] = src68[22 * 2];
            dst[47 * 2 + 1] = src72[46 * 2 + 1];
        }
        {//left eye
            for (int i = 51, j = 36; i < 57; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }
            dst[57 * 2] = src72[15 * 2];
            dst[57 * 2 + 1] = src72[15 * 2 + 1];

            dst[58 * 2] = src72[19 * 2];
            dst[58 * 2 + 1] = src72[19 * 2 + 1];

            dst[59 * 2] = src72[21 * 2];
            dst[59 * 2 + 1] = src72[21 * 2 + 1];

            dst[60 * 2] = (src68[36 * 2] + src68[39 * 2]) / 2.0;
            dst[60 * 2 + 1] = (src68[36 * 2 + 1] + src68[39 * 2 + 1]) / 2.0;
        }
        {//right eye
            for (int i = 61, j = 42; i < 67; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }
            dst[67 * 2] = src72[32 * 2];
            dst[67 * 2 + 1] = src72[32 * 2 + 1];

            dst[68 * 2] = src72[36 * 2];
            dst[68 * 2 + 1] = src72[36 * 2 + 1];

            dst[69 * 2] = src72[38 * 2];
            dst[69 * 2 + 1] = src72[38 * 2 + 1];

            dst[70 * 2] = (src68[42 * 2] + src68[45 * 2]) / 2.0;
            dst[70 * 2 + 1] = (src68[42 * 2 + 1] + src68[45 * 2 + 1]) / 2.0;
        }
        {//nose
            for (int i = 71, j = 27; i < 75; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }
            for (int i = 81, j = 31; i < 86; i++, j++) {
                dst[i * 2] = src68[j * 2];
                dst[i * 2 + 1] = src68[j * 2 + 1];
            }

            dst[75 * 2] = src72[47 * 2];
            dst[75 * 2 + 1] = src72[47 * 2 + 1];

            dst[76 * 2] = src72[56 * 2];
            dst[76 * 2 + 1] = src72[56 * 2 + 1];

            dst[77 * 2] = src72[49 * 2];
            dst[77 * 2 + 1] = src72[49 * 2 + 1];

            dst[78 * 2] = src72[54 * 2];
            dst[78 * 2 + 1] = src72[54 * 2 + 1];

            dst[79 * 2] = dst[81 * 2] - (dst[82 * 2] - dst[81 * 2]);
            dst[79 * 2 + 1] = (dst[77 * 2 + 1] + dst[81 * 2 + 1]) / 2.0;

            dst[80 * 2] = dst[85 * 2] + (dst[85 * 2] - dst[84 * 2]);
            dst[80 * 2 + 1] = (dst[78 * 2 + 1] + dst[85 * 2 + 1]) / 2.0;
        }
        {//mouth

            dst[86 * 2] = src68[48 * 2];
            dst[86 * 2 + 1] = src68[48 * 2 + 1];

            dst[87 * 2] = src68[50 * 2];
            dst[87 * 2 + 1] = src68[50 * 2 + 1];

            dst[88 * 2] = src68[51 * 2];
            dst[88 * 2 + 1] = src68[51 * 2 + 1];

            dst[89 * 2] = src68[52 * 2];
            dst[89 * 2 + 1] = src68[52 * 2 + 1];

            dst[90 * 2] = src68[54 * 2];
            dst[90 * 2 + 1] = src68[54 * 2 + 1];

            dst[91 * 2] = src68[57 * 2];
            dst[91 * 2 + 1] = src68[57 * 2 + 1];

            dst[92 * 2] = src68[49 * 2];
            dst[92 * 2 + 1] = src68[49 * 2 + 1];

            dst[93 * 2] = src68[53 * 2];
            dst[93 * 2 + 1] = src68[53 * 2 + 1];

            dst[94 * 2] = src68[55 * 2];
            dst[94 * 2 + 1] = src68[55 * 2 + 1];

            dst[95 * 2] = src68[59 * 2];
            dst[95 * 2 + 1] = src68[59 * 2 + 1];

            dst[96 * 2] = src68[60 * 2];
            dst[96 * 2 + 1] = src68[60 * 2 + 1];

            dst[97 * 2] = src68[64 * 2];
            dst[97 * 2 + 1] = src68[64 * 2 + 1];

            dst[98 * 2] = src68[61 * 2];
            dst[98 * 2 + 1] = src68[61 * 2 + 1];

            dst[99 * 2] = src68[62 * 2];
            dst[99 * 2 + 1] = src68[62 * 2 + 1];

            dst[100 * 2] = src68[63 * 2];
            dst[100 * 2 + 1] = src68[63 * 2 + 1];

            dst[101 * 2] = src68[67 * 2];
            dst[101 * 2 + 1] = src68[67 * 2 + 1];

            dst[102 * 2] = src68[66 * 2];
            dst[102 * 2 + 1] = src68[66 * 2 + 1];

            dst[103 * 2] = src68[65 * 2];
            dst[103 * 2 + 1] = src68[65 * 2 + 1];

            dst[104 * 2] = src68[58 * 2];
            dst[104 * 2 + 1] = src68[58 * 2 + 1];

            dst[105 * 2] = src68[56 * 2];
            dst[105 * 2 + 1] = src68[56 * 2 + 1];
        }
    }

/*
 draw coordinate according to face rotation
 @img: input image
 @rotation: rotation matrix of face
 @camera_matrix: camera intrincs
 */
    void PdmUtil::draw_face_coor(cv::Mat &img, cv::Matx33f &rotation) {
        int width = img.cols;
        int height = img.rows;
        std::vector<cv::Point> vertex;
        cv::Matx43f m_point3 = {0, 0, 0,
                                100, 0, 0,
                                0, 100, 0,
                                0, 0, 100,};
        float fx = 1500 * img.rows / 640;
        float fy = 1500 * img.cols / 480;
        fx = (fx + fy) / 2.0;
        fy = fx;
        cv::Matx33f camera_matrix = {fx, 0, (float) img.cols / 2, 0, fy, (float) img.rows / 2, 0, 0, 1};
        cv::Matx43f point_rotation = (rotation * m_point3.t()).t();
        cv::Matx43f point_projection = (camera_matrix * point_rotation.t()).t();
        for (int i = 0; i < 4; i++) {
            if (i == 0)
                vertex.push_back(
                        cv::Point(point_projection(i, 0) / fx + width / 2, point_projection(i, 1) / fx + height / 2));
            else
                vertex.push_back(cv::Point(point_projection(i, 0) / (fx + point_rotation(i, 2)) + width / 2,
                                           point_projection(i, 1) / (fx + point_rotation(i, 2)) + height / 2));
        }
#ifdef OPENCV2
        cv::line(img, vertex[0], vertex[1], CV_RGB(255, 0, 0), 3);
        cv::line(img, vertex[0], vertex[2], CV_RGB(0, 255, 0), 3);
        cv::line(img, vertex[0], vertex[3], CV_RGB(0, 0, 255), 3);
#endif
#ifdef OPENCV4

#endif
    }

// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
    cv::Matx33f PdmUtil::euler2rotation_matrix(const cv::Vec3f &eulerAngles) {
        cv::Matx33f rotation_matrix;

        float s1 = sin(eulerAngles[0]);
        float s2 = sin(eulerAngles[1]);
        float s3 = sin(eulerAngles[2]);

        float c1 = cos(eulerAngles[0]);
        float c2 = cos(eulerAngles[1]);
        float c3 = cos(eulerAngles[2]);

        rotation_matrix(0, 0) = c2 * c3;
        rotation_matrix(0, 1) = -c2 * s3;
        rotation_matrix(0, 2) = s2;
        rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
        rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
        rotation_matrix(1, 2) = -c2 * s1;
        rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
        rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
        rotation_matrix(2, 2) = c1 * c2;

        return rotation_matrix;
    }

// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
    cv::Vec3f PdmUtil::rotation_matrix2euler(const cv::Matx33f &rotation_matrix) {
        float q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0f;
        float q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0f * q0);
        float q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0f * q0);
        float q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0f * q0);

        // Slower, but dealing with degenerate cases due to precision
        float t1 = 2.0f * (q0 * q2 + q1 * q3);
        if (t1 > 1) t1 = 1.0f;
        if (t1 < -1) t1 = -1.0f;

        float yaw = asin(t1);
        float pitch = atan2(2.0f * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
        float roll = atan2(2.0f * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3);

        return cv::Vec3f(pitch, yaw, roll);
    }

    cv::Vec3f PdmUtil::euler2axis_angle(const cv::Vec3f &euler) {
        cv::Matx33f rotMatrix = PdmUtil::euler2rotation_matrix(euler);
        cv::Vec3f axis_angle;
#ifdef BUILD_EXPERIMENTAL
        cv::Rodrigues(rotMatrix, axis_angle);
#endif
        return axis_angle;
    }

    cv::Vec3f PdmUtil::axis_angle2euler(const cv::Vec3f &axis_angle) {
        cv::Matx33f rotation_matrix;
#ifdef BUILD_EXPERIMENTAL
        cv::Rodrigues(axis_angle, rotation_matrix);
#endif
        return PdmUtil::rotation_matrix2euler(rotation_matrix);
    }

    cv::Matx33f PdmUtil::axis_angle2rotation_matrix(const cv::Vec3f &axis_angle) {
        cv::Matx33f rotation_matrix;
#ifdef BUILD_EXPERIMENTAL
        cv::Rodrigues(axis_angle, rotation_matrix);
#endif
        return rotation_matrix;
    }

    cv::Vec3f PdmUtil::rotation_matrix2axis_angle(const cv::Matx33f &rotation_matrix) {
        cv::Vec3f axis_angle;
#ifdef BUILD_EXPERIMENTAL
        cv::Rodrigues(rotation_matrix, axis_angle);
#endif
        return axis_angle;
    }

// Generally useful 3D functions
    void PdmUtil::project(cv::Mat_<float> &dest, const cv::Mat_<float> &mesh, float fx, float fy, float cx, float cy) {
        dest = cv::Mat_<float>(mesh.rows, 2, 0.0);

        int num_points = mesh.rows;

        float X, Y, Z;

        cv::Mat_<float>::const_iterator mData = mesh.begin();
        cv::Mat_<float>::iterator projected = dest.begin();

        for (int i = 0; i < num_points; i++) {
            // Get the points
            X = *(mData++);
            Y = *(mData++);
            Z = *(mData++);

            float x;
            float y;

            // if depth is 0 the projection is different
            if (Z != 0) {
                x = ((X * fx / Z) + cx);
                y = ((Y * fy / Z) + cy);
            } else {
                x = X;
                y = Y;
            }

            // Project and store in dest matrix
            (*projected++) = x;
            (*projected++) = y;
        }

    }

//===========================================================================
// Point set and landmark manipulation functions
//===========================================================================
// Using Kabsch's algorithm for aligning shapes
//This assumes that align_from and align_to are already mean normalised
    cv::Matx22f PdmUtil::align_shapes_kabsch2d(const cv::Mat_<float> &align_from, const cv::Mat_<float> &align_to) {

        cv::SVD svd(align_from.t() * align_to);

        // make sure no reflection is there
        // corr ensures that we do only rotaitons and not reflections
        double d = cv::determinant(svd.vt.t() * svd.u.t());

        cv::Matx22f corr = cv::Matx22f::eye();
        if (d > 0) {
            corr(1, 1) = 1;
        } else {
            corr(1, 1) = -1;
        }

        cv::Matx22f R;
        cv::Mat(svd.vt.t() * cv::Mat(corr) * svd.u.t()).copyTo(R);

        return R;
    }

//=============================================================================
// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
    cv::Matx22f PdmUtil::align_shapes_with_scale(cv::Mat_<float> &src, cv::Mat_<float> dst) {
        int n = src.rows;

        // First we mean normalise both src and dst
        float mean_src_x = (float) cv::mean(src.col(0))[0];
        float mean_src_y = (float) cv::mean(src.col(1))[0];

        float mean_dst_x = (float) cv::mean(dst.col(0))[0];
        float mean_dst_y = (float) cv::mean(dst.col(1))[0];

        cv::Mat_<float> src_mean_normed = src.clone();
        src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
        src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

        cv::Mat_<float> dst_mean_normed = dst.clone();
        dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
        dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

        // Find the scaling factor of each
        cv::Mat src_sq;
        cv::pow(src_mean_normed, 2, src_sq);

        cv::Mat dst_sq;
        cv::pow(dst_mean_normed, 2, dst_sq);

        float s_src = (float) sqrt(cv::sum(src_sq)[0] / n);
        float s_dst = (float) sqrt(cv::sum(dst_sq)[0] / n);

        src_mean_normed = src_mean_normed / s_src;
        dst_mean_normed = dst_mean_normed / s_dst;

        float s = s_dst / s_src;

        // Get the rotation
        cv::Matx22f R = align_shapes_kabsch2d(src_mean_normed, dst_mean_normed);

        cv::Matx22f A;
        cv::Mat(s * R).copyTo(A);

        //cv::Mat_<float> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
        //cv::Mat_<float> offset = dst - aligned;

        //float t_x = cv::mean(offset.col(0))[0];
        //float t_y = cv::mean(offset.col(1))[0];

        return A;

    }
}

#endif
