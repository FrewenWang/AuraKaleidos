/*
 * Copyright (c) 2022 Cadence Design Systems Inc. ALL RIGHTS RESERVED.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAAURAS OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef XM_XRP_STRUCT_H_
#define XM_XRP_STRUCT_H_

// total number of function APIs
#define VDSP_REGIST_API_NUM 8

// max number of buffers pass to VDSP
#define XRP_DSP_CMD_INLINE_BUFFER_SIZE 8

// max size of VDSP input data (bytes)
#define XRP_DSP_CMD_INLINE_INDATA_SIZE 256

// max size of VDSP output data (bytes)
#define XRP_DSP_CMD_INLINE_OUTDATA_SIZE 16

// buffer struct define
typedef struct{
	uint32_t flag;
	uint32_t size;
	uint32_t addr;
}xrp_vdsp_buffer;

typedef struct{
    uint32_t in_data_size;
    uint32_t out_data_size;
    uint32_t buffer_num;
    uint8_t in_data[XRP_DSP_CMD_INLINE_INDATA_SIZE];
    uint8_t out_data[XRP_DSP_CMD_INLINE_OUTDATA_SIZE];
    xrp_vdsp_buffer buffer[XRP_DSP_CMD_INLINE_BUFFER_SIZE];
}xrp_vdsp_cmd;

typedef struct
{
    void (*function_handler)(xrp_vdsp_cmd *);
} custom_func_cmd;

typedef struct
{
    uint32_t func_num;
    custom_func_cmd func_cmd[VDSP_REGIST_API_NUM]; // modified by marco
} xm_vdsp_pic_funcs;

#endif //XM_XRP_STRUCT_H_