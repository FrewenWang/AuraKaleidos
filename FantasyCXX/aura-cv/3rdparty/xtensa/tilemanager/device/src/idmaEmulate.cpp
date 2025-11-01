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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__XTENSA__) || (defined(UNIFIED_TEST))
#include <xtensa/tie/xt_ivpn.h>
#include "tileManager.h"

idma_cntrl_t idma_cntrl __attribute__ ((section(".dram0.data"))) __attribute__ ((weak));
idma_buf_t* _idma_buf_ptr __attribute__ ((section(".dram0.data"))) __attribute__ ((weak));

static int32_t idmaCount[MAX_NUM_CHANNEL];
static int32_t idmaDone[MAX_NUM_CHANNEL];

void
_idma_reset()
{
  idma_cntrl.num_outstanding = 0;
  idma_cntrl.newest_task     = 0;
  idma_cntrl.oldest_task     = 0;
}

idma_status_t
idma_init(int32_t ch,
          unsigned init_flags,
          idma_max_block_t max_block_sz,
          unsigned max_pif_req,
          idma_ticks_cyc_t ticks_per_cyc,
          unsigned timeout_ticks,
          idma_err_callback_fn err_cb_func)
{
  idmaCount[ch] = 0;
  idmaDone[ch]  = 0;

  if (max_pif_req > XCHAL_IDMA_MAX_OUTSTANDING_REQ)
  {
    return((idma_status_t) IDMA_ERR_BAD_INIT);
  }
  else
  {
    return((idma_status_t) IDMA_OK);
  }
}

idma_status_t
idma_init_loop(int32_t ch,
               idma_buffer_t *buf,
               idma_type_t type,
               int32_t num_descs,
               void *cb_data,
               idma_callback_fn cb_func)
{
  return((idma_status_t) IDMA_OK);
}

int32_t copy2d(int32_t ch,
               void *dst,
               void *src,
               size_t row_size,
               unsigned flags,
               unsigned num_rows,
               unsigned src_pitch,
               unsigned dst_pitch)
{
  uint32_t indx;

  uint8_t *srcPtr, *dstPtr;

  srcPtr = (uint8_t *) ((int32_t *) src)[0];
  dstPtr = (uint8_t *) ((int32_t *) dst)[0];

  for (indx = 0u; indx < num_rows; indx++)
  {
    memcpy(dstPtr, srcPtr, row_size);
    dstPtr += dst_pitch;
    srcPtr += src_pitch;
  }

  int32_t temp = idmaCount[ch];
  idmaCount[ch] += 1;
  idmaCount[ch] &= 0x7FFFFFFF;

  return(temp);
}

idma_status_t add_and_schedule2d(void* pbuf,
	void *dst,
	void *src,
	size_t row_size,
	unsigned flags,
	unsigned num_rows,
	unsigned src_pitch,
	unsigned dst_pitch)
{
	(void *) pbuf;
	uint32_t indx;

	uint8_t *srcPtr, *dstPtr;

	srcPtr = (uint8_t *)((int32_t *)src)[0];
	dstPtr = (uint8_t *)((int32_t *)dst)[0];

	for (indx = 0u; indx < num_rows; indx++)
	{
		memcpy(dstPtr, srcPtr, row_size);
		dstPtr += dst_pitch;
		srcPtr += src_pitch;
	}

	int32_t temp = idmaCount[0];
	idmaCount[0] += 1;
	idmaCount[0] &= 0x7FFFFFFF;

	return(IDMA_OK);
}

void copy2dNoCountUpdate(int32_t ch,
                         void *dst,
                         void *src,
                         size_t row_size,
                         unsigned flags,
                         unsigned num_rows,
                         unsigned src_pitch,
                         unsigned dst_pitch)
{
  uint32_t indx;

  uint8_t *srcPtr, *dstPtr;

  srcPtr = (uint8_t *) ((int32_t *) src)[0];
  dstPtr = (uint8_t *) ((int32_t *) dst)[0];

  for (indx = 0u; indx < num_rows; indx++)
  {
    memcpy(dstPtr, srcPtr, row_size);
    dstPtr += dst_pitch;
    srcPtr += src_pitch;
  }
}

int32_t idma_copy_2d_pred_desc64_wide(int32_t ch,
                                      void *dst,
                                      void *src,
                                      size_t row_size,
                                      uint32_t flags,
                                      void* pred_mask,
                                      uint32_t num_rows,
                                      uint32_t src_pitch,
                                      uint32_t dst_pitch)
{
  uint32_t indx;

  uint8_t *srcPtr, *dstPtr;

  uint32_t* MaskPtr = (uint32_t * ) pred_mask;

  srcPtr = (uint8_t *) ((int32_t *) src)[0];
  dstPtr = (uint8_t *) ((int32_t *) dst)[0];

  for (indx = 0u; indx < num_rows; indx++)
  {
    if (MaskPtr[indx / 32] & (1 << (indx & 0x01F)))
    {
      memcpy(dstPtr, srcPtr, row_size);
      dstPtr += dst_pitch;
    }
    srcPtr += src_pitch;
  }

  int32_t temp = idmaCount[ch];
  idmaCount[ch] += 1;
  idmaCount[ch] &= 0x7FFFFFFF;

  return(temp);
}

idma_status_t idma_add_2d_pred_desc64_wide(void *bufh,
	void *dst,
	void *src,
	size_t row_size,
	uint32_t flags,
	void* pred_mask,
	uint32_t num_rows,
	uint32_t src_pitch,
	uint32_t dst_pitch)
{
	(void *)bufh;

	uint32_t indx;

	uint8_t *srcPtr, *dstPtr;

	uint32_t* MaskPtr = (uint32_t *)pred_mask;

	srcPtr = (uint8_t *)((int32_t *)src)[0];
	dstPtr = (uint8_t *)((int32_t *)dst)[0];

	for (indx = 0u; indx < num_rows; indx++)
	{
		if (MaskPtr[indx / 32] & (1 << (indx & 0x01F)))
		{
			memcpy(dstPtr, srcPtr, row_size);
			dstPtr += dst_pitch;
		}
		srcPtr += src_pitch;
	}

	int32_t temp = idmaCount[0];
	idmaCount[0] += 1;
	idmaCount[0] &= 0x7FFFFFFF;

	return(IDMA_OK);
}

int32_t idma_copy_3d_desc64_wide(int32_t ch,
                                 void *dst,
                                 void *src,
                                 uint32_t flags,
                                 size_t row_sz,
                                 uint32_t nrows,
                                 uint32_t ntiles,
                                 uint32_t src_row_pitch,
                                 uint32_t dst_row_pitch,
                                 uint32_t src_tile_pitch,
                                 uint32_t dst_tile_pitch)
{
  uint8_t *srcPtr, *dstPtr;
  uint32_t indx;

  srcPtr = (uint8_t *) ((int32_t *) src)[0];
  dstPtr = (uint8_t *) ((int32_t *) dst)[0];

  for (indx = 0u; indx < ntiles; indx++)
  {
    copy2dNoCountUpdate(ch, &dstPtr, &srcPtr, row_sz, flags, nrows, src_row_pitch, dst_row_pitch);

    dstPtr += dst_tile_pitch;
    srcPtr += src_tile_pitch;
  }
  int32_t temp = idmaCount[ch];
  idmaCount[ch] += 1;
  idmaCount[ch] &= 0x7FFFFFFF;

  return(temp);
}

idma_status_t idma_add_3d_desc64_wide(void *bufh,
	void *dst,
	void *src,
	uint32_t flags,
	size_t row_sz,
	uint32_t nrows,
	uint32_t ntiles,
	uint32_t src_row_pitch,
	uint32_t dst_row_pitch,
	uint32_t src_tile_pitch,
	uint32_t dst_tile_pitch)
{
	(void *)bufh;
	uint8_t *srcPtr, *dstPtr;
	uint32_t indx;

	srcPtr = (uint8_t *)((int32_t *)src)[0];
	dstPtr = (uint8_t *)((int32_t *)dst)[0];

	for (indx = 0u; indx < ntiles; indx++)
	{
		copy2dNoCountUpdate(0, &dstPtr, &srcPtr, row_sz, flags, nrows, src_row_pitch, dst_row_pitch);

		dstPtr += dst_tile_pitch;
		srcPtr += src_tile_pitch;
	}
	int32_t temp = idmaCount[0];
	idmaCount[0] += 1;
	idmaCount[0] &= 0x7FFFFFFF;

	return(IDMA_OK);
}

void dma_sleep(int32_t ch)
{
}

int32_t idma_desc_done(int32_t ch, int32_t index)
{
  //To improbe coverage. alternately return 0/1
  int32_t temp = idmaDone[ch];

  idmaDone[ch] ^= 1;

  return(temp);
}
#endif
