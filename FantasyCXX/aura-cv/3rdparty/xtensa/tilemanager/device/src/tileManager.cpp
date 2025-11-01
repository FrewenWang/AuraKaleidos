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

/**********************************************************************************
 * FILE:  tileManager.c
 *
 * DESCRIPTION:
 *
 *    This file contains Tile Manager implementation. It uses xvmem utility for
 *    buffer allocation and idma library for 2D data transfer.
 *
 *
 ********************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xtensa/tie/xt_ivpn.h>
#if defined(__XTENSA__) && (!defined(UNIFIED_TEST))
#include <xtensa/xtensa-versions.h>
#endif
#include "tileManager.h"
#ifdef TM_LOG
xvTileManager * __pxvTM;
#endif

extern application_symbol_tray *g_symbol_tray;

//#define TEST_DTCM23

static void xvExtendEdgesReflect101_I8(xvTile const * tile, int32_t frame_width, int32_t frame_height);
static void xvExtendEdgesReflect101_I16(xvTile const * tile, int32_t frame_width, int32_t frame_height);


/**********************************************************************************
 * FUNCTION: xvTileManagerCopyPaddingData()
 *
 * DESCRIPTION:
 *     Helper function to create a row of padding data that can be copied to
 *     other locations using iDMA engine.
 * 
 * zhonganyu : unused function because of tray functions.
 *
 * INPUTS:
 *     uint8_t      *pBuff              Buffer pointer
 *     int32_t      paddingVal          Padding value to be replicated.
 *     int32_t      pixWidth            Width of pixel in Bytes.
 *     int32_t      numBytes            Number of bytes to replicate using  paddingVal
 *
 * OUTPUTS:
 *     void
 *
 ********************************************************************************** */

// static void __attribute__ ((always_inline)) xvTileManagerCopyPaddingData(uint8_t *pBuff, int32_t paddingVal, int32_t pixWidth, int32_t numBytes)
// {
//   xb_vec2Nx8U dvec1, *pdvecDst = (xb_vec2Nx8U *) (pBuff);
//   valign vas1;
//   int32_t wb;
//   if ((((uint32_t) pixWidth) & 1u) == 1u)
//   {
//     dvec1 = paddingVal;
//   }
//   else if ((((uint32_t) pixWidth) & 2u) == 2u)
//   {
//     xb_vecNx16U vec1 = paddingVal;
//     dvec1 = IVP_MOV2NX8_FROMNX16(vec1);
//   }
//   else
//   {
//     xb_vecN_2x32v vec1 = paddingVal;
//     dvec1 = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(vec1));
//   }
//   vas1 = IVP_ZALIGN();

//   for (wb = numBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
//   {
//     IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
//   }
//   IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
// }

/**********************************************************************************
 * FUNCTION: xvInitIdmaMultiChannel4CH()
 *
 * DESCRIPTION:
 *     Function to initialize iDMA library for 4 iDMA channels.
 *
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object pointer
 *     idma_buffer_t        *buf0               iDMA library handle; contains descriptors and idma library object for channel0
 *     idma_buffer_t        *buf1               iDMA library handle; contains descriptors and idma library object for channel1.
 *                                              For single channel mode, buf1 can be NULL.
 *     idma_buffer_t        *buf2               iDMA library handle; contains descriptors and idma library object for channel2
 *                                              For single channel mode, buf2 can be NULL.
 *     idma_buffer_t        *buf3               iDMA library handle; contains descriptors and idma library object for channel3
 *                                              For single channel mode, buf3 can be NULL.
 *     int32_t              numDescs            Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              maxPifReq           Maximum number of outstanding pif requests
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_err_callback_fn errCallbackFunc1    Callback for dma transfer error for channel 1
 *     idma_err_callback_fn errCallbackFunc2    Callback for dma transfer error for channel 2
 *     idma_err_callback_fn errCallbackFunc3    Callback for dma transfer error for channel 3
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     cbFunc1             Callback for dma transfer completion for channel 1
 *     void                 *cbData1            Data needed for completion callback function for channel 1
 *     idma_callback_fn     cbFunc2             Callback for dma transfer completion for channel 2
 *     void                 *cbData2            Data needed for completion callback function for channel 2
 *     idma_callback_fn     cbFunc3             Callback for dma transfer completion for channel 3
 *     void                 *cbData3            Data needed for completion callback function for channel 3
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitIdmaMultiChannel4CH(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1, idma_buffer_t *buf2, idma_buffer_t *buf3,
                                  int32_t numDescs, int32_t maxBlock, int32_t maxPifReq, idma_err_callback_fn errCallbackFunc0,
                                  idma_err_callback_fn errCallbackFunc1, idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                  idma_callback_fn cbFunc0, void * cbData0, idma_callback_fn cbFunc1, void * cbData1,
                                  idma_callback_fn cbFunc2, void * cbData2, idma_callback_fn cbFunc3, void * cbData3)
{
  return g_symbol_tray->tray_xvInitIdmaMultiChannel4CH(pxvTM, buf0, buf1, buf2, buf3,
                                                         numDescs, maxBlock, maxPifReq, errCallbackFunc0,
                                                         errCallbackFunc1, errCallbackFunc2, errCallbackFunc3,
                                                         cbFunc0, cbData0, cbFunc1, cbData1,
                                                         cbFunc2, cbData2, cbFunc3, cbData3);
}

/**********************************************************************************
 * FUNCTION: xvInitIdmaMultiChannel()
 *
 * DESCRIPTION:
 *     Function to initialize 2 channels of iDMA library.
 *     This is API backward compatible with VisionQ6 processor TM
 *
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object
 *     idma_buffer_t        *buf0               iDMA library handle; contains descriptors and idma library object for channel0
 *     idma_buffer_t        *buf1               iDMA library handle; contains descriptors and idma library object for channel1.
 *                                              For single channel mode, buf1 can be NULL;.
 *     int32_t              numDescs            Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              maxPifReq           Maximum number of outstanding pif requests
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_err_callback_fn errCallbackFunc1    Callback for dma transfer error for channel 1
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     cbFunc1             Callback for dma transfer completion for channel 1
 *     void                 *cbData1            Data needed for completion callback function for channel 1
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitIdmaMultiChannel(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1,
                               int32_t numDescs, int32_t maxBlock, int32_t maxPifReq, idma_err_callback_fn errCallbackFunc0,
                               idma_err_callback_fn errCallbackFunc1, idma_callback_fn cbFunc0, void * cbData0, idma_callback_fn cbFunc1, void * cbData1)
{
  return(xvInitIdmaMultiChannel4CH(pxvTM, buf0, buf1, NULL, NULL,                                                 \
                                   numDescs, maxBlock, maxPifReq, errCallbackFunc0, errCallbackFunc1, NULL, NULL, \
                                   cbFunc0, cbData0, cbFunc1, cbData1, NULL, NULL, NULL, NULL));                  \

}

/**********************************************************************************
 * FUNCTION: xvInitIdma()
 *
 * DESCRIPTION:
 *     Function to initialize single channels of iDMA library.
 *     This is API backward compatible with VisionP6 processor TM.
 *
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object
 *     idma_buffer_t        *buf0               iDMA library handle; contains descriptors and idma library object for channel0
 *     int32_t              numDescs            Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              maxPifReq           Maximum number of outstanding pif requests
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitIdma(xvTileManager *pxvTM, idma_buffer_t *buf0,
                   int32_t numDescs, int32_t maxBlock, int32_t maxPifReq, idma_err_callback_fn errCallbackFunc0,
                   idma_callback_fn cbFunc0, void * cbData0)
{
  return(xvInitIdmaMultiChannel(pxvTM, buf0, NULL,
                                numDescs, maxBlock, maxPifReq, errCallbackFunc0, NULL,
                                cbFunc0, cbData0, NULL, NULL));
}

/**********************************************************************************
 * FUNCTION: xvInitTileManagerMultiChannel4CH()
 *
 * DESCRIPTION:
 *     Function to initialize Tile Manager. Resets Tile Manager's internal structure elements.
 *     Initializes Tile Manager's iDMA object.
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     idma_buffer_t *buf0, *buf1    iDMA object. It should be initialized before calling this function
 *                                   In case of singleChannel DMA, buf1 can be NULL
 *     idma_buffer_t *buf2, *buf3    iDMA object. It should be initialized before calling this function
 *                                   In case of singleChannel or doubleChannel DMA, buf2 and buf3 can be NULL
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitTileManagerMultiChannel4CH(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1,
                                         idma_buffer_t *buf2, idma_buffer_t *buf3)
{
  return g_symbol_tray->tray_xvInitTileManagerMultiChannel4CH(pxvTM, buf0, buf1, buf2, buf3);
}

/**********************************************************************************
 * FUNCTION: xvInitTileManagerMultiChannel()
 *
 * DESCRIPTION:
 *     Function to initialize 2 channel Tile Manager. Resets Tile Manager's internal structure elements.
 *     Initializes Tile Manager's iDMA object. This API is backward compatible with Q6 TM.
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     idma_buffer_t *buf0, *buf1    iDMA object. It should be initialized before calling this function
 *                                   In case of singleChannel DMA, buf1 can be NULL
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitTileManagerMultiChannel(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1)
{
  return(xvInitTileManagerMultiChannel4CH(pxvTM, buf0, buf1, NULL, NULL));
}

/**********************************************************************************
 * FUNCTION: xvInitTileManager()
 *
 * DESCRIPTION:
 *     Function to initialize 1 channel Tile Manager. Resets Tile Manager's internal structure elements.
 *     Initializes Tile Manager's iDMA object. This API is backward compatible with P6 TM.
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     idma_buffer_t *buf0           iDMA object. It should be initialized before calling this function
 *                                   In case of singleChannel DMA, buf1 can be NULL
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvInitTileManager(xvTileManager *pxvTM, idma_buffer_t *buf0)
{
  return(xvInitTileManagerMultiChannel4CH(pxvTM, buf0, NULL, NULL, NULL));
}

/**********************************************************************************
 * FUNCTION: xvResetTileManager()
 *
 * DESCRIPTION:
 *     Function to reset Tile Manager. It closes the log file,
 *     releases the buffers and resets the Tile Manager object
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvResetTileManager(xvTileManager *pxvTM)
{
  return g_symbol_tray->tray_xvResetTileManager(pxvTM);
}

/**********************************************************************************
 * FUNCTION: xvInitMemAllocator()
 *
 * DESCRIPTION:
 *     Function to initialize memory allocator. Tile Manager uses xvmem utility as memory allocator.
 *     It takes array of pointers to memory pool's start addresses and respective sizes as input and
 *     uses the memory pools when memory is allocated.
 *
 * INPUTS:
 *     xvTileManager *pxvTM            Tile Manager object pointer.
 *     int32_t       numMemBanks       Number of memory pools
 *     void          **pBankBuffPool   Array of memory pool start address
 *     int32_t       *buffPoolSize     Array of memory pool sizes
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
#ifndef XVTM_USE_XMEM
int32_t xvInitMemAllocator(xvTileManager *pxvTM, int32_t numMemBanks, void* const* pBankBuffPool, int32_t const* buffPoolSize)
{
  return g_symbol_tray->tray_xvInitMemAllocator(pxvTM, numMemBanks, pBankBuffPool, buffPoolSize);
}


#endif
/**********************************************************************************
 * FUNCTION: xvAllocateBuffer()
 *
 * DESCRIPTION:
 *     Allocates buffer from the given buffer pool. It returns aligned buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM            Tile Manager object
 *     int32_t       buffSize          Size of the requested buffer
 *     int32_t       buffColor         Color/index of requested bufffer
 *     int32_t       buffAlignment     Alignment of requested buffer
 *
 * OUTPUTS:
 *     Returns the buffer with requested parameters. If an error occurs, returns ((void *)(XVTM_ERROR))
 *
 ********************************************************************************** */

void *xvAllocateBuffer(xvTileManager *pxvTM, int32_t buffSize, int32_t buffColor, int32_t buffAlignment)
{
  return g_symbol_tray->tray_xvAllocateBuffer(pxvTM, buffSize, buffColor, buffAlignment); // replaced by tray func.
}

/**********************************************************************************
 * FUNCTION: xvFreeBuffer()
 *
 * DESCRIPTION:
 *     Releases the given buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     void          *pBuff      Buffer that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeBuffer(xvTileManager *pxvTM, void const *pBuff)
{
  return g_symbol_tray->tray_xvFreeBuffer(pxvTM, pBuff); // replaced by tray func.
}

/**********************************************************************************
 * FUNCTION: xvFreeAllBuffers()
 *
 * DESCRIPTION:
 *     Releases all buffers. Reinitializes the memory allocator
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
#ifndef XVTM_USE_XMEM
int32_t xvFreeAllBuffers(xvTileManager *pxvTM)
{
  return g_symbol_tray->tray_xvFreeAllBuffers(pxvTM);
}
#endif
/**********************************************************************************
 * FUNCTION: xvAllocateFrame()
 *
 * DESCRIPTION:
 *     Allocates single frame. It does not allocate buffer required for frame data.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns the pointer to allocated frame.
 *     Returns ((xvFrame *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate frame data buffer.
 *
 ********************************************************************************** */

xvFrame *xvAllocateFrame(xvTileManager *pxvTM)
{
  return g_symbol_tray->tray_xvAllocateFrame(pxvTM);
}

/**********************************************************************************
 * FUNCTION: xvFreeFrame()
 *
 * DESCRIPTION:
 *     Releases the given frame. Does not release associated frame data buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     xvFrame const * pFrame     Frame that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeFrame(xvTileManager *pxvTM, xvFrame const *pFrame)
{
  return g_symbol_tray->tray_xvFreeFrame(pxvTM, pFrame);
}

/**********************************************************************************
 * FUNCTION: xvFreeAllFrames()
 *
 * DESCRIPTION:
 *     Releases all allocated frames.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeAllFrames(xvTileManager *pxvTM)
{
  uint32_t index;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pxvTM->frameCount = 0;
  for (index = 0; index < ((MAX_NUM_FRAMES + 31u) / 32u); index++)
  {
    pxvTM->frameAllocFlags[index] = 0x00000000;
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvAllocateTile()
 *
 * DESCRIPTION:
 *     Allocates single tile. It does not allocate buffer required for tile data.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns the pointer to allocated tile.
 *     Returns ((xvTile *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate tile data buffer
 *
 ********************************************************************************** */

xvTile *xvAllocateTile(xvTileManager *pxvTM)
{
  uint32_t indx, indxArr = 0, indxShift = 0, allocFlags = 0;
  xvTile *pTile = NULL;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, (xvTile *) ((void *) XVTM_ERROR), "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  for (indx = 0; indx < MAX_NUM_TILES; indx++)
  {
    indxArr    = indx >> 5u;
    indxShift  = indx & 0x1Fu;
    allocFlags = pxvTM->tileAllocFlags[indxArr];
    if (((allocFlags >> indxShift) & 0x1u) == 0u)
    {
      break;
    }
  }

  if (indx < MAX_NUM_TILES)
  {
    pTile                          = &(pxvTM->tileArray[indx]);
    pxvTM->tileAllocFlags[indxArr] = allocFlags | (((uint32_t) 0x1u) << indxShift);
    pxvTM->tileCount++;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_TILE_BUFFER_FULL;
    return((xvTile *) ((void *) XVTM_ERROR));
  }

  return(pTile);
}

/**********************************************************************************
 * FUNCTION: xvFreeTile()
 *
 * DESCRIPTION:
 *     Releases the given tile. Does not release associated tile data buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     xvTile const * pTile      Tile that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvFreeTile(xvTileManager *pxvTM, xvTile const *pTile)
{
  uint32_t indx, indxArr, indxShift, allocFlags;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");


  pxvTM->errFlag = XV_ERROR_SUCCESS;
  if (pTile == NULL)
  {
    pxvTM->errFlag = XV_ERROR_TILE_NULL;
    return(XVTM_ERROR);
  }

  for (indx = 0; indx < MAX_NUM_TILES; indx++)
  {
    if (&(pxvTM->tileArray[indx]) == pTile)
    {
      break;
    }
  }

  if (indx < MAX_NUM_TILES)
  {
    indxArr                        = indx >> 5u;
    indxShift                      = indx & 0x1Fu;
    allocFlags                     = pxvTM->tileAllocFlags[indxArr];
    pxvTM->tileAllocFlags[indxArr] = allocFlags & ~(0x1u << indxShift);
    pxvTM->tileCount--;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_BAD_ARG;
    return(XVTM_ERROR);
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvFreeAllTiles()
 *
 * DESCRIPTION:
 *     Releases all allocated tiles.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeAllTiles(xvTileManager *pxvTM)
{
  uint32_t index;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pxvTM->tileCount = 0;
  for (index = 0; index < ((MAX_NUM_TILES + 31u) / 32u); index++)
  {
    pxvTM->tileAllocFlags[index] = 0x00000000;
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvCreateTileManagerMultiChannel4CH()
 *
 * DESCRIPTION:
 *     Creates and initializes Tile Manager, Memory Allocator and iDMA.
 *     Supports 4 iDMA channels.
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object
 *     void                 *buf0               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel0
 *     void                 *buf1               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel1. For single channel mode. buf1 can be NULL;.
 *     void                 *buf2               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel2  For single channel mode. buf2 can be NULL;.
 *     void                 *buf3               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel3. For single channel mode. buf3 can be NULL;.
 *     int32_t              numMemBanks         Number of memory pools
 *     void                 **pBankBuffPool     Array of memory pool start address
 *     int32_t              *buffPoolSize       Array of memory pool sizes
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_err_callback_fn errCallbackFunc1    Callback for dma transfer error for channel 1
 *     idma_err_callback_fn errCallbackFunc2    Callback for dma transfer error for channel 2
 *     idma_err_callback_fn errCallbackFunc3    Callback for dma transfer error for channel 3
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     cbFunc1             Callback for dma transfer completion for channel 1
 *     void                 *cbData1            Data needed for completion callback function for channel 1
 *     idma_callback_fn     cbFunc2             Callback for dma transfer completion for channel 2
 *     void                 *cbData2            Data needed for completion callback function for channel 2
 *     idma_callback_fn     cbFunc3             Callback for dma transfer completion for channel 3
 *     void                 *cbData3            Data needed for completion callback function for channel 3
 *     int32_t              descCount           Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              numOutReq           Maximum number of outstanding pif requests
 *
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvCreateTileManagerMultiChannel4CH(xvTileManager *pxvTM, void *buf0, void *buf1, void *buf2, void *buf3, int32_t numMemBanks, void* const* pBankBuffPool, int32_t const* buffPoolSize,
                                           idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,
                                           idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                           idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,
                                           idma_callback_fn intrCallbackFunc2, void *cbData2, idma_callback_fn intrCallbackFunc3, void *cbData3,
                                           int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  int32_t retVal;

  retVal = xvInitTileManagerMultiChannel4CH(pxvTM, (idma_buffer_t *) buf0, (idma_buffer_t *) buf1,
                                            (idma_buffer_t *) buf2, (idma_buffer_t *) buf3);
  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");
#ifndef XVTM_USE_XMEM

  retVal = xvInitMemAllocator(pxvTM, numMemBanks, pBankBuffPool, buffPoolSize);
  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");
#else
  (void)numMemBanks;
  (void)pBankBuffPool;
  (void)buffPoolSize;
#endif
  retVal = xvInitIdmaMultiChannel4CH(pxvTM, (idma_buffer_t *) buf0, (idma_buffer_t *) buf1, (idma_buffer_t *) buf2, (idma_buffer_t *) buf3,
                                     descCount, maxBlock, numOutReq, errCallbackFunc0, errCallbackFunc1,
                                     errCallbackFunc2, errCallbackFunc3, intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
                                     intrCallbackFunc2, cbData2, intrCallbackFunc3, cbData3);

  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");

  return(retVal);
}

/**********************************************************************************
 * FUNCTION: xvCreateTileManagerMultiChannel()
 *
 * DESCRIPTION:
 *     Creates and initializes Tile Manager, Memory Allocator and 2 channel  iDMA.
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object
 *     void                 *buf0               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel0
 *     void                 *buf1               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel1. For single channel mode. buf1 can be NULL;.
 *     int32_t              numMemBanks         Number of memory pools
 *     void                 **pBankBuffPool     Array of memory pool start address
 *     int32_t              *buffPoolSize       Array of memory pool sizes
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_err_callback_fn errCallbackFunc1    Callback for dma transfer error for channel 1
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     cbFunc1             Callback for dma transfer completion for channel 1
 *     void                 *cbData1            Data needed for completion callback function for channel 1
 *     int32_t              descCount           Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              numOutReq           Maximum number of outstanding pif requests
 *
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
#ifdef TEST_DTCM23
int32_t xvCreateTileManagerMultiChannel(xvTileManager *pxvTM, void *buf0, void *buf1,void *buf2, void *buf3, int32_t numMemBanks,
                                        void **pBankBuffPool, int32_t* buffPoolSize,
                                        idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                        idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,idma_callback_fn intrCallbackFunc2, void *cbData2, idma_callback_fn intrCallbackFunc3, void *cbData3,
                                        int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  return(xvCreateTileManagerMultiChannel4CH(pxvTM, buf0, buf1, buf2, buf3, numMemBanks, pBankBuffPool, buffPoolSize,
                                            errCallbackFunc0, errCallbackFunc1, errCallbackFunc2, errCallbackFunc3,
                                            intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
											intrCallbackFunc2, cbData2, intrCallbackFunc3, cbData3,
                                            descCount, maxBlock, numOutReq));
}
#else
int32_t xvCreateTileManagerMultiChannel(xvTileManager *pxvTM, void *buf0, void *buf1, int32_t numMemBanks,
                                        void* const* pBankBuffPool, int32_t const* buffPoolSize,
                                        idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,
                                        idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,
                                        int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  return(xvCreateTileManagerMultiChannel4CH(pxvTM, buf0, buf1, NULL, NULL, numMemBanks, pBankBuffPool, buffPoolSize,
                                            errCallbackFunc0, errCallbackFunc1, NULL, NULL,
                                            intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
                                            NULL, NULL, NULL, NULL,
                                            descCount, maxBlock, numOutReq));
}
#endif
/**********************************************************************************
 * FUNCTION: xvCreateTileManager()
 *
 * DESCRIPTION:
 *     Creates and initializes Tile Manager, Memory Allocator and 1 channel  iDMA.
 *
 * INPUTS:
 *     xvTileManager        *pxvTM              Tile Manager object
 *     void                 *buf0               iDMA object. It should be initialized before calling this function. Contains descriptors and idma library object for channel0
 *     int32_t              numMemBanks         Number of memory pools
 *     void                 **pBankBuffPool     Array of memory pool start address
 *     int32_t              *buffPoolSize       Array of memory pool sizes
 *     idma_err_callback_fn errCallbackFunc0    Callback for dma transfer error for channel 0
 *     idma_callback_fn     cbFunc0             Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     int32_t              descCount           Number of descriptors that can be added in buffer
 *     int32_t              maxBlock            Maximum block size allowed
 *     int32_t              numOutReq           Maximum number of outstanding pif requests
 *
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvCreateTileManager(xvTileManager *pxvTM, void *buf0, int32_t numMemBanks,
                            void * const* pBankBuffPool, int32_t const* buffPoolSize,
                            idma_err_callback_fn errCallbackFunc0,
                            idma_callback_fn intrCallbackFunc0, void *cbData0,
                            int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  return(xvCreateTileManagerMultiChannel4CH(pxvTM, buf0, NULL, NULL, NULL, numMemBanks, pBankBuffPool, buffPoolSize,
                                            errCallbackFunc0, NULL, NULL, NULL,
                                            intrCallbackFunc0, cbData0, NULL, NULL,
                                            NULL, NULL, NULL, NULL,
                                            descCount, maxBlock, numOutReq));
}

/**********************************************************************************
 * FUNCTION: xvCreateFrame()
 *
 * DESCRIPTION:
 *     Allocates single frame. It does not allocate buffer required for frame data.
 *     Initializes the frame elements
 *
 * INPUTS:
 *     xvTileManager *pxvTM           Tile Manager object
 *     uint64_t       imgBuff         Pointer to iaura buffer
 *     uint32_t       frameBuffSize   Size of allocated iaura buffer
 *     int32_t        width           Width of iaura
 *     int32_t        height          Height of iaura
 *     int32_t        pitch           Pitch of iaura rows
 *     uint8_t        pixRes          Pixel resolution of iaura in bytes
 *     uint8_t        numChannels     Number of channels in the iaura
 *     uint8_t        paddingtype     Supported padding type
 *     uint32_t       paddingVal      Padding value if padding type is edge extension
 *
 * OUTPUTS:
 *     Returns the pointer to allocated frame.
 *     Returns ((xvFrame *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate frame data buffer.
 *
 ********************************************************************************** */

xvFrame *xvCreateFrame(xvTileManager *pxvTM, uint64_t imgBuff, uint32_t frameBuffSize, int32_t width, int32_t height, int32_t pitch, uint8_t pixRes, uint8_t numChannels, uint8_t paddingtype, uint32_t paddingVal)
{
  XV_CHECK_ERROR_NULL(((pxvTM == NULL) || (imgBuff == 0u)), ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR(((width < 0) || (height < 0) || (pitch < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR(((width * numChannels) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR((frameBuffSize < (((uint32_t) pitch) * ((uint32_t) height) * pixRes)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR(((numChannels > MAX_NUM_CHANNEL) || (numChannels <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR((paddingtype > FRAME_PADDING_MAX), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame *) ((void *) XVTM_ERROR)), "XVTM_ERROR");

  xvFrame *pFrame = xvAllocateFrame(pxvTM);

  if ((void *) pFrame == (void *) XVTM_ERROR)
  {
    pxvTM->errFlag = XV_ERROR_BAD_ARG;
    return((xvFrame *) ((void *) XVTM_ERROR));
  }

  SETUP_FRAME(pFrame, imgBuff, frameBuffSize, width, height, pitch, 0, 0, pixRes, numChannels, paddingtype, paddingVal);
  return(pFrame);
}

/**********************************************************************************
 * FUNCTION: xvCreateTile()
 *
 * DESCRIPTION:
 *     Allocates single tile and associated buffer data.
 *     Initializes the elements in tile
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     int32_t       tileBuffSize    Size of allocated tile buffer
 *     int32_t       width           Width of tile
 *     uint16_t      height          Height of tile
 *     int32_t       pitch           Pitch of tile
 *     uint16_t      edgeWidth       Edge width of tile
 *     uint16_t      edgeHeight      Edge height of tile
 *     int32_t       color           Memory pool from which the buffer should be allocated
 *     xvFrame       *pFrame         Frame associated with the tile
 *     uint16_t      tileType       Type of tile
 *     int32_t       alignType       Alignment tpye of tile. could be edge aligned or data aligned
 *
 * OUTPUTS:
 *     Returns the pointer to allocated tile.
 *     Returns ((xvTile *)(XVTM_ERROR)) if it encounters an error.
 *
 ********************************************************************************** */

xvTile *xvCreateTile(xvTileManager *pxvTM, int32_t tileBuffSize, int32_t width, uint16_t height,
                     int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame,
                     uint16_t xvTileType, int32_t alignType)
{
  return g_symbol_tray->tray_xvCreateTile(pxvTM, tileBuffSize, width, height, pitch, edgeWidth, edgeHeight, color, pFrame, xvTileType, alignType); // replaced by tray func.
}

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_predicated_wide()
 *
 * DESCRIPTION:
 *     Add iDMA predicated transfer request using wide addresses
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *     int32_t        *pred_mask               predication mask pointer
 * OUTPUTS:
 *     Returns dmaIndex for this request. It returns XVTM_ERROR if it encounters an error
 *
 ********************************************************************************** */

int32_t xvAddIdmaRequestMultiChannel_predicated_wide(int32_t dmaChannel, xvTileManager *pxvTM, uint64_t pdst64,
                                                     uint64_t psrc64, size_t rowSize,
                                                     int32_t numRows, int32_t srcPitch, int32_t dstPitch,
                                                     int32_t interruptOnCompletion, uint32_t  *pred_mask)
{
  return g_symbol_tray->tray_xvAddIdmaRequestMultiChannel_predicated_wide(dmaChannel, pxvTM, pdst64, psrc64, rowSize,
                                                                            numRows, srcPitch, dstPitch, interruptOnCompletion, pred_mask);
}
#endif

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wide()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request using wide addresses.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request. It returns XVTM_ERROR if it encounters an error
 *
 ********************************************************************************** */


int32_t xvAddIdmaRequestMultiChannel_wide(int32_t dmaChannel, xvTileManager *pxvTM,
                                          uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                          int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvAddIdmaRequestMultiChannel_wide(dmaChannel, pxvTM, pdst64, psrc64, rowSize,
                                                                 numRows, srcPitch, dstPitch, interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: addIdmaRequestInlineMultiChannel_wide()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request using wide addresses.
 *     Inline function without any sanity checks.
 *  * zhonganyu : for PIC compiling, this fuction is unused (static func is used in other functions come with tray)
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request.
 *
 ********************************************************************************** */

// static inline int32_t addIdmaRequestInlineMultiChannel_wide( xvTileManager * const pxvTM, int32_t dmaChannel, uint64_t pdst64,
//                                                             uint64_t psrc64, size_t rowSize, int32_t numRows, int32_t srcPitch,
//                                                             int32_t dstPitch, int32_t interruptOnCompletion)
// {

// }

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wide3D()
 *
 * DESCRIPTION:
 *     Add 3D iDMA transfer request using wide addresses.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *     int32_t        srcTilePitch            Source buffer's 2D tile pitch in bytes
 *     int32_t        dstTilePitch            Destination buffer's 2D tile pitch in bytes
 *     int32_t        numTiles                Number of 2D tiles to transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request. It returns XVTM_ERROR if it encounters an error
 *
 ********************************************************************************** */

int32_t xvAddIdmaRequestMultiChannel_wide3D(int32_t dmaChannel, xvTileManager *pxvTM,
                                            uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                            int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, int32_t srcTilePitch,
                                            int32_t dstTilePitch, int32_t numTiles)
{
  return g_symbol_tray->tray_xvAddIdmaRequestMultiChannel_wide3D(dmaChannel, pxvTM, pdst64, psrc64, rowSize,
                                                                   numRows, srcPitch, dstPitch, interruptOnCompletion, srcTilePitch,
                                                                   dstTilePitch, numTiles);
}

/**********************************************************************************
 * FUNCTION: AddIdmaRequestMultiChannel_wide3D()
 *
 * DESCRIPTION:
 *     Add 3D iDMA transfer request using wide addresses.
 *     Inline function without any sanity checks.
 * zhonganyu : for PIC compiling, this fuction is unused (static func is used in other functions come with tray)
 * 
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *     int32_t        srcTilePitch            Source buffer's 2D tile pitch in bytes
 *     int32_t        dstTilePitch            Destination buffer's 2D tile pitch in bytes
 *     int32_t        numTiles                Number of 2D tiles to transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request.
 *
 ********************************************************************************** */

// static inline int32_t AddIdmaRequestMultiChannel_wide3D(xvTileManager * const pxvTM,int32_t dmaChannel,
//                                                         uint64_t pdst64, uint64_t psrc64, size_t rowSize,
//                                                         int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, int32_t srcTilePitch,
//                                                         int32_t dstTilePitch, int32_t numTiles)
// {

// }

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: addIdmaRequestInlineMultiChannel_predicated_wide()
 *
 * DESCRIPTION:
 *     Add iDMA predicated transfer request using wide addresses
 *     Inline function without any sanity checks.
 * * zhonganyu : for PIC compiling, this fuction is unused (static func is used in other functions come with tray)
 * 
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint64_t       pdst64                  Pointer to destination buffer
 *     uint64_t       psrc64                  Pointer to source buffer
 *     size_t         rowSize                 Number of bytes to transfer in a row
 *     int32_t        numRows                 Number of rows to transfer
 *     int32_t        srcPitch                Source buffer's pitch in bytes
 *     int32_t        dstPitch                Destination buffer's pitch in bytes
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *     int32_t        *pred_mask               predication mask pointer
 * OUTPUTS:
 *     Returns dmaIndex for this request.
 *
 ********************************************************************************** */

// static inline int32_t addIdmaRequestInlineMultiChannel_predicated_wide( xvTileManager * const pxvTM, int32_t dmaChannel,
//                                                                        uint64_t pdst64, uint64_t psrc64, size_t rowSize,
//                                                                        int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, uint32_t* pred_mask)
// {
  
// }
#endif

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request. Request is scheduled as soon as it is added
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     void          const *pdst              Pointer to destination buffer
 *     void          const *psrc              Pointer to source buffer
 *     size_t        rowSize                  Number of bytes to transfer in a row
 *     int32_t       numRows                  Number of rows to transfer
 *     int32_t       srcPitch                 Source buffer's pitch in bytes
 *     int32_t       dstPitch                 Destination buffer's pitch in bytes
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request. It returns XVTM_ERROR if it encounters an error
 *
 ********************************************************************************** */

int32_t xvAddIdmaRequestMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, void const *pdst,
                                     void const *psrc, size_t rowSize, int32_t numRows, int32_t srcPitch,
                                     int32_t dstPitch, int32_t interruptOnCompletion)
{
  uint64_t psrc64, pdst64;
  int32_t dmaIndex;


  psrc64 = (uint64_t) (uint32_t) psrc;
  pdst64 = (uint64_t) (uint32_t) pdst;
  TM_LOG_PRINT("line=%d, src: %llx, dst: %llx, rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, rowSize, numRows, srcPitch, dstPitch, interruptOnCompletion);

  dmaIndex = xvAddIdmaRequestMultiChannel_wide(dmaChannel, pxvTM, pdst64, psrc64, rowSize, numRows, srcPitch, dstPitch, interruptOnCompletion);

  TM_LOG_PRINT("line=%d, dmaIndex: %d\n", __LINE__, dmaIndex);
  return(dmaIndex);
}

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequest()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request for channel 0. Request is scheduled as soon as it is added
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     void          const *pdst              Pointer to destination buffer
 *     void          const *psrc              Pointer to source buffer
 *     size_t        rowSize                  Number of bytes to transfer in a row
 *     int32_t       numRows                  Number of rows to transfer
 *     int32_t       srcPitch                 Source buffer's pitch in bytes
 *     int32_t       dstPitch                 Destination buffer's pitch in bytes
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 * OUTPUTS:
 *     Returns dmaIndex for this request. It returns XVTM_ERROR if it encounters an error
 *
 ********************************************************************************** */

int32_t xvAddIdmaRequest(xvTileManager *pxvTM, void const *pdst, void const *psrc, size_t rowSize,
                         int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion)
{
  return(xvAddIdmaRequestMultiChannel(TM_IDMA_CH0, pxvTM, pdst, psrc, rowSize,
                                      numRows, srcPitch, dstPitch, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: solveForX()
 *
 * DESCRIPTION:
 *     Internal function. Not for end user.
 * zhonganyu : for PIC compiling, this fuction is unused (static func is used in other functions come with tray)
 *
 ********************************************************************************** */

// Part of tile reuse. Checks X direction boundary condition and performs DMA transfers
// static int32_t solveForX(xvTileManager *pxvTM, int32_t dmaChannel, xvTile const *pTile, uint8_t *pCurrBuff, uint8_t *pPrevBuff,
//                          int32_t y1, int32_t y2, int32_t x1, int32_t x2, int32_t px1, int32_t px2, int32_t tp, int32_t ptp, int32_t interruptOnCompletion)
// {

// }

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannelPredicated()
 *
 * DESCRIPTION:
 *     Requests predicated data transfer from frame present in system memory
 *     to local tile memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *     uint32_t      *pred_mask               Pointer to predication mask buffer in DRAM
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferInMultiChannelPredicated(int32_t dmaChannel, xvTileManager *pxvTM,
                                                  xvTile *pTile, int32_t interruptOnCompletion, uint32_t* pred_mask)
{
  return g_symbol_tray->tray_xvReqTileTransferInMultiChannelPredicated(dmaChannel, pxvTM, pTile, interruptOnCompletion, pred_mask);
}
#endif

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannel()
 *
 * DESCRIPTION:
 *     Requests data transfer from frame present in system memory to local tile memory.
 *     If there is an overlap between previous tile and current tile, tile reuse
 *     functionality can be used.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     xvTile        *pPrevTile               Data is copied from this tile to pTile if the buffer overlaps
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferInMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM,
                                        xvTile *pTile, xvTile *pPrevTile, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferInMultiChannel(dmaChannel, pxvTM, pTile, pPrevTile,  interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferIn()
 *
 * DESCRIPTION:
 *     Requests data transfer from frame present in system memory to local tile memory.
 *     Transfer is carried out using idma channel 0
 *     If there is an overlap between previous tile and current tile, tile reuse
 *     functionality can be used.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     xvTile        *pPrevTile               Data is copied from this tile to pTile if the buffer overlaps
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferIn(xvTileManager *pxvTM, xvTile *pTile, xvTile *pPrevTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferInMultiChannel(TM_IDMA_CH0, pxvTM, pTile, pPrevTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFastMultiChannel()
 *
 * DESCRIPTION:
 *     Requests 8b data transfer from frame present in system memory to local tile memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast function. No Sanity checks. Always returns XVTM_SUCCESS
 *
 *
 ********************************************************************************** */

int32_t xvReqTileTransferInFastMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM,
                                            xvTile *pTile, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferInFastMultiChannel(dmaChannel, pxvTM, pTile, interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFast()
 *
 * DESCRIPTION:
 *     Requests 8b data transfer from frame present in system memory to local tile memory.
 *     Transfer is carried out using idma channel 0
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast function. No Sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferInFast(xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferInFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFast16MultiChannel()
 *
 * DESCRIPTION:
 *     Requests 16b data transfer from frame present in system memory to local tile memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast function. No Sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferInFast16MultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferInFastMultiChannel(dmaChannel, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFast16()
 *
 * DESCRIPTION:
 *     Requests 16b data transfer from frame present in system memory to local tile memory.
 *     Transfer is carried out using idma channel 0
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Destination tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast function. No Sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferInFast16(xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferInFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile, interruptOnCompletion));
}

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannelPredicated()
 *
 * DESCRIPTION:
 *     Requests predicated data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *     uint32_t*     pred_mask                Predication mask
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferOutMultiChannelPredicated(int32_t dmaChannel, xvTileManager *pxvTM,
                                                   xvTile *pTile, int32_t interruptOnCompletion, uint32_t *pred_mask)
{
  return g_symbol_tray->tray_xvReqTileTransferOutMultiChannelPredicated(dmaChannel, pxvTM, pTile, interruptOnCompletion, pred_mask);
}
#endif

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannel()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM,
                                         xvTile *pTile, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferOutMultiChannel(dmaChannel, pxvTM, pTile, interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOut()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *     Transfer is carried out using idma channel 0
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOut(xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferOutMultiChannel(TM_IDMA_CH0, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFastMultiChannel()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast variant. No sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvReqTileTransferOutFastMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferOutFastMultiChannel(dmaChannel, pxvTM, pTile, interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFast()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *     Transfer is carrying out using idma channel 0
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast variant. No sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutFast(xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferOutFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFast16MultiChannel()
 *
 * DESCRIPTION:
 *     Requests 16b data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast variant. No sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutFast16MultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferOutFastMultiChannel(dmaChannel, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFast16()
 *
 * DESCRIPTION:
 *     Requests 16b data transfer from tile present in local memory to frame in system memory.
 *     Transfer is carrying out using idma channel 0
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast variant. No sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutFast16(xvTileManager  *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  return(xvReqTileTransferOutFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile, interruptOnCompletion));
}

/**********************************************************************************
 * FUNCTION: xvCheckForIdmaIndexMultiChannel()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer for given index is completed
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     int32_t       index                    Index of the dma transfer request, returned by xvAddIdmaRequest()
 *
 * OUTPUTS:
 *     Returns ONE if transfer is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */

int32_t xvCheckForIdmaIndexMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, int32_t index)
{
  return g_symbol_tray->tray_xvCheckForIdmaIndexMultiChannel(dmaChannel, pxvTM, index);
}

/**********************************************************************************
 * FUNCTION: xvCheckForIdmaIndex()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer for given index of idma channel 0 is completed
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     int32_t       index                    Index of the dma transfer request, returned by xvAddIdmaRequest()
 *
 * OUTPUTS:
 *     Returns ONE if transfer is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */

int32_t xvCheckForIdmaIndex(xvTileManager *pxvTM, int32_t index)
{
  return(xvCheckForIdmaIndexMultiChannel(TM_IDMA_CH0, pxvTM, index));
}

/**********************************************************************************
 * FUNCTION: xvCheckTileReadyMultiChannel()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer for given tile is completed.
 *     It checks all the tile in the transfer request buffer
 *     before the given tile and updates their respective
 *     status. It pads edges wherever required.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */

int32_t xvCheckTileReadyMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile)
{
  int32_t loopInd, index, retVal, dmaIndex, doneCount;
  uint32_t statusFlag, tileHeight;
  xvTile *pTile1;
  xvFrame *pFrame;
  int32_t frameWidth, frameHeight, framePitch, tileWidth, tilePitch, tileDMApendingCount;
  int32_t x1, y1, x2, y2, copyRowBytes, copyHeight;
  uint8_t framePadLeft, framePadRight, framePadTop, framePadBottom;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
  int32_t extraEdgeTop, extraEdgeBottom, extraEdgeLeft, extraEdgeRight;
  int32_t pixWidth;
  uint8_t *srcPtr, *dstPtr;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;

  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_TILE(pTile, pxvTM);
  if (pTile->status == 0u)
  {
    return(1);
  }

  doneCount           = 0;
  index               = pxvTM->tileDMAstartIndex[dmaChannel];
  tileDMApendingCount = pxvTM->tileDMApendingCount[dmaChannel];
  for (loopInd = 0; loopInd < tileDMApendingCount; loopInd++)
  {
    index    = ((int32_t) pxvTM->tileDMAstartIndex[dmaChannel] + (int32_t) loopInd) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
    pTile1   = pxvTM->tileProcQueue[dmaChannel][index % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH)];
    dmaIndex = pTile1->dmaIndex;
    if ((pTile1->status & XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED) == 0u)
    {
      retVal = xvCheckForIdmaIndexMultiChannel(dmaChannel, pxvTM, dmaIndex);
      if (retVal == 1)
      {
        statusFlag = pTile1->status;
        statusFlag = statusFlag & ~XV_TILE_STATUS_DMA_ONGOING;

        pFrame         = pTile1->pFrame;
        pixWidth       = (int32_t) pFrame->pixelRes * (int32_t) pFrame->numChannels;
        frameWidth     = pFrame->frameWidth;
        frameHeight    = pFrame->frameHeight;
        framePitch     = pFrame->framePitch * ((int32_t) pFrame->pixelRes);
        framePadLeft   = pFrame->leftEdgePadWidth;
        framePadRight  = pFrame->rightEdgePadWidth;
        framePadTop    = pFrame->topEdgePadHeight;
        framePadBottom = pFrame->bottomEdgePadHeight;

        tileWidth      = pTile1->width;
        tileHeight     = pTile1->height;
        tilePitch      = pTile1->pitch * ((int32_t) pFrame->pixelRes);
        tileEdgeLeft   = pTile1->tileEdgeLeft;
        tileEdgeRight  = pTile1->tileEdgeRight;
        tileEdgeTop    = pTile1->tileEdgeTop;
        tileEdgeBottom = pTile1->tileEdgeBottom;

        if ((statusFlag & XV_TILE_STATUS_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED) != 0u)
          {
            y1           = pTile1->y - (int32_t) tileEdgeTop;
            extraEdgeTop = -((int32_t) framePadTop) - y1;
            dstPtr       = &((uint8_t *) pTile1->pData)[-(((int32_t) tileEdgeTop * tilePitch) + ((int32_t) tileEdgeLeft * (int32_t) pixWidth))];
            srcPtr       = &dstPtr[(extraEdgeTop * tilePitch)];
            copyRowBytes = (tileEdgeLeft + tileWidth + tileEdgeRight) * XV_TYPE_ELEMENT_SIZE(pTile1->type);
            xvCopyBufferEdgeDataH(srcPtr, dstPtr, copyRowBytes, (int32_t) pixWidth, extraEdgeTop, tilePitch, pFrame->paddingType, pFrame->paddingVal);
          }

          if ((statusFlag & XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED) != 0u)
          {
            y2              = pTile1->y + ((int32_t) tileHeight - 1) + ((int32_t) tileEdgeBottom);
            extraEdgeBottom = y2 - (((int32_t) frameHeight - 1) + (int32_t) framePadBottom);
            y2              = ((int32_t) frameHeight - 1) + (int32_t) framePadBottom;
            dstPtr          = &((uint8_t *) pTile1->pData)[(((y2 - pTile1->y) + 1) * (int32_t) tilePitch) - ((int32_t) tileEdgeLeft * (int32_t) pixWidth)];
            srcPtr          = &dstPtr[-tilePitch];
            copyRowBytes    = (tileEdgeLeft + tileWidth + tileEdgeRight) * XV_TYPE_ELEMENT_SIZE(pTile1->type);
            xvCopyBufferEdgeDataH(srcPtr, dstPtr, copyRowBytes, (int32_t) pixWidth, extraEdgeBottom, tilePitch, pFrame->paddingType, pFrame->paddingVal);
          }

          if ((statusFlag & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
          {
            x1            = pTile1->x - ((int32_t) tileEdgeLeft);
            extraEdgeLeft = (-(int32_t) framePadLeft) - x1;
            dstPtr        = &((uint8_t *) pTile1->pData)[-(((int32_t) tileEdgeTop * tilePitch) + ((int32_t) tileEdgeLeft * (int32_t) pixWidth))];
            srcPtr        = &dstPtr[(extraEdgeLeft * pixWidth)];
            copyHeight    = ((int32_t) tileEdgeTop) + ((int32_t) tileHeight) + ((int32_t) tileEdgeBottom);
            xvCopyBufferEdgeDataV(srcPtr, dstPtr, extraEdgeLeft, (int32_t) pixWidth, copyHeight, tilePitch, pFrame->paddingType, pFrame->paddingVal);
          }

          if ((statusFlag & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
          {
            x2             = pTile1->x + (tileWidth - 1) + ((int32_t) tileEdgeRight);
            extraEdgeRight = x2 - (((int32_t) frameWidth - 1) + (int32_t) framePadRight);
            x2             = (((int32_t) frameWidth) - 1) + ((int32_t) framePadRight);
            dstPtr         = &((uint8_t *) pTile1->pData)[-(((int32_t) tileEdgeTop) * tilePitch) + (((x2 - pTile1->x) + 1) * ((int32_t) pixWidth))];
            srcPtr         = &dstPtr[-((int32_t) pixWidth)];
            copyHeight     = ((int32_t) tileEdgeTop) + ((int32_t) tileHeight) + ((int32_t) tileEdgeBottom);
            xvCopyBufferEdgeDataV(srcPtr, dstPtr, extraEdgeRight, (int32_t) pixWidth, copyHeight, tilePitch, pFrame->paddingType, pFrame->paddingVal);
          }
        }

        statusFlag = statusFlag & ~XV_TILE_STATUS_EDGE_PADDING_NEEDED;

        if ((pTile1->pPrevTile) != NULL)
        {
          pTile1->pPrevTile->reuseCount--;
          pTile1->pPrevTile = NULL;
        }

        pTile1->status = statusFlag;
        doneCount++;
      }
      else       // DMA not done for this tile
      {
        //Need to break the loop but MISRA-C does not allow more than 1
        //break statement. So just do the clean up and return. Also, note that
        // here return will always be 0 i.e. tile not ready.
        pxvTM->tileDMAstartIndex[dmaChannel]   = index;
        pxvTM->tileDMApendingCount[dmaChannel] = pxvTM->tileDMApendingCount[dmaChannel] - doneCount;
        return(0);
      }
    }
    else
    {
      // Tile is not part of frame. Make everything constant
      pFrame         = pTile1->pFrame;
      tileWidth      = pTile1->width;
      tileHeight     = pTile1->height;
      pixWidth       = (int32_t) pFrame->pixelRes * (int32_t) pFrame->numChannels;
      tilePitch      = pTile1->pitch * ((int32_t) pFrame->pixelRes);
      tileEdgeLeft   = pTile1->tileEdgeLeft;
      tileEdgeRight  = pTile1->tileEdgeRight;
      tileEdgeTop    = pTile1->tileEdgeTop;
      tileEdgeBottom = pTile1->tileEdgeBottom;

      dstPtr       = &((uint8_t *) pTile1->pData) [-((((int32_t) tileEdgeTop * tilePitch) + ((int32_t) tileEdgeLeft * (int32_t) pixWidth)))];
      srcPtr       = NULL;
      copyRowBytes = (tileEdgeLeft + tileWidth + tileEdgeRight) * XV_TYPE_ELEMENT_SIZE(pTile1->type);
      copyHeight   = ((int32_t) tileHeight) + ((int32_t) tileEdgeTop) + ((int32_t) tileEdgeBottom);
      if (pFrame->paddingType != FRAME_EDGE_PADDING)
      {
        xvCopyBufferEdgeDataH(srcPtr, dstPtr, copyRowBytes, (int32_t) pixWidth, copyHeight, tilePitch, pFrame->paddingType, pFrame->paddingVal);
      }
      pTile1->status = 0;
      doneCount++;
    }

    // Break if we reached the required tile
    if (pTile1 == pTile)
    {
      index = (index + 1) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
      break;
    }
  }

  pxvTM->tileDMAstartIndex[dmaChannel]   = index;
  pxvTM->tileDMApendingCount[dmaChannel] = pxvTM->tileDMApendingCount[dmaChannel] - doneCount;
  return((pTile->status == 0u) ? 1 : 0);
}

/**********************************************************************************
 * FUNCTION: xvCheckTileReady()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer using idma channel 0  for given tile is completed.
 *     It checks all the tile in the transfer request buffer
 *     before the given tile and updates their respective
 *     status. It pads edges wherever required.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvCheckTileReady(xvTileManager *pxvTM, xvTile const *pTile)
{
  return(xvCheckTileReadyMultiChannel(TM_IDMA_CH0, pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvPadEdges()
 *
 * DESCRIPTION:
 *     Pads edges of the given 8b tile. If FRAME_EDGE_PADDING mode is used,
 *     padding is done using edge values of the frame else if
 *     FRAME_CONSTANT_PADDING or FRAME_ZERO_PADDING mode is used,
 *     constant or zero value is padded
 *    xvPadEdges should be used with Fast functions
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvPadEdges(xvTileManager *pxvTM, xvTile *pTile)
{
  int32_t x1, x2, y1, y2, indy, copyHeight, copyRowBytes, wb, padVal;
  int32_t tileWidth, tilePitch, frameWidth, frameHeight;
  uint32_t tileHeight;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
  int32_t extraEdgeLeft, extraEdgeRight, extraEdgeTop=0, extraEdgeBottom=0;
  uint16_t * __restrict srcPtr_16b, * __restrict dstPtr_16b;
  uint32_t * __restrict srcPtr_32b, * __restrict dstPtr_32b;
  uint8_t * __restrict srcPtr, * __restrict dstPtr;
  xvFrame *pFrame;
  xb_vecNx16 vec1, * __restrict pvecDst;
  xb_vec2Nx8U dvec1, * __restrict pdvecDst;
  xb_vecN_2x32Uv hvec1, * __restrict phvecDst;
  valign vas1;
  valign ald1;
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_TILE(pTile, pxvTM);
  if ((pTile->status & XV_TILE_STATUS_EDGE_PADDING_NEEDED) != 0u)
  {
    pFrame = pTile->pFrame;
    XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
    XV_CHECK_ERROR(((pFrame->pFrameBuff == 0u) || (pFrame->pFrameData == 0u)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

    int32_t channel     = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
    int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
    int32_t bytePerPel;
    bytePerPel = bytesPerPix / channel;

    XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
    XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
    XV_CHECK_ERROR((channel > 4), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

    tileEdgeTop    = pTile->tileEdgeTop;
    tileEdgeBottom = pTile->tileEdgeBottom;
    tileEdgeLeft   = pTile->tileEdgeLeft;
    tileEdgeRight  = pTile->tileEdgeRight;
    tilePitch      = pTile->pitch;
    tileHeight     = pTile->height;
    tileWidth      = pTile->width;
    frameWidth     = pFrame->frameWidth;
    frameHeight    = pFrame->frameHeight;

    if (pFrame->paddingType == FRAME_EDGE_PADDING)
    {
        if ((pTile->status & XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED) != 0u)
        {
            y1 = pTile->y - (int32_t)tileEdgeTop;
            if (y1 > frameHeight)
            {
                y1 = frameHeight;
            }
            if (y1 < 0)
            {
                extraEdgeTop = -y1;
                y1 = 0;
            }

            srcPtr = &((uint8_t*)pTile->pData)[-((((int32_t)tileEdgeTop - (int32_t)extraEdgeTop) * tilePitch) * (int32_t)bytePerPel) - ((int32_t)tileEdgeLeft * (int32_t)bytesPerPix)];
            dstPtr = &srcPtr[-(extraEdgeTop * tilePitch) * (int32_t)bytePerPel];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;
            pdvecDst = (xb_vec2Nx8U*)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeTop; indy++)
            {
		        xb_vec2Nx8U* __restrict srcPtr1 = (xb_vec2Nx8U*)srcPtr;
                for (wb = 0; wb < copyRowBytes; wb += (2 * IVP_SIMD_WIDTH))
                {
                    xb_vec2Nx8U vec;
                    int32_t offset = XT_MIN(copyRowBytes - wb, 2 * IVP_SIMD_WIDTH);
                    ald1 = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(vec, ald1, srcPtr1, offset);
                    IVP_SAV2NX8U_XP(vec, vas1, pdvecDst, offset);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U*)dstPtr;
            }
            pTile->status = pTile->status & ~XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED;
        }
        if ((pTile->status & XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED) != 0u)
        {
            y2 = ((int32_t)pTile->y) + (((int32_t)tileHeight) - 1) + (((int32_t)tileEdgeBottom));
            if (y2 < 0)
            {
                y2 = -1;
            }
            if (y2 > (frameHeight - 1))
            {
                extraEdgeBottom = (y2 - frameHeight) + 1;
                y2 = frameHeight - 1;
            }
            srcPtr = &((uint8_t*)pTile->pData)[-((int32_t)tileEdgeLeft * (int32_t)bytesPerPix) + ((((int32_t)tileHeight + (int32_t)tileEdgeBottom) - extraEdgeBottom - 1) * (int32_t)tilePitch* bytePerPel)];
            dstPtr = &srcPtr[tilePitch * (int32_t)bytePerPel];
            copyRowBytes = ((int32_t)tileEdgeLeft + (int32_t)tileWidth + (int32_t)tileEdgeRight) * (int32_t)bytesPerPix;
            pdvecDst = (xb_vec2Nx8U*)dstPtr;
            vas1 = IVP_ZALIGN();
            for (indy = 0; indy < extraEdgeBottom; indy++)
            {
		        xb_vec2Nx8U* __restrict srcPtr1 = (xb_vec2Nx8U*)srcPtr;
                for (wb = 0; wb < copyRowBytes; wb += (2 * IVP_SIMD_WIDTH))
                {
                    xb_vec2Nx8U vec;
                    int32_t offset = XT_MIN(copyRowBytes - wb, 2 * IVP_SIMD_WIDTH);
                    ald1 = IVP_LA2NX8U_PP(srcPtr1);
                    IVP_LAV2NX8U_XP(vec, ald1, srcPtr1, offset);
                    IVP_SAV2NX8U_XP(vec, vas1, pdvecDst, offset);
                }
                IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
                dstPtr = &dstPtr[tilePitch * bytePerPel];
                srcPtr = &srcPtr[tilePitch * bytePerPel];
                pdvecDst = (xb_vec2Nx8U*)dstPtr;
            }
            pTile->status = pTile->status & ~XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED;
        }

        xb_vecNx16 vs16_sel;
        if (1 == bytePerPel || 2 == bytePerPel)
        {
            switch (channel)
            {
                case 2:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(2));
                    break;
                }
                case 3:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(3));
                    break;
                }
                case 4:
                {
                    vs16_sel = IVP_REMNX16(IVP_SEQNX16(), IVP_MOVVA16(4));
                    break;
                }
                default:
                {
                    XV_CHECK_ERROR((channel > 4), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
                }
            }
        }
      if (bytePerPel == 1)
      {
        xb_vec2Nx8 vs8_sel;
        xb_vec2Nx8U srcVec;

        if (channel == 1)
        {
          vs8_sel = IVP_ZERO2NX8();
        }
        else
        {
          vs8_sel = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(vs16_sel), IVP_MOV2NX8_FROMNX16(vs16_sel), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
        }
        if ((pTile->status & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
        {
          extraEdgeLeft = ((int32_t) tileEdgeLeft) - pTile->x;
          dstPtr        = &((uint8_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) tilePitch) + (int32_t) tileEdgeLeft * channel)];
          srcPtr        = &dstPtr[extraEdgeLeft * channel];
          copyHeight    = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vec2Nx8U *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            pdvecDst = (xb_vec2Nx8U *) dstPtr;
            srcPtr1  = (xb_vec2Nx8U*)srcPtr;
            ald1     = IVP_LA2NX8U_PP(srcPtr1);
            IVP_LAV2NX8U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
            dvec1 = IVP_SHFL2NX8U(srcVec, vs8_sel);
            IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, extraEdgeLeft * bytesPerPix);
            IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
            dstPtr = &dstPtr[tilePitch];
            srcPtr = &srcPtr[tilePitch];
          }
        }

        if ((pTile->status & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
        {
          x2             = pTile->x + ((int32_t) tileWidth - 1) + (int32_t) tileEdgeRight;
          extraEdgeRight = x2 - (frameWidth - 1);
          dstPtr         = &((uint8_t *) pTile->pData)[-((int32_t) tileEdgeTop * (int32_t) tilePitch) + (((int32_t) tileWidth + (int32_t) tileEdgeRight) - extraEdgeRight) * channel];
          srcPtr         = &dstPtr[-channel];
          copyHeight     = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vec2Nx8U *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            srcPtr1 = (xb_vec2Nx8U*)srcPtr;
            pdvecDst = (xb_vec2Nx8U *) dstPtr;
            ald1 = IVP_LA2NX8U_PP(srcPtr1);
            IVP_LAV2NX8U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
            dvec1 = IVP_SHFL2NX8U(srcVec, vs8_sel);
            IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, extraEdgeRight * bytesPerPix);
            IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
            dstPtr = &dstPtr[tilePitch];
            srcPtr = &srcPtr[tilePitch];
          }
        }
      }
      else if (bytePerPel == 2)
      {
        xb_vecNx16U srcVec;
        if (channel == 1)
        {
          vs16_sel = IVP_ZERONX16();
        }
        if ((pTile->status & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
        {
          extraEdgeLeft = (int32_t) tileEdgeLeft - pTile->x;
          dstPtr_16b    = &((uint16_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) pTile->pitch) + (int32_t) tileEdgeLeft * channel)]; // No need of multiplying by 2
          srcPtr_16b    = &dstPtr_16b[extraEdgeLeft * channel];                                                                                  // No need of multiplying by 2 as pointers are uint16_t *
          copyHeight    = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vecNx16 *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            srcPtr1 = (xb_vecNx16 *)srcPtr_16b;
            pvecDst = (xb_vecNx16 *) dstPtr_16b;
            ald1 = IVP_LANX16U_PP(srcPtr1);
            IVP_LAVNX16U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
            vec1 = IVP_SHFLNX16(srcVec, vs16_sel);
            IVP_SAVNX16_XP(vec1, vas1, pvecDst, extraEdgeLeft * bytesPerPix);
            IVP_SAPOSNX16_FP(vas1, pvecDst);
            dstPtr_16b = &dstPtr_16b[pTile->pitch];
            srcPtr_16b = &srcPtr_16b[pTile->pitch];
          }
        }

        if ((pTile->status & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
        {
          x2             = pTile->x + (int32_t) tileWidth + (int32_t) tileEdgeRight;
          extraEdgeRight = x2 - frameWidth;
          dstPtr_16b     = &((uint16_t *) pTile->pData)[-((int32_t) tileEdgeTop * (int32_t) pTile->pitch) + (((int32_t) tileWidth + (int32_t) tileEdgeRight) - extraEdgeRight) * channel];
          srcPtr_16b     = &dstPtr_16b[-channel];
          copyHeight     = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vecNx16 *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            srcPtr1 = (xb_vecNx16 *)srcPtr_16b;
            pvecDst = (xb_vecNx16 *) dstPtr_16b;
            ald1 = IVP_LANX16U_PP(srcPtr1);
            IVP_LAVNX16U_XP(srcVec, ald1, srcPtr1, bytesPerPix);
            vec1 = IVP_SHFLNX16(srcVec, vs16_sel);
            IVP_SAVNX16_XP(vec1, vas1, pvecDst, extraEdgeRight * bytesPerPix);
            IVP_SAPOSNX16_FP(vas1, pvecDst);
            dstPtr_16b = &dstPtr_16b[pTile->pitch];
            srcPtr_16b = &srcPtr_16b[pTile->pitch];
          }
        }
      }
      else
      {
        if ((pTile->status & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
        {
          extraEdgeLeft = (int32_t) tileEdgeLeft - pTile->x;
          dstPtr_32b    = &((uint32_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) pTile->pitch) + (int32_t) tileEdgeLeft * channel)]; // No need of multiplying by 2
          srcPtr_32b    = &dstPtr_32b[channel];                                                                                  // No need of multiplying by 2 as pointers are uint32_t *
          copyHeight    = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vecN_2x32Uv *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            srcPtr1 = (xb_vecN_2x32Uv *)srcPtr_32b;
            phvecDst = (xb_vecN_2x32Uv *) dstPtr_32b;
            ald1 = IVP_LAN_2X32U_PP(srcPtr1);
            IVP_LAVN_2X32U_XP(hvec1, ald1, srcPtr1, extraEdgeLeft * bytesPerPix);
            IVP_SAVN_2X32_XP(hvec1, vas1, phvecDst, extraEdgeLeft * bytesPerPix);
            IVP_SAPOSN_2X32_FP(vas1, (xb_vecN_2x32v *) phvecDst);
            dstPtr_32b = &dstPtr_32b[pTile->pitch];
            srcPtr_32b = &srcPtr_32b[pTile->pitch];
          }
        }

        if ((pTile->status & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
        {
          x2             = (int32_t) pTile->x + (int32_t) tileWidth + (int32_t) tileEdgeRight;
          extraEdgeRight = x2 - frameWidth;
          dstPtr_32b     = &((uint32_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) pTile->pitch)) + (((int32_t) tileWidth + (int32_t) tileEdgeRight) - extraEdgeRight) * channel];
          srcPtr_32b     = &dstPtr_32b[-channel];
          copyHeight     = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          vas1 = IVP_ZALIGN();
          xb_vecN_2x32Uv *__restrict srcPtr1;
          for (indy = 0; indy < copyHeight; indy++)
          {
            srcPtr1 = (xb_vecN_2x32Uv *)srcPtr_32b;
            phvecDst = (xb_vecN_2x32Uv *) dstPtr_32b;
            ald1 = IVP_LAN_2X32U_PP(srcPtr1);
            IVP_LAVN_2X32U_XP(hvec1, ald1, srcPtr1, extraEdgeRight * bytesPerPix);
            IVP_SAVN_2X32_XP(hvec1, vas1, phvecDst, extraEdgeRight * bytesPerPix);
            IVP_SAPOSN_2X32_FP(vas1, (xb_vecN_2x32v *) phvecDst);
            dstPtr_32b = &dstPtr_32b[pTile->pitch];
            srcPtr_32b = &srcPtr_32b[pTile->pitch];
          }
        }
      }
    }
    else if(pFrame->paddingType == FRAME_PADDING_REFLECT_101)
    {
        if (bytePerPel == 1)
    	{
    		xvExtendEdgesReflect101_I8(pTile, frameWidth, frameHeight);

    	}
        else if (bytePerPel == 2)
    	{
    		xvExtendEdgesReflect101_I16(pTile, frameWidth, frameHeight);
    	}
		else
		{
			//default comment for MISRA-C
		}
    }
    else
    {
      padVal = 0;
      if (pFrame->paddingType == FRAME_CONSTANT_PADDING)
      {
        padVal = (int32_t) pFrame->paddingVal;
      }
      dvec1 = padVal;
      if (bytePerPel == 1)
      {
        dvec1 = padVal;
      }
      else if (bytePerPel == 2)
      {
        xb_vecNx16U vec = padVal;
        dvec1 = IVP_MOV2NX8U_FROMNX16(vec);
      }
      else
      {
        xb_vecN_2x32Uv hvec = padVal;
        dvec1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32U(hvec));
      }

      if ((pTile->status & XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED) == 0u)
      {
        if ((pTile->status & XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED) != 0u)
        {
          y1           = pTile->y - (int32_t) tileEdgeTop;
          extraEdgeTop = -y1;
          dstPtr       = &((uint8_t *) pTile->pData)[ -((int32_t) tileEdgeTop * (int32_t) tilePitch * (int32_t) bytePerPel) - (int32_t) tileEdgeLeft * (int32_t) bytesPerPix];
          copyRowBytes = ((int32_t) tileEdgeLeft + (int32_t) tileWidth + (int32_t) tileEdgeRight) * (int32_t) bytesPerPix;

          pdvecDst = (xb_vec2Nx8U *) dstPtr;
          vas1     = IVP_ZALIGN();
          for (indy = 0; indy < extraEdgeTop; indy++)
          {
            for (wb = copyRowBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
            {
              IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
            }
            IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
            dstPtr   = &dstPtr[tilePitch * bytePerPel];
            pdvecDst = (xb_vec2Nx8U *) dstPtr;
          }
        }

        if ((pTile->status & XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED) != 0u)
        {
          y2              = (int32_t) pTile->y + (int32_t) tileHeight + (int32_t) tileEdgeBottom;
          extraEdgeBottom = y2 - (int32_t) frameHeight;
          dstPtr          = &((uint8_t *) pTile->pData)[(((frameHeight - pTile->y) * (int32_t) tilePitch * (int32_t) bytePerPel) - (int32_t) tileEdgeLeft * (int32_t) bytesPerPix)];
          copyRowBytes    = ((int32_t) tileEdgeLeft + (int32_t) tileWidth + (int32_t) tileEdgeRight) * (int32_t) bytesPerPix;

          pdvecDst = (xb_vec2Nx8U *) dstPtr;
          vas1     = IVP_ZALIGN();
          for (indy = 0; indy < extraEdgeBottom; indy++)
          {
            for (wb = copyRowBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
            {
              IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
            }
            IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
            dstPtr   = &dstPtr[tilePitch * bytePerPel];
            pdvecDst = (xb_vec2Nx8U *) dstPtr;
          }
        }
        if ((pTile->status & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
        {
          x1            = pTile->x - (int32_t) tileEdgeLeft;
          extraEdgeLeft = -x1;
          dstPtr        = &((uint8_t *) pTile->pData)[ -(((int32_t) tileEdgeTop * (int32_t) tilePitch * (int32_t) bytePerPel) + (int32_t) tileEdgeLeft * (int32_t) bytesPerPix)];
          copyHeight    = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          pdvecDst      = (xb_vec2Nx8U *) dstPtr;
          vas1          = IVP_ZALIGN();
          for (indy = 0; indy < copyHeight; indy++)
          {
            for (wb = extraEdgeLeft * bytesPerPix; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
            {
              IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
            }
              IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
              dstPtr   = &dstPtr[tilePitch * bytePerPel];
              pdvecDst = (xb_vec2Nx8U *) dstPtr;
          }
        }

        if ((pTile->status & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
        {
          x2             = pTile->x + ((int32_t) tileWidth - 1) + (int32_t) tileEdgeRight;
          extraEdgeRight = x2 - (frameWidth - 1);
          x2             = frameWidth - 1;
          dstPtr         = &((uint8_t *) pTile->pData)[ -(((int32_t) tileEdgeTop) * ((int32_t) tilePitch) * (int32_t) bytePerPel) + ((((int32_t) tileWidth + ((int32_t) tileEdgeRight)) - extraEdgeRight) * ((int32_t) bytesPerPix))];
          copyHeight     = (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom;

          pdvecDst       = (xb_vec2Nx8U *) dstPtr;
          vas1           = IVP_ZALIGN();
          for (indy = 0; indy < copyHeight; indy++)
          {
            for (wb = extraEdgeRight * bytesPerPix; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
            {
              IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
            }
            IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
            dstPtr   = &dstPtr[tilePitch * bytePerPel];
            pdvecDst = (xb_vec2Nx8U *) dstPtr;
          }
        }
      }
      else
      {
        // Tile is not part of frame. Make it constant
        dstPtr       = &((uint8_t *) pTile->pData)[-((((int32_t) tileEdgeTop * (int32_t) tilePitch) + (int32_t) tileEdgeLeft) * (int32_t) bytesPerPix)];
        copyHeight   = (int32_t) tileHeight + (int32_t) tileEdgeTop + (int32_t) tileEdgeBottom;
        copyRowBytes = ((int32_t) tileEdgeLeft + (int32_t) tileWidth + (int32_t) tileEdgeRight) * (int32_t) bytesPerPix;
        xvCopyBufferEdgeDataH(NULL, dstPtr, copyRowBytes, bytesPerPix, copyHeight, (tilePitch * bytesPerPix), pFrame->paddingType, pFrame->paddingVal);
      }
    }

    pTile->status = pTile->status & ~(XV_TILE_STATUS_EDGE_PADDING_NEEDED | XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED);
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvPadEdges16()
 *
 * DESCRIPTION:
 *     Pads edges of the given 16b tile. If FRAME_EDGE_PADDING mode is used,
 *     padding is done using edge values of the frame else if
 *     FRAME_CONSTANT_PADDING or FRAME_ZERO_PADDING mode is used,
 *     constant or zero value is padded
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
// xvPadEdges should be used with Fast functions
int32_t xvPadEdges16(xvTileManager *pxvTM, xvTile *pTile)
{
  return(xvPadEdges(pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvCheckInputTileFree()
 *
 * DESCRIPTION:
 *     A tile is said to be free if all data transfers pertaining to data resue
 *     from this tile is completed
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if input tile is free and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */

int32_t xvCheckInputTileFree(xvTileManager *pxvTM, xvTile const *pTile)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_TILE(pTile, pxvTM);
  if (pTile->reuseCount == 0)
  {
    return(1);
  }
  else
  {
    return(0);
  }
}

/**********************************************************************************
 * FUNCTION: xvWaitForTileMultiChannel()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTileMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_TILE(pTile, pxvTM);
  int32_t status = 0;
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    status = xvCheckTileReadyMultiChannel(dmaChannel, pxvTM, pTile);
  }
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvWaitForTile()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given tile  of iDMA channel 0 is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTile(xvTileManager *pxvTM, xvTile const *pTile)
{
  return(xvWaitForTileMultiChannel(TM_IDMA_CH0, pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileMultiChannel()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile)
{
  return g_symbol_tray->tray_xvSleepForTileMultiChannel(dmaChannel, pxvTM, pTile);
}

/**********************************************************************************
 * FUNCTION: xvSleepForTile()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile of iDMA channel 0  is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTile(xvTileManager *pxvTM, xvTile const *pTile)
{
  return(xvSleepForTileMultiChannel(TM_IDMA_CH0, pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvWaitForiDMAMultiChannel()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given DMA index is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint32_t      dmaIndex                 DMA index
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForiDMAMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex)
{
 return g_symbol_tray->tray_xvWaitForiDMAMultiChannel(dmaChannel, pxvTM, dmaIndex);
}

/**********************************************************************************
 * FUNCTION: xvWaitForiDMA()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given DMA index  of iDMA channel 0 is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint32_t      dmaIndex                 DMA index
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForiDMA(xvTileManager *pxvTM, uint32_t dmaIndex)
{
  return(xvWaitForiDMAMultiChannel(TM_IDMA_CH0, pxvTM, dmaIndex));
}

/**********************************************************************************
 * FUNCTION: xvSleepForiDMAMultiChannel()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given DMA index is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel                       iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint32_t      dmaIndex                 DMA index
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete.
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForiDMAMultiChannel(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex)
{
  return g_symbol_tray->tray_xvSleepForiDMAMultiChannel(dmaChannel, pxvTM, dmaIndex);
}

/**********************************************************************************
 * FUNCTION: xvSleepForiDMA()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given DMA index is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     uint32_t      dmaIndex                 DMA index
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForiDMA(xvTileManager *pxvTM, uint32_t dmaIndex)
{
  return(xvSleepForiDMAMultiChannel(TM_IDMA_CH0, pxvTM, dmaIndex));
}

/**********************************************************************************
 * FUNCTION: xvWaitForTileFastMultiChannel()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns error flag if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTileFastMultiChannel(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile  *pTile)
{
  return g_symbol_tray->tray_xvWaitForTileFastMultiChannel(dmaChannel, pxvTM, pTile);
}

/**********************************************************************************
 * FUNCTION: xvWaitForTileFast()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTileFast(xvTileManager const *pxvTM, xvTile  *pTile)
{
  return(xvWaitForTileFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileFastMultiChannel()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete.
 *     Returns error flag  if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileFastMultiChannel(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile)
{
  return g_symbol_tray->tray_xvSleepForTileFastMultiChannel(dmaChannel, pxvTM, pTile);
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileFast()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile (iDMA channel 0) is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns error flag  if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileFast(xvTileManager const *pxvTM, xvTile *pTile)
{
  return(xvSleepForTileFastMultiChannel(TM_IDMA_CH0, pxvTM, pTile));
}

/**********************************************************************************
 * FUNCTION: xvGetErrorInfo()
 *
 * DESCRIPTION:
 *
 *     Prints the most recent error information.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *
 * OUTPUTS:
 *     It returns the most recent error code.
 *
 ********************************************************************************** */

xvError_t xvGetErrorInfo(xvTileManager const *pxvTM)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XV_ERROR_TILE_MANAGER_NULL, "NULL TM Pointer");
  return(pxvTM->errFlag);
}

/**********************************************************************************
 * FUNCTION: xvAllocateFrame3D()
 *
 * DESCRIPTION:
 *     Allocates single 3D frame. It does not allocate buffer required for 3Dframe data.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns the pointer to allocated 3D frame.
 *     Returns ((xvFrame3D *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate 3D frame data buffer.
 *
 ********************************************************************************** */

xvFrame3D *xvAllocateFrame3D(xvTileManager *pxvTM)
{
  uint32_t indx, indxArr = 0, indxShift = 0, allocFlags = 0;
  xvFrame3D *pFrame3D = NULL;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, ((xvFrame3D *) ((void *) (XVTM_ERROR))), "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;

  for (indx = 0; indx < MAX_NUM_FRAMES3D; indx++)
  {
    indxArr    = indx >> 5u;
    indxShift  = indx & 0x1Fu;
    allocFlags = pxvTM->frame3DAllocFlags[indxArr];
    if (((allocFlags >> indxShift) & 0x1u) == 0u)
    {
      break;
    }
  }

  if (indx < MAX_NUM_FRAMES3D)
  {
    pFrame3D                          = &(pxvTM->frame3DArray[indx]);
    pxvTM->frame3DAllocFlags[indxArr] = allocFlags | (((uint32_t) 0x1u) << indxShift);
    pxvTM->frame3DCount++;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_FRAME_BUFFER_FULL;
    return((xvFrame3D *) ((void *) XVTM_ERROR));
  }

  return(pFrame3D);
}

/**********************************************************************************
 * FUNCTION: xvFreeFrame3D()
 *
 * DESCRIPTION:
 *     Releases the given 3D frame. Does not release associated frame data buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     xvFrame3D       *pFrame3D    3D Frame that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeFrame3D(xvTileManager *pxvTM, xvFrame3D const *pFrame3D)
{
  uint32_t indx, indxArr, indxShift, allocFlags;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(pFrame3D == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

  for (indx = 0; indx < MAX_NUM_FRAMES3D; indx++)
  {
    if (&(pxvTM->frame3DArray[indx]) == pFrame3D)
    {
      break;
    }
  }

  if (indx < MAX_NUM_FRAMES3D)
  {
    indxArr                           = indx >> 5u;
    indxShift                         = indx & 0x1Fu;
    allocFlags                        = pxvTM->frame3DAllocFlags[indxArr];
    pxvTM->frame3DAllocFlags[indxArr] = allocFlags & ~(0x1u << indxShift);
    pxvTM->frame3DCount--;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_BAD_ARG;
    return(XVTM_ERROR);
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvFreeAllFrames3D()
 *
 * DESCRIPTION:
 *     Releases all allocated 3D frames.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeAllFrames3D(xvTileManager *pxvTM)
{
  uint32_t index;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pxvTM->frame3DCount = 0;
  for (index = 0; index < ((MAX_NUM_FRAMES3D + 31u) / 32u); index++)
  {
    pxvTM->frame3DAllocFlags[index] = 0x00000000;
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvAllocateTile3D()
 *
 * DESCRIPTION:
 *     Allocates single 3D tile. It does not allocate buffer required for tile data.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns the pointer to allocated tile.
 *     Returns ((xvTile *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate tile data buffer
 *
 ********************************************************************************** */

xvTile3D *xvAllocateTile3D(xvTileManager *pxvTM)
{
  uint32_t indx, indxArr = 0, indxShift = 0, allocFlags = 0;
  xvTile3D *pTile3D = NULL;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, ((xvTile3D *) ((void *) XVTM_ERROR)), "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  for (indx = 0; indx < MAX_NUM_TILES3D; indx++)
  {
    indxArr    = indx >> 5u;
    indxShift  = indx & 0x1Fu;
    allocFlags = pxvTM->tile3DAllocFlags[indxArr];
    if (((allocFlags >> indxShift) & 0x1u) == 0u)
    {
      break;
    }
  }

  if (indx < MAX_NUM_TILES3D)
  {
    pTile3D                          = &(pxvTM->tile3DArray[indx]);
    pxvTM->tile3DAllocFlags[indxArr] = allocFlags | (((uint32_t) 0x1u) << indxShift);
    pxvTM->tile3DCount++;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_TILE_BUFFER_FULL;
    return((xvTile3D *) ((void *) XVTM_ERROR));
  }

  return(pTile3D);
}

/**********************************************************************************
 * FUNCTION: xvFreeTile3D()
 *
 * DESCRIPTION:
 *     Releases the given 3D tile. Does not release associated tile data buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     xvTile3D        *pTile3D      3D Tile that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvFreeTile3D(xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  uint32_t indx, indxArr, indxShift, allocFlags;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(pTile3D == NULL, pxvTM->errFlag = XV_ERROR_TILE_NULL, XVTM_ERROR, "NULL TM Pointer");

  for (indx = 0; indx < MAX_NUM_TILES3D; indx++)
  {
    if (&(pxvTM->tile3DArray[indx]) == pTile3D)
    {
      break;
    }
  }

  if (indx < MAX_NUM_TILES3D)
  {
    indxArr                          = indx >> 5u;
    indxShift                        = indx & 0x1Fu;
    allocFlags                       = pxvTM->tile3DAllocFlags[indxArr];
    pxvTM->tile3DAllocFlags[indxArr] = allocFlags & ~(0x1u << indxShift);
    pxvTM->tile3DCount--;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_BAD_ARG;
    return(XVTM_ERROR);
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvFreeAllTiles3D()
 *
 * DESCRIPTION:
 *     Releases all allocated 3D tiles.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */

int32_t xvFreeAllTiles3D(xvTileManager *pxvTM)
{
  uint32_t index;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pxvTM->tile3DCount = 0;
  for (index = 0; index < ((MAX_NUM_TILES3D + 31u) / 32u); index++)
  {
    pxvTM->tile3DAllocFlags[index] = 0x00000000;
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannel3D()
 *
 * DESCRIPTION:
 *     Requests data transfer from 3D frame present in system memory to local 3D tile memory.
 *
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager  *pxvTM                  Tile Manager object
 *     xvTile3D       *pTile3D                Destination 3D tile
 *     int32_t        interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferInMultiChannel3D(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferInMultiChannel3D(dmaChannel, pxvTM, pTile3D, interruptOnCompletion);
}

/**********************************************************************************
 * FUNCTION: xvCheckTileReadyMultiChannel3D()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer for given 3D tile is completed.
 *     It checks all the tile in the transfer request buffer
 *     before the given tile and updates their respective
 *     status. It pads edges wherever required.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager  *pxvTM                  Tile Manager object
 *     xvTile3D       *pTile3D                Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ***********************************************************************************/

int32_t xvCheckTileReadyMultiChannel3D(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  return g_symbol_tray->tray_xvCheckTileReadyMultiChannel3D(dmaChannel, pxvTM, pTile3D); //replaced by tray func.
}

/**********************************************************************************
 * FUNCTION: xvCreateFrame3D()
 *
 * DESCRIPTION:
 *     Allocates single 3D frame. It does not allocate buffer required for frame data.
 *     Initializes the frame elements
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     uint64_t      imgBuff         Pointer to iaura buffer
 *     uint32_t      frameBuffSize   Size of allocated iaura buffer
 *     int32_t       width           Width of iaura
 *     int32_t       height          Height of iaura
 *     int32_t       pitch           Row pitch of iaura
 *     int32_t       depth           depth of iaura
 *     int32_t       Frame2Dpitch    2D Frame pitch
 *     uint8_t       pixRes          Pixel resolution of iaura in bytes
 *     uint8_t       numChannels     Number of channels in the iaura
 *     uint8_t       paddingtype     Supported padding type
 *     uint32_t      paddingVal      Padding value if padding type is constant padding
 *
 * OUTPUTS:
 *     Returns the pointer to allocated 3D frame.
 *     Returns ((xvFrame3D *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate frame data buffer.
 *
 ********************************************************************************** */

xvFrame3D *xvCreateFrame3D(xvTileManager *pxvTM, uint64_t imgBuff,
                           uint32_t frameBuffSize, int32_t width, int32_t height, int32_t pitch,
                           int32_t depth, int32_t Frame2Dpitch, uint8_t pixRes, uint8_t numChannels,
                           uint8_t paddingType, uint32_t paddingVal)
{
  XV_CHECK_ERROR_NULL(((pxvTM == NULL) || (((void *) (uint32_t) imgBuff) == NULL)), ((xvFrame3D *) ((void *) XVTM_ERROR)), "NULL TM Pointer");
  XV_CHECK_ERROR(((width < 0) || (height < 0) || (pitch < 0) || (depth < 0) || (Frame2Dpitch < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR(((width * numChannels) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((Frame2Dpitch < (pitch * height)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((frameBuffSize < (((uint32_t) Frame2Dpitch) * ((uint32_t) depth) * pixRes)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR(((numChannels > MAX_NUM_CHANNEL) || (((int8_t) numChannels) <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((paddingType > FRAME_PADDING_MAX), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");

  xvFrame3D *pFrame3D = xvAllocateFrame3D(pxvTM);
  XV_CHECK_ERROR(((void *) pFrame3D == (void *) XVTM_ERROR), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvFrame3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");

  SETUP_FRAME_3D(pFrame3D, imgBuff, frameBuffSize, width, height, depth, pitch, Frame2Dpitch, 0, 0, 0, pixRes, numChannels, paddingType, paddingVal);
  return(pFrame3D);
}

/**********************************************************************************
 * FUNCTION: xvCreateTile3D()
 *
 * DESCRIPTION:
 *     Allocates single 3D tile and associated buffer data.
 *     Initializes the elements in tile
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     int32_t       tileBuffSize    Size of allocated tile buffer
 *     int32_t       width           Width of tile
 *     uint16_t      height          Height of tile
 *     uint16_t      depth           depth of tile
 *     int32_t       pitch           row pitch of tile
 *     int32_t       pitch2D         Pitch of 2D-tile
 *     uint16_t      edgeWidth       Edge width of tile
 *     uint16_t      edgeHeight      Edge height of tile
 *     uint16_t      edgeDepth       Edge depth of tile
 *     int32_t       color           Memory pool from which the buffer should be allocated
 *     xvFrame3D     *pFrame3D       3D Frame associated with the 3D tile
 *     uint16_t      tileType        Type of tile
 *     int32_t       alignType       Alignment tpye of tile. could be edge aligned of data aligned
 *
 * OUTPUTS:
 *     Returns the pointer to allocated 3D tile.
 *     Returns ((xvTile3D *)(XVTM_ERROR)) if it encounters an error.
 *
 ********************************************************************************** */

xvTile3D  *xvCreateTile3D(xvTileManager *pxvTM, int32_t tileBuffSize,
                          int32_t width, uint16_t height, uint16_t depth, int32_t pitch,
                          int32_t pitch2D, uint16_t edgeWidth, uint16_t edgeHeight,
                          uint16_t edgeDepth, int32_t color, xvFrame3D *pFrame3D,
                          uint16_t xvTileType, int32_t alignType)
{
  return g_symbol_tray->tray_xvCreateTile3D(pxvTM, tileBuffSize, width,  height, depth, pitch, pitch2D, edgeWidth, edgeHeight, \
                         edgeDepth, color, pFrame3D, xvTileType, alignType); //replaced by tray func.
}

/**********************************************************************************
 * FUNCTION: xvWaitForTileMultiChannel3D()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given 3D tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel                       iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile3D                 Input 3D tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTileMultiChannel3D(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, (XVTM_ERROR), "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  int32_t status = 0;
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    status = xvCheckTileReadyMultiChannel3D(dmaChannel, pxvTM, pTile3D);
    if (status == XVTM_ERROR)
    {
      return(status);
    }
  }
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileMultiChannel3D()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given 3D-tile is completed.
 *
 * INPUTS:
 *     int32_t         dmaChannel               iDMA channel
 *     xvTileManager   *pxvTM                   Tile Manager object
 *     xvTile3D        *pTile3D               3D-Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input 3D-tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileMultiChannel3D(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  return g_symbol_tray->tray_xvSleepForTileMultiChannel3D(dmaChannel, pxvTM, pTile3D);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannel3D()
 *
 * DESCRIPTION:
 *     Requests data transfer from 3D tile present in local memory to 3D frame in system memory.
 *
 * INPUTS:
 *     int32_t         dmaChannel              DMA channel number
 *     xvTileManager   *pxvTM                  Tile Manager object
 *     xvTile3D        *pTile3D                Source tile
 *     int32_t         interruptOnCompletion   If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutMultiChannel3D(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion)
{
  return g_symbol_tray->tray_xvReqTileTransferOutMultiChannel3D(dmaChannel, pxvTM, pTile3D, interruptOnCompletion);
}


typedef xb_vecNx16 vsaN;
#define XVTM_IVP_SCATTERNX16T(a, b, c, d) IVP_SCATTERNX16T((a), (short int *)(b), (c), (d))
#define XVTM_IVP_SCATTERNX8UT(a, b, c, d) IVP_SCATTERNX8UT((a), (unsigned char *)(b), (c), (d))
#define XVTM_IVP_GATHERANX16T_V(a, b, c, d) IVP_GATHERANX16T_V((const short int *)(a), (b), (c), (d))
#define XVTM_IVP_GATHERANX8UT_V(a, b, c, d) IVP_GATHERANX8UT_V((const unsigned char *)(a), (b), (c), (d))
#define XVTM_IVP_SAVNX16POS_FP IVP_SAPOSNX16_FP
#define XVTM_IVP_MOVVSV(vr,sa) (vr) // sa is always zero in XI, if not zero -> use IVP_MOVVSELNX16
#define XVTM_OFFSET_PTR_2NX8U(  ptr, nrows, stride, in_row_offset) ((xb_vec2Nx8U*)   ((uint8_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))
#define XVTM_OFFSET_PTR_NX8U(   ptr, nrows, stride, in_row_offset) ((xb_vecNx8U*)    ((uint8_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))
#define XVTM_OFFSET_PTR_NX16(   ptr, nrows, stride, in_row_offset) ((xb_vecNx16*)    ((int16_t*) (ptr)+(in_row_offset)+((nrows)*(stride))))

#define XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH  5
#define XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH 8
#define XVTM_IVP_SAV2NX8UPOS_FP IVP_SAPOS2NX8U_FP

static void xvGet2NX8SelVec(int32_t channel, int32_t wid, xb_vec2Nx8U &vu8_sel)
{
    xb_vec2Nx8U vu8_seq = IVP_SEQ2NX8U();
    xb_vecNx16 vs16_seq = IVP_SEQNX16();
    switch (channel)
    {
        case 1:
        {
            vu8_sel = IVP_SUB2NX8U(wid - 1, vu8_seq);
            break;
        }
        case 2:
        {
            vu8_sel = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, IVP_SLLI2NX8U(IVP_SRLI2NX8U(vu8_seq, 1), 1)), IVP_AND2NX8U(IVP_ADD2NX8U(vu8_seq, 1), 1));
            break;
        }
        case 3:
        {
            xb_vecNx16 vs16_thr       = IVP_MOVVA16(3);
            xb_vecNx16 vs16_rem_thr   = IVP_REMNX16(vs16_seq, vs16_thr);
            xb_vecNx16 vs16_div_thr   = IVP_QUONX16(vs16_seq, vs16_thr);
            xb_vec2Nx8U vu8_div_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_div_thr), IVP_MOV2NX8U_FROMNX16(vs16_div_thr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx24 vs24_mul_thr  = IVP_MULUS2NX8(3, vu8_div_thr);
            xb_vecNx16 vs16_mul_thr_l = IVP_CVT16U2NX24L(vs24_mul_thr);
            xb_vecNx16 vs16_mul_thr_h = IVP_CVT16U2NX24H(vs24_mul_thr);
            xb_vec2Nx8U vu8_mul_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_mul_thr_h), IVP_MOV2NX8U_FROMNX16(vs16_mul_thr_l), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx8U vu8_rem_thr   = IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vs16_rem_thr), IVP_MOV2NX8U_FROMNX16(vs16_rem_thr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            xb_vec2Nx8U vu8_sub       = IVP_SUB2NX8U(2, vu8_rem_thr);
            vu8_sel                   = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, vu8_mul_thr), vu8_sub);
            break;
        }
        case 4:
        {
            // a = [0,0,0,0,4,4,4,4,8,8,8,8...] -> [3,3,3,3...], b = [0,1,2,3,4,5,6,7,8...] -> [3,2,1,0,3,2,1,0,3,2,1,0...],a-b=[0,1,2,3...]
            vu8_sel = IVP_SUB2NX8U(IVP_SUB2NX8U(wid - 1, IVP_SLLI2NX8U(IVP_SRLI2NX8U(vu8_seq, 2), 2)), IVP_SUB2NX8U(3, IVP_AND2NX8U(vu8_seq, 3)));
            break;
        }
        default:
        {
            return;
        }
    }
}

static void xvGetNX16SelVec(int32_t channel, int32_t wid, xb_vecNx16 &vs16_sel)
{
    xb_vecNx16 vs16_seq = IVP_SEQNX16();
    switch (channel)
    {
        case 1:
        {
            vs16_sel = IVP_SUBNX16(wid - 1, vs16_seq);
            break;
        }
        case 2:
        {
            vs16_sel = IVP_SUBNX16(IVP_SUBNX16(wid - 1, IVP_SLLINX16(IVP_SRLINX16(vs16_seq, 1), 1)), IVP_ANDNX16(IVP_ADDNX16(vs16_seq, 1), 1));
            break;
        }
        case 3:
        {
            xb_vecNx16 vs16_div_thr      = IVP_QUONX16(vs16_seq, IVP_MOVVA16(3));
            xb_vecNx48 vs48_mul_thr      = IVP_MULNX16(3, vs16_div_thr);
            xb_vecN_2x32v vs32_mul_thr_l = IVP_CVT32UNX48L(vs48_mul_thr);
            xb_vecN_2x32v vs32_mul_thr_h = IVP_CVT32UNX48H(vs48_mul_thr);
            xb_vecNx16 vs16_mul_thr      = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32(vs32_mul_thr_h), IVP_MOVNX16_FROMN_2X32(vs32_mul_thr_l), IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
            xb_vecNx16 vs16_sub          = IVP_SUBNX16(2, IVP_REMNX16(vs16_seq, IVP_MOVVA16(3)));
            vs16_sel                     = IVP_SUBNX16(IVP_SUBNX16(wid - 1, vs16_mul_thr), vs16_sub);
            break;
        }
        case 4:
        {
            // a = [0,0,0,0,4,4,4,4,8,8,8,8...] -> [3,3,3,3...], b = [0,1,2,3,4,5,6,7,8...] -> [3,2,1,0,3,2,1,0,3,2,1,0...],a-b=[0,1,2,3...]
            vs16_sel = IVP_SUBNX16(IVP_SUBNX16(wid - 1, IVP_SLLINX16(IVP_SRLINX16(vs16_seq, 2), 2)), IVP_SUBNX16(3, IVP_ANDNX16(vs16_seq, 3)));
            break;
        }
        default:
        {
            return;
        }
    }
}

static void xvExtendEdgesReflect101_I8(xvTile const * tile, int32_t frame_width, int32_t frame_height)
{
    int32_t channel = XV_FRAME_GET_NUM_CHANNELS(tile->pFrame);

    int32_t w = XV_TILE_GET_EDGE_WIDTH(tile) * channel;
    int32_t h = XV_TILE_GET_EDGE_HEIGHT(tile);

    uint8_t* __restrict src = (uint8_t *)XV_TILE_GET_DATA_PTR(tile);
    xb_vec2Nx8U * vpdst;

    int32_t stride = XV_TILE_GET_PITCH(tile);

    int32_t start_x = XV_TILE_GET_X_COORD(tile) * channel;
    int32_t start_y = XV_TILE_GET_Y_COORD(tile);

    int32_t W_Local = XV_TILE_GET_WIDTH(tile) * channel;
    int32_t H_Local = XV_TILE_GET_HEIGHT(tile);

    frame_width = frame_width * channel;

    // find intersection of tile/frame
    int32_t ixmin = XT_MAX(start_x - w, 0);
    int32_t ixmax = XT_MIN(start_x + W_Local + w - 1 * channel, frame_width - 1 * channel);
    int32_t iymin = XT_MAX(start_y - h, 0);
    int32_t iymax = XT_MIN(start_y + H_Local + h - 1, frame_height - 1);

    int32_t p0x  = ixmin - start_x;
    int32_t ps0y = iymin - start_y;
    int32_t pd0y = iymin - start_y - 1;
    int32_t p0w  = (ixmax - ixmin) + 1 * channel;
    int32_t p0h  = iymin - (start_y - h);

    uint8_t* curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, ps0y, stride, p0x);
    uint8_t* dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, pd0y, stride, p0x);
    valign a_store = IVP_ZALIGN();
    valign a_load;

    int32_t p = XT_MAX(1, 2 * (iymax - iymin + 1) - 2);
    int32_t pmod = (p << 16) + 1;

    int32_t x = 0;
    if(p0h > 0)
    {
		for (; x < (p0w - (2 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vec2Nx8U color, color1;
				xb_vec2Nx8U * src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, k, stride, x);
				a_load = IVP_LA2NX8U_PP(src_1);
				IVP_LA2NX8U_IP (color,  a_load, src_1);
				IVP_LAV2NX8U_XP(color1, a_load, src_1, sizeof(uint8_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vec2Nx8U * vdst = XVTM_OFFSET_PTR_2NX8U(dst, -y, stride, x);
				IVP_SA2NX8U_IP (color,  a_store, vdst);
				IVP_SAV2NX8U_XP(color1, a_store, vdst, sizeof(uint8_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
			}
		}
		if(x < p0w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vec2Nx8U color;
				xb_vec2Nx8U * src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, k, stride, x);
				a_load = IVP_LA2NX8U_PP(src_1);
				IVP_LAV2NX8U_XP(color, a_load, src_1, sizeof(uint8_t) * (p0w - x));

				xb_vec2Nx8U * vdst = XVTM_OFFSET_PTR_2NX8U(dst, -y, stride, x);
				IVP_SAV2NX8U_XP(color, a_store, vdst, sizeof(uint8_t) * (p0w - x));
				XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
			}
		}
    }

    int32_t p1x = ixmin - start_x;
    int32_t ps1y = iymax - start_y;
    int32_t pd1y = (iymax + 1) - start_y;
    int32_t p1w = (ixmax - ixmin) + 1 * channel;
    int32_t p1h = ( start_y + H_Local + h ) - 1 - iymax;

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, ps1y, stride, p1x);
    dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, pd1y, stride, p1x);
    x = 0;
    if(p1h > 0)
    {
		for (; x < (p1w - ( 2 * XCHAL_IVPN_SIMD_WIDTH )); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vec2Nx8U color, color1;
				xb_vec2Nx8U * src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, -k, stride, x);
				a_load = IVP_LA2NX8U_PP(src_1);
				IVP_LA2NX8U_IP (color,  a_load, src_1);
				IVP_LAV2NX8U_XP(color1, a_load, src_1, sizeof(uint8_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vec2Nx8U * vdst = XVTM_OFFSET_PTR_2NX8U(dst, y, stride, x);
				IVP_SA2NX8U_IP (color,  a_store, vdst);
				IVP_SAV2NX8U_XP(color1, a_store, vdst, sizeof(uint8_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
			}
		}
		if(x < p1w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vec2Nx8U color;
				xb_vec2Nx8U * src_1 = XVTM_OFFSET_PTR_2NX8U(curr_src, -k, stride, x);
				a_load = IVP_LA2NX8U_PP(src_1);
				IVP_LAV2NX8U_XP(color, a_load, src_1, sizeof(uint8_t) * (p1w - x));

				xb_vec2Nx8U * vdst = XVTM_OFFSET_PTR_2NX8U(dst, y, stride, x);
				IVP_SAV2NX8U_XP(color, a_store, vdst, sizeof(uint8_t) * (p1w - x));
				XVTM_IVP_SAV2NX8UPOS_FP(a_store, vdst);
			}
		}
    }
    int32_t ps2x = XT_MIN(ixmin - start_x + 1 * channel, ixmax - start_x);
    int32_t pd2x = (ixmin - start_x);
    int32_t p2y = -h;
    int32_t p2w = ixmin - (start_x - w);
    int32_t p2h = H_Local + (2 * h);

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, ps2x);
    dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, pd2x);

    x = 0;

    while ( x < p2w - XT_MIN(XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH, ixmax - ixmin))
    {
    	int32_t wid;
    	xb_vec2Nx8U color, color1, color2, color3;
    	int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p2w-x);
        while ( xtmp < loop_width )
        {

            wid = XT_MIN(p2w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*2 * XCHAL_IVPN_SIMD_WIDTH, 2 * XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);
            xb_vec2Nx8U shuffle_vec;

            xvGet2NX8SelVec(channel, wid, shuffle_vec);

    		xb_vec2Nx8U * vpsrc0 = XVTM_OFFSET_PTR_2NX8U(curr_src, 0, 0, (ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH);
    		xb_vec2Nx8U * vpsrc1 = XVTM_OFFSET_PTR_2NX8U(curr_src, 1, stride, (ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH);
    		vpdst = XVTM_OFFSET_PTR_2NX8U(dst,      0, 0, -x - wid);
    		int32_t y = 0;
    		for (; y < (p2h - 3); y += 4)
    		{
    			a_load = IVP_LA2NX8U_PP(vpsrc0);
    			IVP_LAV2NX8U_XP(color,  a_load, vpsrc0, sizeof(uint8_t) * wid);
    			vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc1);
    			IVP_LAV2NX8U_XP(color1, a_load, vpsrc1, sizeof(uint8_t) * wid);
    			vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc0);
    			IVP_LAV2NX8U_XP(color2, a_load, vpsrc0, sizeof(uint8_t) * wid);
    			vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc1);
    			IVP_LAV2NX8U_XP(color3, a_load, vpsrc1, sizeof(uint8_t) * wid);
    			vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

    			color  = IVP_SHFL2NX8(color,  shuffle_vec);
    			color1 = IVP_SHFL2NX8(color1, shuffle_vec);
    			color2 = IVP_SHFL2NX8(color2, shuffle_vec);
    			color3 = IVP_SHFL2NX8(color3, shuffle_vec);

    			IVP_SAV2NX8U_XP(color,  a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color1, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color2, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color3, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
    		}

    		for (; y < p2h; y++)
    		{
    			a_load = IVP_LA2NX8U_PP(vpsrc0);
    			IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
    			vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, stride - wid, 0);

    			color = IVP_SHFL2NX8(color, shuffle_vec);
    			IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
    		}
    		x += wid;
    		ytmp+=1;
			xtmp+=wid;

    	}
    	curr_src = curr_src - (ixmax);
    }

    if(x < p2w)
    {
    	curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, ps2x);
    	dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p2y, stride, pd2x);

        int32_t wid = XT_MIN(p2w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
        xb_vecNx8U * lsrc = XVTM_OFFSET_PTR_NX8U(curr_src, 0, 0, -x);
        xb_vecNx8U * ldst = XVTM_OFFSET_PTR_NX8U(dst,      0, 0, -x - wid);

    	int32_t q15_inv_w = 1 + ((1<<15)/wid);
    	int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 65536 / stride; // check for gather bound (I8)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;

        xvGetNX16SelVec(channel, wid, shd);

        xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
        IVP_MULANX16PACKL(shs, stride - wid, shy);
        IVP_MULANX16PACKL(shd, stride + wid, shy);

    	for (int32_t s = 0; s < p2h; s += ystep)
    	{
            int32_t line_num = (p2h - s) < ystep ? (p2h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
    		uint8_t * src0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(lsrc, s, stride, 0);
    		uint8_t * dst0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(ldst, s, stride, 0);

            xb_gsr gr0;
            shs = IVP_MOVNX16T(shs, 0, vb);
            shd = IVP_MOVNX16T(shd, 0, vb);
            gr0 = XVTM_IVP_GATHERANX8UT_V(src0, shs, vb, 1);
            xb_vecNx16U v0 = IVP_GATHERDNX8U(gr0);
            XVTM_IVP_SCATTERNX8UT(v0, dst0, shd, vb);
        }
    }

    int32_t ps3x = ixmax - start_x;
    int32_t pd3x = (ixmax + 1 * channel) - start_x;
    int32_t p3y = -h;
    int32_t p3w = ((start_x + W_Local + w) - 1 * channel - ixmax);
    int32_t p3h = H_Local + (2 * h);

    curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, ps3x);
    dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, pd3x);
    x = 0;

    while ( x < p3w - XT_MIN(XVTM_EDGE_REFLECT_I8_SCATTER_MIN_WIDTH, ixmax - ixmin))
    {
    	int32_t wid;
    	xb_vec2Nx8U color, color1, color2, color3;
    	int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p3w-x);
        while ( xtmp < loop_width)
        {
            wid = XT_MIN(p3w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-(ytmp*2 * XCHAL_IVPN_SIMD_WIDTH), 2 * XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);

            xb_vec2Nx8U shuffle_vec;

            xvGet2NX8SelVec(channel, wid, shuffle_vec);

    		xb_vec2Nx8U * vpsrc0 = XVTM_OFFSET_PTR_2NX8U(curr_src, 0, 0, - wid-((ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH));
    		xb_vec2Nx8U * vpsrc1 = XVTM_OFFSET_PTR_2NX8U(curr_src, 1, stride, - wid-((ytmp)*2 * XCHAL_IVPN_SIMD_WIDTH));
    		vpdst  = XVTM_OFFSET_PTR_2NX8U(dst,      0, 0, x);
    		int32_t y = 0;
    		for (; y < (p3h - 3) ; y += 4)
    		{
    			a_load = IVP_LA2NX8U_PP(vpsrc0);
    			IVP_LAV2NX8U_XP(color,  a_load, vpsrc0, sizeof(uint8_t) * wid);
    			vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc1);
    			IVP_LAV2NX8U_XP(color1, a_load, vpsrc1, sizeof(uint8_t) * wid);
    			vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc0);
    			IVP_LAV2NX8U_XP(color2, a_load, vpsrc0, sizeof(uint8_t) * wid);
    			vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, (2 * stride) - wid, 0);

    			a_load = IVP_LA2NX8U_PP(vpsrc1);
    			IVP_LAV2NX8U_XP(color3, a_load, vpsrc1, sizeof(uint8_t) * wid);
    			vpsrc1 = XVTM_OFFSET_PTR_2NX8U(vpsrc1, 1, (2 * stride) - wid, 0);

    			color  = IVP_SHFL2NX8(color,  shuffle_vec);
    			color1 = IVP_SHFL2NX8(color1, shuffle_vec);
    			color2 = IVP_SHFL2NX8(color2, shuffle_vec);
    			color3 = IVP_SHFL2NX8(color3, shuffle_vec);

    			IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color1, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color2, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);

    			IVP_SAV2NX8U_XP(color3, a_store, vpdst, sizeof(uint8_t) * wid);
    			XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    			vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
    		}

    			for (; y < p3h; y++)
    			{
    				a_load = IVP_LA2NX8U_PP(vpsrc0);
    				IVP_LAV2NX8U_XP(color, a_load, vpsrc0, sizeof(uint8_t) * wid);
    				vpsrc0 = XVTM_OFFSET_PTR_2NX8U(vpsrc0, 1, stride - wid, 0);

    				color = IVP_SHFL2NX8(color, shuffle_vec);
    				IVP_SAV2NX8U_XP(color, a_store, vpdst, sizeof(uint8_t) * wid);
    				XVTM_IVP_SAV2NX8UPOS_FP(a_store, vpdst);
    				vpdst = XVTM_OFFSET_PTR_2NX8U(vpdst, 1, stride - wid, 0);
    			}
    			x += wid;
    			ytmp+=1;
				xtmp+=wid;
    	}
    	curr_src = curr_src+(ixmax);

    }

    if(x < p3w)
    {
    	curr_src = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, ps3x);
    	dst      = (uint8_t *)XVTM_OFFSET_PTR_2NX8U(src, p3y, stride, pd3x);

        int32_t wid = XT_MIN(p3w - x, 2 * XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
        xb_vecNx8U * lsrc = XVTM_OFFSET_PTR_NX8U(curr_src, 0, 0, x - wid);
        xb_vecNx8U * ldst = XVTM_OFFSET_PTR_NX8U(dst,      0, 0, x);

    	int32_t q15_inv_w = 1 + ((1<<15)/wid);
    	int32_t ystep = q15_inv_w >> 10;
        int32_t gather_bound = 65536 / stride; // check for gather bound (I8)
        ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;

        xvGetNX16SelVec(channel, wid, shd);

        xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
        IVP_MULANX16PACKL(shs, stride - wid, shy);
        IVP_MULANX16PACKL(shd, stride + wid, shy);

        for (int32_t s = 0; s < p3h; s += ystep)
        {
            int32_t line_num = (p3h - s) < ystep ? (p3h - s) : ystep;
            vboolN vb = IVP_LTRSN(line_num * wid);
            uint8_t * src0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(lsrc, s, stride, 0);
            uint8_t * dst0 = (uint8_t *)XVTM_OFFSET_PTR_NX8U(ldst, s, stride, 0);

            xb_gsr gr0;
            shs = IVP_MOVNX16T(shs, 0, vb);
            shd = IVP_MOVNX16T(shd, 0, vb);
            gr0 = XVTM_IVP_GATHERANX8UT_V(src0, shs, vb, 1);
            xb_vecNx16U v0 = IVP_GATHERDNX8U(gr0);
            XVTM_IVP_SCATTERNX8UT(v0, dst0, shd, vb);
        }
    }
    IVP_SCATTERW();
}

static void xvExtendEdgesReflect101_I16(xvTile const * tile, int32_t frame_width, int32_t frame_height)
{
    int32_t channel = XV_FRAME_GET_NUM_CHANNELS(tile->pFrame);

    int32_t w = XV_TILE_GET_EDGE_WIDTH(tile) * channel;
    int32_t h = XV_TILE_GET_EDGE_HEIGHT(tile);

	int16_t* __restrict src = (int16_t *)XV_TILE_GET_DATA_PTR(tile);
	int32_t stride = XV_TILE_GET_PITCH(tile);

    int32_t start_x = XV_TILE_GET_X_COORD(tile) * channel;
    int32_t start_y = XV_TILE_GET_Y_COORD(tile);

    int32_t W_Local = XV_TILE_GET_WIDTH(tile) * channel;
    int32_t H_Local = XV_TILE_GET_HEIGHT(tile);

    frame_width = frame_width * channel;

    int32_t usr_tmp =0;
    // find intersection of tile/frame
    int32_t ixmin = XT_MAX(start_x - w, 0);
    int32_t ixmax = XT_MIN(start_x + W_Local + w - 1 * channel, frame_width - 1 * channel);
    int32_t iymin = XT_MAX(start_y - h, 0);
    int32_t iymax = XT_MIN(start_y + H_Local + h - 1, frame_height - 1);


    int32_t p0x = ixmin - start_x;
    int32_t ps0y = iymin - start_y;
    int32_t pd0y = iymin - start_y - 1;
    int32_t p0w = (ixmax - ixmin) + 1 * channel;
    int32_t p0h = iymin - (start_y - h);

	int16_t* curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, ps0y, stride, p0x);
	int16_t* dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, pd0y, stride, p0x);

	xb_vecNx16 *vpdst;

	valign a_store = IVP_ZALIGN();
	valign a_load;

	int32_t p = XT_MAX(1, 2 * (iymax - iymin + 1) - 2);
	int32_t pmod = (p << 16) + 1;
	int32_t x = 0;
	if(p0h > 0)
	{
		for (; x < (p0w - (3 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2, color3;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color, a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LANX16_IP (color2, a_load, src_1);
				IVP_LAVNX16_XP(color3, a_load, src_1, sizeof(int16_t) * (p0w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color, a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SANX16_IP (color2, a_store, vdst);
				IVP_SAVNX16_XP(color3, a_store, vdst, sizeof(int16_t) * (p0w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		if(x < (p0w - (2 * XCHAL_IVPN_SIMD_WIDTH)))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LAVNX16_XP(color2, a_load, src_1, sizeof(int16_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SAVNX16_XP(color2, a_store, vdst, sizeof(int16_t) * (p0w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < (p0w - XCHAL_IVPN_SIMD_WIDTH))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color, a_load, src_1);
				IVP_LAVNX16_XP(color1, a_load, src_1, sizeof(int16_t) * (p0w - x - XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SANX16_IP (color, a_store, vdst);
				IVP_SAVNX16_XP(color1, a_store, vdst, sizeof(int16_t) * (p0w - x - XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < p0w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p0h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color;
				xb_vecNx16 *src_1 = XVTM_OFFSET_PTR_NX16(curr_src, k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LAVNX16_XP(color, a_load, src_1, sizeof(int16_t) * (p0w - x));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, -y, stride, x);
				IVP_SAVNX16_XP(color, a_store, vdst, sizeof(int16_t) * (p0w - x));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else
		{
			//do nothing
		}
	}
	int32_t p1x = ixmin - start_x;
	int32_t ps1y = iymax - start_y;
	int32_t pd1y = (iymax + 1) - start_y;
    int32_t p1w = (ixmax - ixmin) + 1 * channel;
	int32_t p1h = (start_y + H_Local + h) - 1 - iymax;

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, ps1y, stride, p1x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, pd1y, stride, p1x);
	x = 0;
	if(p1h > 0)
	{
		for (; x < (p1w - (3 * XCHAL_IVPN_SIMD_WIDTH)); x += 4 * XCHAL_IVPN_SIMD_WIDTH)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2, color3;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LANX16_IP (color2, a_load, src_1);
				IVP_LAVNX16_XP(color3, a_load, src_1, sizeof(int16_t) * (p1w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SANX16_IP (color2, a_store, vdst);
				IVP_SAVNX16_XP(color3, a_store, vdst, sizeof(int16_t) * (p1w - x - 3 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		if(x < (p1w - (2 * XCHAL_IVPN_SIMD_WIDTH)))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1, color2;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LANX16_IP (color,  a_load, src_1);
				IVP_LANX16_IP (color1, a_load, src_1);
				IVP_LAVNX16_XP(color2, a_load, src_1, sizeof(int16_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SANX16_IP (color1, a_store, vdst);
				IVP_SAVNX16_XP(color2, a_store, vdst, sizeof(int16_t) * (p1w - x - 2 * XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < (p1w - XCHAL_IVPN_SIMD_WIDTH))
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color, color1;
				xb_vecNx16 * src_0_t = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_0_t);
				IVP_LANX16_IP (color,  a_load, src_0_t);
				IVP_LAVNX16_XP(color1, a_load, src_0_t, sizeof(int16_t) * (p1w - x - XCHAL_IVPN_SIMD_WIDTH));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SANX16_IP (color,  a_store, vdst);
				IVP_SAVNX16_XP(color1, a_store, vdst, sizeof(int16_t) * (p1w - x - XCHAL_IVPN_SIMD_WIDTH));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else if(x < p1w)
		{
			int32_t z = 0;
			for (int32_t y = 0; y < p1h; y++)
			{
				z = IVP_ADDMOD16U(z, pmod);
				int32_t k = XT_MIN(z, p - z);

				xb_vecNx16 color;
				xb_vecNx16 * src_1 = XVTM_OFFSET_PTR_NX16(curr_src, -k, stride, x);
				a_load = IVP_LANX16_PP(src_1);
				IVP_LAVNX16_XP(color, a_load, src_1, sizeof(int16_t) * (p1w - x));

				xb_vecNx16 *vdst = XVTM_OFFSET_PTR_NX16(dst, y, stride, x);
				IVP_SAVNX16_XP(color, a_store, vdst, sizeof(int16_t) * (p1w - x));
				XVTM_IVP_SAVNX16POS_FP(a_store, vdst);
			}
		}
		else
		{
			//do nothing
		}
	}
    int32_t ps2x = XT_MIN(ixmin - start_x + 1 * channel, ixmax - start_x);
	int32_t pd2x = ixmin - start_x;
	int32_t p2y = -h;
	int32_t p2w = ixmin - (start_x - w);
	int32_t p2h = H_Local + (2 * h);

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, ps2x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, pd2x);
	x = 0;


	while (x < p2w- XT_MIN(XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH, ixmax - ixmin))
	{
		int32_t wid;
		xb_vecNx16 color, color1, color2, color3;

		xb_vecNx16 * vpsrc0;
		xb_vecNx16 * vpsrc1;

		int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p2w-x);
    	while ( xtmp < loop_width)
		{
			wid = XT_MIN(p2w - x, XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*XCHAL_IVPN_SIMD_WIDTH, XCHAL_IVPN_SIMD_WIDTH),wid), 1 * channel);

            xb_vecNx16 ind;
            xvGetNX16SelVec(channel, wid, ind);
			vsaN index = XVTM_IVP_MOVVSV(ind, 0);
			vpsrc0 = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, (ytmp)*XCHAL_IVPN_SIMD_WIDTH);
			vpsrc1 = XVTM_OFFSET_PTR_NX16(curr_src, 1, stride, (ytmp)*XCHAL_IVPN_SIMD_WIDTH);
			vpdst  = XVTM_OFFSET_PTR_NX16(dst,      0, 0, -x - wid);

			int32_t y = 0;
			for (; y < (p2h - 3); y += 4)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color,  a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color1, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color2, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color3, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				color  =  IVP_SELNX16(color,  color,  index);
				color1 =  IVP_SELNX16(color1, color1, index);
				color2 =  IVP_SELNX16(color2, color2, index);
				color3 =  IVP_SELNX16(color3, color3, index);

				IVP_SAVNX16_XP(color,  a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color1, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color2, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color3, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			for (; y < p2h; y++)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, stride - wid, 0);

				color =  IVP_SELNX16(color, color, index);

				IVP_SAVNX16_XP(color, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			x += wid;
			ytmp+=1;
			xtmp+=wid;

		}
		curr_src = curr_src - (ixmax);


	}
	if(x < p2w)
	{
		curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, ps2x);
		dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p2y, stride, pd2x);

		int32_t wid = XT_MIN(p2w - x, XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
		xb_vecNx16 * lsrc = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, -x + ( XCHAL_IVPN_SIMD_WIDTH * usr_tmp));
		xb_vecNx16 * ldst  = XVTM_OFFSET_PTR_NX16(dst,      0, 0, -x - wid);

		int32_t q15_inv_w = 1 + ((1<<15)/wid);
		int32_t ystep = q15_inv_w >> 10;
    int32_t gather_bound = 32768 / stride; // check for gather bound (I16)
    ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;
        xvGetNX16SelVec(channel, wid, shd);
		xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
		IVP_MULANX16PACKL(shs, stride - wid, shy);
		IVP_MULANX16PACKL(shd, stride + wid, shy);

		shd = IVP_SLLINX16(shd, 1);
		shs = IVP_SLLINX16(shs, 1);

		for (int32_t s = 0; s < p2h; s += ystep)
		{
      int32_t line_num = (p2h - s) < ystep ? (p2h - s) : ystep;
      vboolN vb = IVP_LTRSN(line_num * wid);
			int16_t * src0 = (int16_t *)XVTM_OFFSET_PTR_NX16(lsrc, s, stride, 0);
			int16_t * dst0 = (int16_t *)XVTM_OFFSET_PTR_NX16(ldst, s, stride, 0);

			xb_gsr gr0;
			shs = IVP_MOVNX16T(shs, 0, vb);
			shd = IVP_MOVNX16T(shd, 0, vb);
			gr0 = XVTM_IVP_GATHERANX16T_V(src0, shs, vb, 1);
			xb_vecNx16 v0 = IVP_GATHERDNX16(gr0);
			XVTM_IVP_SCATTERNX16T(v0, dst0, shd, vb);
		}
	}

	int32_t ps3x = ixmax - start_x;
    int32_t pd3x = (ixmax + 1 * channel) - start_x;
	int32_t p3y = -h;
    int32_t p3w = (start_x + W_Local + w) - 1 * channel - ixmax;
	int32_t p3h = H_Local + (2 * h);

	curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, ps3x);
	dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, pd3x);

	x = 0;

	while( x < p3w- XT_MIN(XVTM_EDGE_REFLECT_I16_SCATTER_MIN_WIDTH, ixmax - ixmin))
	{
		int32_t wid;
		xb_vecNx16 color, color1, color2, color3;

		xb_vecNx16 * vpsrc0;
		xb_vecNx16 * vpsrc1;

		int32_t ytmp = 0;
		int32_t xtmp = 0;

        int32_t loop_width = XT_MIN(XT_MAX(ixmax, 1 * channel), p3w-x);
    	while ( xtmp < loop_width)
		{
			wid = XT_MIN(p3w - x, XCHAL_IVPN_SIMD_WIDTH);
            wid = XT_MAX(XT_MIN(XT_MIN(ixmax-ytmp*XCHAL_IVPN_SIMD_WIDTH, XCHAL_IVPN_SIMD_WIDTH),wid),1 * channel);

            xb_vecNx16 ind;
            xvGetNX16SelVec(channel, wid, ind);
			vsaN index = XVTM_IVP_MOVVSV(ind, 0);

			vpdst = XVTM_OFFSET_PTR_NX16(dst, 0, 0, x);
			vpsrc0 = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, -wid-((ytmp)*XCHAL_IVPN_SIMD_WIDTH));
			vpsrc1 = XVTM_OFFSET_PTR_NX16(curr_src, 1, stride, -wid-((ytmp)*XCHAL_IVPN_SIMD_WIDTH));

			int32_t y = 0;

			for (; y < (p3h - 3); y += 4)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color,  a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color1, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color2, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, (2 * stride) - wid, 0);

				a_load = IVP_LANX16_PP(vpsrc1);
				IVP_LAVNX16_XP(color3, a_load, vpsrc1, sizeof(int16_t) * wid);
				vpsrc1 = XVTM_OFFSET_PTR_NX16(vpsrc1, 1, (2 * stride) - wid, 0);

				color  =  IVP_SELNX16(color,  color,  index);
				color1 =  IVP_SELNX16(color1, color1, index);
				color2 =  IVP_SELNX16(color2, color2, index);
				color3 =  IVP_SELNX16(color3, color3, index);

				IVP_SAVNX16_XP(color,  a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color1, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color2, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);

				IVP_SAVNX16_XP(color3, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}
			for (; y < p3h; y++)
			{
				a_load = IVP_LANX16_PP(vpsrc0);
				IVP_LAVNX16_XP(color, a_load, vpsrc0, sizeof(int16_t) * wid);
				vpsrc0 = XVTM_OFFSET_PTR_NX16(vpsrc0, 1, stride - wid, 0);

				color =  IVP_SELNX16(color, color, index);

				IVP_SAVNX16_XP(color, a_store, vpdst, sizeof(int16_t) * wid);
				XVTM_IVP_SAVNX16POS_FP(a_store, vpdst);
				vpdst = XVTM_OFFSET_PTR_NX16(vpdst, 1, stride - wid, 0);
			}

			x += wid;
			ytmp+=1;
			xtmp+=wid;
		}
		curr_src = curr_src+(ixmax);
	}

	if(x < p3w)
	{
		curr_src = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, ps3x);
		dst      = (int16_t *)XVTM_OFFSET_PTR_NX16(src, p3y, stride, pd3x);

		int32_t wid = XT_MIN(p3w - x, XCHAL_IVPN_SIMD_WIDTH);
        wid = XT_MAX(XT_MIN(wid, ixmax - ixmin), 1 * channel);
		xb_vecNx16 * lsrc = XVTM_OFFSET_PTR_NX16(curr_src, 0, 0, x - wid);
		xb_vecNx16 * ldst = XVTM_OFFSET_PTR_NX16(dst, 0, 0, x);

		int32_t q15_inv_w = 1 + ((1<<15)/wid);
		int32_t ystep = q15_inv_w >> 10;
    int32_t gather_bound = 32768 / stride; // check for gather bound (I16)
    ystep = ystep > gather_bound ? gather_bound : ystep;

        xb_vecNx16 shs = IVP_SEQNX16();
        xb_vecNx16 shd;
        xvGetNX16SelVec(channel, wid, shd);
		xb_vecNx16 shy = IVP_PACKVRNRNX48(IVP_MULUUNX16(shs, q15_inv_w), 15);
		IVP_MULANX16PACKL(shs, stride - wid, shy);
		IVP_MULANX16PACKL(shd, stride + wid, shy);

		shd = IVP_SLLINX16(shd, 1);
		shs = IVP_SLLINX16(shs, 1);

		for (int32_t s = 0; s < p3h; s += ystep)
		{
      int32_t line_num = (p3h - s) < ystep ? (p3h - s) : ystep;
      vboolN vb = IVP_LTRSN(line_num * wid);
			int16_t * src0 = (int16_t *)XVTM_OFFSET_PTR_NX16(lsrc, s, stride, 0);
			int16_t * dst0 = (int16_t *)XVTM_OFFSET_PTR_NX16(ldst, s, stride, 0);

			xb_gsr gr0;
			shs = IVP_MOVNX16T(shs, 0, vb);
			shd = IVP_MOVNX16T(shd, 0, vb);
			gr0 = XVTM_IVP_GATHERANX16T_V(src0, shs, vb, 1);
			xb_vecNx16 v0 = IVP_GATHERDNX16(gr0);
			XVTM_IVP_SCATTERNX16T(v0, dst0, shd, vb);
		}
	}
	IVP_SCATTERW();
}