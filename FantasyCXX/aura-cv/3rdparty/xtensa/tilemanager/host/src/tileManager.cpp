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

//#define TEST_DTCM23


/**********************************************************************************
 * FUNCTION: xvInitIdmaMultiChannel4CHHost()
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

int32_t xvInitIdmaMultiChannel4CHHost(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1, idma_buffer_t *buf2, idma_buffer_t *buf3,
                                  int32_t numDescs, int32_t maxBlock, int32_t maxPifReq, idma_err_callback_fn errCallbackFunc0,
                                  idma_err_callback_fn errCallbackFunc1, idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                  idma_callback_fn cbFunc0, void * cbData0, idma_callback_fn cbFunc1, void * cbData1,
                                  idma_callback_fn cbFunc2, void * cbData2, idma_callback_fn cbFunc3, void * cbData3)
{
  idma_status_t retVal;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR((((buf0 == NULL) && (buf1 == NULL)) && ((buf2 == NULL) && (buf3 == NULL))), pxvTM->errFlag = XV_ERROR_BUFFER_NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((numDescs < 1), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  (void)maxBlock;
  (void)maxPifReq;
  (void)errCallbackFunc0;
  (void)errCallbackFunc1;
  (void)errCallbackFunc2;
  (void)errCallbackFunc3;

  idma_type_t type             = IDMA_64B_DESC;
  if (buf0 != NULL)
  {

    retVal = idma_init_loop((int32_t) TM_IDMA_CH0, buf0, type, numDescs, cbData0, cbFunc0);
    // No error check on retVal as idma_init_loop() never return a non IDMA_OK value
  }

  if (buf1 != NULL)
  {

    retVal = idma_init_loop((int32_t) TM_IDMA_CH1, buf1, type, numDescs, cbData1, cbFunc1);
    // No error check on retVal as idma_init_loop() never return a non IDMA_OK value
  }

  if (buf2 != NULL)
  {

    retVal = idma_init_loop((int32_t) TM_IDMA_CH2, buf2, type, numDescs, cbData2, cbFunc2);
    // No error check on retVal as idma_init_loop() never return a non IDMA_OK value
  }

  if (buf3 != NULL)
  {

    retVal = idma_init_loop((int32_t) TM_IDMA_CH3, buf3, type, numDescs, cbData3, cbFunc3);
    // No error check on retVal as idma_init_loop() never return a non IDMA_OK value
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvInitTileManagerMultiChannel4CHHost()
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

int32_t xvInitTileManagerMultiChannel4CHHost(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1,
                                         idma_buffer_t *buf2, idma_buffer_t *buf3)
{
  uint32_t index;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pxvTM->errFlag          = XV_ERROR_SUCCESS;
  pxvTM->idmaErrorFlag[0] = XV_ERROR_SUCCESS;
  pxvTM->idmaErrorFlag[1] = XV_ERROR_SUCCESS;
  pxvTM->idmaErrorFlag[2] = XV_ERROR_SUCCESS;
  pxvTM->idmaErrorFlag[3] = XV_ERROR_SUCCESS;

#if (XCHAL_IDMA_NUM_CHANNELS == 1)
  XV_CHECK_ERROR(buf0 == NULL, pxvTM->errFlag = XV_ERROR_BUFFER_NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");
#elif (XCHAL_IDMA_NUM_CHANNELS == 2)
  XV_CHECK_ERROR(((buf0 == NULL) && (buf1 == NULL)), pxvTM->errFlag = XV_ERROR_BUFFER_NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");
#else
  XV_CHECK_ERROR((buf0 == NULL) && (buf1 == NULL) && (buf2 == NULL) && (buf3 == NULL), \
                 pxvTM->errFlag = XV_ERROR_BUFFER_NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");
#endif

  // Initialize DMA related elements
  pxvTM->pdmaObj0 = buf0;
  pxvTM->pdmaObj1 = buf1;
  pxvTM->pdmaObj2 = buf2;
  pxvTM->pdmaObj3 = buf3;

  // Initialize IDMA state variables
  for (index = 0; index < MAX_NUM_CHANNEL; index++)
  {
    pxvTM->tileDMApendingCount[index] = 0;
    pxvTM->tileDMAstartIndex[index]   = 0;
  }

  // Initialize Memory banks related elements
#ifndef XVTM_USE_XMEM
  for (index = 0; index < (uint32_t) MAX_NUM_MEM_BANKS; index++)
  {
    pxvTM->pMemBankStart[index] = NULL;
    pxvTM->memBankSize[index]   = 0;
  }
#endif
  // Reset tile related elements
  pxvTM->tileCount = 0;
  for (index = 0; index < ((MAX_NUM_TILES + 31u) / 32u); index++)
  {
    pxvTM->tileAllocFlags[index] = 0x00000000;
  }

  // Reset frame related elements
  pxvTM->frameCount = 0;
  for (index = 0; index < ((MAX_NUM_FRAMES + 31u) / 32u); index++)
  {
    pxvTM->frameAllocFlags[index] = 0x00000000;
  }

#ifdef FIK_FRAMEWORK
  pxvTM->tmContextSize = 0;
  pxvTM->allocationsIdx = 0;
#endif
  // Reset tile3D related elements
  pxvTM->tile3DCount = 0;
  for (index = 0; index < ((MAX_NUM_TILES3D + 31u) / 32u); index++)
  {
    pxvTM->tile3DAllocFlags[index] = 0x00000000;
  }

  // Reset frame3D related elements
  pxvTM->frame3DCount = 0;
  for (index = 0; index < ((MAX_NUM_FRAMES3D + 31u) / 32u); index++)
  {
    pxvTM->frame3DAllocFlags[index] = 0x00000000;
  }
  // Initialize IDMA state variables
  for (index = 0; index < MAX_NUM_CHANNEL; index++)
  {
    pxvTM->tile3DDMApendingCount[index] = 0;
    pxvTM->tile3DDMAstartIndex[index]   = 0;
  }

#ifdef TM_LOG
  __pxvTM            = pxvTM;
  __pxvTM->tm_log_fp = fopen(TILE_MANAGER_LOG_FILE_NAME, "w");
  if (__pxvTM->tm_log_fp == NULL)
  {
    __pxvTM->errFlag = XV_ERROR_FILE_OPEN;
    return(XVTM_ERROR);
  }
#endif

  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvResetTileManagerHost()
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

int32_t xvResetTileManagerHost(xvTileManager *pxvTM)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  // Close the Tile Manager logging file
#ifdef TM_LOG
  if (__pxvTM != NULL)
  {
    if (__pxvTM->tm_log_fp)
    {
      fclose(__pxvTM->tm_log_fp);
    }
  }
#endif
#ifndef XVTM_USE_XMEM
  // Free all the xvmem allocated buffers
  //Type cast to void to avoid MISRA 17.7 violation
  (void) xvFreeAllBuffersHost(pxvTM);
#endif
  // Resetting the Tile Manager pointer.
  // This will free all allocated tiles and buffers.
  // It will not reset dma object.
  //Type cast to void to avoid MISRA 17.7 violation
  (void) memset(pxvTM, 0, sizeof(xvTileManager));
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvInitMemAllocatorHost()
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
int32_t xvInitMemAllocatorHost(xvTileManager *pxvTM, int32_t numMemBanks, void* const* pBankBuffPool, int32_t const* buffPoolSize)
{
  int32_t indx;
  xvmem_status_t retVal;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pxvTM->errFlag = XV_ERROR_SUCCESS;

  XV_CHECK_ERROR(((numMemBanks <= 0) || (numMemBanks > MAX_NUM_MEM_BANKS)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR(((pBankBuffPool == NULL) || (buffPoolSize == NULL)), pxvTM->errFlag = XV_ERROR_POINTER_NULL, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  xvmem_mgr_t *pmemBankMgr;
  pxvTM->numMemBanks = numMemBanks;

  for (indx = 0; indx < numMemBanks; indx++)
  {
    XV_CHECK_ERROR(((pBankBuffPool[indx] == NULL) || (buffPoolSize[indx] <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
    XV_CHECK_ERROR(!((XV_PTR_START_IN_DRAM(pBankBuffPool[indx])) && (XV_PTR_END_IN_DRAM(((uint64_t) (uint32_t) pBankBuffPool[indx]) + ((uint64_t) buffPoolSize[indx])))), \
                   pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

    pmemBankMgr                = &(pxvTM->memBankMgr[indx]);
    pxvTM->pMemBankStart[indx] = pBankBuffPool[indx];
    pxvTM->memBankSize[indx]   = buffPoolSize[indx];
    retVal                     = xvmem_init(pmemBankMgr, pBankBuffPool[indx], (uint32_t) buffPoolSize[indx], 0, NULL);
    XV_CHECK_ERROR((retVal != XVMEM_OK), pxvTM->errFlag = XV_ERROR_XVMEM_INIT, XVTM_ERROR, "Error");
  }
  return(XVTM_SUCCESS);
}


#endif
/**********************************************************************************
 * FUNCTION: xvAllocateBufferHost()
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

void *xvAllocateBufferHost(xvTileManager *pxvTM, int32_t buffSize, int32_t buffColor, int32_t buffAlignment)
{
  void *buffOut = NULL;
  int32_t currColor;


  XV_CHECK_ERROR_NULL(pxvTM == NULL, (void *) XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  XV_CHECK_ERROR((buffSize <= 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((void *) XVTM_ERROR), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((buffAlignment <= 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((void *) XVTM_ERROR), "XV_ERROR_BAD_ARG");
#ifndef XVTM_USE_XMEM
  int32_t numMemBanks = pxvTM->numMemBanks;
#else
  int32_t numMemBanks = xmem_bank_get_num_banks();
#endif
  XV_CHECK_ERROR((!(((buffColor >= 0) && (buffColor < numMemBanks)) || (buffColor == (int32_t) XV_MEM_BANK_COLOR_ANY))), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((void *) XVTM_ERROR), "XV_ERROR_ALLOC");

  // if the color is XV_MEM_BANK_COLOR_ANY, loop through all the buffers and pick one that meets all criteria
  if (buffColor == (int32_t) XV_MEM_BANK_COLOR_ANY)
  {
    currColor = 0;
    while ((buffOut == NULL) && (currColor < numMemBanks))
    {
#ifdef XVTM_USE_XMEM
      xmem_bank_status_t xmbs;
      buffOut = xmem_bank_alloc(currColor, buffSize, buffAlignment, &xmbs);
#else
      xvmem_status_t errCode;
      buffOut = xvmem_alloc(&(pxvTM->memBankMgr[currColor]), (uint32_t) buffSize, (uint32_t) buffAlignment, &errCode);
#endif
      currColor++;
    }
  }
  else
  {
#ifndef XVTM_USE_XMEM
      xvmem_status_t errCode;
      buffOut = xvmem_alloc(&(pxvTM->memBankMgr[buffColor]), (uint32_t) buffSize, (uint32_t) buffAlignment, &errCode);
#else

	  xmem_bank_status_t xmbs;
	  buffOut = xmem_bank_alloc(buffColor, buffSize, buffAlignment, &xmbs);
#endif
  }
  
  if (buffOut == NULL)
  {
    pxvTM->errFlag = XV_ERROR_ALLOC_FAILED;
    return((void *) XVTM_ERROR);
  }
  
#ifdef FIK_FRAMEWORK
  if (buffOut != NULL)
  {
	  pxvTM->allocatedList[pxvTM->allocationsIdx] = buffOut;
	  pxvTM->allocationsIdx++;
	  pxvTM->allocationsIdx = pxvTM->allocationsIdx % XV_MAX_ALLOCATIONS;
  }
#endif
  return(buffOut);
}

/**********************************************************************************
 * FUNCTION: xvFreeBufferHost()
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

int32_t xvFreeBufferHost(xvTileManager *pxvTM, void const *pBuff)
{
  int32_t index;
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(pBuff == NULL, pxvTM->errFlag = XV_ERROR_BUFFER_NULL, XVTM_ERROR, "XV_ERROR_BUFFER_NULL");
#ifndef XVTM_USE_XMEM
  for (index = 0; index < pxvTM->numMemBanks; index++) {
    if ((pxvTM->pMemBankStart[index] <= pBuff) && (pBuff < (void *) (&((uint8_t *) pxvTM->pMemBankStart[index])[pxvTM->memBankSize[index]]))) {
      xvmem_free(&(pxvTM->memBankMgr[index]), pBuff);
      return(XVTM_SUCCESS);
    }
  }
#else
  int32_t numMemBanks = xmem_bank_get_num_banks();
	uint32_t a = (uint32_t)pBuff;
  	void* p = (void*)a;
  for (index = 0; index < numMemBanks; index++) {
    if (xmem_bank_check_bounds(index,p)==XMEM_BANK_OK) {
    	xmem_bank_status_t xmbs;
    	xmbs = xmem_bank_free(index, p);
    	XV_CHECK_ERROR_NULL(xmbs != XMEM_BANK_OK, XVTM_ERROR, "xmem error");
    	return(XVTM_SUCCESS);
    }
  }
#endif
  pxvTM->errFlag = XV_ERROR_BAD_ARG;
  return(XVTM_ERROR);
}

/**********************************************************************************
 * FUNCTION: xvGetErrorInfoHost()
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

xvError_t xvGetErrorInfoHost(xvTileManager const *pxvTM)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XV_ERROR_TILE_MANAGER_NULL, "NULL TM Pointer");
  return(pxvTM->errFlag);
}

/**********************************************************************************
 * FUNCTION: xvFreeAllBuffersHost()
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
int32_t xvFreeAllBuffersHost(xvTileManager *pxvTM)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  int32_t bankIndex;
#ifndef XVTM_USE_XMEM
  int32_t numMemBanks = pxvTM->numMemBanks;
#else
  int32_t numMemBanks = xmem_bank_get_num_banks();
#endif


  // Resetting all memory manager related elements
  for (bankIndex = 0; bankIndex < numMemBanks; bankIndex++)
  {
#ifndef XVTM_USE_XMEM
    int32_t buffPoolSize;
	xvmem_mgr_t *pmemBankMgr;
	void *pBankBuffPool;
    pmemBankMgr   = &(pxvTM->memBankMgr[bankIndex]);
    pBankBuffPool = pxvTM->pMemBankStart[bankIndex];
    buffPoolSize  = pxvTM->memBankSize[bankIndex];
   // Type cast to void to avoid MISRA 17.7 violation
    (void) xvmem_init(pmemBankMgr, pBankBuffPool, (uint32_t) buffPoolSize, 0u, NULL);
#else

    xmem_bank_status_t xmbs;
    xmbs = xmem_bank_reset(bankIndex);
    XV_CHECK_ERROR_NULL(xmbs != XMEM_BANK_OK, XVTM_ERROR, "xmem error");
#endif
  }

  return(XVTM_SUCCESS);
}
#endif
/**********************************************************************************
 * FUNCTION: xvAllocateFrameHost()
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

xvFrame *xvAllocateFrameHost(xvTileManager *pxvTM)
{
  uint32_t indx, indxArr = 0, indxShift = 0, allocFlags = 0;
  xvFrame *pFrame = NULL;
  XV_CHECK_ERROR_NULL(pxvTM == NULL, (xvFrame *) ((void *) (XVTM_ERROR)), "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;

  for (indx = 0; indx < MAX_NUM_FRAMES; indx++)
  {
    indxArr    = indx >> 5u;
    indxShift  = indx & 0x1Fu;
    allocFlags = pxvTM->frameAllocFlags[indxArr];
    if (((allocFlags >> indxShift) & 0x1u) == 0u)
    {
      break;
    }
  }

  if (indx < MAX_NUM_FRAMES)
  {
    pFrame                          = &(pxvTM->frameArray[indx]);
    pxvTM->frameAllocFlags[indxArr] = allocFlags | (((uint32_t) 0x1u) << indxShift);
    pxvTM->frameCount++;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_FRAME_BUFFER_FULL;
    return((xvFrame *) ((void *) XVTM_ERROR));
  }

  return(pFrame);
}

/**********************************************************************************
 * FUNCTION: xvFreeFrameHost()
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

int32_t xvFreeFrameHost(xvTileManager *pxvTM, xvFrame const *pFrame)
{
  uint32_t indx, indxArr = 0, indxShift = 0, allocFlags = 0;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

  for (indx = 0; indx < MAX_NUM_FRAMES; indx++)
  {
    if (&(pxvTM->frameArray[indx]) == pFrame)
    {
      break;
    }
  }

  if (indx < MAX_NUM_FRAMES)
  {
    indxArr                         = indx >> 5u;
    indxShift                       = indx & 0x1Fu;
    allocFlags                      = pxvTM->frameAllocFlags[indxArr];
    pxvTM->frameAllocFlags[indxArr] = allocFlags & ~(0x1u << indxShift);
    pxvTM->frameCount--;
  }
  else
  {
    pxvTM->errFlag = XV_ERROR_BAD_ARG;
    return(XVTM_ERROR);
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvAllocateTileHost()
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

xvTile *xvAllocateTileHost(xvTileManager *pxvTM)
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
 * FUNCTION: xvCreateTileHost()
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

xvTile *xvCreateTileHost(xvTileManager *pxvTM, int32_t tileBuffSize, int32_t width, uint16_t height,
                     int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame,
                     uint16_t xvTileType, int32_t alignType)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, (xvTile *) ((void *) XVTM_ERROR), "NULL TM Pointer");
  XV_CHECK_ERROR(((width < 0) || (pitch < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR(((alignType != XVTM_EDGE_ALIGNED_N) && (alignType != XVTM_DATA_ALIGNED_N) && (alignType != XVTM_EDGE_ALIGNED_2N) && (alignType != XVTM_DATA_ALIGNED_2N)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
#ifndef XVTM_USE_XMEM
  XV_CHECK_ERROR((((color < (int32_t) 0) || (color >= (int32_t) pxvTM->numMemBanks)) && (color != (int32_t) XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
#else
  XV_CHECK_ERROR((((color < (int32_t) 0) || (color >= (int32_t) xmem_bank_get_num_banks())) && (color != (int32_t) XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
#endif

  int32_t channel     = XV_TYPE_CHANNELS(xvTileType);
  int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(xvTileType);
  int32_t bytePerPel;
  bytePerPel = bytesPerPix / channel;

  XV_CHECK_ERROR((((width + (2 * edgeWidth)) * channel) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  XV_CHECK_ERROR((tileBuffSize < (pitch * (height + (2 * edgeHeight)) * bytePerPel)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");

  if (pFrame != NULL)
  {
    XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
    XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile *) ((void *) XVTM_ERROR)), "XVTM_ERROR");
  }
  void *tileBuff = NULL;
  tileBuff = xvAllocateBufferHost(pxvTM, tileBuffSize, color, 128);

  if (tileBuff == (void *) XVTM_ERROR)
  {
    return((xvTile *) ((void *) XVTM_ERROR));
  }

  xvTile *pTile = xvAllocateTileHost(pxvTM);
  if ((void *) pTile == (void *) XVTM_ERROR)
  {
    //Type cast to void to avoid MISRA 17.7 violation
    (void) xvFreeBufferHost(pxvTM, tileBuff);
    return((xvTile *) ((void *) XVTM_ERROR));
  }

  SETUP_TILE(pTile, tileBuff, tileBuffSize, pFrame, width, height, pitch, xvTileType, edgeWidth, edgeHeight, 0, 0, alignType);
  return(pTile);
}

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_predicated_wideHost()
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

int32_t xvAddIdmaRequestMultiChannel_predicated_wideHost(int32_t dmaChannel, xvTileManager *pxvTM, uint64_t pdst64,
                                                     uint64_t psrc64, size_t rowSize,
                                                     int32_t numRows, int32_t srcPitch, int32_t dstPitch,
                                                     int32_t interruptOnCompletion, uint32_t  *pred_mask)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, (XVTM_ERROR), "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR(((((void *) (uint32_t) pdst64) == NULL) || (((void *) (uint32_t) psrc64) == NULL)), pxvTM->errFlag = XV_ERROR_POINTER_NULL, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR(pred_mask == NULL, pxvTM->errFlag = XV_ERROR_POINTER_NULL, XVTM_ERROR, "XVTM_ERROR");

  XV_CHECK_ERROR(((((int32_t) rowSize) <= 0) || (numRows <= 0) || (srcPitch < 0) || (dstPitch < 0) || (interruptOnCompletion < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR((((((int32_t) rowSize) > srcPitch) && (srcPitch != 0)) || ((((int32_t) rowSize) > dstPitch) && (dstPitch != 0))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR(!(((psrc64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint8_t *) ((uint32_t) (psrc64 & 0x00000000ffffffffLLU))) && XV_PTR_END_IN_DRAM((((uint64_t) (uint32_t) (psrc64 & 0x00000000ffffffffLLU)) + (((uint64_t)srcPitch) * ((uint64_t)numRows))))) || \
                   ((pdst64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint8_t *) ((uint32_t) (pdst64 & 0x00000000ffffffffLLU))) && XV_PTR_END_IN_DRAM((((uint64_t) (uint32_t) (pdst64 & 0x00000000ffffffffLLU)) + (((uint64_t)dstPitch) * ((uint64_t)numRows)))))),  \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");

  //predicated buffer needs to be in local memory
  XV_CHECK_ERROR(!((XV_PTR_START_IN_DRAM((uint8_t *) pred_mask)) && (XV_PTR_END_IN_DRAM(((uint64_t) (uint32_t) (void *)pred_mask) + ((uint64_t)((((uint32_t)numRows) + 7u) >> 3u))))), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");

  XV_CHECK_ERROR(!(((((uint32_t) (void *)pred_mask) & 0x03u)) == 0u), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");

  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }

  TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  pred_buffer: %8.8x, rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, (uintptr_t) pred_mask, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag);

  void *pbuf;
  switch(dmaChannel){
  case 0:
	  pbuf=pxvTM->pdmaObj0;
	  break;
  case 1:
  	  pbuf=pxvTM->pdmaObj1;
  	  break;
  case 2:
  	  pbuf=pxvTM->pdmaObj2;
  	  break;
  case 3:
  	  pbuf=pxvTM->pdmaObj3;
  	  break;
  default:
	   pbuf=pxvTM->pdmaObj0;
   break;
  }
  idma_status_t status;
  status = idma_add_2d_pred_desc64_wide((idma_buffer_t*)pbuf, &pdst64, &psrc64, rowSize, intrCompletionFlag, pred_mask, (uint32_t) numRows, (uint32_t) srcPitch, (uint32_t) dstPitch);
  XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");
  
  dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

  return(dmaIndex);
}
#endif

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: addIdmaRequestInlineMultiChannel_predicated_wide()
 *
 * DESCRIPTION:
 *     Add iDMA predicated transfer request using wide addresses
 *     Inline function without any sanity checks.
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

static inline int32_t addIdmaRequestInlineMultiChannel_predicated_wideHost( xvTileManager * const pxvTM, int32_t dmaChannel,
                                                                       uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                                                       int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, uint32_t* pred_mask)
{
  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }

  TM_LOG_PRINT("line=%d, src: %llx, dst: %llx,  %8.8x, rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, (uintptr_t ) pred_mask, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
   idma_status_t status; 
   status =   idma_add_2d_pred_desc64_wide((idma_buffer_t*)pbuf, &pdst64, &psrc64, rowSize, intrCompletionFlag, pred_mask, (uint32_t) numRows, (uint32_t) srcPitch, (uint32_t) dstPitch);
   XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR"); 
   dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line=%d, dmaIndex: %d\n", __LINE__, dmaIndex);
  return(dmaIndex);
}
#endif

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wideHost()
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


int32_t xvAddIdmaRequestMultiChannel_wideHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                          uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                          int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion)
{
  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, (XVTM_ERROR), "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR(((pdst64 == 0LLu) || (psrc64 == 0LLu)), pxvTM->errFlag = XV_ERROR_POINTER_NULL, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR(((rowSize <= 0) || (numRows <= 0) || (srcPitch < 0) || (dstPitch < 0) || (interruptOnCompletion < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR((((rowSize > ((uint32_t) srcPitch)) && (srcPitch != 0)) || ((rowSize > ((uint32_t) dstPitch)) && (dstPitch != 0))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR((!(((psrc64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint32_t) (psrc64 & 0x00000000ffffffffLLU)) && XV_PTR_END_IN_DRAM(((uint64_t) (uint32_t) (psrc64 & 0x00000000ffffffffLLU)) + (((uint64_t)srcPitch) * ((uint64_t)numRows)))) || \
                    ((pdst64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint32_t) (pdst64 & 0x00000000ffffffffLLU)) && XV_PTR_END_IN_DRAM(((uint64_t) (uint32_t) (pdst64 & 0x00000000ffffffffLLU)) + (((uint64_t)dstPitch) * ((uint64_t)numRows)))))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");


  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }

  TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }

   idma_status_t status;
   status = idma_add_2d_desc64_wide((idma_buffer_t*)pbuf, &pdst64, &psrc64, rowSize, \
     intrCompletionFlag, (uint32_t) numRows, (uint32_t) srcPitch, (uint32_t) dstPitch);
   XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");

  dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

  return(dmaIndex);
}

/**********************************************************************************
 * FUNCTION: addIdmaRequestInlineMultiChannel_wide()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request using wide addresses.
 *     Inline function without any sanity checks.
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

static inline int32_t addIdmaRequestInlineMultiChannel_wideHost( xvTileManager * const pxvTM, int32_t dmaChannel, uint64_t pdst64,
                                                                 uint64_t psrc64, size_t rowSize, int32_t numRows, int32_t srcPitch,
                                                                 int32_t dstPitch, int32_t interruptOnCompletion)
{
  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }
  TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
  idma_status_t status;
  status = idma_add_2d_desc64_wide((idma_buffer_t*)pbuf,  &pdst64, &psrc64, rowSize, \
		    intrCompletionFlag, (uint32_t) numRows, (uint32_t) srcPitch, (uint32_t) dstPitch);
  XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");

  dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

  return(dmaIndex);
}

/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wide3DHost()
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

int32_t xvAddIdmaRequestMultiChannel_wide3DHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                            uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                            int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, int32_t srcTilePitch,
                                            int32_t dstTilePitch, int32_t numTiles)
{
  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR(((pdst64 == 0LLu) || (psrc64 == 0LLu)), pxvTM->errFlag = XV_ERROR_POINTER_NULL, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR((((int32_t) rowSize <= 0) || (numRows <= 0) || (srcPitch < 0) || (dstPitch < 0) || (interruptOnCompletion < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR(((numTiles <= 0) || (srcTilePitch < 0) || (dstTilePitch < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR((((((int32_t) rowSize) > srcPitch) && (srcPitch != 0)) || ((((int32_t) rowSize) > dstPitch) && (dstPitch != 0))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");
  XV_CHECK_ERROR(((((((int32_t) rowSize) * numRows) > (srcTilePitch)) && (srcTilePitch > 0)) || (((((int32_t) rowSize) * numRows) > (dstTilePitch)) && (dstTilePitch > 0))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");

  XV_CHECK_ERROR((!(((psrc64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint8_t *) ((uint32_t) (psrc64 & 0x00000000ffffffffLLU))) && XV_PTR_END_IN_DRAM((((uint64_t) (uint32_t) (psrc64 & 0x00000000ffffffffLLU))) + (((uint64_t)srcTilePitch) * ((uint64_t)numTiles)))) || \
                    ((pdst64 <= 0x00000000ffffffffLLU) && XV_PTR_START_IN_DRAM((uint8_t *) ((uint32_t) (pdst64 & 0x00000000ffffffffLLU))) && XV_PTR_END_IN_DRAM((((uint64_t) (uint32_t) (pdst64 & 0x00000000ffffffffLLU))) + (((uint64_t)dstTilePitch) * ((uint64_t)numTiles)))))), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XVTM_ERROR");



  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }

  TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, \
    srcTilePitch: %d, dstTilePitch: %d", \
               __LINE__, psrc64, pdst64, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag, srcTilePitch, dstTilePitch);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
   idma_status_t status;
   status = idma_add_3d_desc64_wide((idma_buffer_t*)pbuf, &pdst64, &psrc64, intrCompletionFlag, rowSize, (uint32_t) numRows, (uint32_t) numTiles, (uint32_t) srcPitch, (uint32_t) dstPitch,
                                         (uint32_t) srcTilePitch, (uint32_t) dstTilePitch);
   XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");

   dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

  return(dmaIndex);
}

// Part of tile reuse. Checks X direction boundary condition and performs DMA transfers
static int32_t solveForX(xvTileManager *pxvTM, int32_t dmaChannel, xvTile const *pTile, uint8_t *pCurrBuff, uint8_t *pPrevBuff,
                         int32_t y1, int32_t y2, int32_t x1, int32_t x2, int32_t px1, int32_t px2, int32_t tp, int32_t ptp, int32_t interruptOnCompletion)
{
  int32_t pixWidth;
  int32_t dmaIndex, framePitch, bytes, bytesCopy;

  uint64_t pSrc64, pDst64;
  uint8_t *pDst;
  uint8_t *pSrcCopy, *pDstCopy;

  dmaIndex = 0;
  xvFrame *pFrame = pTile->pFrame;
  pixWidth   = (int32_t) pFrame->pixelRes * (int32_t) pFrame->numChannels;
  framePitch = (int32_t) pFrame->framePitch * (int32_t) pFrame->pixelRes;

  // Case 1. Only left most part overlaps
  if ((px1 <= x1) && (px2 < x2))
  {
    pSrcCopy  = &pPrevBuff[((x1 - px1) * pixWidth)];
    pDstCopy  = pCurrBuff;
    bytesCopy = ((px2 - x1) + 1) * pixWidth;
    pSrc64    = (uint64_t) ((uint32_t) (void *)pSrcCopy);
    pDst64    = (uint64_t) ((uint32_t) (void *)pDstCopy);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM,dmaChannel, pDst64, pSrc64, (uint32_t) bytesCopy, (y2 - y1) + 1, ptp, tp, 0);


    pSrc64   = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + (((int64_t) px2 + (int64_t) 1) * (int64_t) pixWidth));
    pDst     = &pCurrBuff[(((px2 - x1) + 1) * pixWidth)];
    bytes    = (x2 - px2) * pixWidth;
    pDst64   = (uint64_t) ((uint32_t) (void *)pDst);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel,pDst64, pSrc64, (uint32_t) bytes, (y2 - y1) + 1, framePitch, tp, interruptOnCompletion);

  }

  // Case 2. Only mid part overlaps
  if ((x1 < px1) && (px2 < x2))
  {
    pSrc64   = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
    pDst     = pCurrBuff;
    bytes    = (px1 - x1) * pixWidth;
    pDst64   = (uint64_t) ((uint32_t) (void *)pDst);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM,dmaChannel, pDst64, pSrc64, (uint32_t) bytes, (y2 - y1) + 1, framePitch, tp, 0);

    pSrcCopy  = pPrevBuff;
    pDstCopy  = &pCurrBuff[((px1 - x1) * pixWidth)];
    bytesCopy = ((px2 - px1) + 1) * pixWidth;
    pSrc64    = (uint64_t) ((uint32_t) (void *)pSrcCopy);
    pDst64    = (uint64_t) ((uint32_t) (void *)pDstCopy);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM,dmaChannel, pDst64, pSrc64, (uint32_t) bytesCopy, (y2 - y1) + 1, ptp, tp, 0);

    pSrc64   = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + (((int64_t) px2 + (int64_t) 1) * (int64_t) pixWidth));
    pDst     = &pCurrBuff[(((px2 - x1) + 1) * pixWidth)];
    bytes    = (x2 - px2) * pixWidth;
    pDst64   = (uint64_t) ((uint32_t) (void *)pDst);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel,pDst64, pSrc64, (uint32_t) bytes, (y2 - y1) + 1, framePitch, tp, interruptOnCompletion);

  }

  // Case 3. Only right part overlaps
  if ((x1 < px1) && (x2 <= px2))
  {
    pSrc64   = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
    pDst     = pCurrBuff;
    bytes    = (px1 - x1) * pixWidth;
    pDst64   = (uint64_t) ((uint32_t) (void *)pDst);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel,pDst64, pSrc64, (uint32_t) bytes, (y2 - y1) + 1, framePitch, tp, 0);

    pSrcCopy  = pPrevBuff;
    pDstCopy  = &pCurrBuff[((px1 - x1) * pixWidth)];
    bytesCopy = ((x2 - px1) + 1) * pixWidth;
    pSrc64    = (uint64_t) ((uint32_t) (void *)pSrcCopy);
    pDst64    = (uint64_t) ((uint32_t) (void *)pDstCopy);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel,pDst64, pSrc64, (uint32_t) bytesCopy, (y2 - y1) + 1, ptp, tp, interruptOnCompletion);
  }

  // Case 4. All three regions overlaps
  if ((px1 <= x1) && (x2 <= px2))
  {
    pSrcCopy  = &pPrevBuff[((x1 - px1) * pixWidth)];
    pDstCopy  = pCurrBuff;
    bytesCopy = ((x2 - x1) + 1) * pixWidth;
    pSrc64    = (uint64_t) ((uint32_t) (void *)pSrcCopy);
    pDst64    = (uint64_t) ((uint32_t) (void *)pDstCopy);

    dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel,pDst64, pSrc64, (uint32_t) bytesCopy, (y2 - y1) + 1, ptp, tp, interruptOnCompletion);
  }

  return(dmaIndex);
}

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannelPredicatedHost()
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

int32_t xvReqTileTransferInMultiChannelPredicatedHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                                  xvTile *pTile, int32_t interruptOnCompletion, uint32_t* pred_mask)
{
  xvFrame *pFrame;
  int32_t frameWidth, frameHeight, framePitch, tileWidth, tilePitch;
  uint32_t tileHeight, statusFlag;
  int32_t x1, y1, x2, y2, dmaHeight, dmaWidthBytes, dmaIndex;
  int32_t tileIndex;
  uint8_t framePadLeft, framePadRight, framePadTop, framePadBottom;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
  int32_t pixWidth, pixRes;
  uint64_t srcPtr64, dstPtr64;
  uint8_t *dstPtr, *edgePtr;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_TILE(pTile, pxvTM);
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_BAD_INTERRUPT_FLAG");
  pFrame = pTile->pFrame;
  XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame->pFrameBuff == 0u) || (pFrame->pFrameData == 0u)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_BUFF_NULL");

  int32_t channel     = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
  int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
  int32_t bytePerPel;
  bytePerPel = bytesPerPix / channel;

  XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PIXEL_MISMATCH");
  XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_CHANNEL_MISMATCH");

  pTile->pPrevTile  = NULL;
  pTile->reuseCount = 0;
  frameWidth        = pFrame->frameWidth;
  frameHeight       = pFrame->frameHeight;
  framePadLeft      = pFrame->leftEdgePadWidth;
  framePadRight     = pFrame->rightEdgePadWidth;
  framePadTop       = pFrame->topEdgePadHeight;
  framePadBottom    = pFrame->bottomEdgePadHeight;
  pixRes            = (int32_t) pFrame->pixelRes;
  pixWidth          = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  framePitch        = pFrame->framePitch * pixRes;

  tileWidth      = pTile->width;
  tileHeight     = pTile->height;
  tilePitch      = pTile->pitch * pixRes;
  tileEdgeLeft   = pTile->tileEdgeLeft;
  tileEdgeRight  = pTile->tileEdgeRight;
  tileEdgeTop    = pTile->tileEdgeTop;
  tileEdgeBottom = pTile->tileEdgeBottom;
  XV_CHECK_ERROR((tileEdgeLeft | tileEdgeRight | tileEdgeTop | tileEdgeBottom), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_NON_ZERO_EDGE");


  statusFlag = pTile->status;

  XV_CHECK_ERROR(pred_mask == NULL, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NULL");

  //predicated buffer needs to be in local memory
  XV_CHECK_ERROR(!((XV_PTR_START_IN_DRAM((uint8_t *) pred_mask)) && (XV_PTR_END_IN_DRAM(((uint8_t *) pred_mask) + ((tileHeight + 7) >> 3)))), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NOT_IN_DRAM");

  //predicated buffer must have a 4 byte alignment.
  XV_CHECK_ERROR(((((uint32_t) (void *)(pred_mask)) & 0x03u) != 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NOT_ALIGNED");


  //Tile can not lie outside frame.
  XV_CHECK_ERROR((pTile->x < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_LEFT_EDGE_OUTSIDE_FRAME");
  XV_CHECK_ERROR((pTile->y < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_TOP_EDGE_OUTSIDE_FRAME");
  //Tile can not lie outside frame.
  XV_CHECK_ERROR(((pTile->x + tileWidth) > frameWidth), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_LEFT_EDGE_OUTSIDE_FRAME");
  XV_CHECK_ERROR((((int32_t) pTile->y + (int32_t) tileHeight) > frameHeight), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_BOTTOM_EDGE_OUTSIDE_FRAME");

  XV_CHECK_ERROR(tileHeight == 0u, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_HEIGHT_ZERO");
  XV_CHECK_ERROR(tileWidth == 0, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_WIDTH_ZERO");


  // 1. CHECK IF EXTRA PADDING NEEDED
  // Check top and bottom borders
  y1 = pTile->y;

  y2 = (int32_t) (pTile->y + ((int32_t) tileHeight - 1));

  // Check left and right borders
  x1 = pTile->x;

  x2 = pTile->x + ((int32_t) tileWidth - 1) + ((int32_t) tileEdgeRight);

  // 2. FILL ALL TILE and DMA RELATED DATA
  // No Need to align srcPtr and dstPtr as DMA does not need aligned start
  srcPtr64      = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
  dmaHeight     = (y2 - y1) + 1;
  dmaWidthBytes = ((x2 - x1) + 1) * pixWidth;
  edgePtr       = &((uint8_t *) pTile->pData)[0];
  dstPtr        = &edgePtr[0];  // For DMA

  dstPtr64 = (uint64_t) ((uint32_t) (void *)dstPtr);

  dmaIndex = addIdmaRequestInlineMultiChannel_predicated_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, interruptOnCompletion, pred_mask);

  pTile->status = statusFlag | XV_TILE_STATUS_DMA_ONGOING;
  tileIndex     = (pxvTM->tileDMAstartIndex[dmaChannel] + pxvTM->tileDMApendingCount[dmaChannel]) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
  pxvTM->tileDMApendingCount[dmaChannel]++;
  pxvTM->tileProcQueue[dmaChannel][tileIndex] = pTile;
  pTile->dmaIndex                             = dmaIndex;
  return(XVTM_SUCCESS);
}
#endif

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannelHost()
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

int32_t xvReqTileTransferInMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                        xvTile *pTile, xvTile *pPrevTile, int32_t interruptOnCompletion)
{
  xvFrame *pFrame;
  int32_t frameWidth, frameHeight, framePitch, tileWidth, tilePitch;
  uint32_t tileHeight, statusFlag;
  int32_t x1, y1, x2, y2, dmaHeight, dmaWidthBytes, dmaIndex = 0;
  int32_t tileIndex;
  uint8_t framePadLeft, framePadRight, framePadTop, framePadBottom;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
  int32_t extraEdgeTop, extraEdgeBottom, extraEdgeLeft, extraEdgeRight;
  int32_t pixWidth, pixRes;
  uint64_t srcPtr64, dstPtr64;
  uint8_t *dstPtr, *pPrevBuff, *pCurrBuff, *edgePtr;
  int32_t px1, px2, py1, py2;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_TILE(pTile, pxvTM);
  pFrame = pTile->pFrame;
  XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame->pFrameBuff == 0u) || (pFrame->pFrameData == 0u)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

  int32_t channel     = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
  int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
  int32_t bytePerPel;
  bytePerPel = bytesPerPix / channel;

  XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pTile->pPrevTile  = NULL;
  pTile->reuseCount = 0;
  frameWidth        = pFrame->frameWidth;
  frameHeight       = pFrame->frameHeight;
  framePadLeft      = pFrame->leftEdgePadWidth;
  framePadRight     = pFrame->rightEdgePadWidth;
  framePadTop       = pFrame->topEdgePadHeight;
  framePadBottom    = pFrame->bottomEdgePadHeight;
  pixRes            = (int32_t) pFrame->pixelRes;
  pixWidth          = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  framePitch        = pFrame->framePitch * pixRes;

  tileWidth      = pTile->width;
  tileHeight     = pTile->height;
  tilePitch      = pTile->pitch * pixRes;
  tileEdgeLeft   = pTile->tileEdgeLeft;
  tileEdgeRight  = pTile->tileEdgeRight;
  tileEdgeTop    = pTile->tileEdgeTop;
  tileEdgeBottom = pTile->tileEdgeBottom;

  statusFlag = pTile->status;

  // 1. CHECK IF EXTRA PADDING NEEDED
  // Check top and bottom borders
  y1 = pTile->y - ((int32_t) tileEdgeTop);
  if (y1 > (frameHeight + (int32_t) framePadBottom))
  {
    y1 = frameHeight + (int32_t) framePadBottom;
  }
  if (y1 < -(int32_t) framePadTop)
  {
    extraEdgeTop = -(int32_t) framePadTop - y1;
    y1           = -(int32_t) framePadTop;
    statusFlag  |= XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeTop = 0;
  }

  y2 = (int32_t) (pTile->y + ((int32_t) tileHeight - 1) + (int32_t) tileEdgeBottom);
  if (y2 < -(int32_t) framePadTop)
  {
    y2 = (int32_t) (-(int32_t) framePadTop - 1);
  }
  if (y2 > (((int32_t) frameHeight - 1) + (int32_t) framePadBottom))
  {
    extraEdgeBottom = y2 - (((int32_t) frameHeight - 1) + (int32_t) framePadBottom);
    y2              = ((int32_t) frameHeight - 1) + (int32_t) framePadBottom;
    statusFlag     |= XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeBottom = 0;
  }

  // Check left and right borders
  x1 = pTile->x - ((int32_t) tileEdgeLeft);
  if (x1 > (((int32_t) frameWidth) + ((int32_t) framePadRight)))
  {
    x1 = ((int32_t) frameWidth) + ((int32_t) framePadRight);
  }
  if (x1 < -((int32_t) framePadLeft))
  {
    extraEdgeLeft = (-(int32_t) framePadLeft) - x1;
    x1            = -(int32_t) framePadLeft;
    statusFlag   |= XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeLeft = 0;
  }

  x2 = pTile->x + ((int32_t) tileWidth - 1) + ((int32_t) tileEdgeRight);
  if (x2 < (-(int32_t) framePadLeft))
  {
    x2 = (-(int32_t) framePadLeft - 1);
  }
  if (x2 > (((int32_t) frameWidth - 1) + (int32_t) framePadRight))
  {
    extraEdgeRight = x2 - (((int32_t) frameWidth - 1) + (int32_t) framePadRight);
    x2             = ((int32_t) frameWidth - 1) + (int32_t) framePadRight;
    statusFlag    |= XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeRight = 0;
  }

  // 2. FILL ALL TILE and DMA RELATED DATA
  // No Need to align srcPtr and dstPtr as DMA does not need aligned start
  srcPtr64      = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
  dmaHeight     = (y2 - y1) + 1;
  dmaWidthBytes = ((x2 - x1) + 1) * pixWidth;
  edgePtr       = &((uint8_t *) pTile->pData)[-(((int32_t) tileEdgeTop * tilePitch) + ((int32_t) tileEdgeLeft * (int32_t) pixWidth))];
  dstPtr        = &edgePtr [(extraEdgeTop * tilePitch) + (extraEdgeLeft * pixWidth)];  // For DMA

  // 3. DATA REUSE FROM PREVIOUS TILE
  if ((dmaHeight > 0) && (dmaWidthBytes > 0))
  {
    if (pPrevTile != NULL)
    {
      XV_CHECK_ERROR(!(XV_TILE_IS_TILE(pPrevTile) > 0), (pxvTM)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "The argument is not a tile");
      XV_CHECK_ERROR(!(XV_TILE_STARTS_IN_DRAM(pPrevTile)), (pxvTM)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not start in DRAM");
      XV_CHECK_ERROR(!(XV_TILE_ENDS_IN_DRAM(pPrevTile)), (pxvTM)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not fit in DRAM");
      XV_CHECK_ERROR(!(XV_TILE_GET_TYPE(pTile) == XV_TILE_GET_TYPE(pPrevTile)), (pxvTM)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Invalid tile types");
      XV_CHECK_ERROR(!(XV_TILE_IS_CONSISTENT(pPrevTile)), (pxvTM)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Invalid buffer");


      py1       = pPrevTile->y - ((int32_t) pPrevTile->tileEdgeTop);
      py2       = ((pPrevTile->y + ((int32_t) pPrevTile->height)) - 1) + ((int32_t) pPrevTile->tileEdgeBottom);
      px1       = pPrevTile->x - ((int32_t) pPrevTile->tileEdgeLeft);
      px2       = ((pPrevTile->x + ((int32_t) pPrevTile->width)) - 1) + ((int32_t) pPrevTile->tileEdgeRight);
      pPrevBuff = &((uint8_t *) pPrevTile->pData)[ -((((int32_t) pPrevTile->tileEdgeTop) * ((int32_t) pPrevTile->pitch) * ((int32_t) pixRes)) + (((int32_t) pPrevTile->tileEdgeLeft) * ((int32_t) pixWidth)))]; // Same pixRes for current and prev tile
      // Case 1. Two tiles are totally independent. DMA entire tile. No copying
      if ((py2 < y1) || (py1 > y2) || (px2 < x1) || (px1 > x2))
      {
        dstPtr64 = (uint64_t) ((uint32_t) (void *)dstPtr);

        dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, interruptOnCompletion);

      }
      else
      {
        pTile->pPrevTile = pPrevTile;
        pPrevTile->reuseCount++;

        // Case 2. Only top part overlaps.
        if ((py1 <= y1) && (py2 < y2))
        {
          // Top part
          pCurrBuff = dstPtr;
          pPrevBuff = &pPrevBuff[(y1 - py1) * pPrevTile->pitch * pixRes];
          // Bottom part
          srcPtr64  = pFrame->pFrameData + (uint64_t) (int64_t) ((((int64_t) py2 + (int64_t) 1) * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
          dstPtr    = &dstPtr[((py2 + 1) - y1) * tilePitch];
          dmaHeight = y2 - py2;           // (y2+1) - (py2+1)
          dstPtr64  = (uint64_t) ((uint32_t) (void *)dstPtr);

          dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, 0);
          dmaIndex = solveForX(pxvTM, dmaChannel, pTile, pCurrBuff, pPrevBuff, y1, py2, x1, x2, px1, px2, tilePitch, pPrevTile->pitch * pixRes, interruptOnCompletion);
        }

        // Case 3. Only mid part overlaps
        if ((y1 < py1) && (py2 < y2))
        {
          // Top part
          dmaHeight = (py1 + 1) - y1;
          dstPtr64  = (uint64_t) ((uint32_t) (void *)dstPtr);

          dmaIndex = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, 0);

          // Mid overlapping part
          pCurrBuff = &dstPtr[((py1 - y1) * tilePitch)];
          // Bottom part
          srcPtr64  = pFrame->pFrameData + (uint64_t) (int64_t) ((((int64_t) py2 + (int64_t) 1) * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
          dstPtr    = &dstPtr[((py2 + 1) - y1) * tilePitch];
          dmaHeight = y2 - py2;           // (y2+1) - (py2+1)
          dstPtr64  = (uint64_t) ((uint32_t) (void *)dstPtr);

          dmaIndex = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, 0);

          dmaIndex = solveForX(pxvTM, dmaChannel, pTile, pCurrBuff, pPrevBuff, py1, py2, x1, x2, px1, px2, tilePitch, pPrevTile->pitch * pixRes, interruptOnCompletion);
        }

        // Case 4. Only bottom part overlaps
        if ((y1 < py1) && (y2 <= py2))
        {
          dmaHeight = py1 - y1;
          dstPtr64  = (uint64_t) ((uint32_t) (void *)dstPtr);

          dmaIndex  = addIdmaRequestInlineMultiChannel_wideHost(pxvTM,dmaChannel,  dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, 0);

          pCurrBuff = &dstPtr[(dmaHeight * tilePitch)];

          dmaIndex  = solveForX(pxvTM, dmaChannel, pTile, pCurrBuff, pPrevBuff, py1, y2, x1, x2, px1, px2, tilePitch, pPrevTile->pitch * pixRes, interruptOnCompletion);

        }

        // Case 5. All the parts overlaps
        if ((py1 <= y1) && (y2 <= py2))
        {
          pCurrBuff = dstPtr;
          pPrevBuff = &pPrevBuff[(y1 - py1) * pPrevTile->pitch * pixRes];

          dmaIndex  = solveForX(pxvTM, dmaChannel, pTile, pCurrBuff, pPrevBuff, y1, y2, x1, x2, px1, px2, tilePitch, pPrevTile->pitch * pixRes, interruptOnCompletion);

        }
      }
    }
    else
    {
      dstPtr64 = (uint64_t) ((uint32_t) (void *)dstPtr);

      dmaIndex = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes, dmaHeight, framePitch, tilePitch, interruptOnCompletion);

    }
  }
  else
  {
    dmaIndex    = XVTM_DUMMY_DMA_INDEX;
    statusFlag |= XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED;
  }
  pTile->status = statusFlag | XV_TILE_STATUS_DMA_ONGOING;
  tileIndex     = (pxvTM->tileDMAstartIndex[dmaChannel] + pxvTM->tileDMApendingCount[dmaChannel]) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
  pxvTM->tileDMApendingCount[dmaChannel]++;
  pxvTM->tileProcQueue[dmaChannel][tileIndex] = pTile;
  pTile->dmaIndex                             = dmaIndex;
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFastMultiChannelHost()
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

int32_t xvReqTileTransferInFastMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                            xvTile *pTile, int32_t interruptOnCompletion)
{
  xvFrame *pFrame;
  int32_t frameWidth, frameHeight, framePitch, tileWidth, tilePitch;
  uint32_t tileHeight, statusFlag;
  int32_t x1, y1, x2, y2, dmaHeight, dmaWidthBytes, dmaIndex;//, copyRowBytes;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;
  int32_t extraEdgeTop, extraEdgeBottom, extraEdgeLeft, extraEdgeRight;
  uint64_t srcPtr64, dstPtr64;
  uint8_t /**srcPtr,*/ *dstPtr, *edgePtr;
  int32_t pixWidth, pixRes;
  int32_t interruptOnCompletionLocal = interruptOnCompletion;

  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pFrame = pTile->pFrame;

  frameWidth  = pFrame->frameWidth;
  frameHeight = pFrame->frameHeight;
  pixRes      = (int32_t) pFrame->pixelRes;
  pixWidth    = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  framePitch  = pFrame->framePitch * pixRes;

  tileWidth      = pTile->width;
  tileHeight     = pTile->height;
  tilePitch      = pTile->pitch * pixRes;
  tileEdgeLeft   = pTile->tileEdgeLeft;
  tileEdgeRight  = pTile->tileEdgeRight;
  tileEdgeTop    = pTile->tileEdgeTop;
  tileEdgeBottom = pTile->tileEdgeBottom;

  statusFlag      = pTile->status;
  extraEdgeTop    = 0;
  extraEdgeBottom = 0;
  extraEdgeLeft   = 0;
  extraEdgeRight  = 0;

  // 1. CHECK IF EXTRA PADDING NEEDED
  // Check top and bottom borders
  y1 = pTile->y - (int32_t) tileEdgeTop;
  if (y1 > frameHeight)
  {
    y1 = frameHeight;
  }
  if (y1 < 0)
  {
    extraEdgeTop = -y1;
    y1           = 0;
    statusFlag  |= XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED;
  }

  y2 = ((int32_t) pTile->y) + (((int32_t) tileHeight) - 1) + (((int32_t) tileEdgeBottom));
  if (y2 < 0)
  {
    y2 = -1;
  }
  if (y2 > (frameHeight - 1))
  {
    extraEdgeBottom = (y2 - frameHeight) + 1;
    y2              = frameHeight - 1;
    statusFlag     |= XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED;
  }

  // Check left and right borders
  x1 = pTile->x - (int32_t) tileEdgeLeft;
  if (x1 > frameWidth)
  {
    x1 = frameWidth;
  }
  if (x1 < 0)
  {
    extraEdgeLeft = -x1;
    x1            = 0;
    statusFlag   |= XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED;
  }

  x2 = pTile->x + ((int32_t) tileWidth - 1) + ((int32_t) tileEdgeRight);
  if (x2 < 0)
  {
    x2 = -1;
  }
  if (x2 > (frameWidth - 1))
  {
    extraEdgeRight = (x2 - frameWidth) + 1;
    x2             = frameWidth - 1;
    statusFlag    |= XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED;
  }

  // 2. FILL ALL TILE and DMA RELATED DATA
  // No Need to align srcPtr as DMA does not need aligned start
  // But dstPtr needs to be aligned
  dmaHeight     = (y2 - y1) + 1;
  dmaWidthBytes = ((x2 - x1) + 1) * pixWidth;

  interruptOnCompletionLocal = (interruptOnCompletionLocal != 0) ? (int32_t) DESC_NOTIFY_W_INT : 0;
  int32_t intrCompletionFlag;




  if ((dmaHeight > 0) && (dmaWidthBytes > 0))
  {
    srcPtr64 = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) y1 * (int64_t) framePitch) + ((int64_t) x1 * (int64_t) pixWidth));
    edgePtr  = &((uint8_t *) pTile->pData)[-(((int32_t) tileEdgeTop * (int32_t) tilePitch) + ((int32_t) tileEdgeLeft * pixWidth))];
    dstPtr   = &edgePtr[((extraEdgeTop * tilePitch) + ((int32_t) extraEdgeLeft * pixWidth))];   // For DMA
    //printf("yjl idma0 %d  %d %d   %d\n", tileEdgeTop,tileEdgeLeft,tileEdgeBottom, tileEdgeRight );
    //printf("yjl idma1 %d  %d %d     %d %d   \n",tileHeight,pTile->y , frameHeight, y2,y1);
    //printf("yjl idma1.1 %d  %d %d     %d    \n",(int32_t)extraEdgeTop,(int32_t) extraEdgeLeft, extraEdgeBottom , extraEdgeRight );

    intrCompletionFlag = interruptOnCompletionLocal;

    dstPtr64 = (uint64_t) ((uint32_t) (void *)dstPtr);

    TM_LOG_PRINT("line=%d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
                 __LINE__, srcPtr64, dstPtr64, dmaWidthBytes, dmaHeight, framePitch, tilePitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
    idma_status_t status;
    status =   idma_add_2d_desc64_wide((idma_buffer_t*)pbuf, &dstPtr64, &srcPtr64, (uint32_t) dmaWidthBytes, \
         (uint32_t) intrCompletionFlag, (uint32_t) dmaHeight, (uint32_t) framePitch, (uint32_t) tilePitch);
    XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");


    dmaIndex = idma_schedule_desc(dmaChannel, 1);

    TM_LOG_PRINT("line=%d, dmaIndex: %d\n", __LINE__, dmaIndex);

  }
  else
  {
    dmaIndex    = XVTM_DUMMY_DMA_INDEX;
    statusFlag |= XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED;
  }
  pTile->status   = statusFlag | XV_TILE_STATUS_DMA_ONGOING;
  pTile->dmaIndex = dmaIndex;



  return(XVTM_SUCCESS);
}

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannelPredicatedHost()
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

int32_t xvReqTileTransferOutMultiChannelPredicatedHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                                   xvTile *pTile, int32_t interruptOnCompletion, uint32_t *pred_mask)
{
  xvFrame *pFrame;
  uint8_t *srcPtr;
  uint64_t srcPtr64, dstPtr64;
  int32_t pixWidth, dmaIndex, tileIndex;
  uint32_t srcHeight;
  int32_t srcWidth, srcPitch;
  int32_t dstPitch, numRows, rowSize;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_TILE(pTile, pxvTM);
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_BAD_INTERRUPT_FLAG");

  pFrame = pTile->pFrame;
  XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame->pFrameBuff == 0u) || (pFrame->pFrameData == 0u)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_BUFF_NULL");

  int32_t channel     = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
  int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
  int32_t bytePerPel;
  bytePerPel = bytesPerPix / channel;

  XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PIXEL_MISMATCH");
  XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_CHANNEL_MISMATCH");




  tileEdgeLeft   = pTile->tileEdgeLeft;
  tileEdgeRight  = pTile->tileEdgeRight;
  tileEdgeTop    = pTile->tileEdgeTop;
  tileEdgeBottom = pTile->tileEdgeBottom;
  srcHeight      = pTile->height;
  srcWidth       = pTile->width;

  XV_CHECK_ERROR((tileEdgeLeft | tileEdgeRight | tileEdgeTop | tileEdgeBottom), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_NON_ZERO_EDGE");


  XV_CHECK_ERROR(pred_mask == NULL, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NULL");

  //predicated buffer needs to be in local memory
  XV_CHECK_ERROR(!((XV_PTR_START_IN_DRAM((uint8_t *) pred_mask)) && (XV_PTR_END_IN_DRAM(((uint8_t *) pred_mask) + ((srcHeight + 7) >> 3)))), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NOT_IN_DRAM");

  //predicated buffer must have a 4 byte alignment.
  XV_CHECK_ERROR(((((uint32_t) (void *)(pred_mask)) & 0x03u) != 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM_PRED_MASK_BUFF_NOT_ALIGNED");


  //Tile can not lie outside frame.
  XV_CHECK_ERROR((pTile->x < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_LEFT_EDGE_OUTSIDE_FRAME");
  XV_CHECK_ERROR((pTile->y < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_TOP_EDGE_OUTSIDE_FRAME");

  XV_CHECK_ERROR(srcHeight == 0u, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_HEIGHT_ZERO");
  XV_CHECK_ERROR(srcWidth == 0, pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TILE_WIDTH_ZERO");


  pixWidth = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  srcPitch = pTile->pitch * (int32_t) pFrame->pixelRes;
  dstPitch = pFrame->framePitch * (int32_t) pFrame->pixelRes;
  numRows  = XVTM_MIN((int32_t) srcHeight, pFrame->frameHeight - pTile->y);
  rowSize  = XVTM_MIN(srcWidth, (pFrame->frameWidth - pTile->x));

  if ((numRows > 0) && (rowSize > 0))
  {
    srcPtr   = (uint8_t *) pTile->pData;
    dstPtr64 = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) (pTile->y) * (int64_t) dstPitch) + ((int64_t) (pTile->x) * (int64_t) pixWidth));

    rowSize *= pixWidth;

    pTile->status = pTile->status | XV_TILE_STATUS_DMA_ONGOING;
    tileIndex     = (pxvTM->tileDMAstartIndex[dmaChannel] + pxvTM->tileDMApendingCount[dmaChannel]) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
    pxvTM->tileDMApendingCount[dmaChannel]++;
    pxvTM->tileProcQueue[dmaChannel][tileIndex] = pTile;
    srcPtr64                                    = (uint64_t) ((uint32_t) (void *)srcPtr);

    dmaIndex                                    = addIdmaRequestInlineMultiChannel_predicated_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) rowSize, numRows, srcPitch, dstPitch, interruptOnCompletion, pred_mask);

    pTile->dmaIndex                             = dmaIndex;
  }
  return(XVTM_SUCCESS);
}
#endif

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannelHost()
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
int32_t xvReqTileTransferOutMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                         xvTile *pTile, int32_t interruptOnCompletion)
{
  xvFrame *pFrame;
  uint8_t *srcPtr;
  uint64_t srcPtr64, dstPtr64;
  uint32_t srcHeight;
  int32_t pixWidth, dmaIndex, tileIndex;
  int32_t srcWidth, srcPitch;
  int32_t dstPitch, numRows, rowSize;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_TILE(pTile, pxvTM);
  pFrame = pTile->pFrame;
  XV_CHECK_ERROR(pFrame == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame->pFrameBuff == 0u) || (pFrame->pFrameData == 0u)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

  int32_t channel     = XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile));
  int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile));
  int32_t bytePerPel;
  bytePerPel = bytesPerPix / channel;

  XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pixWidth  = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  srcHeight = pTile->height;
  srcWidth  = pTile->width;
  srcPitch  = pTile->pitch * (int32_t) pFrame->pixelRes;
  dstPitch  = pFrame->framePitch * (int32_t) pFrame->pixelRes;
  numRows   = XVTM_MIN((int32_t) srcHeight, pFrame->frameHeight - pTile->y);
  rowSize   = XVTM_MIN(srcWidth, (pFrame->frameWidth - pTile->x));

  srcPtr   = (uint8_t *) pTile->pData;
  dstPtr64 = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) (pTile->y) * (int64_t) dstPitch) + ((int64_t) (pTile->x) * (int64_t) pixWidth));

  if (pTile->x < 0)
  {
    rowSize  += pTile->x;    // x is negative;
    srcPtr    = &srcPtr[(-pTile->x * pixWidth)];
    dstPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile->x) * (int64_t) pixWidth);
  }

  if (pTile->y < 0)
  {
    numRows  += pTile->y;    // y is negative;
    srcPtr    = &srcPtr[(-pTile->y * srcPitch)];
    dstPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile->y) * (int64_t) dstPitch);
  }
  rowSize *= pixWidth;

  if ((rowSize > 0) && (numRows > 0))
  {
    pTile->status = pTile->status | XV_TILE_STATUS_DMA_ONGOING;
    tileIndex     = (pxvTM->tileDMAstartIndex[dmaChannel] + pxvTM->tileDMApendingCount[dmaChannel]) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
    pxvTM->tileDMApendingCount[dmaChannel]++;
    pxvTM->tileProcQueue[dmaChannel][tileIndex] = pTile;
    srcPtr64                                    = (uint64_t) ((uint32_t) (void *)srcPtr);

    dmaIndex                                    = addIdmaRequestInlineMultiChannel_wideHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) rowSize, numRows, srcPitch, dstPitch, interruptOnCompletion);

    pTile->dmaIndex                             = dmaIndex;
  }
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFastMultiChannelHost()
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

int32_t xvReqTileTransferOutFastMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion)
{
  xvFrame *pFrame;
  uint64_t srcPtr64, dstPtr64;
  uint8_t *srcPtr;
  int32_t dmaIndex, pixWidth;
  uint32_t srcHeight;
  int32_t srcWidth, srcPitch;
  int32_t dstPitch, numRows, rowSize;

  pxvTM->errFlag = XV_ERROR_SUCCESS;

  pFrame = pTile->pFrame;

  pixWidth  = ((int32_t) pFrame->pixelRes) * ((int32_t) pFrame->numChannels);
  srcHeight = pTile->height;
  srcWidth  = pTile->width;
  srcPitch  = pTile->pitch * (int32_t) pFrame->pixelRes;
  dstPitch  = pFrame->framePitch * (int32_t) pFrame->pixelRes;
  numRows   = XVTM_MIN((int32_t) srcHeight, pFrame->frameHeight - pTile->y);
  rowSize   = XVTM_MIN(srcWidth, (pFrame->frameWidth - pTile->x));

  srcPtr   = (uint8_t *) pTile->pData;
  dstPtr64 = pFrame->pFrameData + (uint64_t) (int64_t) (((int64_t) (pTile->y) * (int64_t) dstPitch) + ((int64_t) (pTile->x) * (int64_t) pixWidth));

  if (pTile->x < 0)
  {
    rowSize  += pTile->x;    // x is negative;
    srcPtr    = &srcPtr[(-pTile->x * pixWidth)];
    dstPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile->x) * (int64_t) pixWidth);
  }

  if (pTile->y < 0)
  {
    numRows  += pTile->y;    // y is negative;
    srcPtr    = &srcPtr[(-pTile->y * srcPitch)];
    dstPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile->y) * (int64_t) dstPitch);
  }
  rowSize *= pixWidth;
  uint32_t intrCompletionFlag = (interruptOnCompletion != 0) ? ((uint32_t) DESC_NOTIFY_W_INT) : 0u;


  if ((rowSize > 0) && (numRows > 0))
  {
    pTile->status = pTile->status | XV_TILE_STATUS_DMA_ONGOING;
    srcPtr64      = (uint64_t) ((uint32_t) (void *)srcPtr);

    TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, srcPitch: %d, dstPitch: %d, flags: 0x%x, ",
                 __LINE__, srcPtr64, dstPtr64, rowSize, numRows, srcPitch, dstPitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
    idma_status_t status;
    status =   idma_add_2d_desc64_wide((idma_buffer_t*)pbuf, &dstPtr64, &srcPtr64,
		      (uint32_t) rowSize, intrCompletionFlag, (uint32_t) numRows,
		      (uint32_t) srcPitch, (uint32_t) dstPitch);
    XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");

  dmaIndex = idma_schedule_desc(dmaChannel, 1);

    TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

    pTile->dmaIndex = dmaIndex;
  }


  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvCheckForIdmaIndexMultiChannelHost()
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

int32_t xvCheckForIdmaIndexMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, int32_t index)
{
  int32_t retVal = 1;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR_NULL((index < 0), XVTM_ERROR, "XVTM_ERROR");

  retVal = idma_desc_done(dmaChannel, index);

  return(retVal);
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

int32_t xvCheckTileReadyMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile)
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
      retVal = xvCheckForIdmaIndexMultiChannelHost(dmaChannel, pxvTM, dmaIndex);
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
      retVal = xvCheckForIdmaIndexMultiChannelHost(dmaChannel, pxvTM, dmaIndex);
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
 * FUNCTION: xvSleepForTileMultiChannelHost()
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
int32_t xvSleepForTileMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_TILE(pTile, pxvTM);
#if (defined (XTENSA_SWVERSION_RH_2018_6)) && (defined(XTENSA_SWVERSION)) && ((XTENSA_SWVERSION > XTENSA_SWVERSION_RH_2018_6))
  DECLARE_PS();
#endif
  IDMA_DISABLE_INTS();
  int32_t status = xvCheckTileReadyMultiChannelHost(dmaChannel, (pxvTM), (pTile));
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    idma_sleep(dmaChannel);
    status = xvCheckTileReadyMultiChannelHost(dmaChannel, (pxvTM), (pTile));
  }
  IDMA_ENABLE_INTS();
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvWaitForiDMAMultiChannelHost()
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
int32_t xvWaitForiDMAMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  int32_t status = 0;

  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    status = idma_desc_done(dmaChannel, (int32_t) dmaIndex);
  }
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvSleepForiDMAMultiChannelHost()
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
int32_t xvSleepForiDMAMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

#if defined(XTENSA_SWVERSION_RH_2018_6) && defined(XTENSA_SWVERSION) && (XTENSA_SWVERSION > XTENSA_SWVERSION_RH_2018_6)
  DECLARE_PS();
#endif
  IDMA_DISABLE_INTS();
  int32_t status = idma_desc_done(dmaChannel, (int32_t) dmaIndex);
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    idma_sleep(dmaChannel);
    status = idma_desc_done(dmaChannel, (int32_t) dmaIndex);
  }
  IDMA_ENABLE_INTS();
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvWaitForTileFastMultiChannelHost()
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
int32_t xvWaitForTileFastMultiChannelHost(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile  *pTile)
{
  int32_t status = 0;

  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    status = idma_desc_done(dmaChannel, (pTile)->dmaIndex);
  }
  (pTile)->status = (pTile)->status & ~XV_TILE_STATUS_DMA_ONGOING;
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileFastMultiChannelHost()
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
int32_t xvSleepForTileFastMultiChannelHost(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile)
{
#if defined (XTENSA_SWVERSION_RH_2018_6) && defined(XTENSA_SWVERSION) && (XTENSA_SWVERSION > XTENSA_SWVERSION_RH_2018_6)
  DECLARE_PS();
#endif
  IDMA_DISABLE_INTS();
  int32_t status = idma_desc_done(dmaChannel, (pTile)->dmaIndex);
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    idma_sleep(dmaChannel);
    status = idma_desc_done(dmaChannel, (pTile)->dmaIndex);
  }
  IDMA_ENABLE_INTS();
  (pTile)->status = (pTile)->status & ~XV_TILE_STATUS_DMA_ONGOING;
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}


static void __attribute__ ((always_inline)) xvTileManagerCopyPaddingDataHost(uint8_t *pBuff, int32_t paddingVal, int32_t pixWidth, int32_t numBytes)
{
  xb_vec2Nx8U dvec1, *pdvecDst = (xb_vec2Nx8U *) (pBuff);
  valign vas1;
  int32_t wb;
  if ((((uint32_t) pixWidth) & 1u) == 1u)
  {
    dvec1 = paddingVal;
  }
  else if ((((uint32_t) pixWidth) & 2u) == 2u)
  {
    xb_vecNx16U vec1 = paddingVal;
    dvec1 = IVP_MOV2NX8_FROMNX16(vec1);
  }
  else
  {
    xb_vecN_2x32v vec1 = paddingVal;
    dvec1 = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(vec1));
  }
  vas1 = IVP_ZALIGN();

  for (wb = numBytes; wb > 0; wb -= (2 * IVP_SIMD_WIDTH))
  {
    IVP_SAV2NX8U_XP(dvec1, vas1, pdvecDst, wb);
  }
  IVP_SAPOS2NX8U_FP(vas1, pdvecDst);
}

/**********************************************************************************
 * FUNCTION: AddIdmaRequestMultiChannel_wide3D()
 *
 * DESCRIPTION:
 *     Add 3D iDMA transfer request using wide addresses.
 *     Inline function without any sanity checks.
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

static inline int32_t AddIdmaRequestMultiChannel_wide3DHost(xvTileManager * const pxvTM,int32_t dmaChannel,
                                                        uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                                        int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, int32_t srcTilePitch,
                                                        int32_t dstTilePitch, int32_t numTiles)
{
  int32_t dmaIndex;
  uint32_t intrCompletionFlag;

  if (interruptOnCompletion != 0)
  {
    intrCompletionFlag = DESC_NOTIFY_W_INT;
  }
  else
  {
    intrCompletionFlag = 0;
  }

  TM_LOG_PRINT("line = %d, src: %llx, dst: %llx,  rowsize: %d, numRows: %d, numTiles: %d, srcPitch: %d, dstPitch: %d, source2DTilePitch: %d, dst2DTilePitch: %d, flags: 0x%x, ",
               __LINE__, psrc64, pdst64, rowSize, numRows, numTiles, srcPitch, dstPitch, srcTilePitch, dstTilePitch, intrCompletionFlag);

  void *pbuf;
   switch(dmaChannel){
   case 0:
 	  pbuf=pxvTM->pdmaObj0;
 	  break;
   case 1:
   	  pbuf=pxvTM->pdmaObj1;
   	  break;
   case 2:
   	  pbuf=pxvTM->pdmaObj2;
   	  break;
   case 3:
   	  pbuf=pxvTM->pdmaObj3;
   	  break;
	default:
	   pbuf=pxvTM->pdmaObj0;
   break;
   }
   idma_status_t status;
   status =   idma_add_3d_desc64_wide((idma_buffer_t*)pbuf,&pdst64, &psrc64, intrCompletionFlag, rowSize, (uint32_t) numRows, (uint32_t) numTiles, (uint32_t) srcPitch, (uint32_t) dstPitch,
           (uint32_t) srcTilePitch, (uint32_t) dstTilePitch);
   XV_CHECK_ERROR(status != IDMA_OK, pxvTM->errFlag = XV_ERROR_IDMA, XVTM_ERROR, "XVTM_ERROR");
   
   dmaIndex = idma_schedule_desc(dmaChannel, 1);

  TM_LOG_PRINT("line = %d, dmaIndex: %d\n", __LINE__, dmaIndex);

  return(dmaIndex);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannel3DHost()
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

int32_t xvReqTileTransferInMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion)
{
  xvFrame3D *pFrame3D;
  int32_t frameWidth, frameHeight, frameDepth, framePitch, tileWidth, tilePitch;
  uint32_t tileHeight, tileDepth, statusFlag, num2DTiles;
  int32_t x1, y1, x2, y2, z1, z2, dmaHeight, dmaWidthBytes, dmaIndex;
  int32_t tileIndex;
  uint16_t framePadLeft, framePadRight, framePadTop, framePadBottom, framePadFront, framePadBack;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom, tileEdgeFront, tileEdgeBack;
  int32_t extraEdgeTop, extraEdgeBottom, extraEdgeLeft, extraEdgeRight, extraEdgeFront, extraEdgeBack;
  int32_t pixWidth, pixRes, framePadType;
  int32_t framePadVal;
  uint64_t srcPtr64, dstPtr64;
  uint8_t *dstPtr, *edgePtr;
  int32_t framePitchTileBytes, tilePitchTileBytes;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  XV_CHECK_TILE_3D(pTile3D, pxvTM);

  pFrame3D = pTile3D->pFrame;
  XV_CHECK_ERROR(pFrame3D == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame3D->pFrameBuff == 0LLu) || (pFrame3D->pFrameData == 0LLu)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

#if (XV_ERROR_LEVEL != XV_ERROR_LEVEL_NO_ERROR)
  int32_t channel     = (int32_t) (uint16_t) XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile3D));
  int32_t bytesPerPix = (int32_t) (uint16_t) XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile3D));
  int32_t bytePerPel  = bytesPerPix / channel;
#endif

  XV_CHECK_ERROR((pFrame3D->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((pFrame3D->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  frameWidth          = pFrame3D->frameWidth;
  frameHeight         = pFrame3D->frameHeight;
  framePadLeft        = pFrame3D->leftEdgePadWidth;
  framePadRight       = pFrame3D->rightEdgePadWidth;
  framePadTop         = pFrame3D->topEdgePadHeight;
  framePadBottom      = pFrame3D->bottomEdgePadHeight;
  frameDepth          = pFrame3D->frameDepth;
  framePadFront       = pFrame3D->frontEdgePadDepth;
  framePadBack        = pFrame3D->backEdgePadDepth;
  pixRes              = (int32_t) pFrame3D->pixelRes;
  pixWidth            = ((int32_t) pFrame3D->pixelRes) * ((int32_t) pFrame3D->numChannels);
  framePitch          = pFrame3D->framePitch * pixRes;
  framePitchTileBytes = (int32_t) pFrame3D->frame2DFramePitch * (int32_t) pFrame3D->pixelRes;

  tileWidth          = pTile3D->width;
  tileHeight         = pTile3D->height;
  tilePitch          = pTile3D->pitch * pixRes;
  tileEdgeLeft       = pTile3D->tileEdgeLeft;
  tileEdgeRight      = pTile3D->tileEdgeRight;
  tileEdgeTop        = pTile3D->tileEdgeTop;
  tileEdgeBottom     = pTile3D->tileEdgeBottom;
  tileEdgeFront      = pTile3D->tileEdgeFront;
  tileEdgeBack       = pTile3D->tileEdgeBack;
  tileDepth          = pTile3D->depth;
  tilePitchTileBytes = (int32_t) pTile3D->Tile2Dpitch * (int32_t) pFrame3D->pixelRes;

  statusFlag = pTile3D->status;

  // Check front and back borders
  z1 = pTile3D->z - (int32_t) tileEdgeFront;
  if (z1 > ((int32_t) frameDepth + (int32_t) framePadBack))
  {
    z1 = (int32_t) frameDepth + (int32_t) framePadBack;
  }
  if (z1 < -((int32_t) framePadFront))
  {
    extraEdgeFront = -((int32_t) framePadFront) - z1;
    z1             = -((int32_t) framePadFront);
    statusFlag    |= XV_TILE_STATUS_FRONT_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeFront = 0;
  }

  z2 = (int32_t) pTile3D->z + ((int32_t) tileDepth - 1) + (int32_t) tileEdgeBack;
  if (z2 < -((int32_t) framePadFront))
  {
    z2 = (int32_t) (-((int32_t) framePadFront) - 1);
  }
  if (z2 > (((int32_t) frameDepth - 1) + (int32_t) framePadBack))
  {
    extraEdgeBack = z2 - (((int32_t) frameDepth - 1) + (int32_t) framePadBack);
    z2            = ((int32_t) frameDepth - 1) + (int32_t) framePadBack;
    statusFlag   |= XV_TILE_STATUS_BACK_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeBack = 0;
  }

  // 1. CHECK IF EXTRA PADDING NEEDED
  // Check top and bottom borders
  y1 = pTile3D->y - (int32_t) tileEdgeTop;
  if (y1 > ((int32_t) frameHeight + (int32_t) framePadBottom))
  {
    y1 = (int32_t) frameHeight + (int32_t) framePadBottom;
  }
  if (y1 < -((int32_t) framePadTop))
  {
    extraEdgeTop = -((int32_t) framePadTop) - y1;
    y1           = -((int32_t) framePadTop);
    statusFlag  |= XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeTop = 0;
  }

  y2 = (int32_t) pTile3D->y + ((int32_t) tileHeight - 1) + (int32_t) tileEdgeBottom;
  if (y2 < -((int32_t) framePadTop))
  {
    y2 = (int32_t) (-((int32_t) framePadTop) - 1);
  }
  if (y2 > (((int32_t) frameHeight - 1) + (int32_t) framePadBottom))
  {
    extraEdgeBottom = y2 - (((int32_t) frameHeight - 1) + (int32_t) framePadBottom);
    y2              = ((int32_t) frameHeight - 1) + (int32_t) framePadBottom;
    statusFlag     |= XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeBottom = 0;
  }

  // Check left and right borders
  x1 = pTile3D->x - (int32_t) tileEdgeLeft;
  if (x1 > ((int32_t) frameWidth + (int32_t) framePadRight))
  {
    x1 = (int32_t) frameWidth + (int32_t) framePadRight;
  }
  if (x1 < -((int32_t) framePadLeft))
  {
    extraEdgeLeft = -((int32_t) framePadLeft) - x1;
    x1            = -((int32_t) framePadLeft);
    statusFlag   |= XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeLeft = 0;
  }

  x2 = pTile3D->x + ((int32_t) tileWidth - 1) + (int32_t) tileEdgeRight;
  if (x2 < -((int32_t) framePadLeft))
  {
    x2 = (int32_t) (-((int32_t) framePadLeft) - 1);
  }
  if (x2 > (((int32_t) frameWidth - 1) + (int32_t) framePadRight))
  {
    extraEdgeRight = x2 - (((int32_t) frameWidth - 1) + (int32_t) framePadRight);
    x2             = ((int32_t) frameWidth - 1) + (int32_t) framePadRight;
    statusFlag    |= XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED;
  }
  else
  {
    extraEdgeRight = 0;
  }

  // 2. FILL ALL TILE and DMA RELATED DATA
  // No Need to align srcPtr and dstPtr as DMA does not need aligned start

  int32_t addtTrmp = (x1 * pixWidth) + (y1 * framePitch) + (z1 * framePitchTileBytes);
  srcPtr64      = pFrame3D->pFrameData + ((uint64_t) ((int64_t) addtTrmp));
  dmaHeight     = (y2 - y1) + 1;
  dmaWidthBytes = ((x2 - x1) + 1) * pixWidth;
  num2DTiles    = (uint32_t) (int32_t) ((z2 - z1) + (int32_t) 1);

  edgePtr = &((uint8_t *) pTile3D->pData)[ -(((int32_t) tileEdgeFront * tilePitchTileBytes) + ((int32_t) tileEdgeTop * tilePitch) + \
                                             ((int32_t) tileEdgeLeft * (int32_t) pixWidth))];
  dstPtr = &edgePtr[(extraEdgeTop * tilePitch) + (extraEdgeLeft * pixWidth) + (extraEdgeFront * tilePitchTileBytes)];          // For DMA

  if ((dmaHeight > (int32_t) 0) && (dmaWidthBytes > (int32_t) 0) && (num2DTiles > 0u))
  {
    {
      dstPtr64 = (uint64_t) ((uint32_t) (void *)dstPtr);

      dmaIndex = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) dmaWidthBytes,
                                                         dmaHeight, framePitch, tilePitch, interruptOnCompletion, framePitchTileBytes, tilePitchTileBytes, (int32_t) num2DTiles);

    }
  }
  else
  {
    //Tile is completely outside frame boundaries.
    //NOTE: EDGE PADDING NOT SUPPORTED for this scenario.
    //initialize first row of first 2D tile with zero/constant padding.

    statusFlag   = XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED;
    framePadVal  = (int32_t) pFrame3D->paddingVal;
    framePadType = (int32_t) pFrame3D->paddingType;
    if (framePadType != (int32_t) FRAME_CONSTANT_PADDING)
    {
      framePadVal = 0;
    }

    int32_t NumBytes = pixWidth * ((int32_t) tileWidth + (int32_t) tileEdgeLeft + (int32_t) tileEdgeRight);
    xvTileManagerCopyPaddingDataHost(edgePtr, framePadVal, pixWidth, NumBytes);
    //now set up a 2D DMA to copy first row to remaining rows of the tile
    srcPtr64 = (uint64_t) ((uint32_t) (void *)edgePtr);
    dstPtr64 = srcPtr64 + (uint64_t) ((int64_t) (int32_t) tilePitch);

    dmaIndex = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) NumBytes,
                                                 ((((int32_t) tileEdgeTop) + ((int32_t) tileHeight)) + ((int32_t) tileEdgeBottom - 1)), 0, tilePitch, interruptOnCompletion, 0, 0, 1);

  }
  statusFlag     |= (((((uint32_t) interruptOnCompletion & 1u) == 1u) ? 1u : 0) << XV_TILE_STATUS_INTERRUPT_ON_COMPLETION_SHIFT);
  pTile3D->status = statusFlag | XV_TILE_STATUS_DMA_ONGOING;
  tileIndex       = (pxvTM->tile3DDMAstartIndex[dmaChannel] + pxvTM->tile3DDMApendingCount[dmaChannel]) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
  pxvTM->tile3DDMApendingCount[dmaChannel]++;
  pxvTM->tile3DProcQueue[dmaChannel][tileIndex] = pTile3D;
  pTile3D->dmaIndex                             = dmaIndex;
  return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvCheckTileReadyMultiChannel3DHost()
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

int32_t xvCheckTileReadyMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  int32_t loopInd, index, retVal, dmaIndex=-1, dmaIndexTemp=-1, doneCount;
  uint32_t statusFlag, tileHeight, tileDepth;
  xvTile3D *pTile13D;
  xvFrame3D *pFrame3D;
  int32_t frameWidth, frameHeight, frameDepth, framePitch, tileWidth;
  int32_t tilePitch, tilePitch2DtileBytes, tile3DDMApendingCount;
  int32_t x1, y1, x2, y2, z1, z2;
  uint16_t framePadLeft, framePadRight, framePadTop, framePadBottom, framePadFront, framePadBack;
  uint16_t tileEdgeLeft, tileEdgeRight, tileEdgeTop, tileEdgeBottom, tileEdgeFront, tileEdgeBack;
  int32_t extraEdgeTop, extraEdgeBottom, extraEdgeLeft, extraEdgeRight, extraEdgeFront, extraEdgeBack;
  uint8_t framePadType;
  int32_t framePadVal, pixWidth;
  uint64_t srcPtr64, dstPtr64;
  uint32_t src2DtilePitch;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");


  XV_CHECK_TILE_3D(pTile3D, pxvTM);
  pFrame3D = pTile3D->pFrame;
  XV_CHECK_ERROR(pFrame3D == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame3D->pFrameBuff == 0LLu) || (pFrame3D->pFrameData == 0LLu)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

#if (XV_ERROR_LEVEL != XV_ERROR_LEVEL_NO_ERROR)
  int32_t channel     = (int32_t) (uint16_t) XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(pTile3D));
  int32_t bytesPerPix = (int32_t) (uint16_t) XV_TYPE_ELEMENT_SIZE(XV_TILE_GET_TYPE(pTile3D));
  int32_t bytePerPel  = bytesPerPix / channel;
#endif

  XV_CHECK_ERROR((pFrame3D->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((pFrame3D->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  if (pTile3D->status == 0u)
  {
    return(1);
  }

  doneCount             = 0;
  index                 = pxvTM->tile3DDMAstartIndex[dmaChannel];
  tile3DDMApendingCount = pxvTM->tile3DDMApendingCount[dmaChannel];
  for (loopInd = 0; loopInd < tile3DDMApendingCount; loopInd++)
  {
    index    = (((int32_t) pxvTM->tile3DDMAstartIndex[dmaChannel]) + ((int32_t) loopInd)) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
    pTile13D = pxvTM->tile3DProcQueue[dmaChannel][index];
    dmaIndex = pTile13D->dmaIndex;

    retVal = xvCheckForIdmaIndexMultiChannelHost(dmaChannel, pxvTM, dmaIndex);
    if (retVal == 1)
    {
      statusFlag = pTile13D->status;
      uint32_t InterruptFlag = (statusFlag >> XV_TILE_STATUS_INTERRUPT_ON_COMPLETION_SHIFT) & 1u;

      statusFlag = statusFlag & ~(XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_INTERRUPT_ON_COMPLETION);

      pFrame3D       = pTile13D->pFrame;
      pixWidth       = (int32_t) pFrame3D->pixelRes * (int32_t) pFrame3D->numChannels;
      frameWidth     = pFrame3D->frameWidth;
      frameHeight    = pFrame3D->frameHeight;
      frameDepth     = pFrame3D->frameDepth;
      framePitch     = (int32_t) pFrame3D->framePitch * (int32_t) pFrame3D->pixelRes;
      framePadLeft   = pFrame3D->leftEdgePadWidth;
      framePadRight  = pFrame3D->rightEdgePadWidth;
      framePadTop    = pFrame3D->topEdgePadHeight;
      framePadBottom = pFrame3D->bottomEdgePadHeight;
      framePadType   = pFrame3D->paddingType;
      framePadVal    = (int32_t) pFrame3D->paddingVal;
      framePadFront  = pFrame3D->frontEdgePadDepth;
      framePadBack   = pFrame3D->backEdgePadDepth;

      tileWidth            = pTile13D->width;
      tileHeight           = pTile13D->height;
      tileDepth            = pTile13D->depth;
      tilePitch            = pTile13D->pitch * (int32_t) pFrame3D->pixelRes;
      tileEdgeLeft         = pTile13D->tileEdgeLeft;
      tileEdgeRight        = pTile13D->tileEdgeRight;
      tileEdgeTop          = pTile13D->tileEdgeTop;
      tileEdgeBottom       = pTile13D->tileEdgeBottom;
      tilePitch2DtileBytes = pTile13D->Tile2Dpitch * (int32_t) pFrame3D->pixelRes;
      tileEdgeFront        = pTile13D->tileEdgeFront;
      tileEdgeBack         = pTile13D->tileEdgeBack;

      while (((statusFlag & XV_TILE_STATUS_EDGE_PADDING_NEEDED) != 0u) || ((statusFlag & XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED) != 0u))
      {
        // Check front and back borders
        z1 = pTile13D->z - (int32_t) tileEdgeFront;
        if (z1 > ((int32_t) frameDepth + (int32_t) framePadBack))
        {
          z1 = (int32_t) frameDepth + (int32_t) framePadBack;
        }
        if (z1 < -((int32_t) framePadFront))
        {
          extraEdgeFront = -((int32_t) framePadFront) - z1;
          z1             = -((int32_t) framePadFront);
        }
        else
        {
          extraEdgeFront = 0;
        }

        z2 = ((int32_t) pTile13D->z) + (((int32_t) tileDepth) - 1) + ((int32_t) tileEdgeBack);
        if (z2 < -((int32_t) framePadFront))
        {
          z2 = (int32_t) (-((int32_t) framePadFront) - 1);
        }
        if (z2 > (((int32_t) frameDepth - 1) + (int32_t) framePadBack))
        {
          extraEdgeBack = z2 - (((int32_t) frameDepth - 1) + (int32_t) framePadBack);
          z2            = ((int32_t) frameDepth - 1) + (int32_t) framePadBack;
        }
        else
        {
          extraEdgeBack = 0;
        }

        // 1. CHECK IF EXTRA PADDING NEEDED
        // Check top and bottom borders
        y1 = pTile13D->y - (int32_t) tileEdgeTop;
        if (y1 > ((int32_t) frameHeight + (int32_t) framePadBottom))
        {
          y1 = (int32_t) frameHeight + (int32_t) framePadBottom;
        }
        if (y1 < -((int32_t) framePadTop))
        {
          extraEdgeTop = -((int32_t) framePadTop) - y1;
          y1           = -((int32_t) framePadTop);
        }
        else
        {
          extraEdgeTop = 0;
        }

        y2 = ((int32_t) pTile13D->y) + (((int32_t) tileHeight) - 1) + ((int32_t) tileEdgeBottom);
        if (y2 < -((int32_t) framePadTop))
        {
          y2 = (int32_t) (-((int32_t) framePadTop) - 1);
        }
        if (y2 > (((int32_t) frameHeight - 1) + (int32_t) framePadBottom))
        {
          extraEdgeBottom = y2 - (((int32_t) frameHeight - 1) + (int32_t) framePadBottom);
          y2              = ((int32_t) frameHeight - 1) + (int32_t) framePadBottom;
        }
        else
        {
          extraEdgeBottom = 0;
        }

        // Check left and right borders
        x1 = pTile13D->x - (int32_t) tileEdgeLeft;
        if (x1 > ((int32_t) frameWidth + (int32_t) framePadRight))
        {
          x1 = (int32_t) frameWidth + (int32_t) framePadRight;
        }
        if (x1 < -((int32_t) framePadLeft))
        {
          extraEdgeLeft = -((int32_t) framePadLeft) - x1;
          x1            = -((int32_t) framePadLeft);
        }
        else
        {
          extraEdgeLeft = 0;
        }

        x2 = pTile13D->x + ((int32_t) tileWidth - 1) + (int32_t) tileEdgeRight;
        if (x2 < -((int32_t) framePadLeft))
        {
          x2 = (int32_t) (-((int32_t) framePadLeft) - 1);
        }
        if (x2 > (((int32_t) frameWidth - 1) + (int32_t) framePadRight))
        {
          extraEdgeRight = x2 - (((int32_t) frameWidth - 1) + (int32_t) framePadRight);
          x2             = ((int32_t) frameWidth - 1) + (int32_t) framePadRight;
        }
        else
        {
          extraEdgeRight = 0;
        }

        if ((statusFlag & XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_LEFT_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_LEFT_EDGE_PADDING_ONGOING | XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, extraEdgeLeft * pixWidth, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, extraEdgeLeft * pixWidth);

              //use same source buffer for all 2D tiles and all rows in wach 2D tile.
              //simply set source row pitch and 2Dtile pitch to zero.

              int32_t addrTemp = ((((int32_t) tileEdgeLeft) * pixWidth) + (((int32_t) tileEdgeTop - extraEdgeTop) * tilePitch)) + \
                                 ((((int32_t) tileEdgeFront - extraEdgeFront)) * ((int32_t) tilePitch2DtileBytes));
              dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp);
              srcPtr64 = (uint64_t) (uint32_t) pTile13D->pTemp;


              dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint64_t)((uint32_t) extraEdgeLeft * (uint32_t) pixWidth),
                                                                          ((y2 - y1) + 1), 0, tilePitch, (int32_t) InterruptFlag, 0, tilePitch2DtileBytes, ((z2 - z1) + 1));

            }
            else
            {
              //edge padding.
              int32_t i;
              int32_t addrTemp = (((((int32_t) tileEdgeLeft) - extraEdgeLeft) * pixWidth) + ((((int32_t) tileEdgeTop) - extraEdgeTop) * (int32_t) tilePitch)) + \
                                 (((((int32_t) tileEdgeFront) - extraEdgeFront)) * ((int32_t) tilePitch2DtileBytes));
              srcPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp);
              dstPtr64 = srcPtr64 - (uint64_t) (int64_t) pixWidth;


              for (i = 0; i < extraEdgeLeft; i++)
              {

                dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM,  dmaChannel, dstPtr64, srcPtr64, (uint32_t) pixWidth,
                      ((y2 - y1) + 1), tilePitch, tilePitch, (int32_t) InterruptFlag, tilePitch2DtileBytes, tilePitch2DtileBytes, ((z2 - z1) + 1));

                dstPtr64 -= (uint64_t) (int64_t) pixWidth;
              }
            }
            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_LEFT_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((statusFlag & XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_RIGHT_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_RIGHT_EDGE_PADDING_ONGOING | XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, extraEdgeRight * pixWidth, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, extraEdgeRight * pixWidth);

              int32_t addrTemp = ((((int32_t) tileEdgeTop) - extraEdgeTop) * ((int32_t) tilePitch)) + \
                                 (((((int32_t) tileEdgeFront) - extraEdgeFront) * ((int32_t) tilePitch2DtileBytes)) - (((x2 - pTile13D->x) + 1) * pixWidth));
              dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp);

              srcPtr64 = (uint64_t) (uint32_t) pTile13D->pTemp;

              dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint64_t)((uint32_t) extraEdgeRight * (uint32_t) pixWidth),
                                                                             ((y2 - y1) + 1), 0, tilePitch, (int32_t) InterruptFlag, 0, tilePitch2DtileBytes, ((z2 - z1) + 1));

            }
            else
            {
              //edge padding.
              int32_t i;

              int32_t addrTemp = ((((int32_t) tileEdgeTop) - extraEdgeTop) * ((int32_t) tilePitch)) + \
                                 (((((int32_t) tileEdgeFront) - extraEdgeFront) * ((int32_t) tilePitch2DtileBytes)) - ((((((int32_t) tileWidth) + ((int32_t) tileEdgeRight)) - extraEdgeRight) - 1) * ((int32_t) pixWidth)));
              srcPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp);
              dstPtr64 = srcPtr64 + (uint64_t) (int64_t) pixWidth;

              for (i = 0; i < extraEdgeRight; i++)
              {

                dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) pixWidth,
																((y2 - y1) + 1), tilePitch, tilePitch, (int32_t) InterruptFlag, tilePitch2DtileBytes, tilePitch2DtileBytes, ((z2 - z1) + 1));

                dstPtr64 += (uint64_t) (int64_t) pixWidth;
              }
            }

            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_RIGHT_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((statusFlag & XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_TOP_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_TOP_EDGE_PADDING_ONGOING | XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            int32_t RowSize = ((int32_t) tileWidth + (int32_t) tileEdgeRight + (int32_t) tileEdgeLeft) * pixWidth;

            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, RowSize, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, RowSize);
              srcPtr64       = (uint64_t) (uint32_t) pTile13D->pTemp;
              src2DtilePitch = 0;
            }
            else
            {
              int32_t addrTemp1 = ((((int32_t) tileEdgeTop) - extraEdgeTop) * tilePitch) + ((((int32_t) tileEdgeFront) - extraEdgeFront) * (int32_t) tilePitch2DtileBytes) + \
                                  (((int32_t) tileEdgeLeft) * pixWidth);
              srcPtr64       = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp1);
              src2DtilePitch = (uint32_t) tilePitch2DtileBytes;
            }

            int32_t addrTemp2 = ((((int32_t) tileEdgeTop) * ((int32_t) tilePitch)) + ((((int32_t) tileEdgeFront - extraEdgeFront)) * ((int32_t) tilePitch2DtileBytes))) + \
                                (((int32_t) tileEdgeLeft) * ((int32_t) pixWidth));
            dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp2);

			dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel,  dstPtr64, srcPtr64, (uint32_t) RowSize,
                        extraEdgeTop, 0, tilePitch, (int32_t) InterruptFlag, (int32_t) src2DtilePitch, tilePitch2DtileBytes, (z2 - z1) + 1);


            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_TOP_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((statusFlag & XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_BOTTOM_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_BOTTOM_EDGE_PADDING_ONGOING | XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            int32_t RowSize = ((int32_t) tileWidth + (int32_t) tileEdgeRight + (int32_t) tileEdgeLeft) * pixWidth;

            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, RowSize, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, RowSize);
              srcPtr64       = (uint64_t) (uint32_t) pTile13D->pTemp;
              src2DtilePitch = 0;
            }
            else
            {
              int32_t addrTemp3 = (((((int32_t) tileHeight) + ((int32_t) tileEdgeBottom)) - ((int32_t) extraEdgeBottom + 1)) * tilePitch) - \
                                  ((((int32_t) tileEdgeLeft) * ((int32_t) pixWidth)) + ((((int32_t) tileEdgeFront) - ((int32_t) extraEdgeFront)) * tilePitch2DtileBytes));
              srcPtr64       = ((uint64_t) (uint32_t) pTile13D->pData) + (uint64_t) ((int64_t) addrTemp3);
              src2DtilePitch = (uint32_t) tilePitch2DtileBytes;
            }
            int32_t addrTemp4 = (((((int32_t) tileHeight) + ((int32_t) tileEdgeBottom)) - extraEdgeBottom) * ((int32_t) tilePitch)) - \
                                ((((int32_t) tileEdgeLeft) * pixWidth) + ((((int32_t) tileEdgeFront) - extraEdgeFront) * ((int32_t) tilePitch2DtileBytes)));
            dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) + (uint64_t) ((int64_t) addrTemp4);

			dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) RowSize,
                        extraEdgeBottom, 0, tilePitch, (int32_t) InterruptFlag, (int32_t) src2DtilePitch, tilePitch2DtileBytes, (z2 - z1) + 1);


            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_BOTTOM_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((statusFlag & XV_TILE_STATUS_FRONT_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_FRONT_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_FRONT_EDGE_PADDING_ONGOING | XV_TILE_STATUS_FRONT_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            int32_t RowSize = ((int32_t) tileWidth + (int32_t) tileEdgeRight + (int32_t) tileEdgeLeft) * (int32_t) pixWidth;
            int32_t srcPitch;

            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, RowSize, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, RowSize);
              srcPtr64 = (uint64_t) (uint32_t) pTile13D->pTemp;
              srcPitch = 0;
            }
            else
            {
              int32_t addrTemp5 = ((int32_t) tileEdgeTop * (int32_t) tilePitch) + \
                                  ((int32_t) tileEdgeLeft * (int32_t) pixWidth) + (((int32_t) tileEdgeFront - extraEdgeFront) * (int32_t) tilePitch2DtileBytes);
              srcPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp5);
              srcPitch = tilePitch;
            }

            int32_t addrTemp6 = ((int32_t) tileEdgeTop * (int32_t) tilePitch) + \
                                ((int32_t) tileEdgeLeft * (int32_t) pixWidth) + ((int32_t) tileEdgeFront * (int32_t) tilePitch2DtileBytes);
            dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp6);

			dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM,dmaChannel,  dstPtr64, srcPtr64, (uint32_t) RowSize,
                        (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom, srcPitch, tilePitch, (int32_t) InterruptFlag, 0, tilePitch2DtileBytes, extraEdgeFront);

            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_FRONT_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((statusFlag & XV_TILE_STATUS_BACK_EDGE_PADDING_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_BACK_EDGE_PADDING_ONGOING) != 0u)
          {
            //We just finished this iDMA operation.
            statusFlag &= ~(XV_TILE_STATUS_BACK_EDGE_PADDING_ONGOING | XV_TILE_STATUS_BACK_EDGE_PADDING_NEEDED);

            //free temp buffer
            if (pTile13D->pTemp != NULL)
            {
              (void) xvFreeBufferHost(pxvTM, pTile13D->pTemp);
              pTile13D->pTemp = NULL;
            }
          }
          else
          {
            int32_t RowSize = ((int32_t) tileWidth + (int32_t) tileEdgeRight + (int32_t) tileEdgeLeft) * (int32_t) pixWidth;
            int32_t srcPitch;

            if (framePadType != FRAME_EDGE_PADDING)
            {
              //zero or constant padding.
              //allocate buffer
              pTile13D->pTemp = xvAllocateBufferHost(pxvTM, RowSize, (int32_t) XV_MEM_BANK_COLOR_ANY, 128);
              if (((int32_t) pTile13D->pTemp) == XVTM_ERROR)
              {
                return(XVTM_ERROR);
              }

              if (framePadType == FRAME_ZERO_PADDING)
              {
                framePadVal = 0;
              }

              xvTileManagerCopyPaddingDataHost((uint8_t *) pTile13D->pTemp, framePadVal, pixWidth, RowSize);
              srcPtr64 = (uint64_t) (uint32_t) pTile13D->pTemp;
              srcPitch = 0;
            }
            else
            {
              int32_t addrTemp7 = (((int32_t) tileEdgeTop) * ((int32_t) tilePitch)) + \
                                  ((((int32_t) tileEdgeLeft) * ((int32_t) pixWidth)) - (((((int32_t) tileDepth) + ((int32_t) tileEdgeBack)) - (extraEdgeBack + 1)) * tilePitch2DtileBytes));
              srcPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp7);
              srcPitch = tilePitch;
            }
            int32_t addrTemp8 = (((int32_t) tileEdgeTop) * ((int32_t) tilePitch)) + \
                                ((((int32_t) tileEdgeLeft) * ((int32_t) pixWidth)) - (((((int32_t) tileDepth) + ((int32_t) tileEdgeBack)) - extraEdgeBack) * ((int32_t) tilePitch2DtileBytes)));
            dstPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp8);

			dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) RowSize,
                        (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom, srcPitch, tilePitch, (int32_t) InterruptFlag, 0, tilePitch2DtileBytes, extraEdgeBack);

            statusFlag        |= XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_BACK_EDGE_PADDING_ONGOING;
            pTile13D->dmaIndex = dmaIndexTemp;
            break;
          }
        }

        if ((pTile13D->status & XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED) != 0u)
        {
          if ((statusFlag & XV_TILE_STATUS_DUMMY_DMA_ONGOING) != 0u)
          {
            //3D DMA is complete.
            statusFlag = 0;
          }
          else
          {
            // Tile is not part of frame. Make everything constant
            //first 2D tile in stack is ready with required padding data.
            //Set up DMA to copy it to rest of the stack
            int32_t addrTemp = ((int32_t) tileEdgeTop * (int32_t) tilePitch) + \
                               ((int32_t) tileEdgeLeft * (int32_t) pixWidth) + ((int32_t) tileEdgeFront * (int32_t) tilePitch2DtileBytes);
            srcPtr64 = ((uint64_t) (uint32_t) pTile13D->pData) - (uint64_t) ((int64_t) addrTemp);
            dstPtr64 = srcPtr64 + (uint64_t) (int64_t) (int32_t) tilePitch2DtileBytes;

            int32_t num2DTiles = ((((int32_t) tileEdgeFront) + (((int32_t) tileEdgeBack) + ((int32_t) tileDepth))) - 1);

            int32_t RowSize = ((int32_t) tileWidth + (int32_t) tileEdgeRight + (int32_t) tileEdgeLeft) * pixWidth;

            if (num2DTiles > 0)
            {

			  dmaIndexTemp = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) RowSize,
                        (int32_t) tileEdgeTop + (int32_t) tileHeight + (int32_t) tileEdgeBottom, tilePitch, tilePitch, (int32_t) InterruptFlag, tilePitch2DtileBytes, tilePitch2DtileBytes, num2DTiles);

              statusFlag         = XV_TILE_STATUS_DMA_ONGOING | XV_TILE_STATUS_DUMMY_DMA_ONGOING | XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED;
              pTile13D->dmaIndex = dmaIndexTemp;
              break;
            }
            else
            {
              //here we are done.
              statusFlag = 0;
            }
          }
        }
      }   //End of While PADDING loop

      pTile13D->status = statusFlag;
      if (statusFlag == 0u)
      {
        doneCount++;
      }
      else
      {
        //restore interrupt flag
        pTile13D->status |= (InterruptFlag << XV_TILE_STATUS_INTERRUPT_ON_COMPLETION_SHIFT);
        // Need to break the loop but MISRA-C does not allow more than 1
        // break statement. So just do the clean up and return. Also, note that
        // here return will always be 0 i.e. tile not ready.
        pxvTM->tile3DDMAstartIndex[dmaChannel]   = index;
        pxvTM->tile3DDMApendingCount[dmaChannel] = pxvTM->tile3DDMApendingCount[dmaChannel] - doneCount;
        return(0);
        // break;
      }
    }
    else         // DMA not done for this tile
    {
      // Need to break the loop but MISRA-C does not allow more than 1
      // break statement. So just do the clean up and return. Also, note that
      // here return will always be 0 i.e. tile not ready.
      pxvTM->tile3DDMAstartIndex[dmaChannel]   = index;
      pxvTM->tile3DDMApendingCount[dmaChannel] = pxvTM->tile3DDMApendingCount[dmaChannel] - doneCount;
      return(0);
      // break;
    }


    // Break if we reached the required tile
    if (pTile13D == pTile3D)
    {
      index = ((int32_t) index + 1) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
      break;
    }
  }  // End of outermost for loop

  pxvTM->tile3DDMAstartIndex[dmaChannel]   = index;
  pxvTM->tile3DDMApendingCount[dmaChannel] = pxvTM->tile3DDMApendingCount[dmaChannel] - doneCount;
  return((pTile3D->status == 0u) ? 1 : 0);
}

/**********************************************************************************
 * FUNCTION: xvAllocateTile3DHost()
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

xvTile3D *xvAllocateTile3DHost(xvTileManager *pxvTM)
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
 * FUNCTION: xvCreateTile3DHost()
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

xvTile3D  *xvCreateTile3DHost(xvTileManager *pxvTM, int32_t tileBuffSize,
                          int32_t width, uint16_t height, uint16_t depth, int32_t pitch,
                          int32_t pitch2D, uint16_t edgeWidth, uint16_t edgeHeight,
                          uint16_t edgeDepth, int32_t color, xvFrame3D *pFrame3D,
                          uint16_t xvTileType, int32_t alignType)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, ((xvTile3D *) ((void *) XVTM_ERROR)), "NULL TM Pointer");
  XV_CHECK_ERROR(((width <= 0) || (pitch <= 0) || (pitch2D <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR(((((int16_t) height) <= 0) || (((int16_t) depth) <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR(((alignType != XVTM_EDGE_ALIGNED_N) && (alignType != XVTM_DATA_ALIGNED_N) && (alignType != XVTM_EDGE_ALIGNED_2N) && (alignType != XVTM_DATA_ALIGNED_2N)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
#ifndef XVTM_USE_XMEM
  XV_CHECK_ERROR((((color < 0) || (color >= (int32_t) pxvTM->numMemBanks)) && (color != (int32_t) XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
#else
  XV_CHECK_ERROR((((color < 0) || (color >= (int32_t) xmem_bank_get_num_banks())) && (color != (int32_t) XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
#endif

#if (XV_ERROR_LEVEL != XV_ERROR_LEVEL_NO_ERROR)
  int32_t channel     = (int32_t) (uint16_t) XV_TYPE_CHANNELS(xvTileType);
  int32_t bytesPerPix = (int32_t) (uint16_t) XV_TYPE_ELEMENT_SIZE(xvTileType);
  int32_t bytePerPel  = bytesPerPix / channel;
#endif
  XV_CHECK_ERROR((((width + (2 * edgeWidth)) * channel) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((((width + (2 * edgeWidth)) * (height + (2 * edgeHeight))) > pitch2D), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((tileBuffSize < (pitch2D * (depth + (2 * edgeDepth)) * bytePerPel)), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");


  if (pFrame3D != NULL)
  {
    XV_CHECK_ERROR((pFrame3D->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
    XV_CHECK_ERROR((pFrame3D->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((xvTile3D *) ((void *) XVTM_ERROR)), "XV_ERROR_BAD_ARG");
  }
  void *tileBuff = NULL;
  tileBuff = xvAllocateBufferHost(pxvTM, tileBuffSize, color, 128);

  if (tileBuff == (void *) XVTM_ERROR)
  {
    return((xvTile3D *) ((void *) XVTM_ERROR));
  }

  xvTile3D *pTile3D = xvAllocateTile3DHost(pxvTM);
  if ((void *) pTile3D == (void *) XVTM_ERROR)
  {
    //Type cast to void to avoid MISRA 17.7 violation
    (void) xvFreeBufferHost(pxvTM, tileBuff);
    return((xvTile3D *) ((void *) XVTM_ERROR));
  }

  SETUP_TILE_3D(pTile3D, tileBuff, tileBuffSize, pFrame3D, width, height, depth, pitch, pitch2D, xvTileType, edgeWidth, edgeHeight, edgeDepth, 0, 0, 0, alignType);
  return(pTile3D);
}

/**********************************************************************************
 * FUNCTION: xvSleepForTileMultiChannel3DHost()
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
int32_t xvSleepForTileMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");

  pxvTM->errFlag = XV_ERROR_SUCCESS;

#if defined (XTENSA_SWVERSION_RH_2018_6) && defined(XTENSA_SWVERSION) && (XTENSA_SWVERSION > XTENSA_SWVERSION_RH_2018_6)
  DECLARE_PS();
#endif
  IDMA_DISABLE_INTS();
  int32_t status = xvCheckTileReadyMultiChannel3DHost(dmaChannel, (pxvTM), (pTile3D));
  if (status == XVTM_ERROR)
  {
    IDMA_ENABLE_INTS();
    return(status);
  }
  while ((status == 0) && (pxvTM->idmaErrorFlag[dmaChannel] == XV_ERROR_SUCCESS))
  {
    idma_sleep(dmaChannel);
    status = xvCheckTileReadyMultiChannel3DHost(dmaChannel, (pxvTM), (pTile3D));
    if (status == XVTM_ERROR)
    {
      IDMA_ENABLE_INTS();
      return(status);
    }
  }
  IDMA_ENABLE_INTS();
  return((int32_t) pxvTM->idmaErrorFlag[dmaChannel]);
}

/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannel3DHost()
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
int32_t xvReqTileTransferOutMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion)
{
  xvFrame3D *pFrame3D;
  uint64_t srcPtr64, dstPtr64;
  int32_t pixWidth, tileIndex, dmaIndex;
  int32_t srcHeight, srcWidth, srcDepth, srcPitch, srcTilePitch;
  int32_t dstPitch, dstTilePitch, numRows, rowSize, numTiles;

  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  pxvTM->errFlag = XV_ERROR_SUCCESS;
  XV_CHECK_ERROR(((dmaChannel < 0) || (dmaChannel >= XCHAL_IDMA_NUM_CHANNELS)), \
                 pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "TM BAD DMA_CHANNEL");

  XV_CHECK_ERROR((interruptOnCompletion < 0), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  XV_CHECK_TILE_3D(pTile3D, pxvTM);
  pFrame3D = pTile3D->pFrame;
  XV_CHECK_ERROR(pFrame3D == NULL, pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");
  XV_CHECK_ERROR(((pFrame3D->pFrameBuff == 0LLu) || (pFrame3D->pFrameData == 0LLu)), pxvTM->errFlag = XV_ERROR_FRAME_NULL, XVTM_ERROR, "XV_ERROR_FRAME_NULL");

#if (XV_ERROR_LEVEL != XV_ERROR_LEVEL_NO_ERROR)
  int32_t channel     = (int32_t) (uint32_t) (XV_TYPE_CHANNELS(((uint32_t) XV_TILE_GET_TYPE(pTile3D))));
  int32_t bytesPerPix = (int32_t) (uint32_t) (XV_TYPE_ELEMENT_SIZE(((uint32_t) XV_TILE_GET_TYPE(pTile3D))));
  int32_t bytePerPel  = bytesPerPix / channel;
#endif
  XV_CHECK_ERROR((pFrame3D->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");
  XV_CHECK_ERROR((pFrame3D->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "XV_ERROR_BAD_ARG");

  pixWidth     = ((int32_t) pFrame3D->pixelRes) * ((int32_t) pFrame3D->numChannels);
  srcHeight    = (int32_t) pTile3D->height;
  srcWidth     = pTile3D->width;
  srcPitch     = pTile3D->pitch * (int32_t) pFrame3D->pixelRes;
  srcTilePitch = pTile3D->Tile2Dpitch * (int32_t) pFrame3D->pixelRes;
  srcDepth     = (int32_t) pTile3D->depth;
  dstPitch     = pFrame3D->framePitch * (int32_t) pFrame3D->pixelRes;
  dstTilePitch = pFrame3D->frame2DFramePitch * (int32_t) pFrame3D->pixelRes;

  numRows  = XVTM_MIN(srcHeight, pFrame3D->frameHeight - pTile3D->y);
  rowSize  = XVTM_MIN(srcWidth, (pFrame3D->frameWidth - pTile3D->x));
  numTiles = XVTM_MIN(srcDepth, pFrame3D->frameDepth - pTile3D->z);

  srcPtr64 = (uint64_t) (uint32_t) pTile3D->pData;
  dstPtr64 = pFrame3D->pFrameData + (uint64_t) (int64_t) (((int64_t) (pTile3D->y) * (int64_t) dstPitch) + ((int64_t) (pTile3D->x) * (int64_t) pixWidth) + ((int64_t) (pTile3D->z) * (int64_t) dstTilePitch));

  if (pTile3D->x < 0)
  {
    rowSize  += pTile3D->x;    // x is negative;
    srcPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile3D->x) * (int64_t) pixWidth);
    dstPtr64 += (uint64_t) (int64_t) ((int64_t) (-pTile3D->x) * (int64_t) pixWidth);
  }

  if (pTile3D->y < 0)
  {
    numRows  += pTile3D->y;    // y is negative;
    srcPtr64 += (uint64_t) (int64_t) (((int64_t) (-pTile3D->y)) * ((int64_t) srcPitch));
    dstPtr64 += (uint64_t) (int64_t) (((int64_t) (-pTile3D->y)) * ((int64_t) dstPitch));
  }

  if (pTile3D->z < 0)
  {
    numTiles += pTile3D->z;     // z is negative;
    srcPtr64 += (uint64_t) (int64_t) (((int64_t) (-pTile3D->z)) * ((int64_t) srcTilePitch));
    dstPtr64 += (uint64_t) (int64_t) (((int64_t) (-pTile3D->z)) * ((int32_t) dstTilePitch));
  }


  rowSize *= pixWidth;

  if ((rowSize > 0) && (numRows > 0) && (numTiles > 0))
  {
    pTile3D->status = pTile3D->status | XV_TILE_STATUS_DMA_ONGOING;
    tileIndex       = ((int32_t) ((pxvTM->tile3DDMAstartIndex[dmaChannel])) + (int32_t) (pxvTM->tile3DDMApendingCount[dmaChannel])) % ((int32_t) MAX_NUM_DMA_QUEUE_LENGTH);
    pxvTM->tile3DDMApendingCount[dmaChannel]++;
    pxvTM->tile3DProcQueue[dmaChannel][tileIndex] = pTile3D;

    dmaIndex = AddIdmaRequestMultiChannel_wide3DHost(pxvTM, dmaChannel, dstPtr64, srcPtr64, (uint32_t) rowSize,
                        numRows, srcPitch, dstPitch, interruptOnCompletion, srcTilePitch,
                        dstTilePitch, numTiles);

    pTile3D->dmaIndex = dmaIndex;
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

int32_t xvCreateTileManagerMultiChannel4CHHost(xvTileManager *pxvTM, void *buf0, void *buf1, void *buf2, void *buf3, int32_t numMemBanks, void* const* pBankBuffPool, int32_t const* buffPoolSize,
                                               idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,
                                               idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                               idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,
                                               idma_callback_fn intrCallbackFunc2, void *cbData2, idma_callback_fn intrCallbackFunc3, void *cbData3,
                                               int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  int32_t retVal;

  retVal = xvInitTileManagerMultiChannel4CHHost(pxvTM, (idma_buffer_t *) buf0, (idma_buffer_t *) buf1,
                                            (idma_buffer_t *) buf2, (idma_buffer_t *) buf3);
  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");
#ifndef XVTM_USE_XMEM

  retVal = xvInitMemAllocatorHost(pxvTM, numMemBanks, pBankBuffPool, buffPoolSize);
  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");
#else
  (void)numMemBanks;
  (void)pBankBuffPool;
  (void)buffPoolSize;
#endif
  retVal = xvInitIdmaMultiChannel4CHHost(pxvTM, (idma_buffer_t *) buf0, (idma_buffer_t *) buf1, (idma_buffer_t *) buf2, (idma_buffer_t *) buf3,
                                     descCount, maxBlock, numOutReq, errCallbackFunc0, errCallbackFunc1,
                                     errCallbackFunc2, errCallbackFunc3, intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
                                     intrCallbackFunc2, cbData2, intrCallbackFunc3, cbData3);

  XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");

  return(retVal);
}

/**********************************************************************************
 * FUNCTION: xvCreateTileManagerMultiChannelHost()
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
int32_t xvCreateTileManagerMultiChannelHost(xvTileManager *pxvTM, void *buf0, void *buf1,void *buf2, void *buf3, int32_t numMemBanks,
                                        void **pBankBuffPool, int32_t* buffPoolSize,
                                        idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                        idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,idma_callback_fn intrCallbackFunc2, void *cbData2, idma_callback_fn intrCallbackFunc3, void *cbData3,
                                        int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  return(xvCreateTileManagerMultiChannel4CHHost(pxvTM, buf0, buf1, buf2, buf3, numMemBanks, pBankBuffPool, buffPoolSize,
                                            errCallbackFunc0, errCallbackFunc1, errCallbackFunc2, errCallbackFunc3,
                                            intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
											intrCallbackFunc2, cbData2, intrCallbackFunc3, cbData3,
                                            descCount, maxBlock, numOutReq));
}
#else
int32_t xvCreateTileManagerMultiChannelHost(xvTileManager *pxvTM, void *buf0, void *buf1, int32_t numMemBanks,
                                        void* const* pBankBuffPool, int32_t const* buffPoolSize,
                                        idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,
                                        idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,
                                        int32_t descCount, int32_t maxBlock, int32_t numOutReq)
{
  return(xvCreateTileManagerMultiChannel4CHHost(pxvTM, buf0, buf1, NULL, NULL, numMemBanks, pBankBuffPool, buffPoolSize,
                                            errCallbackFunc0, errCallbackFunc1, NULL, NULL,
                                            intrCallbackFunc0, cbData0, intrCallbackFunc1, cbData1,
                                            NULL, NULL, NULL, NULL,
                                            descCount, maxBlock, numOutReq));
}
#endif


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

