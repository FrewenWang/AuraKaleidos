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
#include <xtensa/tie/xt_ivpn.h>
#include "tileManager.h"
#include "tileManager_FIK_api.h"
#ifdef FIK_FRAMEWORK
#if DBG_PROFILE

#include <xtensa/tie/xt_timer.h>
#if defined (__XTENSA__)
#include <xtensa/sim.h>
#include <xtensa/xt_profiling.h>

#define USER_DEFINED_HOOKS_START()             \
  {                                            \
    xt_iss_switch_mode(XT_ISS_CYCLE_ACCURATE); \
    xt_iss_trace_level(3);                     \
    xt_iss_client_command("all", "enable");    \
  }

#define USER_DEFINED_HOOKS_STOP()            \
  {                                          \
    xt_iss_switch_mode(XT_ISS_FUNCTIONAL);   \
    xt_iss_trace_level(0);                   \
    xt_iss_client_command("all", "disable"); \
  }
#else
#define USER_DEFINED_HOOKS_START()
#define USER_DEFINED_HOOKS_STOP()
#endif

#endif
#define INTERRUPT_ON_COMPLETION  0

#define XV_ERR_PTR XVTM_ERROR

extern application_symbol_tray *g_symbol_tray;

#if defined (__cplusplus)
extern "C"
{
#endif
/**********************************************************************************
 * FUNCTION: xvSetupFrame()
 *
 * DESCRIPTION:
 *     Allocates single frame. It does not allocate buffer required for frame data.
 *     Initializes the frame elements
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     void          *imgBuff        Pointer to iaura buffer
 *     uint32_t       frameBuffSize   Size of allocated iaura buffer
 *     int32_t       width           Width of iaura
 *     int32_t       height          Height of iaura
 *     int32_t       pitch           Pitch of iaura
 *     uint8_t       pixRes          Pixel resolution of iaura in bytes
 *     uint8_t       numChannels     Number of channels in the iaura
 *     uint8_t       paddingtype     Supported padding type
 *     uint32_t       paddingVal      Padding value if padding type is edge extension
 *
 * OUTPUTS:
 *     Returns the pointer to allocated frame.
 *     Returns ((xvFrame *)(XVTM_ERROR)) if it encounters an error.
 *     Does not allocate frame data buffer.
 *
 ********************************************************************************** */

int32_t xvSetupFrame(xvTileManager *pxvTM, xvFrame * pFrame, uint64_t imgBuff, int32_t width, int32_t height, int32_t pitch, uint8_t pixRes, uint8_t numChannels, uint8_t paddingtype, uint32_t paddingVal){
	XV_CHECK_ERROR_NULL(((pxvTM == NULL) || (imgBuff == 0u) || (pFrame == NULL)),  (XVTM_ERROR), "XVTM_ERROR");
	XV_CHECK_ERROR(((width < 0) || (height < 0) || (pitch < 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG,   (XVTM_ERROR), "XVTM_ERROR");
	XV_CHECK_ERROR(((width * numChannels) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG,   (XVTM_ERROR), "XVTM_ERROR");
	XV_CHECK_ERROR(((numChannels > MAX_NUM_CHANNEL) || (numChannels <= 0)), pxvTM->errFlag = XV_ERROR_BAD_ARG,   (XVTM_ERROR), "XVTM_ERROR");
	XV_CHECK_ERROR((paddingtype > FRAME_PADDING_MAX), pxvTM->errFlag = XV_ERROR_BAD_ARG,   (XVTM_ERROR), "XVTM_ERROR");
	uint32_t frameBuffSize = (pitch * height * pixRes);
	SETUP_FRAME(pFrame, imgBuff, frameBuffSize, width, height, pitch, 0, 0, pixRes, numChannels, paddingtype, paddingVal);
	return XVTM_SUCCESS;
}

int32_t xvSetupTile(xvTileManager *pxvTM, xvTile *pTile, int32_t tileBuffSize, int32_t width, uint16_t height, int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame, uint16_t xvTileType, int32_t alignType)
{
	return g_symbol_tray->tray_xvSetupTile(pxvTM, pTile, tileBuffSize, width, height, pitch, edgeWidth, edgeHeight, color, pFrame, xvTileType, alignType); // replaced by tray func.
}

/**********************************************************************************
 * FUNCTION: xvAllocateAndSetupTile()
 *
 * DESCRIPTION:
 *     Allocates single tile and associated buffer data.
 *     Initializes the elements in tile
 *
 * INPUTS:
 *     xvTileManager *pxvTM          Tile Manager object
 *     int32_t       width           Width of tile
 *     uint16_t       height          Height of tile
 *     uint16_t       edgeWidth       Edge width of tile
 *     uint16_t       edgeHeight      Edge height of tile
 *     int32_t       color           Memory pool from which the buffer should be allocated
 *     xvFrame       *pFrame         Frame associated with the tile
 *     uint16_t       tileType        Type of tile
 *
 * OUTPUTS:
 *     Returns the pointer to allocated tile.
 *     Returns ((xvTile *)(XVTM_ERROR)) if it encounters an error.
 *
 ********************************************************************************** */

int32_t xvRegisterTile(xvTileManager *pxvTM, xvTile* pTile, void *pBuff, xvFrame *pFrame, uint32_t DMAInOut){
	return g_symbol_tray->tray_xvRegisterTile(pxvTM, pTile, pBuff, pFrame, DMAInOut); // replaced by tray func.
}

int32_t xvBufferCheckPointRestore(xvTileManager *pxvTM, uint32_t Idx)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  int i;
  int retVal = XVTM_SUCCESS;
  for (i = Idx; i != (int)pxvTM->allocationsIdx; i = ((i + 1) % XV_MAX_ALLOCATIONS))
  {
    retVal |= xvFreeBuffer(pxvTM, pxvTM->allocatedList[i]);
  }
  pxvTM->allocationsIdx = Idx;
  return(retVal);
}

int32_t xvBufferCheckPointSave(xvTileManager *pxvTM)
{
  XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
  return(pxvTM->allocationsIdx);
}


/**********************************************************************************
 * FUNCTION: xvTileManagerContextSave()
 *
 * DESCRIPTION:
 *
 *     Stores the current tile and frame allocation state
 *
 * INPUTS:
 *     xvTileManager * pxvTM                   Tile Manager object
 *
 * OUTPUTS:
 *     It returns XVTM_SUCCESS if success or else returns XVTM_ERROR
 *
 ********************************************************************************** */
int32_t xvTileManagerContextSave(xvTileManager * pxvTM)
{
	XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
	XV_CHECK_ERROR_NULL(pxvTM->tmContextSize >= MAX_TM_CONTEXT_SIZE, XVTM_ERROR, "Tile Manager Context Buffer is Full");
		uint32_t tmContextIdx = pxvTM->tmContextSize;

	for (int32_t i = 0; i < (int32_t)((MAX_NUM_TILES + 31) / 32); i++)
	{
		pxvTM->tmContext[tmContextIdx].tileAllocFlags[i] = pxvTM->tileAllocFlags[i];
	}
	for (int32_t i = 0; i < (int32_t)((MAX_NUM_TILES + 31) / 32); i++)
	{
		pxvTM->tmContext[tmContextIdx].frameAllocFlags[i] = pxvTM->frameAllocFlags[i];
	}
	pxvTM->tmContext[tmContextIdx].tileCount = pxvTM->tileCount;
	pxvTM->tmContext[tmContextIdx].frameCount = pxvTM->frameCount;
	pxvTM->tmContextSize = pxvTM->tmContextSize + 1;
	return(XVTM_SUCCESS);
}

/**********************************************************************************
 * FUNCTION: xvTileManagerContextRestore()
 *
 * DESCRIPTION:
 *
 *     Restores the tile and frame allocation state based on the last
 *     stored context
 *
 * INPUTS:
 *     xvTileManager * pxvTM                   Tile Manager object
 *
 * OUTPUTS:
 *     It returns XVTM_SUCCESS if the context is restored successfully else returns
 *     XVTM_ERROR
 *
 ********************************************************************************** */
int32_t xvTileManagerContextRestore(xvTileManager * pxvTM)
{
	XV_CHECK_ERROR_NULL(pxvTM == NULL, XVTM_ERROR, "NULL TM Pointer");
	XV_CHECK_ERROR_NULL((pxvTM->tmContextSize <= 0), XVTM_ERROR, "Tile Manager Context Buffer is Empty");

	uint32_t tmContextIdx = pxvTM->tmContextSize - 1;
	for (int32_t i = 0; i < (int32_t)((MAX_NUM_TILES + 31u) / 32u); i++)
	{
		pxvTM->tileAllocFlags[i] = pxvTM->tmContext[tmContextIdx].tileAllocFlags[i];
	}
	for (int32_t i = 0; i < (int32_t)((MAX_NUM_TILES + 31u) / 32u); i++)
	{
		pxvTM->frameAllocFlags[i] = pxvTM->tmContext[tmContextIdx].frameAllocFlags[i];
	}
	pxvTM->tileCount = pxvTM->tmContext[tmContextIdx].tileCount;
	pxvTM->frameCount = pxvTM->tmContext[tmContextIdx].frameCount;
	pxvTM->tmContextSize = tmContextIdx;
	return(XVTM_SUCCESS);
}

int32_t xvCheckPointSave(xvTileManager *pxvTM)
{
	uint32_t bufferIdx = xvBufferCheckPointSave(pxvTM);
	int32_t retVal;
	retVal = xvTileManagerContextSave(pxvTM);
	XV_CHECK_ERROR_NULL(((int32_t)bufferIdx == (int32_t)XVTM_ERROR) || (retVal == (int32_t)XVTM_ERROR), XVTM_ERROR, "Check Point Save Failed");
	pxvTM->tmContext[pxvTM->tmContextSize - 1].allocationsIdx = bufferIdx;
	return (XVTM_SUCCESS);
}

int32_t xvCheckPointRestore(xvTileManager *pxvTM)
{
	int32_t retVal = xvBufferCheckPointRestore(pxvTM, pxvTM->tmContext[pxvTM->tmContextSize - 1].allocationsIdx);
	retVal |= xvTileManagerContextRestore(pxvTM);
	return retVal;
}


int32_t xvGetArgParamsContext(xvTileManager *pxvTM, RefTile *pRefTile, uint32_t ArgSize,void* pComArgs,
		 void (*SetupUpdatesTiles)(xvTileManager *pxvTM, RefTile *pRefTile, void* CommonArgs, void* TileArgs, int updateOnlyFlag),
		 int32_t (*ProcessKernel)(void* CommonArgs, void* TileArgs),
		uint16_t CoreID, uint16_t NumCores, uint32_t flags, FIK_Context_t** dpWorkPacket)
{
	XV_CHECK_ERROR_NULL((pxvTM == NULL || pRefTile==NULL || SetupUpdatesTiles ==NULL  || ProcessKernel==NULL || dpWorkPacket==NULL), XVTM_ERROR, "NULL TM Pointer");
	uint32_t allocationStartIdx;
	FIK_Context_t* pWorkPacket = * dpWorkPacket;
	void* ArgsPing, *ArgsPong=NULL;
	uint32_t allocateSingleBank = flags & XV_ALLOCATE_SINGLE_BANK;
	int32_t SingleTileFlag;
	if((pRefTile->frameHeight <= pRefTile->tileHeight &&	pRefTile->frameWidth <=  pRefTile->tileWidth)||(flags & XV_SEQ_FLAG)==XV_SEQ_FLAG)
	{
		SingleTileFlag = 1;
	}
	else
	{
		SingleTileFlag = 0;
	}

	allocationStartIdx = pxvTM->allocationsIdx;
	ArgsPing = (void*)xvAllocateBuffer(pxvTM, ArgSize+ArgSize+128, XV_MEM_BANK_COLOR_ANY, 128);
 	if ((int32_t) ArgsPing == XVTM_ERROR && pxvTM->errFlag==XV_ERROR_ALLOC_FAILED) { return(XVTM_ERROR_MEMALLOC); }
  	if ((int32_t) ArgsPing == XVTM_ERROR) { return(XVTM_ERROR); }

	if(SingleTileFlag!=1)
	{
		ArgsPong = (void*)((   (uint32_t)(  ((uint8_t*)ArgsPing) +127 ) & (~127) ) + ArgSize);
		//ArgsPong = (void*)xvAllocateBuffer(pxvTM, ArgSize, XV_MEM_BANK_COLOR_ANY, 64);
		//if((int32_t)ArgsPong == XVTM_ERROR) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return XVTM_ERROR;}
	}

	pWorkPacket = (FIK_Context_t*)xvAllocateBuffer(pxvTM, sizeof(FIK_Context_t), XV_MEM_BANK_COLOR_ANY, 128);
  	if ((int32_t) pWorkPacket == XVTM_ERROR && pxvTM->errFlag==XV_ERROR_ALLOC_FAILED) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return(XVTM_ERROR_MEMALLOC); }
  	if ((int32_t) pWorkPacket == XVTM_ERROR) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return(XVTM_ERROR); }

	pWorkPacket->allocationStartIdx = allocationStartIdx;
	pWorkPacket->numInTiles[0] = 0;pWorkPacket->numInTiles[1] = 0;
	pWorkPacket->numOutTiles[0] = 0;pWorkPacket->numOutTiles[1] = 0;


	pxvTM->AllocateErrorState = XVTM_SUCCESS;
	pxvTM->pWorkPacket = pWorkPacket;

	pxvTM->PingPongState = PING;
	pxvTM->AllocateColor = allocateSingleBank == 0 ? XV_MEM_BANK_COLOR_1 : XV_MEM_BANK_COLOR_ANY;
	pxvTM->InterimTileCounter = 0;
	SetupUpdatesTiles(pxvTM, pRefTile, pComArgs, ArgsPing, 0);
	if(SingleTileFlag!=1)
	{
		pxvTM->PingPongState = PONG;
		pxvTM->AllocateColor = allocateSingleBank == 0 ? XV_MEM_BANK_COLOR_0 : XV_MEM_BANK_COLOR_ANY;
		pxvTM->InterimTileCounter = 0;
		SetupUpdatesTiles(pxvTM, pRefTile, pComArgs, ArgsPong, 0);
	}

	while(pxvTM->AllocateErrorState == XVTM_ERROR_MEMALLOC)
	{
		xvBufferCheckPointRestore(pxvTM, pWorkPacket->allocationStartIdx);

		int32_t THeight = pRefTile->tileHeight;
		uint16_t Hstep = pRefTile->tileHeight>>4;
		if(Hstep < 2)
			THeight -=2;
		else
			THeight -=Hstep;

		pRefTile->tileHeight = THeight;

		if(THeight <=0 || (flags & XV_AUTOSIZE_FLAG) == 0)
		{
			return XVTM_ERROR_MEMALLOC;
		}
		if((pRefTile->frameHeight <= pRefTile->tileHeight &&	pRefTile->frameWidth <=  pRefTile->tileWidth)||(flags & XV_SEQ_FLAG)==XV_SEQ_FLAG)
		{
			SingleTileFlag = 1;
		}
		else
		{
			SingleTileFlag = 0;
		}


		allocationStartIdx = pxvTM->allocationsIdx;
		ArgsPing = (void*)xvAllocateBuffer(pxvTM, ArgSize+ArgSize+128, XV_MEM_BANK_COLOR_ANY, 128);
    	if ((int32_t) ArgsPing == XVTM_ERROR && pxvTM->errFlag==XV_ERROR_ALLOC_FAILED) { return(XVTM_ERROR_MEMALLOC); }
		if ((int32_t) ArgsPing == XVTM_ERROR) { return(XVTM_ERROR); }		

		if(SingleTileFlag!=1)
		{
			ArgsPong = (void*)((   (uint32_t)(  ((uint8_t*)ArgsPing) +127 ) & (~127) ) + ArgSize);
			//ArgsPong = (void*)xvAllocateBuffer(pxvTM, ArgSize, XV_MEM_BANK_COLOR_ANY, 64);
			//if((int32_t)ArgsPong == XVTM_ERROR) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return XVTM_ERROR;}

		}
		pWorkPacket = (FIK_Context_t*)xvAllocateBuffer(pxvTM, sizeof(FIK_Context_t), XV_MEM_BANK_COLOR_ANY, 128);
 		if ((int32_t) pWorkPacket == XVTM_ERROR && pxvTM->errFlag==XV_ERROR_ALLOC_FAILED) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return(XVTM_ERROR_MEMALLOC); }
		if ((int32_t) pWorkPacket == XVTM_ERROR) { xvBufferCheckPointRestore(pxvTM, allocationStartIdx); return(XVTM_ERROR); }


		pWorkPacket->allocationStartIdx = allocationStartIdx;
		pWorkPacket->numInTiles[0] = 0;pWorkPacket->numInTiles[1] = 0;
		pWorkPacket->numOutTiles[0] = 0;pWorkPacket->numOutTiles[1] = 0;


		pxvTM->AllocateErrorState = XVTM_SUCCESS;
		pxvTM->pWorkPacket = pWorkPacket;

		pxvTM->PingPongState = PING;
		pxvTM->AllocateColor = allocateSingleBank == 0 ? XV_MEM_BANK_COLOR_1 : XV_MEM_BANK_COLOR_ANY;
		pxvTM->InterimTileCounter = 0;
		SetupUpdatesTiles(pxvTM, pRefTile, pComArgs, ArgsPing, 0);
		if(SingleTileFlag!=1)
		{
			pxvTM->PingPongState = PONG;
			pxvTM->AllocateColor = allocateSingleBank == 0 ? XV_MEM_BANK_COLOR_0 : XV_MEM_BANK_COLOR_ANY;
			pxvTM->InterimTileCounter = 0;
			SetupUpdatesTiles(pxvTM, pRefTile, pComArgs, ArgsPong, 0);
		}
	}

	* dpWorkPacket = pWorkPacket;
	/* Update Number of cores and current core ID*/
	pWorkPacket->CoreID = CoreID;
	pWorkPacket->numCores = NumCores;
	pWorkPacket->numTiles = 0;

	pWorkPacket->refTile.x = pRefTile->x;
	pWorkPacket->refTile.y = pRefTile->y;
	pWorkPacket->refTile.frameWidth = pRefTile->frameWidth;
	pWorkPacket->refTile.frameHeight = pRefTile->frameHeight;
	pWorkPacket->refTile.tileWidth =  pRefTile->tileWidth;
	pWorkPacket->refTile.tileHeight =  pRefTile->tileHeight;

	/* update pointers to argument structures for ping/pong kernel calls */
	pWorkPacket->KernelArgs[PING] = ArgsPing;
	pWorkPacket->KernelArgs[PONG] = ArgsPong;

	pWorkPacket->CommonArgs = pComArgs;

	/* Update pointer to funciton calling a kernel, this funciton will get called for every tile*/
	pWorkPacket->ProcessKernel = ProcessKernel;

	pWorkPacket->flags = flags;

	pWorkPacket->SetupUpdatesTiles = (void(*)(void* pxvTM, RefTile *pRefTile, void* _CommonArgs, void* TileArgs, int updateOnlyFlag))SetupUpdatesTiles;
  	if(pxvTM->AllocateErrorState==XVTM_ERROR) { xvBufferCheckPointRestore(pxvTM, pWorkPacket->allocationStartIdx); return(XVTM_ERROR);}

	return XVTM_SUCCESS;
}

 int32_t xvExecuteFullIauraKernel(xvTileManager *pxvTM, RefTile *pRefTile, uint32_t ArgSize,void* pComArgs,
		 	 	 void (*SetupUpdatesTiles)(xvTileManager *pxvTM, RefTile *pRefTile, void* CommonArgs, void* TileArgs, int updateOnlyFlag),
		 	 	int32_t (*ProcessKernel)(void* CommonArgs, void* TileArgs),
		 		uint16_t CoreID, uint16_t NumCores, uint32_t Flags)
 {

	FIK_Context_t *Context;

	int32_t retVal = xvGetArgParamsContext(pxvTM, pRefTile, ArgSize, pComArgs, SetupUpdatesTiles, ProcessKernel, CoreID, NumCores, Flags, &Context);
    XV_CHECK_ERROR_NULL(retVal != XVTM_SUCCESS, retVal, "XVTM_ERROR");

#ifdef XVTM_MULTITHREADING_SUPPORT
  xos_preemption_enable();
#endif
	retVal = xvProcessTileWiseFast(Context, pxvTM);
#ifdef XVTM_MULTITHREADING_SUPPORT
  xos_preemption_disable();
#endif
	XV_CHECK_ERROR(retVal != XVTM_SUCCESS, xvBufferCheckPointRestore(pxvTM, Context->allocationStartIdx), retVal, "XVTM_ERROR");

	retVal = xvBufferCheckPointRestore(pxvTM, Context->allocationStartIdx);

	return retVal;

 }

 int32_t xvMemCpy(xvTileManager *pxvTM, void * destination, void * source, int32_t num )
 {
	 int32_t retVal = xvAddIdmaRequest(pxvTM, destination, source, num, 1, num, num, 0);
	 XV_CHECK_ERROR_NULL(retVal == XVTM_ERROR, XVTM_ERROR, "XVTM_ERROR");
	 retVal = xvWaitForiDMA(pxvTM, retVal);
	 return retVal;

 }

//int xvProcessTileWiseFastParallel(FIK_Context_t *WPacket, xvTileManager* pxvTM) __attribute__((section(".iram0.text")));
int xvProcessTileWiseFastParallel(FIK_Context_t *WPacket, xvTileManager* pxvTM)
{
#if DBG_PROFILE
	uint32_t Setup, Total = XT_RSR_CCOUNT();
	uint32_t CycleStamp0, CycleStamp1;
	uint32_t DMAWait = 0;
	uint32_t KernelCycle = 0;
	uint32_t TileParaSetup = 0;
	uint32_t InDMAConfig = 0;
	uint32_t TileMapfun = 0;
	uint32_t OutDMAConfig = 0;
	uint32_t PadEdges = 0;
#endif
	int32_t CoreID = WPacket->CoreID;
	int32_t numCores = WPacket->numCores;
	int32_t pingPongFlag = 0;
	int32_t indx;
	int32_t retVal;
	int32_t numInTiles = WPacket->numInTiles[0];
	int32_t numOutTiles = WPacket->numOutTiles[0];
	int32_t (*ProcessKernel)(void* CommonArgs_, void* TileArgs)= WPacket->ProcessKernel;
	void (*SetupUpdatesTiles)(void*pxvTM, RefTile *pRefTile, void* _CommonArgs, void* TileArgs, int updateOnlyFlag) = WPacket->SetupUpdatesTiles;
	int32_t frameWidth;
	int32_t frameHeight;
	
	int32_t RefTileWidth = WPacket->refTile.tileWidth;
	int32_t RefTileHeight = WPacket->refTile.tileHeight;

	frameWidth = WPacket->refTile.frameWidth;
	frameHeight = WPacket->refTile.frameHeight;

	RefTile ReTmp;
	ReTmp = WPacket->refTile;

	const uint32_t Yinc =ReTmp.tileHeight;
	const uint32_t Xinc1 = ReTmp.tileWidth;

	uint32_t Hor_Tiles =  frameWidth/Xinc1;
	if(frameWidth%Xinc1!=0)
		Hor_Tiles++;
	uint32_t Ver_Tiles =  frameHeight/Yinc;
	if(frameHeight%Yinc!=0)
		Ver_Tiles++;
#if XCHAL_HAVE_NX
	int32_t DMAInCH = TM_IDMA_CH0;
	int32_t DMAOutCH = TM_IDMA_CH1;
	if((WPacket->flags & XV_DMA_OVERLAP) == XV_DMA_OVERLAP)
		DMAOutCH = TM_IDMA_CH0;
#endif

	uint32_t TilesPerCore = (Hor_Tiles*Ver_Tiles)/numCores;
	uint32_t TileIdx = CoreID * TilesPerCore;
	uint32_t TileIdxEnd = ((CoreID+1)==numCores)? (Hor_Tiles*Ver_Tiles) : ((CoreID+1)* TilesPerCore);
	if(TileIdx==TileIdxEnd) return XVTM_SUCCESS;

	uint32_t TileIdxX = TileIdx % Hor_Tiles;
	uint32_t TileIdxY = TileIdx / Hor_Tiles;
	TileIdxY = XT_MIN(TileIdxY,Ver_Tiles-1);
#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	Setup = CycleStamp0 - Total;
#pragma no_reorder
#endif

	ReTmp.x = RefTileWidth * TileIdxX;
	ReTmp.y = RefTileHeight * TileIdxY;
	ReTmp.tileWidth = XT_MIN(RefTileWidth,\
			(frameWidth - ReTmp.x));
	ReTmp.tileHeight = XT_MIN(RefTileHeight,\
			(frameHeight - ReTmp.y));
#if DBG_PROFILE
#pragma no_reorder
	CycleStamp1 = XT_RSR_CCOUNT();
	TileParaSetup += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
	/*Call to map function mapping output tile x y width height to input tile x y width height*/
	(*SetupUpdatesTiles)(pxvTM, &ReTmp, WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag], 1);

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	TileMapfun += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif

	for(indx = 0; indx < numInTiles; indx++)
	{
#if XCHAL_HAVE_NX
		retVal = xvReqTileTransferInFastMultiChannel(DMAInCH, pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
		retVal = xvReqTileTransferInFast(pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
		XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");

	}

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp1 = XT_RSR_CCOUNT();
	InDMAConfig += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
	TileIdx += 1;
	TileIdxX = TileIdx % Hor_Tiles;
	TileIdxY = TileIdx / Hor_Tiles;
	int32_t LastTile = (TileIdx<TileIdxEnd) ? 1 : 0;
	TileIdxY = XT_MIN(TileIdxY,Ver_Tiles-1);
	pingPongFlag = pingPongFlag ^ 1;

	ReTmp.x = RefTileWidth * TileIdxX;
	ReTmp.y = RefTileHeight * TileIdxY;
	ReTmp.tileWidth = XT_MIN(RefTileWidth,\
			(frameWidth - ReTmp.x));
	ReTmp.tileHeight = XT_MIN(RefTileHeight,\
			(frameHeight - ReTmp.y));

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	TileParaSetup += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
	/*Call to map function mapping output tile x y width height to input tile x y width height*/
	if(LastTile == 1)
	{
		(*SetupUpdatesTiles)(pxvTM, &ReTmp, WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag], 1);
	}

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp1 = XT_RSR_CCOUNT();
	TileMapfun += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
	for(indx = 0; indx < numInTiles && LastTile == 1; indx++)
	{
#if XCHAL_HAVE_NX
		// g_symbol_tray->tray_printf("PIC : XCHAL_HAVE_NX Multichannel");
		retVal = xvReqTileTransferInFastMultiChannel(DMAInCH, pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
		// g_symbol_tray->tray_printf("PIC : XCHAL_HAVE_NX No multichannel");
		retVal = xvReqTileTransferInFast(pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif

		XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");
	}
#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	InDMAConfig += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
	pingPongFlag = pingPongFlag ^ 0x1;
	int32_t Tiles;
	for (Tiles = CoreID * TilesPerCore; Tiles < (int32_t)TileIdxEnd ; Tiles += 1)
	{
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
#pragma no_reorder
#endif
		if(Tiles >= (int32_t)((CoreID * TilesPerCore)+2))
		{
			for(indx = 0; indx < numOutTiles; indx++)
			{
#if XCHAL_HAVE_NX
				xvWaitForTileFastMultiChannel(DMAOutCH, pxvTM, WPacket->OutTiles[pingPongFlag][indx]);
#else
				xvWaitForTileFast((pxvTM), (WPacket->OutTiles[pingPongFlag][indx]));
#endif
			}
		}
		XV_CHECK_ERROR_NULL(pxvTM->errFlag != XVTM_SUCCESS, XVTM_ERROR, "Error DMA");
		for(indx = 0; indx < numInTiles; indx++)
		{
#if XCHAL_HAVE_NX
			xvWaitForTileFastMultiChannel(DMAInCH, pxvTM, WPacket->InTiles[pingPongFlag][indx]);
#else
			xvWaitForTileFast((pxvTM), (WPacket->InTiles[pingPongFlag][indx]));
#endif
		}
			XV_CHECK_ERROR_NULL(pxvTM->errFlag != XVTM_SUCCESS, XVTM_ERROR, "Error DMA");
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			DMAWait += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		for(indx = 0; indx < numInTiles; indx++)
		{
			xvPadEdges(pxvTM, WPacket->InTiles[pingPongFlag][indx]);
		}

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			PadEdges += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
        retVal = ProcessKernel(WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag]);
        XV_CHECK_ERROR_NULL(retVal != XVTM_SUCCESS, XVTM_ERROR, "Error processes function");

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			KernelCycle += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		for(indx = 0; indx < numOutTiles; indx++)
		{
#if XCHAL_HAVE_NX
			retVal = xvReqTileTransferOutFastMultiChannel(DMAOutCH, pxvTM, WPacket->OutTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
			retVal = xvReqTileTransferOutFast(pxvTM, WPacket->OutTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
			XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");
		}
			WPacket->numTiles++;
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			OutDMAConfig += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
		TileIdx += 1;
		TileIdxX = TileIdx % Hor_Tiles;
		TileIdxY = TileIdx / Hor_Tiles;
		LastTile = (TileIdx<TileIdxEnd) ? 1 : 0;
		TileIdxY = XT_MIN(TileIdxY,Ver_Tiles-1);

		ReTmp.x = RefTileWidth * TileIdxX;
		ReTmp.y = RefTileHeight * TileIdxY;
		ReTmp.tileWidth = XT_MIN(RefTileWidth,\
				(frameWidth - ReTmp.x));
		ReTmp.tileHeight = XT_MIN(RefTileHeight,\
				(frameHeight - ReTmp.y));

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			TileParaSetup += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		if(LastTile == 1)
		{
			/*Call to map function mapping output tile x y width height to input tile x y width height*/
			(*SetupUpdatesTiles)(pxvTM, &ReTmp, WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag], 1);
		}

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			TileMapfun += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif

		for(indx = 0; indx < numInTiles && LastTile == 1; indx++)
		{
#if XCHAL_HAVE_NX
			retVal = xvReqTileTransferInFastMultiChannel(DMAInCH, pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
			retVal = xvReqTileTransferInFast(pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
			XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");
		}
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			InDMAConfig += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		pingPongFlag = pingPongFlag ^ 0x1;
	}

	for(indx = 0; indx < numOutTiles; indx++)
	{
		// Wait for the last output tile transfer
#if XCHAL_HAVE_NX
		xvWaitForTileFastMultiChannel(DMAOutCH, pxvTM, WPacket->OutTiles[pingPongFlag ^ 0x01][indx]);
#else
		xvWaitForTileFast(pxvTM, WPacket->OutTiles[pingPongFlag ^ 0x01][indx]);
#endif
	}
	  //xvFrame *pFrame;
	  //xvTile *pTile = (xvTile *)WPacket->InTiles[pingPongFlag][indx];

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	USER_DEFINED_HOOKS_STOP();
	DMAWait += CycleStamp0 - CycleStamp1;
	Total = CycleStamp0 - Total;

	WPacket->Cycles[0]=Total;
	WPacket->Cycles[1]=KernelCycle;
	WPacket->Cycles[2]=DMAWait;
	WPacket->Cycles[3]=TileMapfun;
	WPacket->Cycles[4]=PadEdges;
	USER_DEFINED_HOOKS_START();
#pragma no_reorder
#endif
	return (XVTM_SUCCESS);
}


int xvProcessTileWiseFastSequential(FIK_Context_t *WPacket, xvTileManager* pxvTM)
{
#if DBG_PROFILE
	uint32_t Setup, Total = XT_RSR_CCOUNT();
	uint32_t CycleStamp0, CycleStamp1;
	uint32_t DMAWait = 0;
	uint32_t KernelCycle = 0;
	uint32_t TileParaSetup = 0;
	uint32_t InDMAConfig = 0;
	uint32_t TileMapfun = 0;
	uint32_t OutDMAConfig = 0;
	uint32_t PadEdges = 0;
#endif
	int32_t CoreID = WPacket->CoreID;
	int32_t numCores = WPacket->numCores;
	int32_t pingPongFlag = 0;
	int32_t indx;
	int32_t retVal;
	int32_t numInTiles = WPacket->numInTiles[0];
	int32_t numOutTiles = WPacket->numOutTiles[0];
	int32_t (*ProcessKernel)(void* CommonArgs_, void* TileArgs)= WPacket->ProcessKernel;
	void (*SetupUpdatesTiles)(void*pxvTM, RefTile *pRefTile, void* _CommonArgs, void* TileArgs, int updateOnlyFlag) = WPacket->SetupUpdatesTiles;
	int32_t frameWidth;
	int32_t frameHeight;

	int32_t RefTileWidth = WPacket->refTile.tileWidth;
	int32_t RefTileHeight = WPacket->refTile.tileHeight;

	frameWidth = WPacket->refTile.frameWidth;
	frameHeight = WPacket->refTile.frameHeight;

	const uint32_t Yinc = WPacket->refTile.tileHeight;
	const uint32_t Xinc1 =  WPacket->refTile.tileWidth;
	RefTile ReTmp;
	ReTmp = WPacket->refTile;
	uint32_t Hor_Tiles =  frameWidth/Xinc1;
	if(frameWidth%Xinc1!=0)
		Hor_Tiles++;
	uint32_t Ver_Tiles =  frameHeight/Yinc;
	if(frameHeight%Yinc!=0)
		Ver_Tiles++;

	uint32_t TilesPerCore = (Hor_Tiles*Ver_Tiles)/numCores;
	uint32_t TileIdx = CoreID * TilesPerCore;
	uint32_t TileIdxEnd = ((CoreID+1)==numCores)? (Hor_Tiles*Ver_Tiles) : ((CoreID+1)* TilesPerCore);
	if(TileIdx==TileIdxEnd) return XVTM_SUCCESS;

	uint32_t TileIdxX = TileIdx % Hor_Tiles;
	uint32_t TileIdxY = TileIdx / Hor_Tiles;
	TileIdxY = XT_MIN(TileIdxY,Ver_Tiles-1);
#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	Setup = CycleStamp0 - Total;
#pragma no_reorder
#endif

	ReTmp.x = RefTileWidth * TileIdxX;
	ReTmp.y = RefTileHeight * TileIdxY;
	ReTmp.tileWidth = XT_MIN(RefTileWidth,\
			(frameWidth - ReTmp.x));
	ReTmp.tileHeight = XT_MIN(RefTileHeight,\
			(frameHeight - ReTmp.y));


#if DBG_PROFILE
#pragma no_reorder
	CycleStamp1 = XT_RSR_CCOUNT();
	TileParaSetup += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
	/*Call to map function mapping output tile x y width height to input tile x y width height*/
	(*SetupUpdatesTiles)(pxvTM, &ReTmp, WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag], 1);

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	TileMapfun += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
	for(indx = 0; indx < numInTiles; indx++)
	{
#if XCHAL_HAVE_NX
		retVal = xvReqTileTransferInFastMultiChannel(TM_IDMA_CH0, pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
		retVal = xvReqTileTransferInFast(pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
		XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");

	}

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp1 = XT_RSR_CCOUNT();
	InDMAConfig += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif

	int32_t Tiles;
	for (Tiles = CoreID * TilesPerCore; Tiles < (int32_t)TileIdxEnd ; Tiles += 1)
	{
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
#pragma no_reorder
#endif
		if(Tiles >= (int32_t)((CoreID * TilesPerCore)+1))
		{
			for(indx = 0; indx < numOutTiles; indx++)
			{
#if XCHAL_HAVE_NX
				xvWaitForTileFastMultiChannel(TM_IDMA_CH1, pxvTM, WPacket->OutTiles[pingPongFlag][indx]);
#else
				xvWaitForTileFast((pxvTM), (WPacket->OutTiles[pingPongFlag][indx]));
#endif
			}
		}
		XV_CHECK_ERROR_NULL(pxvTM->errFlag != XVTM_SUCCESS, XVTM_ERROR, "Error DMA");
		for(indx = 0; indx < numInTiles; indx++)
		{
#if XCHAL_HAVE_NX
			xvWaitForTileFastMultiChannel(TM_IDMA_CH0, pxvTM, WPacket->InTiles[pingPongFlag][indx]);
#else
			xvWaitForTileFast((pxvTM), (WPacket->InTiles[pingPongFlag][indx]));
#endif
		}
			XV_CHECK_ERROR_NULL(pxvTM->errFlag != XVTM_SUCCESS, XVTM_ERROR, "Error DMA");
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			DMAWait += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		for(indx = 0; indx < numInTiles; indx++)
		{
			xvPadEdges(pxvTM, WPacket->InTiles[pingPongFlag][indx]);
		}

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			PadEdges += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
			retVal = ProcessKernel(WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag]);
			XV_CHECK_ERROR_NULL(retVal != XVTM_SUCCESS, XVTM_ERROR, "Error processes function");

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			KernelCycle += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		for(indx = 0; indx < numOutTiles; indx++)
		{
#if XCHAL_HAVE_NX
			retVal = xvReqTileTransferOutFastMultiChannel(TM_IDMA_CH1, pxvTM, WPacket->OutTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
			retVal = xvReqTileTransferOutFast(pxvTM, WPacket->OutTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
			XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");
		}
			WPacket->numTiles++;
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			OutDMAConfig += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif
		TileIdx += 1;
		TileIdxX = TileIdx % Hor_Tiles;
		TileIdxY = TileIdx / Hor_Tiles;
		int32_t LastTile = (TileIdx<TileIdxEnd) ? 1 : 0;
		TileIdxY = XT_MIN(TileIdxY,Ver_Tiles-1);

		ReTmp.x = RefTileWidth * TileIdxX;
		ReTmp.y = RefTileHeight * TileIdxY;
		ReTmp.tileWidth = XT_MIN(RefTileWidth,\
				(frameWidth - ReTmp.x));
		ReTmp.tileHeight = XT_MIN(RefTileHeight,\
				(frameHeight - ReTmp.y));

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			TileParaSetup += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
		if(LastTile == 1)
		{
			/*Call to map function mapping output tile x y width height to input tile x y width height*/
			(*SetupUpdatesTiles)(pxvTM, &ReTmp, WPacket->CommonArgs, WPacket->KernelArgs[pingPongFlag], 1);
		}

#if DBG_PROFILE
#pragma no_reorder
			CycleStamp0 = XT_RSR_CCOUNT();
			TileMapfun += CycleStamp0 - CycleStamp1;
#pragma no_reorder
#endif

		for(indx = 0; indx < numInTiles && LastTile == 1; indx++)
		{
#if XCHAL_HAVE_NX
			retVal = xvReqTileTransferInFastMultiChannel(TM_IDMA_CH0, pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#else
			retVal = xvReqTileTransferInFast(pxvTM, WPacket->InTiles[pingPongFlag][indx], INTERRUPT_ON_COMPLETION);
#endif
			XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfo(pxvTM), XVTM_ERROR, "Error in Tile In");
		}
#if DBG_PROFILE
#pragma no_reorder
			CycleStamp1 = XT_RSR_CCOUNT();
			InDMAConfig += CycleStamp1 - CycleStamp0;
#pragma no_reorder
#endif
	}

	for(indx = 0; indx < numOutTiles; indx++)
	{
		// Wait for the last output tile transfer
#if XCHAL_HAVE_NX
		xvWaitForTileFastMultiChannel(TM_IDMA_CH1, pxvTM, WPacket->OutTiles[pingPongFlag][indx]);
#else
		xvWaitForTileFast(pxvTM, WPacket->OutTiles[pingPongFlag][indx]);
#endif
	}

#if DBG_PROFILE
#pragma no_reorder
	CycleStamp0 = XT_RSR_CCOUNT();
	USER_DEFINED_HOOKS_STOP();
	DMAWait += CycleStamp0 - CycleStamp1;
	Total = CycleStamp0 - Total;

	WPacket->Cycles[0]=Total;
	WPacket->Cycles[1]=KernelCycle;
	WPacket->Cycles[2]=DMAWait;
	WPacket->Cycles[3]=TileMapfun;
	WPacket->Cycles[4]=PadEdges;
	USER_DEFINED_HOOKS_START();
#pragma no_reorder
#endif
	return (XVTM_SUCCESS);
}




/*Note: workpacket should have atleast one output tile as tile loop bound are calculated based on first output tile*/
/*Fast function does not Support edge/constant edge extensions except 8/16/32 bit element bit widths zero padding is supported for all*/
int xvProcessTileWiseFast(FIK_Context_t *WPacket, xvTileManager* pxvTM)
{
	int err_val;
	if((WPacket->flags & XV_SEQ_FLAG) == XV_SEQ_FLAG)
	{
		err_val = xvProcessTileWiseFastSequential(WPacket, pxvTM);
	}
	else
	{
		err_val = xvProcessTileWiseFastParallel(WPacket, pxvTM);

	}
	return (err_val);
}
#if defined (__cplusplus)
}
#endif
#endif
