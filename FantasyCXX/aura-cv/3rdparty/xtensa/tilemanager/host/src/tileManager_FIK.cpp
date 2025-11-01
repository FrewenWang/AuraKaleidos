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
    xt_iss_trace_level(3);                     \
    xt_iss_client_command("all", "enable");    \
  }

#define USER_DEFINED_HOOKS_STOP()            \
  {                                          \
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

int32_t xvSetupTileHost(xvTileManager *pxvTM, xvTile *pTile, int32_t tileBuffSize, int32_t width, uint16_t height, int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame, uint16_t xvTileType, int32_t alignType)
{
	XV_CHECK_ERROR_NULL((pxvTM == NULL || pTile==NULL), ( XVTM_ERROR), "NULL TM Pointer");
	XV_CHECK_ERROR(((width < 0) || (pitch < 0) ), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( ( XVTM_ERROR)), "XVTM_ERROR");
	XV_CHECK_ERROR(((alignType != XVTM_EDGE_ALIGNED_N) && (alignType != XVTM_DATA_ALIGNED_N) && (alignType != XVTM_EDGE_ALIGNED_2N) && (alignType != XVTM_DATA_ALIGNED_2N)), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( ( XVTM_ERROR)), "XVTM_ERROR");
#ifndef XVTM_USE_XMEM
  	XV_CHECK_ERROR((((color < 0) || (color >= pxvTM->numMemBanks)) && (color != XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( ( XVTM_ERROR)), "XVTM_ERROR");
#else
  	XV_CHECK_ERROR((((color < (int32_t)0) || (color >= (int32_t)xmem_bank_get_num_banks())) && (color != (int32_t)XV_MEM_BANK_COLOR_ANY)), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( ( XVTM_ERROR)), "XVTM_ERROR");
#endif

    int32_t channel = XV_TYPE_CHANNELS(xvTileType);
    int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(xvTileType);
    int32_t bytePerPel;
	bytePerPel	= bytesPerPix / channel;

	XV_CHECK_ERROR((((width + (2 * edgeWidth)) * channel) > pitch), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( ( XVTM_ERROR)), "XVTM_ERROR");
	XV_CHECK_ERROR((tileBuffSize < (pitch * (height + (2 * edgeHeight)) * bytePerPel)), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( XVTM_ERROR), "XVTM_ERROR");

	if (pFrame != NULL) {
		XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG,  (XVTM_ERROR), "XVTM_ERROR");
		XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG,  ( XVTM_ERROR), "XVTM_ERROR");
	}
    void *tileBuff = NULL;
    tileBuff = xvAllocateBufferHost(pxvTM, tileBuffSize, color, 128);

    if (tileBuff == (void *) XVTM_ERROR)
    {
        return(XVTM_ERROR);
    }

    SETUP_TILE(pTile, tileBuff, tileBuffSize, pFrame, width, height, pitch, xvTileType, edgeWidth, edgeHeight, 0, 0, alignType);
    return(XVTM_SUCCESS);
}

int32_t xvRegisterTileHost(xvTileManager *pxvTM, xvTile* pTile, void *pBuff, xvFrame *pFrame, uint32_t DMAInOut){
	XV_CHECK_ERROR_NULL(pxvTM == NULL, ( XVTM_ERROR), "NULL TM Pointer");
	XV_CHECK_ERROR_NULL(pTile == NULL, ( XVTM_ERROR), "NULL TM Pointer");
#ifndef XVTM_USE_XMEM
  int32_t numMemBanks = pxvTM->numMemBanks;
#else
  int32_t numMemBanks = xmem_bank_get_num_banks();
#endif


	FIK_Context_t* pWorkPacket = pxvTM->pWorkPacket;
	uint32_t PingPong = pxvTM->PingPongState;

	int32_t alignType = XVTM_TILE_UNALIGNED;
	uint16_t xvTileType = pTile->type;
	uint16_t edgeWidth = pTile->tileEdgeLeft;
	uint16_t edgeHeight = pTile->tileEdgeTop;
	int32_t width = pTile->width;
	uint16_t height = pTile->height;

	int32_t channel = XV_TYPE_CHANNELS(xvTileType);
	int32_t bytesPerPix = XV_TYPE_ELEMENT_SIZE(xvTileType);
	int32_t bytePerPel;
	bytePerPel	= bytesPerPix / channel;

	int32_t pitch = (width + 2*edgeWidth)* channel;
	int32_t tileBuffSize;
	tileBuffSize = (pitch * (height + (2 * edgeHeight)) * bytePerPel);

	if (pFrame != NULL) {
		XV_CHECK_ERROR((pFrame->pixelRes != bytePerPel), pxvTM->errFlag = XV_ERROR_BAD_ARG, (( XVTM_ERROR)), "XVTM_ERROR");
		XV_CHECK_ERROR((pFrame->numChannels != channel), pxvTM->errFlag = XV_ERROR_BAD_ARG, ((XVTM_ERROR)), "XVTM_ERROR");
	}
	void *tileBuff;
	tileBuff = pBuff;
	uint32_t bufAvalibleFlag = 0;
	if(tileBuff!= NULL)
		bufAvalibleFlag = 1;
	if(tileBuff == NULL)
	{
		if((DMAInOut != XV_INTERIM_TILE || PingPong == PING ))
		{
			int32_t color = pxvTM->AllocateColor;
			if (color == (int32_t)XV_MEM_BANK_COLOR_ANY)
				color = 0;// pxvTM->numMemBanks - 1;
			if (DMAInOut == XV_INTERIM_TILE)
				color = 0;
			int32_t i;
			for( i = 0; i < numMemBanks; i++)
			{
				color = color %  numMemBanks;
				tileBuff = xvAllocateBufferHost(pxvTM, tileBuffSize, color, 128);
				if (tileBuff != (void *) XVTM_ERROR)
				{
					break;
				}
				color++;
			}

   			if (tileBuff == (void *) XVTM_ERROR && pxvTM->errFlag==XV_ERROR_ALLOC_FAILED) {
        		pxvTM->AllocateErrorState = XVTM_ERROR_MEMALLOC;
        		return(XVTM_ERROR_MEMALLOC);
      		}
			if (tileBuff == (void *) XVTM_ERROR) {
				pxvTM->AllocateErrorState = XVTM_ERROR;
				return(XVTM_ERROR);
			}
		}
		else if(DMAInOut == XV_INTERIM_TILE)
		{
			tileBuff = pxvTM->InterimTileList[pxvTM->InterimTileCounter];
			pxvTM->InterimTileCounter++;

		}
	}
	SETUP_TILE(pTile, tileBuff, tileBuffSize, pFrame, width, height, pitch, xvTileType, edgeWidth, edgeHeight, 0, 0, alignType);

	if((DMAInOut & XV_INPUT_TILE) != 0)
	{
		pWorkPacket->InTiles[PingPong][pWorkPacket->numInTiles[PingPong]]=pTile;
		pWorkPacket->numInTiles[PingPong]++;
	}
	if((DMAInOut & XV_OUTPUT_TILE) != 0)
	{
		pWorkPacket->OutTiles[PingPong][pWorkPacket->numOutTiles[PingPong]]=pTile;
		pWorkPacket->numOutTiles[PingPong]++;
	}
	else if(DMAInOut == XV_INTERIM_TILE && PingPong==PING && bufAvalibleFlag==0)
	{
		pxvTM->InterimTileList[pxvTM->InterimTileCounter] = tileBuff;
		pxvTM->InterimTileCounter++;

	}



	return XVTM_SUCCESS;
}

#endif
