/*
 * Copyright (c) 2020-2022 Cadence Design Systems Inc. ALL RIGHTS RESERVED.
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
#include <stdint.h>
#include <string.h>

#include "tileManager.h"

#include <xtensa/config/core.h>
#include <xtensa/idma.h>

//#define TEST_DTCM23

// #include <xtensa\config\core-isa.h>
#if !defined(NON_POOL_MEM)
#define NON_POOL_MEM 			(17*1024)
#endif
#define POOL_SIZE1                ((XCHAL_DATARAM1_SIZE) - NON_POOL_MEM)
#define POOL_SIZE0                ((XCHAL_DATARAM0_SIZE) - NON_POOL_MEM)
#define DMA_DESCR_CNT            32 // number of DMA descriptors
#ifdef XCHAL_IDMA_MAX_OUTSTANDING_REQ
#define MAX_PIF                  XCHAL_IDMA_MAX_OUTSTANDING_REQ
#else
#define MAX_PIF                  64
#endif


#ifndef __XTENSA__
#define _LOCAL_RAM0_
#define _LOCAL_RAM1_
#define ALIGN64
#if XCHAL_HAVE_NX
IDMA_BUFFER_DEFINE(idmaObjBuff, DMA_DESCR_CNT, IDMA_64B_DESC);
#else
IDMA_BUFFER_DEFINE(idmaObjBuff, DMA_DESCR_CNT, IDMA_2D_DESC);
#endif
#if XCHAL_HAVE_NX
IDMA_BUFFER_DEFINE(idmaObjBuff1, DMA_DESCR_CNT, IDMA_64B_DESC);
#endif

#else
#define _LOCAL_RAM0_  __attribute__((section(".dram0.data")))
#define _LOCAL_RAM1_  __attribute__((section(".dram1.data")))
#define ALIGN64  __attribute__((aligned(64)))
#if XCHAL_HAVE_NX
IDMA_BUFFER_DEFINE(idmaObjBuff, DMA_DESCR_CNT, IDMA_64B_DESC);
#else
IDMA_BUFFER_DEFINE(idmaObjBuff, DMA_DESCR_CNT, IDMA_2D_DESC);
#endif
#if XCHAL_HAVE_NX
IDMA_BUFFER_DEFINE(idmaObjBuff1, DMA_DESCR_CNT, IDMA_64B_DESC);
#ifdef TEST_DTCM23
IDMA_BUFFER_DEFINE(idmaObjBuff2, DMA_DESCR_CNT, IDMA_64B_DESC);
IDMA_BUFFER_DEFINE(idmaObjBuff3, DMA_DESCR_CNT, IDMA_64B_DESC);
#endif
#endif

#endif // MISRA_C_TEST
#ifndef XVTM_USE_XMEM
uint8_t ALIGN64 pBankBuffPool0[POOL_SIZE0]  _LOCAL_RAM0_;
uint8_t ALIGN64 pBankBuffPool1[POOL_SIZE1] 	_LOCAL_RAM1_;
#else
#include <xtensa/xmem_bank.h>
#endif
xvTileManager *gxvpTM;

typedef struct intrCallbackDataStruct
{
	int32_t intrCount;
} intrCbDataStruct;
intrCbDataStruct cbData0 _LOCAL_RAM0_;
intrCbDataStruct cbData1 _LOCAL_RAM0_;
#ifdef TEST_DTCM23
intrCbDataStruct cbData2 _LOCAL_RAM0_;
intrCbDataStruct cbData3 _LOCAL_RAM0_;
#endif
// IDMA error callback function
void errCallbackFunc0(idma_error_details_t* data)
{
	(void)(data);
	printf("ERROR CALLBACK: iDMA in Error\n");
	gxvpTM->errFlag = (xvError_t)XVTM_ERROR;
#if XCHAL_HAVE_NX
	gxvpTM->idmaErrorFlag[0] = (xvError_t)XVTM_ERROR;
#else
	gxvpTM->idmaErrorFlag = (xvError_t)XVTM_ERROR;
#endif
	//printf("COPY FAILED, Error %d at desc:%p, PIF src/dst=%x/%x\n", data->err_type, (void *) data->currDesc, data->srcAddr, data->dstAddr);
	return;
}

// IDMA error callback function
void errCallbackFunc1(idma_error_details_t* data)
{
	(void)(data);
	printf("ERROR CALLBACK: iDMA in Error\n");
	gxvpTM->errFlag = (xvError_t)XVTM_ERROR;
#if XCHAL_HAVE_NX
	gxvpTM->idmaErrorFlag[1] = (xvError_t)XVTM_ERROR;
#else
	gxvpTM->idmaErrorFlag = (xvError_t)XVTM_ERROR;
#endif
	//printf("COPY FAILED, Error %d at desc:%p, PIF src/dst=%x/%x\n", data->err_type, (void *) data->currDesc, data->srcAddr, data->dstAddr);
	return;
}

// IDMA Interrupt callback function
void intrCallbackFunc0(void *pCallBackStr)
{
	printf("INTERRUPT CALLBACK : processing iDMA interrupt\n");
	((intrCbDataStruct *) pCallBackStr)->intrCount++;

	return;
}

void intrCallbackFunc1(void *pCallBackStr)
{
	printf("INTERRUPT CALLBACK : processing iDMA interrupt\n");
	((intrCbDataStruct *) pCallBackStr)->intrCount++;

	return;
}
#ifdef TEST_DTCM23
void errCallbackFunc2(idma_error_details_t* data)
{
	(void)(data);
	printf("ERROR CALLBACK: iDMA in Error\n");
	gxvpTM->errFlag = (xvError_t)XVTM_ERROR;
#if XCHAL_HAVE_NX
	gxvpTM->idmaErrorFlag[2] = (xvError_t)XVTM_ERROR;
#else
	gxvpTM->idmaErrorFlag = (xvError_t)XVTM_ERROR;
#endif
	//printf("COPY FAILED, Error %d at desc:%p, PIF src/dst=%x/%x\n", data->err_type, (void *) data->currDesc, data->srcAddr, data->dstAddr);
	return;
}
void errCallbackFunc3(idma_error_details_t* data)
{
	(void)(data);
	printf("ERROR CALLBACK: iDMA in Error\n");
	gxvpTM->errFlag = (xvError_t)XVTM_ERROR;
#if XCHAL_HAVE_NX
	gxvpTM->idmaErrorFlag[3] = (xvError_t)XVTM_ERROR;
#else
	gxvpTM->idmaErrorFlag = (xvError_t)XVTM_ERROR;
#endif
	//printf("COPY FAILED, Error %d at desc:%p, PIF src/dst=%x/%x\n", data->err_type, (void *) data->currDesc, data->srcAddr, data->dstAddr);
	return;
}
void intrCallbackFunc2(void *pCallBackStr)
{
	printf("INTERRUPT CALLBACK : processing iDMA interrupt\n");
	((intrCbDataStruct *) pCallBackStr)->intrCount++;

	return;
}
void intrCallbackFunc3(void *pCallBackStr)
{
	printf("INTERRUPT CALLBACK : processing iDMA interrupt\n");
	((intrCbDataStruct *) pCallBackStr)->intrCount++;

	return;
}
#endif//#ifdef TEST_DTCM23
#define STACK_SIZE 8*1024
int32_t  xvfInitTileManager(xvTileManager *pxvTM)
{
#ifdef XVTM_MULTITHREADING_SUPPORT
 xos_start_main_ex("main", 1 /*priority*/, STACK_SIZE);
#define TICK_CYCLES (xos_get_clock_freq()/10)
 xos_start_system_timer(-1, TICK_CYCLES);
#endif

#if XCHAL_HAVE_NX
 //idma_init(IDMA_CHANNEL_0, 0, MAX_BLOCK_8, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc0);
 //idma_init(IDMA_CHANNEL_1, 0, MAX_BLOCK_8, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc1);
 idma_init(IDMA_CHANNEL_0, 0, MAX_BLOCK_4, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc0);
 idma_init(IDMA_CHANNEL_1, 0, MAX_BLOCK_4, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc1);
#ifdef TEST_DTCM23
 idma_init(IDMA_CHANNEL_2, 0, MAX_BLOCK_8, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc2);
 idma_init(IDMA_CHANNEL_3, 0, MAX_BLOCK_8, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc3);
#endif
#else
 idma_init(0, MAX_BLOCK_8, MAX_PIF, TICK_CYCLES_2, 0, (idma_err_callback_fn)errCallbackFunc0);
#endif

#ifndef XVTM_USE_XMEM
	void *buffPool[2];
	int32_t buffSize[2];
	cbData0.intrCount = 0;
	cbData1.intrCount = 0;
	// Initialize the DMA, Memory Allocator and Tile Manager
	buffPool[0] = pBankBuffPool0;
	buffPool[1] = pBankBuffPool1;
	buffSize[0] = POOL_SIZE0;
	buffSize[1] = POOL_SIZE1;
	gxvpTM = pxvTM;
#if XCHAL_HAVE_NX
	uint32_t retVal = xvCreateTileManagerMultiChannelHost(pxvTM, idmaObjBuff, idmaObjBuff1, 2, buffPool, buffSize,
			(idma_err_callback_fn)errCallbackFunc0, (idma_err_callback_fn)errCallbackFunc1,
			intrCallbackFunc0, (void *) &cbData0,
			intrCallbackFunc1, (void *) &cbData1,
			DMA_DESCR_CNT, MAX_BLOCK_8, MAX_PIF);
#else
	uint32_t retVal = xvCreateTileManagerHost(pxvTM, idmaObjBuff, 2, buffPool, buffSize,
			(idma_err_callback_fn)errCallbackFunc0, intrCallbackFunc0, (void *) &cbData0, DMA_DESCR_CNT, MAX_BLOCK_8, MAX_PIF);
#endif
#else

	xmem_bank_status_t xmbs;
    xmbs = xmem_init_local_mem(XMEM_BANK_HEAP_ALLOC, STACK_SIZE/*stack_size*/);
	
	if(XMEM_BANK_ERR_STACK_RESERVE_FAIL==xmbs)
	{
		//printf("Stack not in DRAM\n");

	}
	
    if(xmbs != XMEM_BANK_OK && XMEM_BANK_ERR_STACK_RESERVE_FAIL!=xmbs)
    {
		//printf("xmem_init_local_mem() failed\n");

    }
	

#if XCHAL_HAVE_NX
#ifdef TEST_DTCM23
	uint32_t retVal = xvCreateTileManagerMultiChannelHost(pxvTM, idmaObjBuff, idmaObjBuff1,idmaObjBuff2, idmaObjBuff3, 0, NULL, NULL,
			(idma_err_callback_fn)errCallbackFunc0, (idma_err_callback_fn)errCallbackFunc1,(idma_err_callback_fn)errCallbackFunc2, (idma_err_callback_fn)errCallbackFunc3,
			intrCallbackFunc0, (void *) &cbData0,
			intrCallbackFunc1, (void *) &cbData1,intrCallbackFunc2, (void *) &cbData2,intrCallbackFunc3, (void *) &cbData3,
			DMA_DESCR_CNT, MAX_BLOCK_8, MAX_PIF);
#else
	uint32_t retVal = xvCreateTileManagerMultiChannelHost(pxvTM, idmaObjBuff, idmaObjBuff1, 0, NULL, NULL,
			(idma_err_callback_fn)errCallbackFunc0, (idma_err_callback_fn)errCallbackFunc1,
			intrCallbackFunc0, (void *) &cbData0,
			intrCallbackFunc1, (void *) &cbData1,
			DMA_DESCR_CNT, MAX_BLOCK_8, MAX_PIF);
#endif
#else
	uint32_t retVal = xvCreateTileManagerHost(pxvTM, idmaObjBuff, 0, NULL, NULL,
			(idma_err_callback_fn)errCallbackFunc0, intrCallbackFunc0, (void *) &cbData0, DMA_DESCR_CNT, MAX_BLOCK_8, MAX_PIF);
#endif
#endif
	XV_CHECK_ERROR(retVal != XVTM_SUCCESS,xvGetErrorInfoHost(pxvTM), XVTM_ERROR, "Error in Init TM");
	return(retVal);
}
