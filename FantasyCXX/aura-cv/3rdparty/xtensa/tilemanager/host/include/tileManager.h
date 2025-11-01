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

#ifndef TILE_MANAGER_H__
#define TILE_MANAGER_H__

#if defined (__cplusplus)
extern "C"
{
#endif
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#define IDMA_USE_MULTICHANNEL          1
#define IDMA_USE_WIDE_ADDRESS_COMPILE  //Enable this for cores with >32bit data memory

//#define XVTM_MULTITHREADING_SUPPORT
#ifdef __XTENSA__
#define XVTM_USE_XMEM
#endif

#include "tmUtils.h"

#ifdef XVTM_USE_XMEM
#include <xtensa/xmem_bank.h>
#endif

#include <xtensa/config/core-isa.h>
#if (!defined(__XTENSA__)) || (defined(UNIFIED_TEST))
#include "dummy.h"
#endif
#ifndef XVTM_MULTITHREADING_SUPPORT
#define IDMA_APP_USE_XTOS
#else
#include <xtensa/xos.h>
#endif


#if defined(__XTENSA__) && (!defined(UNIFIED_TEST))
#include <xtensa/config/core.h>
#include <xtensa/idma.h>

#if (IDMA_USE_WIDE_API == 0)

#ifdef __XTENSA__
#define idma_copy_2d_desc64_wide(dmaChannel, pdst, psrc, rowSize, intrCompletionFlag, numRows, srcPitch, dstPitch) \
  idma_copy_2d_desc64((dmaChannel), (pdst), (psrc), (rowSize), (intrCompletionFlag), (numRows), (srcPitch), (dstPitch))
#define idma_copy_2d_pred_desc64_wide(dmaChannel, pdst, psrc, rowSize, intrCompletionFlag, pred_mask, numRows, srcPitch, dstPitch) \
  idma_copy_2d_pred_desc64((dmaChannel), (pdst), (psrc), (rowSize), (intrCompletionFlag), (pred_mask), (numRows), (srcPitch), (dstPitch))
#define idma_copy_3d_desc64_wide(dmaChannel, pdst, psrc, intrCompletionFlag, rowSize, numRows, numTiles, srcPitch, dstPitch, srcTilePitch, dstTilePitch) \
  idma_copy_3d_desc64((dmaChannel), (pdst), (psrc), (intrCompletionFlag), (rowSize), (numRows), (numTiles), (srcPitch), (dstPitch), (srcTilePitch), (dstTilePitch))

#define idma_add_2d_desc64_wide(buf,  pdst, psrc, rowSize, intrCompletionFlag, numRows, srcPitch, dstPitch) \
  idma_add_2d_desc64((buf), (pdst), (psrc), (rowSize), (intrCompletionFlag), (numRows), (srcPitch), (dstPitch))
#define idma_add_2d_pred_desc64_wide(buf,  pdst, psrc, rowSize, intrCompletionFlag, pred_mask, numRows, srcPitch, dstPitch) \
  idma_add_2d_pred_desc64((buf),  (pdst), (psrc), (rowSize), (intrCompletionFlag), (pred_mask), (numRows), (srcPitch), (dstPitch))
#define idma_add_3d_desc64_wide(buf,  pdst, psrc, intrCompletionFlag, rowSize, numRows, numTiles, srcPitch, dstPitch, srcTilePitch, dstTilePitch) \
  idma_add_3d_desc64((buf),  (pdst), (psrc), (intrCompletionFlag), (rowSize), (numRows), (numTiles), (srcPitch), (dstPitch), (srcTilePitch), (dstTilePitch))


#endif
#endif
#endif //__XTENSA__

#if defined(__XTENSA__) && (!defined(UNIFIED_TEST))
#include <xtensa/tie/xt_misc.h>
#endif //__XTENSA__

// #define TM_LOG
#include "tileManager_api.h"

#ifdef TM_LOG
extern xvTileManager * __pxvTM;
#define TILE_MANAGER_LOG_FILE_NAME  "tileManager.log"
#define TM_LOG_PRINT(fmt, ...)  do { fprintf(__pxvTM->tm_log_fp, (fmt), __VA_ARGS__); } while (0)
#else
#define TM_LOG_PRINT(fmt, ...)  do {} while (0)
#endif

#ifndef IVP_SIMD_WIDTH
#define IVP_SIMD_WIDTH  XCHAL_IVPN_SIMD_WIDTH
#endif
#define IVP_ALIGNMENT   0x1F
#define XVTM_MIN(a, b)  (((a) < (b)) ? (a) : (b))

#define XVTM_DUMMY_DMA_INDEX  -2

#define XVTM_ERROR_MEMALLOC   -2
#define XVTM_ERROR            -1
#define XVTM_SUCCESS          0


#define XVTM_RAISE_EXCEPTION(ch, pxvTM)  ((pxvTM)->idmaErrorFlag[(ch)] = XV_ERROR_IDMA)
#define XVTM_RESET_EXCEPTION(ch, pxvTM)  ((pxvTM)->idmaErrorFlag[(ch)] = XV_ERROR_SUCCESS)
#define IS_TRANSFER_SUCCESS(ch, pxvTM)   (((pxvTM)->idmaErrorFlag[(ch)] == XV_ERROR_SUCCESS) ? 1 : 0)

/***********************************
*    Other Marcos
***********************************/
#define WAIT_FOR_TILE_MULTICHANNEL(ch, pxvTM, pTile)        xvWaitForTileMultiChannel((ch), (pxvTM), (pTile))
#define SLEEP_FOR_TILE_MULTICHANNEL(ch, pxvTM, pTile)       xvSleepForTileMultiChannel((ch), (pxvTM), (pTile))
#define WAIT_FOR_DMA_MULTICHANNEL(ch, pxvTM, dmaIndex)      xvWaitForiDMAMultiChannel((ch), (pxvTM), (dmaIndex))
#define SLEEP_FOR_DMA_MULTICHANNEL(ch, pxvTM, dmaIndex)     xvSleepForiDMAMultiChannel((ch), (pxvTM), (dmaIndex))
#define WAIT_FOR_TILE_FAST_MULTICHANNEL(ch, pxvTM, pTile)   xvWaitForTileFastMultiChannel((ch), (pxvTM), (pTile))
#define SLEEP_FOR_TILE_FAST_MULTICHANNEL(ch, pxvTM, pTile)  xvSleepForTileFastMultiChannel((ch), (pxvTM), (pTile))

#define XV_TILE_CHECK_VIRTUAL_FRAME(pTile)                  ((pTile)->pFrame->pFrameBuff == NULL)
#define XV_FRAME_CHECK_VIRTUAL_FRAME(pFrame)                ((pFrame)->pFrameBuff == NULL)

#define SETUP_TILE(pTile, pBuf, bufSize, pFrame, width, height, pitch, type, edgeWidth, edgeHeight, x, y, alignType)              \
  {                                                                                                                               \
    int32_t tileType, bytesPerPixel, channels, bytesPerPel;                                                                       \
    uint8_t *edgePtr  = (uint8_t *) (pBuf), *dataPtr;                                                                             \
    int32_t alignment = 127;                                                                                                      \
    if (((alignType) == XVTM_EDGE_ALIGNED_N) || ((alignType) == XVTM_DATA_ALIGNED_N)) { (alignment) = 63; }                               \
    tileType      = (type);                                                                                                         \
    bytesPerPixel = XV_TYPE_ELEMENT_SIZE(tileType);                                                                               \
    channels      = XV_TYPE_CHANNELS(tileType);                                                                                   \
    bytesPerPel   = bytesPerPixel / channels;                                                                                     \
    XV_TILE_SET_FRAME_PTR((xvTile *) (pTile), ((xvFrame *) (pFrame)));                                                            \
    XV_TILE_SET_BUFF_PTR((xvTile *) (pTile), (pBuf));                                                                             \
    XV_TILE_SET_BUFF_SIZE((xvTile *) (pTile), (bufSize));                                                                         \
    if (((alignType) == XVTM_EDGE_ALIGNED_N) || ((alignType) == XVTM_EDGE_ALIGNED_2N))                                                    \
    {                                                                                                                             \
      edgePtr = &((uint8_t *) (pBuf))[alignment - (((uint32_t) (pBuf) + (uint32_t)alignment) & (alignment))];                                \
    }                                                                                                                             \
    XV_TILE_SET_DATA_PTR((xvTile *) (pTile), (uint32_t)edgePtr + (((edgeHeight) * (pitch) * bytesPerPel)) + ((edgeWidth) * bytesPerPixel)); \
    if (((alignType) == XVTM_DATA_ALIGNED_N) || ((alignType) == XVTM_DATA_ALIGNED_2N))                                                     \
    {                                                                                                                             \
      dataPtr = (uint8_t *) XV_TILE_GET_DATA_PTR((xvTile *) (pTile));                                                             \
      dataPtr = (uint8_t *) (void *)(((long) ((void *)(dataPtr)) + alignment) & (~alignment));                                                      \
      XV_TILE_SET_DATA_PTR((xvTile *) (pTile), dataPtr);                                                                          \
    }                                                                                                                             \
    XV_TILE_SET_WIDTH((xvTile *) (pTile), (width));                                                                               \
    XV_TILE_SET_HEIGHT((xvTile *) (pTile), (height));                                                                             \
    XV_TILE_SET_PITCH((xvTile *) (pTile), (pitch));                                                                               \
    XV_TILE_SET_TYPE((xvTile *) (pTile), (tileType | XV_TYPE_TILE_BIT));                                                          \
    XV_TILE_SET_EDGE_WIDTH((xvTile *) (pTile), (edgeWidth));                                                                      \
    XV_TILE_SET_EDGE_HEIGHT((xvTile *) (pTile), (edgeHeight));                                                                    \
    XV_TILE_SET_X_COORD((xvTile *) (pTile), (x));                                                                                 \
    XV_TILE_SET_Y_COORD((xvTile *) (pTile), (y));                                                                                 \
    XV_TILE_SET_STATUS_FLAGS((xvTile *) (pTile), 0);                                                                              \
    XV_TILE_RESET_DMA_INDEX((xvTile *) (pTile));                                                                                  \
    XV_TILE_RESET_PREVIOUS_TILE((xvTile *) (pTile));                                                                              \
    XV_TILE_RESET_REUSE_COUNT((xvTile *) (pTile));                                                                                \
  }


#define SETUP_TILE_3D(pTile3D, pBuf, bufSize, pFrame3D, width, height, depth, pitch, pitch2D, type, edgeWidth, edgeHeight, edgeDepth, x, y, z, alignType) \
  {                                                                                                                                                       \
    int32_t tileType, bytesPerPixel, channels, bytesPerPel;                                                                                               \
    uint8_t *edgePtr  = (uint8_t *) (pBuf), *dataPtr;                                                                                                     \
    int32_t alignment = 127;                                                                                                                               \
    if (((alignType) == XVTM_EDGE_ALIGNED_N) || ((alignType) == XVTM_DATA_ALIGNED_N)) { (alignment) = 63; }                                                       \
    tileType      = (type);                                                                                                                                 \
    bytesPerPixel = XV_TYPE_ELEMENT_SIZE(tileType);                                                                                                       \
    channels      = XV_TYPE_CHANNELS(tileType);                                                                                                           \
    bytesPerPel   = bytesPerPixel / channels;                                                                                                             \
    XV_TILE_3D_SET_FRAME_3D_PTR((pTile3D), (pFrame3D));                                                                                                       \
    XV_TILE_SET_BUFF_PTR((pTile3D), (pBuf));                                                                                                              \
    XV_TILE_SET_BUFF_SIZE((pTile3D), (bufSize));                                                                                                          \
    if (((alignType) == XVTM_EDGE_ALIGNED_N) || ((alignType) == XVTM_EDGE_ALIGNED_2N))                                                                             \
    {                                                                                                                                                     \
      edgePtr = &((uint8_t *) (pBuf))[alignment - (((int32_t) (pBuf) + alignment) & (alignment))];                                                        \
    }                                                                                                                                                     \
    XV_TILE_SET_DATA_PTR((pTile3D), edgePtr + (((edgeHeight) * (pitch) * bytesPerPel)) + ((edgeWidth) * bytesPerPixel) +                                  \
                         ((edgeDepth) * (pitch2D) * bytesPerPel));                                                                                        \
    if (((alignType) == XVTM_DATA_ALIGNED_N) || ((alignType) == XVTM_DATA_ALIGNED_2N))                                                                             \
    {                                                                                                                                                     \
      dataPtr = (uint8_t *) XV_TILE_GET_DATA_PTR(pTile3D);                                                                                                \
      dataPtr = (uint8_t *) (((long) (dataPtr) + alignment) & (~alignment));                                                                              \
      XV_TILE_SET_DATA_PTR((pTile3D), dataPtr);                                                                                                             \
    }                                                                                                                                                     \
    XV_TILE_SET_WIDTH((pTile3D), (width));                                                                                                                \
    XV_TILE_SET_HEIGHT((pTile3D), (height));                                                                                                              \
    XV_TILE_SET_PITCH((pTile3D), (pitch));                                                                                                                \
    XV_TILE_SET_TYPE((pTile3D), (tileType | XV_TYPE_TILE_BIT));                                                                                           \
    XV_TILE_SET_EDGE_WIDTH((pTile3D), (edgeWidth));                                                                                                       \
    XV_TILE_SET_EDGE_HEIGHT((pTile3D), (edgeHeight));                                                                                                     \
    XV_TILE_SET_X_COORD((pTile3D), (x));                                                                                                                  \
    XV_TILE_SET_Y_COORD((pTile3D), (y));                                                                                                                  \
    XV_TILE_SET_STATUS_FLAGS((pTile3D), 0);                                                                                                               \
    XV_TILE_RESET_DMA_INDEX((pTile3D));                                                                                                                   \
    XV_TILE_SET_Z_COORD((pTile3D), (z));                                                                                                                      \
    XV_TILE_SET_TILE_PITCH((pTile3D), (pitch2D));                                                                                                             \
    XV_TILE_SET_EDGE_DEPTH((pTile3D), (edgeDepth));                                                                                                           \
    XV_TILE_SET_DEPTH((pTile3D), (depth));                                                                                                                    \
    (pTile3D)->pTemp = (void *)NULL;                                                                                                                                \
  }

#define SETUP_FRAME(pFrame, pFrameBuffer, buffSize, width, height, pitch, padWidth, padHeight, pixRes, numCh, paddingType, paddingVal)                   \
  {                                                                                                                                                      \
    XV_FRAME_SET_BUFF_PTR((xvFrame *) (pFrame), (pFrameBuffer));                                                                                         \
    XV_FRAME_SET_BUFF_SIZE((xvFrame *) (pFrame), (buffSize));                                                                                            \
    XV_FRAME_SET_WIDTH((xvFrame *) (pFrame), (width));                                                                                                   \
    XV_FRAME_SET_HEIGHT((xvFrame *) (pFrame), (height));                                                                                                 \
    XV_FRAME_SET_PITCH((xvFrame *) (pFrame), (pitch));                                                                                                   \
    XV_FRAME_SET_PIXEL_RES((xvFrame *) (pFrame), (pixRes));                                                                                              \
    XV_FRAME_SET_DATA_PTR((xvFrame *) (pFrame), (uint64_t) (pFrameBuffer) + (uint64_t) ((((pitch) * (padHeight)) + ((padWidth) * (numCh))) * (pixRes))); \
    XV_FRAME_SET_EDGE_WIDTH((xvFrame *) (pFrame), (padWidth));                                                                                           \
    XV_FRAME_SET_EDGE_HEIGHT((xvFrame *) (pFrame), (padHeight));                                                                                         \
    XV_FRAME_SET_NUM_CHANNELS((xvFrame *) (pFrame), (numCh));                                                                                            \
    XV_FRAME_SET_PADDING_TYPE((xvFrame *) (pFrame), (paddingType));                                                                                      \
    XV_FRAME_SET_PADDING_VALUE((xvFrame *) (pFrame), (paddingVal));                                                                                      \
  }



#define SETUP_FRAME_3D(pFrame3D, pFrameBuffer, buffSize, width, height, depth, pitch, pitchFrame2D, padWidth, padHeight, padDepth, pixRes, numCh, paddingType, paddingVal) \
  {                                                                                                                                                                        \
    XV_FRAME_SET_BUFF_PTR((pFrame3D), (pFrameBuffer));                                                                                                                     \
    XV_FRAME_SET_BUFF_SIZE((pFrame3D), (buffSize));                                                                                                                        \
    XV_FRAME_SET_WIDTH((pFrame3D), (width));                                                                                                                               \
    XV_FRAME_SET_HEIGHT((pFrame3D), (height));                                                                                                                             \
    XV_FRAME_SET_PITCH((pFrame3D), (pitch));                                                                                                                               \
    XV_FRAME_SET_PIXEL_RES((pFrame3D), (pixRes));                                                                                                                          \
    XV_FRAME_SET_DATA_PTR((pFrame3D), ((uint64_t) (pFrameBuffer)) + ((((pitch) * (padHeight)) +                                                                            \
                                                                      ((padWidth) * (numCh))) * (pixRes)) + (((pitchFrame2D) * (padDepth)) * (pixRes)));                   \
    XV_FRAME_SET_EDGE_WIDTH((pFrame3D), (padWidth));                                                                                                                       \
    XV_FRAME_SET_EDGE_HEIGHT((pFrame3D), (padHeight));                                                                                                                     \
    XV_FRAME_SET_NUM_CHANNELS((pFrame3D), (numCh));                                                                                                                        \
    XV_FRAME_SET_PADDING_TYPE((pFrame3D), (paddingType));                                                                                                                  \
    XV_FRAME_SET_PADDING_VALUE((pFrame3D), (paddingVal));                                                                                                                  \
    XV_FRAME_SET_EDGE_DEPTH((pFrame3D), (padDepth));                                                                                                                           \
    XV_FRAME_SET_DEPTH((pFrame3D), (depth));                                                                                                                                   \
    XV_FRAME_SET_FRAME_PITCH((pFrame3D), (pitchFrame2D));                                                                                                                      \
  }


#define WAIT_FOR_TILE(pxvTM, pTile)        WAIT_FOR_TILE_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (pTile))
#define SLEEP_FOR_TILE(pxvTM, pTile)       SLEEP_FOR_TILE_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (pTile))
#define WAIT_FOR_DMA(pxvTM, dmaIndex)      WAIT_FOR_DMA_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (dmaIndex))
#define SLEEP_FOR_DMA(pxvTM, dmaIndex)     SLEEP_FOR_DMA_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (dmaIndex))
#define WAIT_FOR_TILE_FAST(pxvTM, pTile)   WAIT_FOR_TILE_FAST_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (pTile))
#define SLEEP_FOR_TILE_FAST(pxvTM, pTile)  SLEEP_FOR_TILE_FAST_MULTICHANNEL(TM_IDMA_CH0, (pxvTM), (pTile))


#define WAIT_FOR_TILE_3D(pxvTM, pTile3D)                   WAIT_FOR_TILE_MULTICHANNEL_3D(TM_IDMA_CH0, (pxvTM), (pTile3D))
#define WAIT_FOR_TILE_MULTICHANNEL_3D(ch, pxvTM, pTile3D)  xvWaitForTileMultiChannel3D((ch), (pxvTM), (pTile3D))



// Assumes both top and bottom edges are equal before and after the update.
#define XV_TILE_UPDATE_EDGE_HEIGHT(pTile, newEdgeHeight)                              \
  {                                                                                   \
    uint32_t tileType       = XV_TILE_GET_TYPE(pTile);                                \
    uint32_t bytesPerPixel  = XV_TYPE_ELEMENT_SIZE(tileType);                         \
    uint32_t channels       = XV_TYPE_CHANNELS(tileType);                             \
    uint32_t bytesPerPel    = bytesPerPixel / channels;                               \
    uint16_t currEdgeHeight = (uint16_t) XV_TILE_GET_EDGE_HEIGHT(pTile);              \
    uint32_t tilePitch      = (uint32_t) XV_TILE_GET_PITCH(pTile);                    \
    uint32_t tileHeight     = (uint32_t) XV_TILE_GET_HEIGHT(pTile);                   \
    uint32_t dataU32        = (uint32_t) XV_TILE_GET_DATA_PTR(pTile);                 \
    dataU32 = dataU32 + tilePitch * bytesPerPel * ((newEdgeHeight) - currEdgeHeight); \
    XV_TILE_SET_DATA_PTR((pTile), (void *) dataU32);                                  \
    XV_TILE_SET_EDGE_HEIGHT((pTile), (newEdgeHeight));                                \
    XV_TILE_SET_HEIGHT((pTile), tileHeight + 2 * (currEdgeHeight - (newEdgeHeight))); \
  }

// Assumes both left and right edges are equal before and after the update.
#define XV_TILE_UPDATE_EDGE_WIDTH(pTile, newEdgeWidth)                            \
  {                                                                               \
    uint32_t tileType      = (pTile)->type;                                       \
    uint32_t bytesPerPixel = XV_TYPE_ELEMENT_SIZE(tileType);                      \
    uint16_t currEdgeWidth = (uint16_t) XV_TILE_GET_EDGE_WIDTH(pTile);            \
    uint32_t tileWidth     = (uint32_t) XV_TILE_GET_WIDTH(pTile);                 \
    uint32_t dataU32       = (uint32_t) XV_TILE_GET_DATA_PTR(pTile);              \
    dataU32 = dataU32 + ((newEdgeWidth) - currEdgeWidth) * bytesPerPixel;         \
    XV_TILE_SET_DATA_PTR((pTile), (void *) dataU32);                              \
    XV_TILE_SET_EDGE_WIDTH((pTile), (newEdgeWidth));                              \
    XV_TILE_SET_WIDTH((pTile), tileWidth + 2 * (currEdgeWidth - (newEdgeWidth))); \
  }

#define XV_TILE_UPDATE_DIMENSIONS(pTile, x, y, w, h, p) \
  {                                                     \
    XV_TILE_SET_X_COORD((pTile), (x));                  \
    XV_TILE_SET_Y_COORD((pTile), (y));                  \
    XV_TILE_SET_WIDTH((pTile), (w));                    \
    XV_TILE_SET_HEIGHT((pTile), (h));                   \
    XV_TILE_SET_PITCH((pTile), (p));                    \
  }

#define XV_TILE_GET_CHANNEL(t)  XV_TYPE_CHANNELS(XV_TILE_GET_TYPE(t))
#define XV_TILE_PEL_SIZE(t)     (XV_TILE_GET_ELEMENT_SIZE(t) / XV_TILE_GET_CHANNEL(t))

#define XV_ARRAY_STARTS_IN_DRAM(t) \
  (((unsigned int) XV_ARRAY_GET_BUFF_PTR(t)) >= ((unsigned int) XVTM_DRAM_LOWER_ADDR))
#define XV_ARRAY_ENDS_IN_DRAM(t) \
  ((((uint64_t) (uint32_t) XV_ARRAY_GET_BUFF_PTR(t)) + ((uint64_t) XV_ARRAY_GET_BUFF_SIZE(t))) <= ((uint64_t) XVTM_DRAM_HIGHER_ADDR))
#define XV_TILE_STARTS_IN_DRAM(t) \
  (((unsigned int) XV_TILE_GET_BUFF_PTR(t)) >= ((unsigned int) XVTM_DRAM_LOWER_ADDR))
#define XV_TILE_ENDS_IN_DRAM(t) \
  ((((uint64_t) (uint32_t) XV_TILE_GET_BUFF_PTR(t)) + ((uint64_t) XV_TILE_GET_BUFF_SIZE(t))) <= ((uint64_t) XVTM_DRAM_HIGHER_ADDR))

#define XV_PTR_START_IN_DRAM(t) \
  (((unsigned int) (t)) >= (((unsigned int ) XVTM_DRAM_LOWER_ADDR)))

#define XV_PTR_END_IN_DRAM(t) \
  (((uint64_t) (t)) <= (((unsigned int ) XVTM_DRAM_HIGHER_ADDR)))


#define XV_TILE_PITCH_IS_CONSISTENT(t) \
  ((XV_TILE_GET_PITCH(t)) >= (((XV_TILE_GET_WIDTH(t) + XV_TILE_GET_EDGE_LEFT(t)) + XV_TILE_GET_EDGE_RIGHT(t)) * XV_TILE_GET_CHANNEL(t)))

#define XV_TILE_BASE_IS_CONSISTENT(t)                                                                                                                                          \
  (((uint8_t *) XV_TILE_GET_DATA_PTR(t) - ((XV_TILE_GET_EDGE_LEFT(t) * XV_TILE_GET_ELEMENT_SIZE(t)) + (XV_TILE_GET_PITCH(t) * (XV_TILE_GET_EDGE_TOP(t) * XV_TILE_PEL_SIZE(t))))) \
   >= ((uint8_t *) XV_TILE_GET_BUFF_PTR(t)))
#define XV_TILE_END_IS_CONSISTENT(t)                                                                                                                                                                        \
  ((((uint32_t ) XV_TILE_GET_DATA_PTR(t) - (int32_t)(XV_TILE_GET_EDGE_LEFT(t) * XV_TILE_GET_ELEMENT_SIZE(t))) + (XV_TILE_GET_PITCH(t) * (int32_t)((int32_t)XV_TILE_GET_HEIGHT(t) + (int32_t)XV_TILE_GET_EDGE_BOTTOM(t)) * XV_TILE_PEL_SIZE(t))) \
   <= ((uint32_t ) XV_TILE_GET_BUFF_PTR(t) + XV_TILE_GET_BUFF_SIZE(t)))

#define XV_TILE_IS_CONSISTENT(t)  (   \
    ((XV_TILE_PITCH_IS_CONSISTENT(t) && \
    XV_TILE_BASE_IS_CONSISTENT(t)) &&  \
    XV_TILE_END_IS_CONSISTENT(t)))
#define XV_TILE_2D_PITCH_IS_CONSISTENT(t) \
  ((XV_TILE_GET_TILE_PITCH(t)) >= ((int32_t)((int32_t)(XV_TILE_GET_HEIGHT(t) + XV_TILE_GET_EDGE_TOP(t)) + (int32_t)XV_TILE_GET_EDGE_BOTTOM(t)) * XV_TILE_GET_PITCH(t)))

#define XV_TILE_3D_BASE_IS_CONSISTENT(t)                                                                                                                                      \
  (((uint32_t) XV_TILE_GET_DATA_PTR(t) - ((XV_TILE_GET_EDGE_LEFT(t) * XV_TILE_GET_ELEMENT_SIZE(t)) + ((XV_TILE_GET_PITCH(t) * XV_TILE_GET_EDGE_TOP(t)) * XV_TILE_PEL_SIZE(t)) + \
                                          ((XV_TILE_GET_TILE_PITCH(t) * XV_TILE_GET_EDGE_FRONT(t)) * XV_TILE_PEL_SIZE(t)))) >= ((uint32_t) XV_TILE_GET_BUFF_PTR(t)))
#define XV_TILE_3D_END_IS_CONSISTENT(t)                                                                                                                                      \
  ((((uint32_t) XV_TILE_GET_DATA_PTR(t) - (int32_t)(XV_TILE_GET_EDGE_LEFT(t) * XV_TILE_GET_ELEMENT_SIZE(t)) - ((XV_TILE_GET_PITCH(t) * (int32_t)XV_TILE_GET_EDGE_TOP(t)) * XV_TILE_PEL_SIZE(t))) + \
    (XV_TILE_GET_TILE_PITCH(t) * (int32_t)(XV_TILE_GET_DEPTH(t) + XV_TILE_GET_EDGE_BACK(t)) * XV_TILE_PEL_SIZE(t))) <= ((uint32_t) XV_TILE_GET_BUFF_PTR(t) + XV_TILE_GET_BUFF_SIZE(t)))

#define XV_TILE_IS_CONSISTENT_3D(t)  (   \
    (((XV_TILE_PITCH_IS_CONSISTENT(t) &&    \
    XV_TILE_2D_PITCH_IS_CONSISTENT(t)) && \
    XV_TILE_3D_BASE_IS_CONSISTENT(t)) &&  \
    XV_TILE_3D_END_IS_CONSISTENT(t)))

#define XV_CHECK_POINTER(pointer, statment) \
  XV_CHECK_ERROR((pointer) == (void *) NULL, (statment), XVTM_ERROR, " NULL pointer error")

#define XV_CHECK_TILE(tile, TileMgr)                                                                                                          \
  XV_CHECK_POINTER((tile), ((TileMgr)->errFlag = XV_ERROR_BAD_ARG));                                                                          \
  XV_CHECK_ERROR(!(XV_TILE_IS_TILE(tile) > 0), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "The argument is not a tile");            \
  XV_CHECK_ERROR(!(XV_TILE_STARTS_IN_DRAM(tile)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not start in DRAM"); \
  XV_CHECK_ERROR(!(XV_TILE_ENDS_IN_DRAM(tile)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not fit in DRAM");     \
  XV_CHECK_ERROR(!(XV_TILE_IS_CONSISTENT(tile)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Invalid buffer")


#define XV_CHECK_TILE_3D(tile3D, TileMgr)                                                                                                       \
  XV_CHECK_POINTER((tile3D), (TileMgr)->errFlag = XV_ERROR_BAD_ARG);                                                                            \
  XV_CHECK_ERROR(!(XV_TILE_IS_TILE(tile3D) > 0), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "The argument is not a tile");            \
  XV_CHECK_ERROR(!(XV_TILE_STARTS_IN_DRAM(tile3D)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not start in DRAM"); \
  XV_CHECK_ERROR(!(XV_TILE_ENDS_IN_DRAM(tile3D)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Data buffer does not fit in DRAM");     \
  XV_CHECK_ERROR(!(XV_TILE_IS_CONSISTENT_3D(tile3D)), (TileMgr)->errFlag = XV_ERROR_BAD_ARG, XVTM_ERROR, "Invalid buffer")


/* do not check arguments for errors */
#define XV_ERROR_LEVEL_NO_ERROR                     0
/* call exit(-1) in case of error */
#define XV_ERROR_LEVEL_TERMINATE_ON_ERROR           1
/* return corresponding error code on error without any processing (recommended)*/
#define XV_ERROR_LEVEL_RETURN_ON_ERROR              2
/* capture error but attempt continue processing (dangerous!) */
#define XV_ERROR_LEVEL_CONTINUE_ON_ERROR            3
/* print error message to stdout and return without any processing */
#define XV_ERROR_LEVEL_PRINT_ON_ERROR               4
/* print error message but attempt continue processing (dangerous!) */
#define XV_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR  5

#ifndef XV_ERROR_LEVEL
#define XV_ERROR_LEVEL  XV_ERROR_LEVEL_RETURN_ON_ERROR
#endif

#if XV_ERROR_LEVEL == XV_ERROR_LEVEL_TERMINATE_ON_ERROR
#  define XV_CHECK_ERROR(condition, statement, code, wide_description) \
  if (((condition) == 0)) {} else { exit(-1); }
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description) \
  if (((condition) == 0)) {} else { exit(-1); }

#elif XV_ERROR_LEVEL == XV_ERROR_LEVEL_RETURN_ON_ERROR
#  define XV_CHECK_ERROR(condition, statement, code, wide_description) \
  if (((condition) == 0)) {} else { (statement); return (code); }
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description) \
  if (((condition) == 0)) {} else { return (code); }

#elif XV_ERROR_LEVEL == XV_ERROR_LEVEL_CONTINUE_ON_ERROR
#  define XV_CHECK_ERROR(condition, statement, code, wide_description) \
  if (((condition) == 0)) {} else { (statement); }
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description)

#elif XV_ERROR_LEVEL == XV_ERROR_LEVEL_PRINT_ON_ERROR
#  define XV_CHECK_ERROR(condition, statement, code, wide_description)          \
  do { if (condition) { (statement); printf("%s:%d: Error # in function %s: %s\n", \
                                        __FILE__, __LINE__, __func__, (wide_description)); fflush(stdout); return (code); } } while (0)
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description)          \
  do { if (condition) { printf("%s:%d: Error # in function %s: %s\n", \
                                        __FILE__, __LINE__, __func__, (wide_description)); fflush(stdout); return (code); } } while (0)

#elif XV_ERROR_LEVEL == XV_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR
#  define XV_CHECK_ERROR(condition, statement, code, wide_description)  \
  do { if (condition) { printf("%s:%d: Error #  in function %s: %s\n", \
                               __FILE__, __LINE__, __func__, (wide_description)); fflush(stdout); } } while (0)
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description)  \
  do { if (condition) { printf("%s:%d: Error #  in function %s: %s\n", \
                               __FILE__, __LINE__, __func__, (wide_description)); fflush(stdout); } } while (0)

#else
#  define XV_CHECK_ERROR(condition, statment, code, wide_description)
#  define XV_CHECK_ERROR_NULL(condition, code, wide_description)
#endif


#if defined (__cplusplus)
}
/*
 * A tray to define all symbols in the application that should be
 * visible in the library.  A pointer to this tray is passed as an
 * argument to the start function.
 *
 * This is useful to pass pointers to certain standard library functions
 * that otherwise would be unusable from within the library.
 */
typedef struct application_symbol_tray
{
    xvTileManager* pTMObj;

    int32_t (*tray_xvSetupTile)(xvTileManager *pxvTM, xvTile *pTile, int32_t tileBuffSize, int32_t width, uint16_t height, int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame, uint16_t xvTileType, int32_t alignType);
    int32_t (*tray_xvRegisterTile)(xvTileManager *pxvTM, xvTile *pTile, void *pBuff, xvFrame *pFrame, uint32_t DMAInOut);
    void *(*tray_xvAllocateBuffer)(xvTileManager *pxvTM, int32_t buffSize, int32_t buffColor, int32_t buffAlignment);
    int32_t (*tray_xvFreeBuffer)(xvTileManager *pxvTM, void const *pBuff);
    xvTile *(*tray_xvCreateTile)(xvTileManager *pxvTM, int32_t tileBuffSize, int32_t width, uint16_t height, int32_t pitch, uint16_t edgeWidth, uint16_t edgeHeight, int32_t color, xvFrame *pFrame, uint16_t xvTileType, int32_t alignType);
    int32_t (*tray_xvCheckTileReadyMultiChannel3D)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D);
    xvTile3D *(*tray_xvCreateTile3D)(xvTileManager *pxvTM, int32_t tileBuffSize, int32_t width, uint16_t height, uint16_t depth, int32_t pitch, int32_t pitch2D, uint16_t edgeWidth, uint16_t edgeHeight,
                                     uint16_t edgeDepth, int32_t color, xvFrame3D *pFrame3D, uint16_t xvTileType, int32_t alignType);
    int32_t (*tray_xvInitIdmaMultiChannel4CH)(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1, idma_buffer_t *buf2, idma_buffer_t *buf3,
                                              int32_t numDescs, int32_t maxBlock, int32_t maxPifReq, idma_err_callback_fn errCallbackFunc0,
                                              idma_err_callback_fn errCallbackFunc1, idma_err_callback_fn errCallbackFunc2, idma_err_callback_fn errCallbackFunc3,
                                              idma_callback_fn cbFunc0, void *cbData0, idma_callback_fn cbFunc1, void *cbData1,
                                              idma_callback_fn cbFunc2, void *cbData2, idma_callback_fn cbFunc3, void *cbData3);
    int32_t (*tray_xvInitTileManagerMultiChannel4CH)(xvTileManager *pxvTM, idma_buffer_t *buf0, idma_buffer_t *buf1,
                                                     idma_buffer_t *buf2, idma_buffer_t *buf3);
    int32_t (*tray_xvResetTileManager)(xvTileManager *pxvTM);
    int32_t (*tray_xvInitMemAllocator)(xvTileManager *pxvTM, int32_t numMemBanks, void *const *pBankBuffPool, int32_t const *buffPoolSize);
    int32_t (*tray_xvFreeAllBuffers)(xvTileManager *pxvTM);
    xvFrame *(*tray_xvAllocateFrame)(xvTileManager *pxvTM);
    int32_t (*tray_xvFreeFrame)(xvTileManager *pxvTM, xvFrame const *pFrame);
    int32_t (*tray_xvAddIdmaRequestMultiChannel_predicated_wide)(int32_t dmaChannel, xvTileManager *pxvTM, uint64_t pdst64,
                                                                 uint64_t psrc64, size_t rowSize,
                                                                 int32_t numRows, int32_t srcPitch, int32_t dstPitch,
                                                                 int32_t interruptOnCompletion, uint32_t *pred_mask);
    int32_t (*tray_xvAddIdmaRequestMultiChannel_wide)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                      uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                                      int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion);
    int32_t (*tray_xvAddIdmaRequestMultiChannel_wide3D)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                        uint64_t pdst64, uint64_t psrc64, size_t rowSize,
                                                        int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion, int32_t srcTilePitch,
                                                        int32_t dstTilePitch, int32_t numTiles);
    int32_t (*tray_xvReqTileTransferInFastMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                        xvTile *pTile, int32_t interruptOnCompletion);
    int32_t (*tray_xvReqTileTransferOutFastMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion);
    int32_t (*tray_xvCheckForIdmaIndexMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, int32_t index);
    int32_t (*tray_xvSleepForTileMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile);
    int32_t (*tray_xvWaitForiDMAMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex);
    int32_t (*tray_xvSleepForiDMAMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex);
    int32_t (*tray_xvWaitForTileFastMultiChannel)(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile);
    int32_t (*tray_xvSleepForTileFastMultiChannel)(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile);
    int32_t (*tray_xvSleepForTileMultiChannel3D)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D);
    int32_t (*tray_xvReqTileTransferInMultiChannelPredicated)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                              xvTile *pTile, int32_t interruptOnCompletion, uint32_t *pred_mask);
    int32_t (*tray_xvReqTileTransferInMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                    xvTile *pTile, xvTile *pPrevTile, int32_t interruptOnCompletion);
    int32_t (*tray_xvReqTileTransferOutMultiChannelPredicated)(int32_t dmaChannel, xvTileManager *pxvTM,
                                                               xvTile *pTile, int32_t interruptOnCompletion, uint32_t *pred_mask);
    int32_t (*tray_xvReqTileTransferInMultiChannel3D)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion);
    int32_t (*tray_xvReqTileTransferOutMultiChannel3D)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion);
    int32_t (*tray_xvReqTileTransferOutMultiChannel)(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion);

    // 刷cache接口
    void (*tray_xthal_dcache_region_invalidate)(void* addr, uint32_t size);
    void (*tray_xthal_dcache_region_writeback_inv)(void* addr, uint32_t size);
    
    int32_t (*tray_vdsplog)(int level, const char *func, const int line, const char *fmt, ...);
    void (*tray_forceSyncDfxLog)(void);

    // idma接口
    idma_status_t (*tray_idma_init_task)(int32_t ch, idma_buffer_t *taskh, idma_type_t type, int32_t ndescs, idma_callback_fn cb_func, void *cb_data);
    idma_status_t (*tray_idma_add_2d_desc64)(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch);
    idma_status_t (*tray_idma_add_2d_desc64_wide)(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch);
    idma_status_t (*tray_idma_schedule_task)(idma_buffer_t *taskh);
    int32_t (*tray_idma_schedule_desc)(int32_t ch, uint32_t count);
    idma_status_t (*tray_idma_process_tasks)(int32_t ch);
    int32_t (*tray_idma_desc_done)(int32_t ch, int32_t index);

    int32_t (*tray_printf)(const char *, ...);
    int32_t (*tray_vprintf)(const char *format, va_list arg);
    void (*tray_report_error)(const char *);

    int32_t (*tray_strcmp)(const char *, const char *);
    void* (*tray_memcpy)(void *, const void *, size_t);
    void* (*tray_memset)(void *, int, size_t);
    size_t (*tray_strlen)(const char *);
    char* (*tray_strcpy)(char *, const char *);
    void* (*tray_memmove)(void *, const void *, size_t);
    const char *(*tray_strstr)(const char *, const char *);
    float (*tray_modff)(float, float*);
    double (*tray_modf)(double, double*);
    float (*tray_fabsf)(float);
    double (*tray_fabs)(double);
    float (*tray_sqrtf)(float);
    double (*tray_sqrt)(double);
    float (*tray_expf)(float);
    double (*tray_exp)(double);
    float (*tray_exp2f)(float);
    double (*tray_exp2)(double);
    float (*tray_logf)(float);
    double (*tray_log)(double);
    float (*tray_log2f)(float);
    double (*tray_log2)(double);
    float (*tray_log10f)(float);
    double (*tray_log10)(double);
    float (*tray_powf)(float, float);
    double (*tray_pow)(double, double);
    float (*tray_sinf)(float);
    double (*tray_sin)(double);
    float (*tray_cosf)(float);
    double (*tray_cos)(double);
    float (*tray_tanf)(float);
    double (*tray_tan)(double);
    float (*tray_asinf)(float);
    double (*tray_asin)(double);
    float (*tray_acosf)(float);
    double (*tray_acos)(double);
    float (*tray_atanf)(float);
    double (*tray_atan)(double);
    float (*tray_atan2f)(float, float);
    double (*tray_atan2)(double, double);
    float (*tray_floorf)(float);
    double (*tray_floor)(double);
} application_symbol_tray;

int32_t xvfInitTileManager(xvTileManager *pxvTM);
xvError_t xvGetErrorInfoHost(xvTileManager const *pxvTM);

#endif
#endif


