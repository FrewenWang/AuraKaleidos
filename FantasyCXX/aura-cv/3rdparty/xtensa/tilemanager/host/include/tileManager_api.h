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

#ifndef TILE_MANAGER_API_H__
#define TILE_MANAGER_API_H__

#if defined (__cplusplus)
extern "C"
{
#endif

#include <xtensa/config/core-isa.h>

#include "tmUtils.h"

/*****************************************
*   Tile Manager Version
*****************************************/
#define XVTM_VERSION_MAJOR    8
#define XVTM_VERSION_MINOR    1
#define XVTM_VERSION_PATCH    2

#define XVTM_MAKE_LIBRARY_VERSION(major, minor, patch) ((major)*100000 + (minor)*1000 + (patch)*10)
#define XVTM_VERSION     (XVTM_MAKE_LIBRARY_VERSION(XVTM_VERSION_MAJOR, XVTM_VERSION_MINOR, XVTM_VERSION_PATCH))
#define XVTM_VERSION_STR "8.1.2"

#if defined(__XTENSA__)
#if defined(XCHAL_DATARAM1_VADDR)

#if (XCHAL_DATARAM1_VADDR < XCHAL_DATARAM0_VADDR)
#define XVTM_DRAM_LOWER_ADDR   XCHAL_DATARAM1_VADDR
#define XVTM_DRAM_HIGHER_ADDR  (((uint64_t) (XCHAL_DATARAM0_VADDR)) + ((uint64_t) (XCHAL_DATARAM0_SIZE)))
#else // (XCHAL_DATARAM1_VADDR < XCHAL_DATARAM0_VADDR)
#define XVTM_DRAM_LOWER_ADDR   XCHAL_DATARAM0_VADDR
#define XVTM_DRAM_HIGHER_ADDR  (((uint64_t) (XCHAL_DATARAM1_VADDR)) + ((uint64_t) (XCHAL_DATARAM1_SIZE)))
#endif // (XCHAL_DATARAM1_VADDR < XCHAL_DATARAM0_VADDR)

#else // defined(XCHAL_DATARAM1_VADDR)

#define XVTM_DRAM_LOWER_ADDR   XCHAL_DATARAM0_VADDR
#define XVTM_DRAM_HIGHER_ADDR  (((uint64_t) (XCHAL_DATARAM0_VADDR)) + ((uint64_t) (XCHAL_DATARAM0_SIZE)))
#endif // defined(XCHAL_DATARAM1_VADDR)

#else // defined(__XTENSA__)
#define XVTM_DRAM_LOWER_ADDR   ((uint64_t) 0x00001000)
#define XVTM_DRAM_HIGHER_ADDR  ((uint64_t) 0x00001000 + ((uint64_t) (0xFFFFFFFFU - 0x2000)))
#endif // defined(__XTENSA__)


#ifndef FIK_FRAMEWORK
#define FIK_FRAMEWORK                    /*FIK framework */
#endif

#define DEFAULT_IDMA_CHANNEL  IDMA_CHANNEL_0
#define TM_IDMA_CH0           IDMA_CHANNEL_0
#if defined XCHAL_IDMA_NUM_CHANNELS && (XCHAL_IDMA_NUM_CHANNELS >= 2)
#define TM_IDMA_CH1           IDMA_CHANNEL_1
#else
#define TM_IDMA_CH1           IDMA_CHANNEL_0
#endif

#if defined XCHAL_IDMA_NUM_CHANNELS && (XCHAL_IDMA_NUM_CHANNELS == 4)
#define TM_IDMA_CH2  IDMA_CHANNEL_2
#define TM_IDMA_CH3  IDMA_CHANNEL_3
#else
#define TM_IDMA_CH2  TM_IDMA_CH0
#define TM_IDMA_CH3  TM_IDMA_CH1
#endif

// MAX limits for number of tiles, frames memory banks and dma queue length
// Allow user to change the values build time.

#if !defined(MAX_NUM_MEM_BANKS)
#define MAX_NUM_MEM_BANKS         8
#endif
#if !defined(MAX_NUM_TILES)
#define MAX_NUM_TILES             52u
#endif
#if !defined(MAX_NUM_FRAMES)
#define MAX_NUM_FRAMES            16u
#endif
#if !defined(MAX_NUM_DMA_QUEUE_LENGTH)
#define MAX_NUM_DMA_QUEUE_LENGTH  32u // Optimization, multiple of 2
#endif
#if !defined(MAX_NUM_CHANNEL)
#define MAX_NUM_CHANNEL           4u
#endif

#if (XCHAL_IDMA_NUM_CHANNELS > MAX_NUM_CHANNEL)
#error "XCHAL_IDMA_NUM_CHANNE exceeds supported limit"
#endif

//support fro 3D Frame and Tile

#if !defined(MAX_NUM_FRAMES3D)
#define MAX_NUM_FRAMES3D  4u
#endif

#if !defined(MAX_NUM_TILES3D)
#define MAX_NUM_TILES3D  8u
#endif

#if !defined(MAX_TM_CONTEXT_SIZE)
#define MAX_TM_CONTEXT_SIZE           4u
#endif


#ifndef XCHAL_IDMA_HAVE_2DPRED
/* This is defined in RI.9.1 and onwards in core-isa.h. Definition to zero/non-zero depends on build configuration
 * In RI.9 and previous releases, this macro wasn't defined at all in core-isa.h nor in any other header files,
 * and the predicated apis were open available without this macro protection in idma lib
 * So, setting this to 1 below.
*/
#define XVTM_IDMA_HAVE_2DPRED	(1)
#else
#define XVTM_IDMA_HAVE_2DPRED	(XCHAL_IDMA_HAVE_2DPRED)
#endif


// Bank colors. XV_MEM_BANK_COLOR_ANY is an unlikely enum value
#define XV_MEM_BANK_COLOR_0    0x0u
#define XV_MEM_BANK_COLOR_1    0x1u
#define XV_MEM_BANK_COLOR_2    0x2u
#define XV_MEM_BANK_COLOR_3    0x3u
#define XV_MEM_BANK_COLOR_4    0x4u
#define XV_MEM_BANK_COLOR_5    0x5u
#define XV_MEM_BANK_COLOR_6    0x6u
#define XV_MEM_BANK_COLOR_7    0x7u
#define XV_MEM_BANK_COLOR_ANY  0xBEEDDEAFU

// Edge padding format in cadence
// #define FRAME_ZERO_PADDING          0u
// #define FRAME_CONSTANT_PADDING      1u
// #define FRAME_EDGE_PADDING          2u
// #define FRAME_PADDING_REFLECT_101   3u
// #define FRAME_PADDING_REFLECT       4u
// #define FRAME_PADDING_MAX           5u

// Modify the macro of padding to be consistent with aura
#define FRAME_CONSTANT_PADDING      (0u)
#define FRAME_EDGE_PADDING          (1u)
#define FRAME_PADDING_REFLECT_101   (2u)
#define FRAME_PADDING_REFLECT       (3u)
#define FRAME_ZERO_PADDING          (4u)
#define FRAME_PADDING_MAX           (5u)


#define XVTM_DUMMY_DMA_INDEX    -2
#define XVTM_ERROR              -1
#define XVTM_SUCCESS            0

//#define TEST_DTCM23

/*****************************************
*   Individual status flags definitions
*****************************************/

#define XV_TILE_STATUS_DMA_ONGOING                 (0x01u)
#define XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED    (0x02u)
#define XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED   (0x04u)
#define XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED     (0x08u)
#define XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED  (0x10u)
#define XV_TILE_STATUS_FRONT_EDGE_PADDING_NEEDED   (0x20u)
#define XV_TILE_STATUS_BACK_EDGE_PADDING_NEEDED    (0x40u)


#define XV_TILE_STATUS_LEFT_EDGE_PADDING_ONGOING      (0x100u)
#define XV_TILE_STATUS_RIGHT_EDGE_PADDING_ONGOING     (0x200u)
#define XV_TILE_STATUS_TOP_EDGE_PADDING_ONGOING       (0x400u)
#define XV_TILE_STATUS_BOTTOM_EDGE_PADDING_ONGOING    (0x800u)
#define XV_TILE_STATUS_FRONT_EDGE_PADDING_ONGOING     (0x1000u)
#define XV_TILE_STATUS_BACK_EDGE_PADDING_ONGOING      (0x2000u)
#define XV_TILE_STATUS_DUMMY_DMA_ONGOING              (0x4000u)
#define XV_TILE_STATUS_DUMMY_IDMA_INDEX_NEEDED        (0x8000u)
#define XV_TILE_STATUS_INTERRUPT_ON_COMPLETION        (0x10000u)
#define XV_TILE_STATUS_INTERRUPT_ON_COMPLETION_SHIFT  (16u)

#define XV_TILE_STATUS_EDGE_PADDING_NEEDED     \
  (XV_TILE_STATUS_LEFT_EDGE_PADDING_NEEDED |   \
   XV_TILE_STATUS_RIGHT_EDGE_PADDING_NEEDED |  \
   XV_TILE_STATUS_TOP_EDGE_PADDING_NEEDED |    \
   XV_TILE_STATUS_BOTTOM_EDGE_PADDING_NEEDED | \
   XV_TILE_STATUS_FRONT_EDGE_PADDING_NEEDED |  \
   XV_TILE_STATUS_BACK_EDGE_PADDING_NEEDED)



/*****************************************
*   Data type definitions
*****************************************/

#define XV_TYPE_SIGNED_BIT         (0x8000u)
#define XV_TYPE_TILE_BIT           (0x4000u)

#define XV_TYPE_ELEMENT_SIZE_BITS  ((uint16_t) 10)
#define XV_TYPE_ELEMENT_SIZE_MASK  ((((uint16_t) 1) << XV_TYPE_ELEMENT_SIZE_BITS) - (uint16_t) 1u)
#define XV_TYPE_CHANNELS_BITS      ((uint16_t) 2)
#define XV_TYPE_CHANNELS_MASK      ((uint16_t) ((((uint16_t) 1u << XV_TYPE_CHANNELS_BITS) - (uint32_t) 1u) << XV_TYPE_ELEMENT_SIZE_BITS))

// XV_MAKETYPE accepts 3 parameters
// 1: flag: Denotes whether the entity is a tile (XV_TYPE_TILE_BIT is set) or an array (XV_TYPE_TILE_BIT is not set),
//    and also if the data is a signed(XV_TYPE_SIGNED_BIT is set) or unsigned(XV_TYPE_SIGNED_BIT is not set).
// 2: depth: Denotes number of bytes per pel.
//    1 implies the data is 8bit, 2 implies the data is 16bit and 4 implies the data is 32bit.
// 3: Denotes number of channels.
//    1 implies gray scale, 3 implies RGB

#define XV_MAKETYPE(flags, depth, channels)  (((depth) * (channels)) | (((channels) - 1) << XV_TYPE_ELEMENT_SIZE_BITS) | (flags))
#define XV_CUSTOMTYPE(type)                  XV_MAKETYPE(0, sizeof(type), 1)

#define XV_TYPE_ELEMENT_SIZE(type)           ((type) & (XV_TYPE_ELEMENT_SIZE_MASK))
#define XV_TYPE_ELEMENT_TYPE(type)           ((type) & (XV_TYPE_SIGNED_BIT | XV_TYPE_CHANNELS_MASK | XV_TYPE_ELEMENT_SIZE_MASK))
#define XV_TYPE_IS_TILE(type)                ((type) & (XV_TYPE_TILE_BIT))
#define XV_TYPE_IS_SIGNED(type)              ((type) & (XV_TYPE_SIGNED_BIT))
#define XV_TYPE_CHANNELS(type)               ((((type) & (XV_TYPE_CHANNELS_MASK)) >> (XV_TYPE_ELEMENT_SIZE_BITS)) + ((uint16_t) 1))

// Common XV_MAKETYPEs
#define XV_U8            XV_MAKETYPE(0, 1, 1)
#define XV_U16           XV_MAKETYPE(0, 2, 1)
#define XV_U32           XV_MAKETYPE(0, 4, 1)

#define XV_S8            XV_MAKETYPE(XV_TYPE_SIGNED_BIT, 1, 1)
#define XV_S16           XV_MAKETYPE(XV_TYPE_SIGNED_BIT, 2, 1)
#define XV_S32           XV_MAKETYPE(XV_TYPE_SIGNED_BIT, 4, 1)
#define XV_F32           XV_MAKETYPE(XV_TYPE_SIGNED_BIT, 4, 1)

#define XV_ARRAY_U8      XV_U8
#define XV_ARRAY_S8      XV_S8
#define XV_ARRAY_U16     XV_U16
#define XV_ARRAY_S16     XV_S16
#define XV_ARRAY_U32     XV_U32
#define XV_ARRAY_S32     XV_S32

#define XV_TILE_U8       (XV_U8 | XV_TYPE_TILE_BIT)
#define XV_TILE_S8       (XV_S8 | XV_TYPE_TILE_BIT)
#define XV_TILE_U16      (XV_U16 | XV_TYPE_TILE_BIT)
#define XV_TILE_S16      (XV_S16 | XV_TYPE_TILE_BIT)
#define XV_TILE_U32      (XV_U32 | XV_TYPE_TILE_BIT)
#define XV_TILE_S32      (XV_S32 | XV_TYPE_TILE_BIT)
#define XV_TILE_F32      (XV_F32 | XV_TYPE_TILE_BIT)

#define XV_TILE_RGB_U8   XV_MAKETYPE(XV_TYPE_TILE_BIT, 1, 3)
#define XV_TILE_RGB_U16  XV_MAKETYPE(XV_TYPE_TILE_BIT, 2, 3)
#define XV_TILE_RGB_U32  XV_MAKETYPE(XV_TYPE_TILE_BIT, 4, 3)
  
#define XV_FLOAT         XV_MAKETYPE(0, 4, 1)
#define XV_TILE_FLOAT 	 (XV_FLOAT | XV_TYPE_TILE_BIT)

/*****************************************
*    Frame Access Macros
*****************************************/

#define XV_FRAME_GET_BUFF_PTR(pFrame)                   ((pFrame)->pFrameBuff)
#define XV_FRAME_SET_BUFF_PTR(pFrame, pBuff)            (pFrame)->pFrameBuff = ((uint64_t) (pBuff))

#define XV_FRAME_GET_BUFF_SIZE(pFrame)                  ((pFrame)->frameBuffSize)
#define XV_FRAME_SET_BUFF_SIZE(pFrame, buffSize)        (pFrame)->frameBuffSize = ((uint32_t) (buffSize))

#define XV_FRAME_GET_DATA_PTR(pFrame)                   ((pFrame)->pFrameData)
#define XV_FRAME_SET_DATA_PTR(pFrame, pData)            (pFrame)->pFrameData = ((uint64_t) (pData))

#define XV_FRAME_GET_WIDTH(pFrame)                      ((pFrame)->frameWidth)
#define XV_FRAME_SET_WIDTH(pFrame, width)               (pFrame)->frameWidth = ((int32_t) (width))

#define XV_FRAME_GET_HEIGHT(pFrame)                     ((pFrame)->frameHeight)
#define XV_FRAME_SET_HEIGHT(pFrame, height)             (pFrame)->frameHeight = ((int32_t) (height))

#define XV_FRAME_GET_PITCH(pFrame)                      ((pFrame)->framePitch)
#define XV_FRAME_SET_PITCH(pFrame, pitch)               (pFrame)->framePitch = ((int32_t) (pitch))
#define XV_FRAME_GET_PITCH_IN_BYTES(pFrame)             ((pFrame)->framePitch * (pFrame)->pixelRes)

#define XV_FRAME_GET_PIXEL_RES(pFrame)                  ((pFrame)->pixelRes)
#define XV_FRAME_SET_PIXEL_RES(pFrame, pixRes)          (pFrame)->pixelRes = ((uint8_t) (pixRes))

#define XV_FRAME_GET_NUM_CHANNELS(pFrame)               ((pFrame)->numChannels)
#define XV_FRAME_SET_NUM_CHANNELS(pFrame, pixelFormat)  (pFrame)->numChannels = ((uint8_t) (pixelFormat))

#define XV_FRAME_GET_EDGE_WIDTH(pFrame)                 ((pFrame)->leftEdgePadWidth < (pFrame)->rightEdgePadWidth ? (pFrame)->leftEdgePadWidth : (pFrame)->rightEdgePadWidth)
#define XV_FRAME_SET_EDGE_WIDTH(pFrame, padWidth)         \
  {                                                       \
    (pFrame)->leftEdgePadWidth  = ((uint8_t) (padWidth)); \
    (pFrame)->rightEdgePadWidth = ((uint8_t) (padWidth)); \
  }

#define XV_FRAME_GET_EDGE_HEIGHT(pFrame)  ((pFrame)->topEdgePadHeight < (pFrame)->bottomEdgePadHeight ? (pFrame)->topEdgePadHeight : (pFrame)->bottomEdgePadHeight)
#define XV_FRAME_SET_EDGE_HEIGHT(pFrame, padHeight)          \
  {                                                          \
    (pFrame)->topEdgePadHeight    = ((uint8_t) (padHeight)); \
    (pFrame)->bottomEdgePadHeight = ((uint8_t) (padHeight)); \
  }

#define XV_FRAME_GET_EDGE_LEFT(pFrame)               ((pFrame)->leftEdgePadWidth)
#define XV_FRAME_SET_EDGE_LEFT(pFrame, padWidth)     (pFrame)->leftEdgePadWidth = ((uint8_t) (padWidth))

#define XV_FRAME_GET_EDGE_RIGHT(pFrame)              ((pFrame)->rightEdgePadWidth)
#define XV_FRAME_SET_EDGE_RIGHT(pFrame, padWidth)    (pFrame)->rightEdgePadWidth = ((uint8_t) (padWidth))

#define XV_FRAME_GET_EDGE_TOP(pFrame)                ((pFrame)->topEdgePadHeight)
#define XV_FRAME_SET_EDGE_TOP(pFrame, padHeight)     (pFrame)->topEdgePadHeight = ((uint8_t) (padHeight))

#define XV_FRAME_GET_EDGE_BOTTOM(pFrame)             ((pFrame)->bottomEdgePadHeight)
#define XV_FRAME_SET_EDGE_BOTTOM(pFrame, padHeight)  (pFrame)->bottomEdgePadHeight = ((uint8_t) (padHeight))

#define XV_FRAME_GET_PADDING_TYPE(pFrame)            ((pFrame)->paddingType)
#define XV_FRAME_SET_PADDING_TYPE(pFrame, padType)   (pFrame)->paddingType = (padType)

#define XV_FRAME_GET_PADDING_VALUE(pFrame)           ((pFrame)->paddingVal)
#define XV_FRAME_SET_PADDING_VALUE(pFrame, padVal)   (pFrame)->paddingVal = (padVal)

/*****************************************
*    Array Access Macros
*****************************************/

#define XV_ARRAY_GET_BUFF_PTR(pArray)              ((pArray)->pBuffer)
#define XV_ARRAY_SET_BUFF_PTR(pArray, pBuff)       (pArray)->pBuffer = ((void *) (pBuff))

#define XV_ARRAY_GET_BUFF_SIZE(pArray)             ((pArray)->bufferSize)
#define XV_ARRAY_SET_BUFF_SIZE(pArray, buffSize)   (pArray)->bufferSize = ((uint32_t) (buffSize))

#define XV_ARRAY_GET_DATA_PTR(pArray)              ((pArray)->pData)
#define XV_ARRAY_SET_DATA_PTR(pArray, pArrayData)  (pArray)->pData = ((void *) (pArrayData))

#define XV_ARRAY_GET_WIDTH(pArray)                 ((pArray)->width)
#define XV_ARRAY_SET_WIDTH(pArray, value)          (pArray)->width = ((int32_t) (value))

#define XV_ARRAY_GET_PITCH(pArray)                 ((pArray)->pitch)
#define XV_ARRAY_SET_PITCH(pArray, value)          (pArray)->pitch = ((int32_t) (value))

#define XV_ARRAY_GET_HEIGHT(pArray)                ((pArray)->height)
#define XV_ARRAY_SET_HEIGHT(pArray, value)         (pArray)->height = ((uint16_t) (value))

#define XV_ARRAY_GET_STATUS_FLAGS(pArray)          ((pArray)->status)
#define XV_ARRAY_SET_STATUS_FLAGS(pArray, value)   (pArray)->status = ((uint8_t) (value))

#define XV_ARRAY_GET_TYPE(pArray)                  ((pArray)->type)
#define XV_ARRAY_SET_TYPE(pArray, value)           (pArray)->type = ((uint16_t) (value))

#define XV_ARRAY_GET_CAPACITY(pArray)              XV_ARRAY_GET_PITCH(pArray)
#define XV_ARRAY_SET_CAPACITY(pArray, value)       XV_ARRAY_SET_PITCH((pArray), (value))

#define XV_ARRAY_GET_ELEMENT_TYPE(pArray)          XV_TYPE_ELEMENT_TYPE(XV_ARRAY_GET_TYPE(pArray))
#define XV_ARRAY_GET_ELEMENT_SIZE(pArray)          XV_TYPE_ELEMENT_SIZE(XV_ARRAY_GET_TYPE(pArray))
#define XV_ARRAY_IS_TILE(pArray)                   XV_TYPE_IS_TILE(XV_ARRAY_GET_TYPE(pArray) & (XV_TYPE_TILE_BIT))

#define XV_ARRAY_GET_AREA(pArray)                  (((pArray)->width) * ((int32_t) (pArray)->height))

/*****************************************
*    Tile Access Macros
*****************************************/

#define XV_TILE_GET_BUFF_PTR   XV_ARRAY_GET_BUFF_PTR
#define XV_TILE_SET_BUFF_PTR   XV_ARRAY_SET_BUFF_PTR

#define XV_TILE_GET_BUFF_SIZE  XV_ARRAY_GET_BUFF_SIZE
#define XV_TILE_SET_BUFF_SIZE  XV_ARRAY_SET_BUFF_SIZE

#define XV_TILE_GET_DATA_PTR   XV_ARRAY_GET_DATA_PTR
#define XV_TILE_SET_DATA_PTR   XV_ARRAY_SET_DATA_PTR

#define XV_TILE_GET_WIDTH      XV_ARRAY_GET_WIDTH
#define XV_TILE_SET_WIDTH      XV_ARRAY_SET_WIDTH

#define XV_TILE_GET_PITCH      XV_ARRAY_GET_PITCH
#define XV_TILE_SET_PITCH      XV_ARRAY_SET_PITCH
#define XV_TILE_GET_PITCH_IN_BYTES(pTile)  ((pTile)->pitch * (int32_t) ((pTile)->pFrame->pixelRes))

#define XV_TILE_GET_HEIGHT        XV_ARRAY_GET_HEIGHT
#define XV_TILE_SET_HEIGHT        XV_ARRAY_SET_HEIGHT

#define XV_TILE_GET_STATUS_FLAGS  XV_ARRAY_GET_STATUS_FLAGS
#define XV_TILE_SET_STATUS_FLAGS  XV_ARRAY_SET_STATUS_FLAGS

#define XV_TILE_GET_TYPE          XV_ARRAY_GET_TYPE
#define XV_TILE_SET_TYPE          XV_ARRAY_SET_TYPE

#define XV_TILE_GET_ELEMENT_TYPE  XV_ARRAY_GET_ELEMENT_TYPE
#define XV_TILE_GET_ELEMENT_SIZE  XV_ARRAY_GET_ELEMENT_SIZE
#define XV_TILE_IS_TILE           XV_ARRAY_IS_TILE

#define XV_TILE_RESET_DMA_INDEX(pTile)              ((pTile)->dmaIndex = 0)
#define XV_TILE_RESET_PREVIOUS_TILE(pTile)          (pTile)->pPrevTile = ((xvTile *) (NULL))
#define XV_TILE_RESET_REUSE_COUNT(pTile)            ((pTile)->reuseCount = 0)

#define XV_TILE_GET_FRAME_PTR(pTile)                ((pTile)->pFrame)
#define XV_TILE_SET_FRAME_PTR(pTile, ptrFrame)      (pTile)->pFrame = ((xvFrame *) (ptrFrame))

#define XV_TILE_GET_X_COORD(pTile)                  ((pTile)->x)
#define XV_TILE_SET_X_COORD(pTile, xcoord)          (pTile)->x = ((int32_t) (xcoord))

#define XV_TILE_GET_Y_COORD(pTile)                  ((pTile)->y)
#define XV_TILE_SET_Y_COORD(pTile, ycoord)          (pTile)->y = ((int32_t) (ycoord))

#define XV_TILE_GET_EDGE_LEFT(pTile)                ((pTile)->tileEdgeLeft)
#define XV_TILE_SET_EDGE_LEFT(pTile, edgeWidth)     (pTile)->tileEdgeLeft = ((uint16_t) (edgeWidth))

#define XV_TILE_GET_EDGE_RIGHT(pTile)               ((pTile)->tileEdgeRight)
#define XV_TILE_SET_EDGE_RIGHT(pTile, edgeWidth)    (pTile)->tileEdgeRight = ((uint16_t) (edgeWidth))

#define XV_TILE_GET_EDGE_TOP(pTile)                 ((pTile)->tileEdgeTop)
#define XV_TILE_SET_EDGE_TOP(pTile, edgeHeight)     (pTile)->tileEdgeTop = ((uint16_t) (edgeHeight))

#define XV_TILE_GET_EDGE_BOTTOM(pTile)              ((pTile)->tileEdgeBottom)
#define XV_TILE_SET_EDGE_BOTTOM(pTile, edgeHeight)  (pTile)->tileEdgeBottom = ((uint16_t) (edgeHeight))

#define XV_TILE_GET_EDGE_WIDTH(pTile)               (((pTile)->tileEdgeLeft < (pTile)->tileEdgeRight) ? (pTile)->tileEdgeLeft : (pTile)->tileEdgeRight)
#define XV_TILE_SET_EDGE_WIDTH(pTile, edgeWidth)       \
  {                                                    \
    (pTile)->tileEdgeLeft  = ((uint16_t) (edgeWidth)); \
    (pTile)->tileEdgeRight = ((uint16_t) (edgeWidth)); \
  }

#define XV_TILE_GET_EDGE_HEIGHT(pTile)  (((pTile)->tileEdgeTop < (pTile)->tileEdgeBottom) ? (pTile)->tileEdgeTop : (pTile)->tileEdgeBottom)
#define XV_TILE_SET_EDGE_HEIGHT(pTile, edgeHeight)       \
  {                                                      \
    (pTile)->tileEdgeTop    = ((uint16_t) (edgeHeight)); \
    (pTile)->tileEdgeBottom = ((uint16_t) (edgeHeight)); \
  }

#define XV_TILE_CHECK_STATUS_FLAGS_DMA_ONGOING(pTile)          (((pTile)->status & XV_TILE_STATUS_DMA_ONGOING) > 0)
#define XV_TILE_CHECK_STATUS_FLAGS_EDGE_PADDING_NEEDED(pTile)  (((pTile)->status & XV_TILE_STATUS_EDGE_PADDING_NEEDED) > 0)




#define XV_TILE_3D_GET_FRAME_3D_PTR(pTile3D)              ((pTile3D)->pFrame)
#define XV_TILE_3D_SET_FRAME_3D_PTR(pTile3D, ptrFrame3D)  (pTile3D)->pFrame = ((ptrFrame3D))

#define XV_FRAME_GET_DEPTH(pFrame3D)                      ((pFrame3D)->frameDepth)
#define XV_FRAME_GET_FRAME_PITCH(pFrame3D)                ((pFrame3D)->frame2DFramePitch)


#define XV_FRAME_SET_DEPTH(pFrame3D, numTiles)            ((pFrame3D)->frameDepth = (numTiles))
#define XV_FRAME_SET_FRAME_PITCH(pFrame3D, Frame2DPitch)  ((pFrame3D)->frame2DFramePitch = (Frame2DPitch))

#define XV_FRAME_SET_EDGE_DEPTH(pFrame3D, edgeDepth) \
  {                                                  \
    ((pFrame3D)->frontEdgePadDepth = (edgeDepth));     \
    ((pFrame3D)->backEdgePadDepth = (edgeDepth));      \
  }

#define XV_FRAME_SET_EDGE_FRONT(pFrame3D, edgeDepth)  ((pFrame3D)->frontEdgePadDepth = (uint16_t) (edgeDepth))
#define XV_FRAME_SET_EDGE_BACK(pFrame3D, edgeDepth)   ((pFrame3D)->backEdgePadDepth = (uint16_t) (edgeDepth))

#define XV_FRAME_GET_EDGE_FRONT(pFrame3D, edgeDepth)  ((pFrame3D)->frontEdgePadDepth)
#define XV_FRAME_GET_EDGE_BACK(pFrame3D, edgeDepth)   ((pFrame3D)->backEdgePadDepth)

//MACROs for 16bit padding values.
#define XV_FRAME3D_SET_EDGE_LEFT(pFrame3D, padWidth)     (pFrame3D)->leftEdgePadWidth    = ((uint16_t) (padWidth))
#define XV_FRAME3D_SET_EDGE_RIGHT(pFrame3D, padWidth)    (pFrame3D)->rightEdgePadWidth   = ((uint16_t) (padWidth))
#define XV_FRAME3D_SET_EDGE_TOP(pFrame3D, padHeight)     (pFrame3D)->topEdgePadHeight    = ((uint16_t) (padHeight))
#define XV_FRAME3D_SET_EDGE_BOTTOM(pFrame3D, padHeight)  (pFrame3D)->bottomEdgePadHeight = ((uint16_t) (padHeight))



#define XV_TILE_GET_DEPTH(pTile3D)            ((pTile3D)->depth)
#define XV_TILE_SET_DEPTH(pTile3D, Tdepth)    ((pTile3D)->depth = (Tdepth))
#define XV_TILE_GET_Z_COORD(pTile3D)          ((pTile3D)->z)
#define XV_TILE_SET_Z_COORD(pTile3D, zcoord)  (pTile3D)->z = ((int32_t) (zcoord))

#define XV_TILE_GET_EDGE_FRONT(pTile3D)       ((pTile3D)->tileEdgeFront)
#define XV_TILE_GET_EDGE_BACK(pTile3D)        ((pTile3D)->tileEdgeBack)
#define XV_TILE_GET_EDGE_DEPTH(pTile3D)       ((pTile3D)->tileEdgeBack < (pTile3D)->tileEdgeFront ? (pTile3D)->tileEdgeFront : (pTile3D)->tileEdgeBack)


#define XV_TILE_SET_EDGE_FRONT(pTile3D, frontEdgePad)  (pTile3D)->tileEdgeFront = ((uint16_t) (frontEdgePad))
#define XV_TILE_SET_EDGE_BACK(pTile3D, backEdgePad)    (pTile3D)->tileEdgeBack  = ((uint16_t) (backEdgePad))
#define XV_TILE_SET_EDGE_DEPTH(pTile3D, DepthPad)       \
  {                                                     \
    (pTile3D)->tileEdgeFront = ((uint16_t) (DepthPad)); \
    (pTile3D)->tileEdgeBack  = ((uint16_t) (DepthPad)); \
  }

#define XV_TILE_SET_TILE_PITCH(pTile3D, Tile2DPitch)  ((pTile3D)->Tile2Dpitch = (Tile2DPitch))
#define XV_TILE_GET_TILE_PITCH(pTile3D)               ((pTile3D)->Tile2Dpitch)



/*Structures and enums*/

typedef enum
{
  /* CRITICAL ORDER, DO NOT CHANGE. */
  XVTM_TILE_UNALIGNED = 0,
  XVTM_EDGE_ALIGNED_N,
  XVTM_DATA_ALIGNED_N,
  XVTM_EDGE_ALIGNED_2N,
  XVTM_DATA_ALIGNED_2N,
} buffer_align_type_t;

#define TILE_UNALIGNED   XVTM_TILE_UNALIGNED  
#define EDGE_ALIGNED_64  XVTM_EDGE_ALIGNED_N  
#define DATA_ALIGNED_64  XVTM_DATA_ALIGNED_N  
#define EDGE_ALIGNED_128 XVTM_EDGE_ALIGNED_2N 
#define DATA_ALIGNED_128 XVTM_DATA_ALIGNED_2N 

typedef enum
{
  XV_ERROR_SUCCESS            = 0,
  XV_ERROR_TILE_MANAGER_NULL  = 1,
  XV_ERROR_POINTER_NULL       = 2,
  XV_ERROR_FRAME_NULL         = 3,
  XV_ERROR_TILE_NULL          = 4,
  XV_ERROR_BUFFER_NULL        = 5,
  XV_ERROR_ALLOC_FAILED       = 6,
  XV_ERROR_FRAME_BUFFER_FULL  = 7,
  XV_ERROR_TILE_BUFFER_FULL   = 8,
  XV_ERROR_DIMENSION_MISMATCH = 9,
  XV_ERROR_BUFFER_OVERFLOW    = 10,
  XV_ERROR_BAD_ARG            = 11,
  XV_ERROR_FILE_OPEN          = 12,
  XV_ERROR_DMA_INIT           = 13,
  XV_ERROR_XVMEM_INIT         = 14,
  XV_ERROR_IDMA               = 15
}xvError_t;


typedef struct  xvFrameStruct
{
  uint64_t pFrameBuff;
  uint32_t frameBuffSize;
  uint64_t pFrameData;
  int32_t  frameWidth;
  int32_t  frameHeight;
  int32_t  framePitch;
  uint8_t  pixelRes;
  uint8_t  numChannels;
  uint8_t  leftEdgePadWidth;
  uint8_t  topEdgePadHeight;
  uint8_t  rightEdgePadWidth;
  uint8_t  bottomEdgePadHeight;
  uint8_t  paddingType;
  uint32_t paddingVal;
#if defined(XI_XV_TILE_COMPATIBILITY_TEST)
} __attribute__((packed))  xvFrame, *xvpFrame;
#else
} xvFrame, *xvpFrame;
#endif

#define XV_ARRAY_FIELDS \
  void     *pBuffer;    \
  uint32_t bufferSize;  \
  void *pData;          \
  int32_t width;        \
  int32_t pitch;        \
  uint32_t status;      \
  uint16_t type;        \
  uint16_t height;






typedef struct xvArrayStruct
{
  XV_ARRAY_FIELDS
} xvArray, *xvpArray;


typedef struct  xvTileStruct
{
  XV_ARRAY_FIELDS
  xvFrame             *pFrame;
  int32_t             x;
  int32_t             y;
  uint16_t            tileEdgeLeft;
  uint16_t            tileEdgeTop;
  uint16_t            tileEdgeRight;
  uint16_t            tileEdgeBottom;
  int32_t             dmaIndex;
  int32_t             reuseCount;
  struct xvTileStruct *pPrevTile;
#if defined(XI_XV_TILE_COMPATIBILITY_TEST)
} __attribute__((packed)) xvTile, *xvpTile;
#else
} xvTile, *xvpTile;
#endif


typedef struct xvFrame3DStruct
{
  uint64_t pFrameBuff;
  uint32_t frameBuffSize;
  uint64_t pFrameData;
  int32_t frameWidth;
  int32_t frameHeight;
  int32_t framePitch;
  uint8_t pixelRes;
  uint8_t numChannels;
  uint16_t leftEdgePadWidth;
  uint16_t rightEdgePadWidth;
  uint16_t topEdgePadHeight;
  uint16_t bottomEdgePadHeight;
  uint16_t frontEdgePadDepth;
  uint16_t backEdgePadDepth;
  uint8_t paddingType;

  int32_t frame2DFramePitch;
  int32_t frameDepth;
  int32_t dataOrder;
  uint32_t paddingVal;
}  xvFrame3D, *xvpFrame3D;

typedef struct xvTileStruct3D
{
  void       *pBuffer;
  uint32_t bufferSize;
  void       *pData;
  int32_t width;
  int32_t pitch;
  uint32_t status;
  uint16_t type;
  uint16_t height;
  xvpFrame3D pFrame;
  int32_t x;
  int32_t y;
  uint16_t tileEdgeLeft;
  uint16_t tileEdgeTop;
  uint16_t tileEdgeRight;
  uint16_t tileEdgeBottom;

  int32_t Tile2Dpitch;
  uint16_t depth;
  uint32_t dataOrder;
  int32_t z;

  uint16_t tileEdgeFront;
  uint16_t tileEdgeBack;
  int32_t dmaIndex;
  void      *pTemp;
} xvTile3D, *xvpTile3D;


#ifdef FIK_FRAMEWORK
#define XV_INTERIM_TILE 0
#define XV_INPUT_TILE 1
#define XV_OUTPUT_TILE 2

#define XV_AUTOSIZE_FLAG (1)
#define XV_SEQ_FLAG (1<<10)
#define XV_ALLOCATE_SINGLE_BANK (1<<11)
#define XV_DMA_OVERLAP (1<<12)

#define XV_MAX_ALLOCATIONS 64

#define MAX_TILES 20
#define INTERIM_MAX_TILES 16

#define PING 0
#define PONG 1
#ifdef __XTENSA__
#define DBG_PROFILE 0
#else
#define DBG_PROFILE 0
#endif
typedef struct
{
  int32_t  frameWidth;
  int32_t  frameHeight;
  int32_t  tileWidth;
  uint16_t tileHeight;
  int32_t  x;
  int32_t  y;

} RefTile;
/*Structure holds overall tile based processing requirements*/
typedef struct
{
	int32_t numInTiles[2];					/*Number of input tiles for DMA in configuration*/
	xvTile* InTiles[2][MAX_TILES]; 	/* Ping/Pong input tile pointers for DMA in configuration*/
	int32_t numOutTiles[2];				/*Number of output tiles for DMA in configuration*/
	xvTile* OutTiles[2][MAX_TILES];	/* Ping/Pong output tile pointers for DMA in configuration*/
	void *KernelArgs[2];		/* Pointer to kernel arguments struct for each kernel ping/pong */
	void *CommonArgs;
	int32_t (*ProcessKernel)(void* CommonArgs_, void* TileArgs);		/* pointer to kernel fucntion provided by used inside which user can call kernel[s] for every tile */
	void (*SetupUpdatesTiles)(void* pxvTM, RefTile *pRefTile, void* _CommonArgs, void* TileArgs, int32_t updateOnlyFlag);/*Coordinate map function, Need to pass a function pointer incase of constom input to output tile mapping e.g. Transpose*/
	int32_t CoreID;					/* ID of current core should start from 0 to numCores -1 */
	int32_t numCores;				/* Total number of cores */
	uint32_t numTiles;
	uint32_t allocationStartIdx;
	RefTile refTile;
	uint32_t flags;
#if DBG_PROFILE
	uint32_t Cycles[5];
	/*uint32_t TotalCycles;
	uint32_t KernelCycles;
	uint32_t DMAWaitCycles;
	uint32_t MapFunCycles;
	uint32_t PadEdgesCycles;*/
#endif
}FIK_Context_t;

typedef struct xvTileManagerContextStruct
{
	uint32_t  tileAllocFlags[(MAX_NUM_TILES + 31) / 32];      // Each bit of tileAllocFlags and frameAllocFlags
	uint32_t  frameAllocFlags[(MAX_NUM_FRAMES + 31) / 32];    // indicates if a particular tile/frame is allocated
	int32_t   tileCount;
	int32_t   frameCount;
	uint32_t  allocationsIdx;
}xvTileManagerContext;
#endif // FIK_FRAMEWORK

typedef struct xvTileManagerStruct
{
  // iDMA related
  void    *pdmaObj0;
  void    *pdmaObj1;
  void    *pdmaObj2;
  void    *pdmaObj3;

  int32_t tileDMApendingCount[MAX_NUM_CHANNEL];       // Incremented when new request is added. Decremented when request is completed.
  int32_t tileDMAstartIndex[MAX_NUM_CHANNEL];         // Incremented when request is completed
  xvTile  *tileProcQueue[MAX_NUM_CHANNEL][MAX_NUM_DMA_QUEUE_LENGTH];
#ifndef XVTM_USE_XMEM
  // Mem Banks
  int32_t numMemBanks;                           // Number of memory banks/pools
  xvmem_mgr_t memBankMgr[MAX_NUM_MEM_BANKS];     // xvmem memory manager, one for each bank
  void        *pMemBankStart[MAX_NUM_MEM_BANKS]; // Start address of bank
  int32_t memBankSize[MAX_NUM_MEM_BANKS];        // size of each bank
#endif
#ifdef TM_LOG
  FILE *tm_log_fp;
#endif
  // Tiles and frame allocation
  xvTile tileArray[MAX_NUM_TILES];
  xvFrame frameArray[MAX_NUM_FRAMES];
  uint32_t tileAllocFlags[(MAX_NUM_TILES + 31u) / 32u];        // Each bit of tileAllocFlags and frameAllocFlags
  uint32_t frameAllocFlags[(MAX_NUM_FRAMES + 31u) / 32u];      // indicates if a particular tile/frame is allocated

  int32_t tileCount;
  int32_t frameCount;


  xvError_t errFlag;
  xvError_t idmaErrorFlag[MAX_NUM_CHANNEL];               //Allocate for MAX iDMA channels.


  xvTile3D tile3DArray[MAX_NUM_TILES3D];
  xvFrame3D frame3DArray[MAX_NUM_FRAMES3D];
  uint32_t tile3DAllocFlags[(MAX_NUM_TILES3D + 31) / 32];        // Each bit of tile3DAllocFlags and frame3DAllocFlags
  uint32_t frame3DAllocFlags[(MAX_NUM_FRAMES3D + 31) / 32];      // indicates if a particular tile3D/frame3D is allocated

  int32_t tile3DDMApendingCount[MAX_NUM_CHANNEL];       // Incremented when new request is added. Decremented when request is completed.
  int32_t tile3DDMAstartIndex[MAX_NUM_CHANNEL];         // Incremented when request is completed
  xvTile3D  *tile3DProcQueue[MAX_NUM_CHANNEL][MAX_NUM_DMA_QUEUE_LENGTH];

  int32_t tile3DCount;
  int32_t frame3DCount;
#ifdef FIK_FRAMEWORK
  xvTileManagerContext tmContext[MAX_TM_CONTEXT_SIZE];
  uint32_t             tmContextSize;
  void                 * allocatedList[XV_MAX_ALLOCATIONS];
  uint32_t             allocationsIdx;
  int32_t              AllocateErrorState;
  int32_t              AllocateColor;
  uint32_t             PingPongState;
  void                 * InterimTileList[INTERIM_MAX_TILES];
  uint32_t             InterimTileCounter;
  FIK_Context_t        * pWorkPacket;
#endif
} xvTileManager;


/***********************************
*    Function  Prototypes
***********************************/
/**********************************************************************************
 * FUNCTION: xvCreateTileManagerMultiChannel4CHHost()
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
 *     idma_callback_fn     intrCallbackFunc0   Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     intrCallbackFunc1   Callback for dma transfer completion for channel 1
 *     void                 *cbData1            Data needed for completion callback function for channel 1
 *     idma_callback_fn     intrCallbackFunc2   Callback for dma transfer completion for channel 2
 *     void                 *cbData2            Data needed for completion callback function for channel 2
 *     idma_callback_fn     intrCallbackFunc3   Callback for dma transfer completion for channel 3
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
                                               int32_t descCount, int32_t maxBlock, int32_t numOutReq);

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
 *     idma_callback_fn     intrCallbackFunc0   Callback for dma transfer completion for channel 0
 *     void                 *cbData0            Data needed for completion callback function for channel 0
 *     idma_callback_fn     intrCallbackFunc1   Callback for dma transfer completion for channel 1
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
                                            int32_t descCount, int32_t maxBlock, int32_t numOutReq);

#else
int32_t xvCreateTileManagerMultiChannelHost(xvTileManager *pxvTM, void *buf0, void *buf1, int32_t numMemBanks,
                                            void * const* pBankBuffPool, int32_t const* buffPoolSize,
                                            idma_err_callback_fn errCallbackFunc0, idma_err_callback_fn errCallbackFunc1,
                                            idma_callback_fn intrCallbackFunc0, void *cbData0, idma_callback_fn intrCallbackFunc1, void *cbData1,
                                            int32_t descCount, int32_t maxBlock, int32_t numOutReq);
#endif
/**********************************************************************************
 * FUNCTION: xvCreateTileManagerHost()
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
 *     idma_callback_fn     intrCallbackFunc0   Callback for dma transfer completion for channel 0
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
int32_t xvCreateTileManagerHost(xvTileManager *pxvTM, void *buf0, int32_t numMemBanks,
                                void * const* pBankBuffPool, int32_t const* buffPoolSize,
                                idma_err_callback_fn errCallbackFunc0,
                                idma_callback_fn intrCallbackFunc0, void *cbData0,
                                int32_t descCount, int32_t maxBlock, int32_t numOutReq);


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
                                  idma_callback_fn cbFunc2, void * cbData2, idma_callback_fn cbFunc3, void * cbData3);

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
                                         idma_buffer_t *buf2, idma_buffer_t *buf3);

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
int32_t xvResetTileManagerHost(xvTileManager *pxvTM);

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
int32_t xvInitMemAllocatorHost( xvTileManager *pxvTM, int32_t numMemBanks, void* const* pBankBuffPool, int32_t const* buffPoolSize);


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
void *xvAllocateBufferHost(xvTileManager *pxvTM, int32_t buffSize, int32_t buffColor, int32_t buffAlignment);


/**********************************************************************************
 * FUNCTION: xvFreeBufferHost()
 *
 * DESCRIPTION:
 *     Releases the given buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     void const    *pBuff      Buffer that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvFreeBufferHost(xvTileManager *pxvTM, void const *pBuff);


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
int32_t xvFreeAllBuffersHost(xvTileManager *pxvTM);


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
xvFrame *xvAllocateFrameHost(xvTileManager *pxvTM);


/**********************************************************************************
 * FUNCTION: xvFreeFrameHost()
 *
 * DESCRIPTION:
 *     Releases the given frame. Does not release associated frame data buffer.
 *
 * INPUTS:
 *     xvTileManager *pxvTM      Tile Manager object
 *     xvFrame const *pFrame     Frame that needs to be released
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvFreeFrameHost(xvTileManager *pxvTM, xvFrame const *pFrame);

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
                     uint16_t xvTileType, int32_t alignType);

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_predicated_wideHost()
 *
 * DESCRIPTION:
 *     Add iDMA predicated transfer request using wide addresses
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager  *pxvTM                  Tile Manager object
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
                                                     int32_t interruptOnCompletion, uint32_t  *pred_mask);
#endif


/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wideHost()
 *
 * DESCRIPTION:
 *     Add iDMA transfer request using wide addresses.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager  *pxvTM                  Tile Manager object
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
                                          int32_t numRows, int32_t srcPitch, int32_t dstPitch, int32_t interruptOnCompletion);


/**********************************************************************************
 * FUNCTION: xvAddIdmaRequestMultiChannel_wide3DHost()
 *
 * DESCRIPTION:
 *     Add 3D iDMA transfer request using wide addresses.
 *
 * INPUTS:
 *     int32_t        dmaChannel              DMA channel number
 *     xvTileManager  *pxvTM                  Tile Manager object
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
                                            int32_t dstTilePitch, int32_t numTiles);


#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferInMultiChannelPredicatedHost()
 *
 * DESCRIPTION:
 *     Requests predicated data transfer from frame present in system memory
 *     to local tile memory.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
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
                                                  xvTile *pTile, int32_t interruptOnCompletion, uint32_t* pred_mask);
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
                                        xvTile *pTile, xvTile *pPrevTile, int32_t interruptOnCompletion);

/**********************************************************************************
 * FUNCTION: xvReqTileTransferInFastMultiChannelHost()
 *
 * DESCRIPTION:
 *     Requests 8b data transfer from frame present in system memory to local tile memory.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
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
                                            xvTile *pTile, int32_t interruptOnCompletion);

#if (XVTM_IDMA_HAVE_2DPRED > 0)
/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannelPredicatedHost()
 *
 * DESCRIPTION:
 *     Requests predicated data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
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
                                                   xvTile *pTile, int32_t interruptOnCompletion, uint32_t *pred_mask);
#endif


/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutMultiChannelHost()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Returns XVTM_ERROR if it encounters an error, else it returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM,
                                         xvTile *pTile, int32_t interruptOnCompletion);

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
int32_t xvReqTileTransferOut(xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion);


/**********************************************************************************
 * FUNCTION: xvReqTileTransferOutFastMultiChannelHost()
 *
 * DESCRIPTION:
 *     Requests data transfer from tile present in local memory to frame in system memory.
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile        *pTile                   Source tile
 *     int32_t       interruptOnCompletion    If it is set, iDMA will interrupt after completing transfer
 *
 * OUTPUTS:
 *     Fast variant. No sanity checks. Always returns XVTM_SUCCESS
 *
 ********************************************************************************** */
int32_t xvReqTileTransferOutFastMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile *pTile, int32_t interruptOnCompletion);

/**********************************************************************************
 * FUNCTION: xvCheckForIdmaIndexMultiChannelHost()
 *
 * DESCRIPTION:
 *     Checks if DMA transfer for given index is completed
 *
 * INPUTS:
 *     int32_t       dmaChannel               DMA channel number
 *     xvTileManager *pxvTM                   Tile Manager object
 *     int32_t       index                    Index of the dma transfer request, returned by xvAddIdmaRequest()
 *
 * OUTPUTS:
 *     Returns ONE if transfer is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvCheckForIdmaIndexMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, int32_t index);

/**********************************************************************************
 * FUNCTION: xvSleepForTileMultiChannelHost()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t       dmaChannel               iDMA channel
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile const  *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile const *pTile);

/**********************************************************************************
 * FUNCTION: xvSleepForTile()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile of iDMA channel 0  is completed.
 *
 * INPUTS:
 *     xvTileManager *pxvTM                   Tile Manager object
 *     xvTile const  *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTile(xvTileManager *pxvTM, xvTile const *pTile);

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
int32_t xvWaitForiDMAMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex);


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
int32_t xvSleepForiDMAMultiChannelHost(int32_t dmaChannel, xvTileManager *pxvTM, uint32_t dmaIndex);


/**********************************************************************************
 * FUNCTION: xvWaitForTileFastMultiChannelHost()
 *
 * DESCRIPTION:
 *     Waits till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t              dmaChannel               DMA channel
 *     xvTileManager const  *pxvTM                   Tile Manager object
 *     xvTile               *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns error flag if an error occurs
 *
 ********************************************************************************** */
int32_t xvWaitForTileFastMultiChannelHost(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile);

/**********************************************************************************
 * FUNCTION: xvSleepForTileFastMultiChannelHost()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given tile is completed.
 *
 * INPUTS:
 *     int32_t             dmaChannel               DMA channel
 *     xvTileManager const *pxvTM                   Tile Manager object
 *     xvTile              *pTile                   Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete.
 *     Returns error flag  if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileFastMultiChannelHost(int32_t dmaChannel, xvTileManager const *pxvTM, xvTile *pTile);

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
int32_t xvReqTileTransferInMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion);



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
 *     xvTile3D const *pTile3D                Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ***********************************************************************************/
int32_t xvCheckTileReadyMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D);


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
 *     uint16_t      xvTileType      Type of tile
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
                          uint16_t xvTileType, int32_t alignType);


/**********************************************************************************
 * FUNCTION: xvSleepForTileMultiChannel3DHost()
 *
 * DESCRIPTION:
 *     Sleeps till DMA transfer for given 3D-tile is completed.
 *
 * INPUTS:
 *     int32_t         dmaChannel               iDMA channel
 *     xvTileManager   *pxvTM                   Tile Manager object
 *     xvTile3D const  *pTile3D               3D-Input tile
 *
 * OUTPUTS:
 *     Returns ONE if dma transfer for input 3D-tile is complete and ZERO if it is not
 *     Returns XVTM_ERROR if an error occurs
 *
 ********************************************************************************** */
int32_t xvSleepForTileMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D const *pTile3D);


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
int32_t xvReqTileTransferOutMultiChannel3DHost(int32_t dmaChannel, xvTileManager *pxvTM, xvTile3D *pTile3D, int32_t interruptOnCompletion);

/**********************************************************************************
 * FUNCTION: xvGetErrorInfoHost()
 *
 * DESCRIPTION:
 *
 *     Prints the most recent error information.
 *
 * INPUTS:
 *     xvTileManager const *pxvTM                   Tile Manager object
 *
 * OUTPUTS:
 *     It returns the most recent error code.
 *
 ********************************************************************************** */
xvError_t xvGetErrorInfoHost(xvTileManager const *pxvTM);


#if defined (__cplusplus)
}
#endif
#endif


