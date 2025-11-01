/*
 * xm_cascade_utils.h
 *
 *  Created on: Feb 5, 2024
 *      Author: zhonganyu
 */

#ifndef XM_CASCADE_UTILS_H_
#define XM_CASCADE_UTILS_H_

#include "tileManager_api.h"

#define PADDING_LEFT  1
#define PADDING_RIGHT 2
#define PADDING_TOP   4
#define PADDING_DOWN  8

#define TILE_I8  1
#define TILE_I16 2
#define TILE_I32 4

typedef int32_t XI_ERR_TYPE;
#define XVF_SUCCESS XVTM_SUCCESS

typedef struct TileInfoStruct
{
    void* base_ptr;
    int32_t base_width;
    int32_t base_height;
    int32_t base_x;
    int32_t base_y;
    int32_t tiletype;
    int32_t edge_width;
    int32_t edge_height;
    int32_t padding_type;
    int32_t padding_value;
} TileInfo;

XI_ERR_TYPE TilePadding(xi_tile* p_tile, int32_t padding_type, int32_t padding_value);
XI_ERR_TYPE TilePaddingWithSize(xi_tile* p_tile, int32_t padding_type, int32_t padding_width, int32_t padding_height, int32_t padding_value);
XI_ERR_TYPE TileResetting(xi_tile* p_tile, TileInfo* p_tileinfo, int32_t extra_width, int32_t extra_height, int32_t edge_width, int32_t edge_height, int32_t edgeflags);
void TileResetToOrigin(xi_tile* p_tile, TileInfo* p_tileinfo);
void ExtractTileInfo(TileInfo* p_tileinfo, xi_tile* p_tile, int32_t tiletype);
int32_t GetEdgeFlags(xi_tile* p_tile, xi_frame* p_frame);

#endif /*  */
