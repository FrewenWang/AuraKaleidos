/** @brief      : hexagon instructions uint test head for aura
 *  @file       : hexagon_instructions_unit_test.hpp
 *  @author     : fankai1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 13, 2023
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_HEXAGON_INSTRUCTIONS_UINT_TEST_HPP__
#define AURA_RUNTIME_HEXAGON_INSTRUCTIONS_UINT_TEST_HPP__

#include "aura/runtime/core.h"
#include "aura/tools/unit_test.h"

using namespace aura;

static Status CheckVectorEqual(Context *ctx, void *dst, void *ref, DT_S32 size,
                               const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    Status ret = Status::OK;

    DT_U8 *dst_u8 = static_cast<DT_U8*>(dst);
    DT_U8 *ref_u8 = static_cast<DT_U8*>(ref);

    for (DT_S32 i = 0; i < size; i++)
    {
        ret |= TestCheckEQ(ctx, dst_u8[i], ref_u8[i], "CheckVectorEqual failed\n", file, func, line);
        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "error %d != %d\n", static_cast<DT_S32>(dst_u8[i]), static_cast<DT_S32>(ref_u8[i]));
            return ret;
        }
    }

    return ret;
}

#define CHECK_CMP_VECTOR(ctx, dst, ref, size) CheckVectorEqual(ctx, dst, ref, size, __FILE__, __FUNCTION__, __LINE__)

#endif // AURA_RUNTIME_HEXAGON_INSTRUCTIONS_UINT_TEST_HPP__