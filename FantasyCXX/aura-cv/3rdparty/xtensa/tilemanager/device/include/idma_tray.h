/*
 * xm_xrp_struct.h
 *
 *  Created on: Aug 30, 2024
 *      Author: zhonganyu
 */

// provide some iMDA function for PIL lib. these functions are from symbol tray.
#ifndef IDMA_TRAY_API_H__
#define IDMA_TRAY_API_H__

#if defined (__cplusplus)
extern "C"
{
#endif

#include <stdint.h>

#if (!defined(__XTENSA__)) || (defined(UNIFIED_TEST))
#include "dummy.h"
#else
#include <xtensa/idma.h>
#endif //__XTENSA__

idma_status_t tray_idma_init_task(int32_t ch, idma_buffer_t *taskh, idma_type_t type, int32_t ndescs, idma_callback_fn cb_func, void *cb_data);

idma_status_t tray_idma_add_2d_desc64(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch);

idma_status_t tray_idma_add_2d_desc64_wide(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch);

idma_status_t tray_idma_schedule_task(idma_buffer_t *taskh);

int32_t tray_idma_schedule_desc(int32_t ch, uint32_t count);

idma_status_t tray_idma_process_tasks(int32_t ch);

int32_t tray_idma_desc_done(int32_t ch, int32_t index);

#if defined (__cplusplus)
}
#endif

#endif
