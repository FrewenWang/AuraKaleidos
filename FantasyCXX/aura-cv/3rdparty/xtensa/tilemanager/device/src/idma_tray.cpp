#include "idma_tray.h"
#include "tileManager.h"

extern application_symbol_tray* app_symbol_tray;

idma_status_t tray_idma_init_task(int32_t ch, idma_buffer_t *taskh, idma_type_t type, int32_t ndescs, idma_callback_fn cb_func, void *cb_data)
{
    return app_symbol_tray->tray_idma_init_task(ch, taskh, type, ndescs, cb_func, cb_data);
}

idma_status_t tray_idma_add_2d_desc64(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch)
{
    return app_symbol_tray->tray_idma_add_2d_desc64(bufh, dst, src, row_sz, flags, nrows, src_pitch, dst_pitch);
}

idma_status_t tray_idma_add_2d_desc64_wide(idma_buffer_t *bufh, void *dst, void *src, size_t row_sz, uint32_t flags, uint32_t nrows, uint32_t src_pitch, uint32_t dst_pitch)
{
    return app_symbol_tray->tray_idma_add_2d_desc64_wide(bufh, dst, src, row_sz, flags, nrows, src_pitch, dst_pitch);
}

idma_status_t tray_idma_schedule_task(idma_buffer_t *taskh)
{
    return app_symbol_tray->tray_idma_schedule_task(taskh);
}

int32_t tray_idma_schedule_desc(int32_t ch, uint32_t count)
{
    return app_symbol_tray->tray_idma_schedule_desc(ch, count);
}

idma_status_t tray_idma_process_tasks(int32_t ch)
{
    return app_symbol_tray->tray_idma_process_tasks(ch);
}

int32_t tray_idma_desc_done(int32_t ch, int32_t index)
{
    return app_symbol_tray->tray_idma_desc_done(ch, index);
}
