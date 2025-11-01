#ifndef _VDSP_INTERFACE_C_H_
#define _VDSP_INTERFACE_C_H_

#include "vdsp_common.h"

typedef void* vdsp_ctx_handle_t;

#ifdef __cplusplus
extern "C" {
#endif

/*
@brief Init vdsp context.
@param ctx_handle, return vdsp context handle.
@param para, vdsp init para.
@param op_ids, return op id when custom op needs to be registered.
*/
int vdsp_init(vdsp_ctx_handle_t* ctx_handle, vdsp_init_para* para, vdsp_init_response* op_ids);


/*
@brief Create a node object.
@param ctx_handle, vdsp context handle.
@param para, op common param.
@param in_data, Points to a buffer containing an op custom parameter.
@param in_size, the size of in_data.
@param out_data, Points to a buffer containing the data that the user wants to return.
@param out_size, the size of out_data.
@param out_node, return the created node object.
*/
int vdsp_create_node(vdsp_ctx_handle_t ctx_handle,
                                common_param* para,
                                void* in_data, uint32_t in_size,
                                void* out_data, uint32_t out_size,
                                vdsp_task_node** out_node);


/*
@brief Create io buffer for a node.
@param ctx_handle, vdsp context handle.
@param node, the node with buffer info.
@param index, the buffer index.
@param buffer, return the buffer object.
*/
int vdsp_create_io_buffer(vdsp_ctx_handle_t ctx_handle, vdsp_task_node* node, uint32_t index,
                                vdsp_buffer_descriptor* buffer);


/*
@brief Create a buffer that has CPU VA and VDSP VA mapped.
@param ctx_handle, vdsp context handle.
@param size, The buffer size needs to be created.
@param to_vdsp, Indicates whether the created buffer need send be to VDSP.
@param fd, return the handle of buffer.
@param cpu_va, return the CPU VA of buffer.
@param dsp_va, return the VDSP VA of buffer.
*/
int vdsp_create_buffer(vdsp_ctx_handle_t ctx_handle,
                                uint32_t size, bool to_vdsp,
                                uint32_t* fd, uint8_t** cpu_va, uint32_t* dsp_va);


/*
@brief Free a buffer created with vdsp_create_buffer().
@param ctx_handle, vdsp context handle.
@param fd, the buffer handle.
*/
int vdsp_free_buffer(vdsp_ctx_handle_t ctx_handle, uint32_t fd);


/*
@brief Map a buffer to VDSP.
@param ctx_handle, vdsp context handle.
@param fd, the buffer handle.
@param size, The buffer size needs to be mapped.
@param to_vdsp, Indicates whether the created buffer need send be to VDSP.
@param dsp_va, return the VDSP VA of buffer.
*/
int vdsp_map_buffer(vdsp_ctx_handle_t ctx_handle,
                                uint32_t fd, uint32_t size,
                                bool to_vdsp, uint32_t* dsp_va);


/*
@brief Unmap a buffer mapped with vdsp_map_buffer().
@param ctx_handle, vdsp context handle.
@param fd, the buffer handle.
*/
int vdsp_unmap_buffer(vdsp_ctx_handle_t ctx_handle, uint32_t fd);


/*
@brief Set a input buffer to the node.
@param ctx_handle, vdsp context handle.
@param node, the node.
@param index, the buffer index.
@param buffer, the buffer need to be set.
*/
int vdsp_set_input(vdsp_ctx_handle_t ctx_handle, vdsp_task_node* node, uint32_t input_idx,
                                vdsp_buffer_descriptor* buffer);


/*
@brief Set a output buffer to the node.
@param ctx_handle, vdsp context handle.
@param node, the node.
@param index, the buffer index.
@param buffer, the buffer need to be set.
*/
int vdsp_set_output(vdsp_ctx_handle_t ctx_handle, vdsp_task_node* node, uint32_t output_idx,
                                vdsp_buffer_descriptor* buffer);


/*
@brief Flush cache.
@param ctx_handle, vdsp context handle.
@param fd, the buffer handle.
*/
int vdsp_cache_start(vdsp_ctx_handle_t ctx_handle, uint32_t fd);


/*
@brief Invalidate cache.
@param ctx_handle, vdsp context handle.
@param fd, the buffer handle.
*/
int vdsp_cache_end(vdsp_ctx_handle_t ctx_handle, uint32_t fd);


/*
@brief Trigger(async) an op directly with the parameters, use with vdsp_wait().
@param ctx_handle, vdsp context handle.
@param op_id, the op ID.
@param cycle, op's estimated cycle.
@param in_data, points to a buffer containing an op custom parameter.
@param in_size, the size of in_data.
@param out_data, points to a buffer containing the data that the user wants to return.
@param out_size, the size of out_data.
@param msg_id, return the message ID after trigger success.
*/
int vdsp_trigger_node(vdsp_ctx_handle_t ctx_handle,
                                uint32_t op_id, uint32_t cycle,
                                void* in_data, uint32_t in_size,
                                void* out_data, uint32_t out_size,
                                uint32_t* msg_id); //异步,trigger一个node


/*
@brief Run(sync) an op directly with the parameters.
@param ctx_handle, vdsp context handle.
@param op_id, the op ID.
@param cycle, op's estimated cycle.
@param in_data, points to a buffer containing an op custom parameter.
@param in_size, the size of in_data.
@param out_data, points to a buffer containing the data that the user wants to return.
@param out_size, the size of out_data.
*/
int vdsp_run_node(vdsp_ctx_handle_t ctx_handle,
                                uint32_t op_id, uint32_t cycle,
                                void* in_data, uint32_t in_size,
                                void* out_data, uint32_t out_size);


/*
@brief Run(sync) the nodes created by vdsp_create_node().
@param ctx_handle, vdsp context handle.
*/
int vdsp_sync_run(vdsp_ctx_handle_t ctx_handle);


/*
@brief Trigger(async) the nodes created by vdsp_create_node(), use with vdsp_wait().
@param ctx_handle, vdsp context handle.
@param msg_id, return the message ID after trigger success.
*/
int vdsp_trigger(vdsp_ctx_handle_t ctx_handle, uint32_t* msg_id);


/*
@brief Async wait a message from vdsp_trigger_node() or vdsp_trigger().
@param ctx_handle, vdsp context handle.
@param msg_id, message ID.
*/
int vdsp_wait(vdsp_ctx_handle_t ctx_handle, uint32_t msg_id);


/*
@brief Async run the nodes created by vdsp_create_node() with callback.
@param ctx_handle, vdsp context handle.
@param callback, the callback function, this function is automatically called when the result is obtained from vdsp.
*/
int vdsp_async_run(vdsp_ctx_handle_t ctx_handle, async_callback& callback);


/*
@brief Get profiling data of the op created by vdsp_create_node().
@param ctx_handle, vdsp context handle.
@param idx, the op index.
*/
int vdsp_get_profiling(vdsp_ctx_handle_t ctx_handle, vdsp_profiling_data *data, uint32_t idx);


/*
@brief Set vdsp power frequency level.
@param ctx_handle, vdsp context handle.
@param level, point to the specific level.
*/
int vdsp_set_power(vdsp_ctx_handle_t ctx_handle, uint32_t* level);


/*
@brief Set dump information.
@param ctx_handle, vdsp context handle.
@param dump_type, dump type.
@param dump_path, indicates that data is dumped to the path.
*/
int vdsp_save_dump_to_file(vdsp_ctx_handle_t ctx_handle, uint32_t* dump_type, char* dump_path);


/*
@brief Release vdsp context object.
@param ctx_handle, vdsp context handle.
*/
int vdsp_release(vdsp_ctx_handle_t ctx_handle);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif /* _VDSP_INTERFACE_C_H_ */