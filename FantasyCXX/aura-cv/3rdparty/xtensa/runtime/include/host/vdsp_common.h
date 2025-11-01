#ifndef _COMMOM_H_
#define _COMMOM_H_

#include <string>
#include <unistd.h>

typedef unsigned char        uint8_t;
typedef unsigned short       uint16_t;
typedef unsigned int         uint32_t;
typedef signed char          int8_t;
typedef signed short         int16_t;
typedef signed int           int32_t;
typedef float                float32_t;
typedef double               double64_t;

#define VDSP_OP_INLINE_INDATA_SIZE 256
#define VDSP_OP_INLINE_OUTDATA_SIZE 16
#define VDSP_OP_MAX_NUM 8
#define VDSP_OP_INLINE_BUFFER_MAX_NUM 8
#define VDSP_CUSTOM_OP_START_ID 101
#define NODE_NUM_MAX VDSP_OP_MAX_NUM
#define OP_IAURA_NUM_MAX VDSP_OP_MAX_NUM
#define OP_FUNC_NUM_MAX VDSP_OP_MAX_NUM

// Error codes returned by VDSP
#define VDSP_ERROR_MBX_TYPE           0x0102
#define VDSP_ERROR_CV_TYPE            0x0103
#define VDSP_ERROR_CUSTOM_CV_TYPE     0x0104 // AP-side sends a custom CV type that is not supported (not in the registration list) --- used during calculation
#define VDSP_ERROR_CUSTOM_FUNC_NUM    0x0105 // AP-side registers a number of custom functions that exceeds the upper limit --- used during registration -- placed in cmd0
#define VDSP_ERROR_CUSTOM_CANCEL_ID   0x0106 // AP-side cancels a custom function ID that is not in the registration list --- used during termination -- each
#define VDSP_ERROR_PIL_SIZE           0x0107 // AP-side provides a custom library size that is incorrect --- used during registration -- placed in cmd0
#define VDSP_ERROR_PIL_START          0x0108 // AP-side provides a custom library parsing that is incorrect --- used during registration -- placed in cmd0

// Definitions for managing the entire module
typedef enum
{
    VDSP_STATUS_SUCCESS                    = 0x0, // AP-side sends a VDSP_FLAG type that is not supported --- used for each message -- 0
    VDSP_STATUS_ERROR_CMD_STATUS_FAIL      = 0x1, // VDSP sends a status=1 failure
    VDSP_STATUS_ERROR_PARA_NULL            = 0x2,
    VDSP_STATUS_ERROR_PARA_SET_ILLEGAL     = 0x3,
    VDSP_STATUS_ERROR_OPEN_DSPDEV_FAIL     = 0x4,
    VDSP_STATUS_ERROR_DEV_ABNORMAL         = 0x5,
    VDSP_STATUS_ERROR_MALLOC_MEM_FAIL      = 0x6,
    VDSP_STATUS_ERROR_RPC_MSG_TIMEOUT      = 0x7,
    VDSP_STATUS_ERROR_RPC_MSG_FAIL         = 0x8, // VDSP sends information that is incorrect
    VDSP_STATUS_ERROR_API_CALL_ILLEGAL     = 0x9,
    VDSP_STATUS_ERROR_SET_PWCTL_FAIL       = 0xA,
    VDSP_STATUS_ERROR_DUMP_DATA_FAIL       = 0xB,
}vdsp_status_t;

typedef std::function<void(int status)> async_callback;

typedef enum
{
    VDSP_TASK_PRIORITY_LOW    = 0,
    VDSP_TASK_PRIORITY_MIDDLE = 1,
    VDSP_TASK_PRIORITY_HIGH   = 2,
    VDSP_TASK_PRIORITY_MAX    = 3,
}vdsp_task_priority;

typedef enum
{
    VDSP_DATA_DUMP_OFF              = 0, // Do not enable dump
    VDSP_DATA_DUMP_ENQUEUE_CMD      = 1, // Dump enqueue cmd
    VDSP_DATA_DUMP_OUTPUT           = 2, // Dump output
    VDSP_DATA_DUMP_CMD_AND_OUT      = 3, // Dump output + enqueue cmd
}vdsp_data_dump_type;

struct vdsp_custom_para
{
    uint32_t         op_mirror_num = 0; // Number of mirrors 1 or N are placed in the first structure
    uint32_t         op_func_num = 0; // Number of function entries == cmd number
    uint32_t         op_mirror_sizes[OP_IAURA_NUM_MAX] = {0};// Size of each mirror
    void*            op_mirror_data[OP_IAURA_NUM_MAX] = {nullptr};// Mirror CPU_VA
    std::string      op_fun_entry[OP_FUNC_NUM_MAX]; // Entry function | operator name
};

struct vdsp_init_para
{
    uint32_t         log_close = 0;
    uint32_t         priority = 0; // Priority
    uint32_t         profiling = 0; // 0-close 1-Output basic information 2-Output detailed information
    uint32_t         time_out = 0;
    uint32_t         is_custom = 0;
    vdsp_custom_para custom_para;
};

struct vdsp_init_response
{
    uint32_t out_op_ids[OP_FUNC_NUM_MAX] = {0};
};

typedef struct
{
    uint32_t op_id;
    uint32_t input_num;
    uint32_t output_num;
    uint32_t io_size[VDSP_OP_INLINE_BUFFER_MAX_NUM];
    uint32_t cycle;
}common_param;

typedef struct
{
    uint32_t total_cycle;
}vdsp_profiling_data;

typedef enum
{
    VDSP_BUFFER_TYPE_INPUT  = 1,
    VDSP_BUFFER_TYPE_OUTPUT = 2,
    VDSP_BUFFER_TYPE_OTHERS = 3,
}vdsp_buffer_type;

struct vdsp_buffer_descriptor
{
    uint32_t  mem_handle = 0;
    uint8_t*  cpu_va = nullptr;
    uint32_t  dsp_va = 0;
    uint32_t  size = 0;
    uint32_t  buffer_type = 0;
    uint32_t  is_mapped = 0; // Whether vdsp_va has been mapped

    bool      is2vdsp = true;
    bool      cpu_map = true;
    bool      vdsp_map = true;
    vdsp_buffer_descriptor(){};
    vdsp_buffer_descriptor(uint32_t buf_type):
        buffer_type(buf_type){};
    vdsp_buffer_descriptor(uint32_t buf_type, bool cpu_map_flag, bool vdsp_map_flag):
        buffer_type(buf_type),cpu_map(cpu_map_flag),vdsp_map(vdsp_map_flag){};
};

typedef struct
{
    common_param    common_info;
    uint32_t       in_data_size; 
    uint8_t*       in_data_cpu_va;
    uint32_t       out_data_size;
    uint8_t*       out_data_cpu_va;
    vdsp_buffer_descriptor io_buffers[VDSP_OP_INLINE_BUFFER_MAX_NUM];
}vdsp_task_node;

#endif //_COMMOM_H_