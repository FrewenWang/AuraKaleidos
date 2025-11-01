#include "aura/runtime/array/host/cl_mem.hpp"
#include "aura/runtime/opencl/cl_engine.hpp"
#include "aura/runtime/logger.h"

#include <fstream>

namespace aura
{

CLMem::CLMem() : m_cl_sync_method(CLMemSyncMethod::INVALID), m_data(MI_NULL)
{
    m_array_type = ArrayType::CL_MEMORY;
}

CLMem::CLMem(Context *ctx, const CLMemParam &cl_param, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides)
             : Array(ctx, elem_type, sizes, strides), m_cl_param(cl_param), m_data(MI_NULL)
{
    m_array_type = ArrayType::CL_MEMORY;

    do
    {
        if ((MI_NULL == ctx) || (CLMemType::INVALID == m_cl_param.m_type) || (!Array::IsValid()))
        {
            Clear();
            break;
        }

        if (ctx->GetCLEngine() == MI_NULL)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCLEngine failed");
            Clear();
            break;
        }

        m_cl_rt = ctx->GetCLEngine()->GetCLRuntime();
        if (MI_NULL == m_cl_rt)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCLRuntime failed");
            Clear();
            break;
        }

        m_cl_cmd = m_cl_rt->GetCommandQueue();
        if (MI_NULL == m_cl_cmd)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCommandQueue failed");
            Clear();
            break;
        }

        if (InitRefCount() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "InitRefCount failed");
            Clear();
            break;
        }

        if (CreateCLMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreateCLMem failed");
            AURA_FREE(m_ctx, m_refcount);
            Clear();
            break;
        }

    } while (0);
}

CLMem::CLMem(Context *ctx, const CLMemParam &cl_param, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides)
             : Array(ctx, elem_type, sizes, strides, buffer), m_cl_param(cl_param), m_data(MI_NULL)
{
    m_array_type = ArrayType::CL_MEMORY;

    do
    {
        if ((MI_NULL == ctx) || (CLMemType::INVALID == m_cl_param.m_type) || (!m_buffer.IsValid()) || (!Array::IsValid()))
        {
            Clear();
            break;
        }

        if (ctx->GetCLEngine() == MI_NULL)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCLEngine failed");
            Clear();
            break;
        }

        if (m_total_bytes > m_buffer.m_size)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the memory of buffer does not meet the requirements of CLMem");
            Clear();
            break;
        }

        m_buffer.m_size = m_total_bytes;

        m_cl_rt = ctx->GetCLEngine()->GetCLRuntime();
        if (MI_NULL == m_cl_rt)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCLRuntime failed");
            Clear();
            break;
        }

        m_cl_cmd = m_cl_rt->GetCommandQueue();
        if (MI_NULL == m_cl_cmd)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetCommandQueue failed");
            Clear();
            break;
        }

        if (InitRefCount() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "InitRefCount failed");
            Clear();
            break;
        }

        if (InitCLMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "InitCLMem failed");
            AURA_FREE(m_ctx, m_refcount);
            Clear();
            break;
        }

    } while (0);
}

CLMem::CLMem(const CLMem &cl_mem)
             : Array(cl_mem), m_cl_param(cl_mem.m_cl_param), m_cl_sync_method(cl_mem.m_cl_sync_method),
               m_data(cl_mem.m_data), m_cl_rt(cl_mem.m_cl_rt), m_cl_cmd(cl_mem.m_cl_cmd)
{
    if (!cl_mem.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid cl_mem");
        Clear();
    }
    else
    {
        AddRefCount(1);
    }
}

CLMem& CLMem::operator=(const CLMem &cl_mem)
{
    if (this == &cl_mem)
    {
        return *this;
    }

    Release();
    if (!cl_mem.IsValid())
    {
        return *this;
    }

    Array::operator=(cl_mem);
    AddRefCount(1);

    m_array_type     = cl_mem.m_array_type;
    m_cl_param       = cl_mem.m_cl_param;
    m_cl_sync_method = cl_mem.m_cl_sync_method;
    m_data           = cl_mem.m_data;
    m_cl_rt          = cl_mem.m_cl_rt;
    m_cl_cmd         = cl_mem.m_cl_cmd;

    return *this;
}

AURA_VOID CLMem::Clear()
{
    Array::Clear();
    m_array_type     = ArrayType::INVALID;
    m_cl_param       = CLMemParam();
    m_cl_sync_method = CLMemSyncMethod::INVALID;

    m_cl_rt.reset();
    m_cl_cmd.reset();
}

CLMem::~CLMem()
{
    Release();
}

AURA_VOID CLMem::Release()
{
    if (m_refcount)
    {
        if (AddRefCount(-1) == 0)
        {
            AURA_FREE(m_ctx, m_refcount);
            m_cl_rt->DeleteCLMem(&m_data);
        }
    }

    Clear();
}

CLMem CLMem::FromArray(Context *ctx, const Array &array, const CLMemParam &cl_param)
{
    if (MI_NULL == ctx)
    {
        return CLMem();
    }

    if (CLMemType::INVALID == cl_param.m_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "CLMemParam CLMemType is invalid");
        return CLMem();
    }

    if (ArrayType::MAT == array.GetArrayType())
    {
        const Mat *mat = dynamic_cast<const Mat *>(&array);
        if (!(mat && mat->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "mat is invalid");
            return CLMem();
        }

        CLMem cl_mem = CLMem(ctx, cl_param, mat->GetElemType(), mat->GetSizes(), mat->GetBuffer(), mat->GetStrides());
        if (!cl_mem.IsValid())
        {
            cl_mem = CLMem(ctx, cl_param, mat->GetElemType(), mat->GetSizes(), mat->GetStrides());
            if (!cl_mem.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "CLMem is invalid");
                return CLMem();
            }

            if (cl_mem.BindBuffer(mat->GetBuffer()) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "CLMem BindBuffer failed");
                return CLMem();
            }
        }

        return cl_mem;
    }
    else if(ArrayType::CL_MEMORY == array.GetArrayType())
    {
        const CLMem *cl_mem = dynamic_cast<const CLMem *>(&array);
        if (!(cl_mem && cl_mem->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "mem is invalid");
            return CLMem();
        }

        const CLMemParam &cl_param_in = cl_mem->GetCLMemParam();

        // if buffer to buffer, return directly
        if (CLMemType::BUFFER == cl_param.m_type && CLMemType::BUFFER == cl_param_in.m_type)
        {
            return *cl_mem;
        }

        // if iaura2d to iaura2d, check if param is exactly the same
        if (CLMemType::IAURA2D == cl_param.m_type &&
            CLMemType::IAURA2D == cl_param_in.m_type &&
            cl_param == cl_param_in)
        {
            return *cl_mem;
        }

        // if iaura3d to iaura3d, check if param is exactly the same
        if (CLMemType::IAURA3D == cl_param.m_type &&
            CLMemType::IAURA3D == cl_param_in.m_type &&
            cl_param == cl_param_in)
        {
            return *cl_mem;
        }

        // TODO: add support buffer to iaura2d/3d
        // TODO: add support iaura2d/3d to buffer
    }

    AURA_ADD_ERROR_STRING(ctx, "FromArray failed");
    return CLMem();
}

MI_BOOL CLMem::IsValid() const
{
    return (Array::IsValid() && m_data != MI_NULL && ArrayType::CL_MEMORY == m_array_type);
}

AURA_VOID CLMem::Show() const
{
    if (m_ctx)
    {
        std::string info = Array::ToString();

        std::stringstream oss;
        oss << "cl_param            : " << m_cl_param << std::endl;
        oss << "m_cl_sync_method    : " << CLMemSyncMethodToString(m_cl_sync_method) << std::endl;
        oss << "data                : " << m_data << std::endl;
        oss << "================= CLMem Info End =================" << std::endl << std::endl;
        info += oss.str();

        AURA_LOGD(m_ctx, AURA_TAG, "%s\n", info.c_str());
    }
}

AURA_VOID CLMem::Dump(const std::string &fname) const
{
    if (!fname.empty())
    {
        std::ofstream fout(fname.c_str(), std::ios::out | std::ios::binary);
        if (fout.is_open())
        {
            cl_int cl_err    = CL_SUCCESS;
            MI_S32 elem_size = ElemTypeSize(m_elem_type);

            switch (m_cl_param.m_type)
            {
                case CLMemType::BUFFER:
                {
                    AURA_VOID *host_ptr = m_cl_cmd->enqueueMapBuffer(GetCLMemRef<cl::Buffer>(), CL_TRUE, CL_MAP_READ,
                                                                    0, m_total_bytes, NULL, NULL, &cl_err);
                    if ((cl_err != CL_SUCCESS) || (MI_NULL == host_ptr))
                    {
                        if (m_buffer.IsValid())
                        {
                            host_ptr = m_buffer.m_data;
                        }
                    }

                    if (MI_NULL == host_ptr)
                    {
                        return;
                    }

                    for (MI_S32 i = 0; i < m_sizes.m_height; i++)
                    {
                        const MI_CHAR *src = (MI_CHAR *)host_ptr + i * m_strides.m_width;
                        fout.write(src, m_sizes.m_width * m_sizes.m_channel * elem_size);
                    }
                    fout.close();

                    if (host_ptr != m_buffer.m_data)
                    {
                        cl::Event cl_event;
                        cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Buffer>(), host_ptr, NULL, &cl_event);
                        if (CL_SUCCESS == cl_err)
                        {
                            cl_event.wait();
                        }
                    }

                    break;
                }
                case CLMemType::IAURA2D:
                {
                    cl_iaura_format fmt;
                    fmt.iaura_channel_order     = m_cl_param.m_param.iaura2d.cl_ch_order;
                    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura2d.is_norm);

                    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
                    if (0 == iaura_width)
                    {
                        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
                        return;
                    }

                    std::array<size_t, 3> origin = {0, 0, 0};
                    std::array<size_t, 3> region = {iaura_width, static_cast<size_t>(m_strides.m_height), 1};
                    size_t row_pitch = 0;
                    AURA_VOID *host_ptr = m_cl_cmd->enqueueMapIaura(GetCLMemRef<cl::Iaura2D>(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
                                                                    &row_pitch, 0, NULL, NULL);
                    if ((cl_err != CL_SUCCESS) || (MI_NULL == host_ptr))
                    {
                        if (m_buffer.IsValid())
                        {
                            host_ptr = m_buffer.m_data;
                            row_pitch = m_strides.m_width;
                        }
                    }

                    if (MI_NULL == host_ptr)
                    {
                        return;
                    }

                    for (MI_S32 i = 0; i < m_sizes.m_height; i++)
                    {
                        const MI_CHAR *src = (MI_CHAR *)host_ptr + i * row_pitch;
                        fout.write(src, m_sizes.m_width * m_sizes.m_channel * elem_size);
                    }
                    fout.close();

                    if (host_ptr != m_buffer.m_data)
                    {
                        cl::Event cl_event;
                        cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Iaura2D>(), host_ptr, NULL, &cl_event);
                        if (CL_SUCCESS == cl_err)
                        {
                            cl_event.wait();
                        }
                    }

                    break;
                }
                case CLMemType::IAURA3D:
                {
                    cl_iaura_format fmt;
                    fmt.iaura_channel_order     = m_cl_param.m_param.iaura3d.cl_ch_order;
                    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura3d.is_norm);

                    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
                    if (0 == iaura_width)
                    {
                        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
                        return;
                    }

                    size_t depth                 = m_cl_param.m_param.iaura3d.depth;
                    size_t iaura_height          = m_strides.m_height / depth;
                    std::array<size_t, 3> origin = {0, 0, 0};
                    std::array<size_t, 3> region = {iaura_width, iaura_height, depth};
                    size_t row_pitch             = 0;
                    size_t slice_pitch           = 0;

                    AURA_VOID *host_ptr = m_cl_cmd->enqueueMapIaura(GetCLMemRef<cl::Iaura3D>(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
                                                                  &row_pitch, &slice_pitch, NULL, NULL);
                    if (cl_err != CL_SUCCESS)
                    {
                        std::string info = "enqueueMapIaura failed, Error: " + GetCLErrorInfo(cl_err);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        return;
                    }

                    for (MI_S32 i = 0; i < static_cast<MI_S32>(depth); i++)
                    {
                        for (MI_S32 j = 0; j < static_cast<MI_S32>(iaura_height); j++)
                        {
                            const MI_CHAR *src = (MI_CHAR *)host_ptr + i * slice_pitch + j * row_pitch;
                            fout.write(src, m_sizes.m_width * m_sizes.m_channel * elem_size);
                        }
                    }
                    fout.close();

                    cl::Event cl_event;
                    cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Iaura3D>(), host_ptr, NULL, &cl_event);
                    if (cl_err != CL_SUCCESS)
                    {
                        std::string info = "enqueueUnmapMemObject failed, Error: " + GetCLErrorInfo(cl_err);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        return;
                    }
                    cl_event.wait();

                    break;
                }
                default:
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "m_type error, onlys suppose BUFFER/IAURA2D/IAURA3D");
                    return;
                }
            }
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "fname fopen failed", "%s\n", fname.c_str());
            return;
        }
    }
    else
    {
        AURA_LOGE(m_ctx, AURA_TAG, "fname is null\n");
    }
}

Status CLMem::InitCLMem()
{
    Status ret = Status::ERROR;

    if (CLMemType::BUFFER == m_cl_param.m_type)
    {
        ret = InitCLBuffer();
    }
    else if (CLMemType::IAURA2D == m_cl_param.m_type)
    {
        ret = InitCLIaura2D();
    }
    else if (CLMemType::IAURA3D == m_cl_param.m_type)
    {
        ret = InitCLIaura3D();
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_type is invalid");
    }

    return ret;
}

Status CLMem::InitCLBuffer()
{
    cl::Buffer *cl_buffer = m_cl_rt->InitCLBuffer(m_buffer, m_cl_param.m_param.buffer.cl_flags, m_cl_sync_method);

    if (cl_buffer != MI_NULL)
    {
        m_data = cl_buffer;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitCLBuffer failed");
        return Status::ERROR;
    }
}

Status CLMem::InitCLIaura2D()
{
    cl_iaura_format fmt;
    fmt.iaura_channel_order     = m_cl_param.m_param.iaura2d.cl_ch_order;
    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura2d.is_norm);

    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
    if (0 == iaura_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
        return Status::ERROR;
    }

    cl::Iaura2D *cl_iaura2d = m_cl_rt->InitCLIaura2D(m_buffer, m_cl_param.m_param.iaura2d.cl_flags, fmt, iaura_width,
                                                     m_strides.m_height, m_strides.m_width, m_cl_sync_method);

    if (cl_iaura2d != MI_NULL)
    {
        m_data = cl_iaura2d;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitCLIaura2D failed");
        return Status::ERROR;
    }
}

Status CLMem::InitCLIaura3D()
{
    size_t depth = m_cl_param.m_param.iaura3d.depth;

    if ((m_strides.m_height % depth != 0) || (depth <= 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "depth is invalid");
        return Status::ERROR;
    }

    cl_iaura_format fmt;
    fmt.iaura_channel_order     = m_cl_param.m_param.iaura3d.cl_ch_order;
    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura3d.is_norm);

    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
    if (0 == iaura_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
        return Status::ERROR;
    }

    size_t iaura_height = m_strides.m_height / depth;
    size_t slice_pitch  = iaura_height * m_strides.m_width;

    cl::Iaura3D *cl_iaura3d = m_cl_rt->InitCLIaura3D(m_buffer, m_cl_param.m_param.iaura3d.cl_flags, fmt, iaura_width,
                                                     iaura_height, depth, m_strides.m_width, slice_pitch, m_cl_sync_method);

    if (cl_iaura3d != MI_NULL)
    {
        m_data = cl_iaura3d;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitCLIaura3D failed");
        return Status::ERROR;
    }
}

Status CLMem::CreateCLMem()
{
    Status ret = Status::ERROR;

    if (CLMemType::BUFFER == m_cl_param.m_type)
    {
        ret = CreateCLBuffer();
    }
    else if (CLMemType::IAURA2D == m_cl_param.m_type)
    {
        ret = CreateCLIaura2D();
    }
    else if (CLMemType::IAURA3D == m_cl_param.m_type)
    {
        ret = CreateCLIaura3D();
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_type is invalid");
    }

    return ret;
}

Status CLMem::CreateCLBuffer()
{
    cl::Buffer *cl_buffer = m_cl_rt->CreateCLBuffer(m_cl_param.m_param.buffer.cl_flags, m_total_bytes);

    if (cl_buffer != MI_NULL)
    {
        m_data        = cl_buffer;
        m_cl_sync_method = CLMemSyncMethod::AUTO;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateCLBuffer failed");
        return Status::ERROR;
    }
}

Status CLMem::CreateCLIaura2D()
{
    cl_iaura_format fmt;
    fmt.iaura_channel_order     = m_cl_param.m_param.iaura2d.cl_ch_order;
    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura2d.is_norm);

    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
    if (0 == iaura_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
        return Status::ERROR;
    }

    cl::Iaura2D *cl_iaura2d = m_cl_rt->CreateCLIaura2D(m_cl_param.m_param.iaura2d.cl_flags, fmt, iaura_width,
                                                       m_strides.m_height);
    if (cl_iaura2d != MI_NULL)
    {
        m_data        = cl_iaura2d;
        m_cl_sync_method = CLMemSyncMethod::AUTO;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateCLIaura2D failed");
        return Status::ERROR;
    }
}

Status CLMem::CreateCLIaura3D()
{
    size_t depth = m_cl_param.m_param.iaura3d.depth;

    if ((m_strides.m_height % depth != 0) || (depth <= 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "depth is invalid");
        return Status::ERROR;
    }

    cl_iaura_format fmt;
    fmt.iaura_channel_order     = m_cl_param.m_param.iaura3d.cl_ch_order;
    fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura3d.is_norm);

    size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
    if (0 == iaura_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
        return Status::ERROR;
    }

    size_t iaura_height = m_strides.m_height / depth;

    cl::Iaura3D *cl_iaura3d = m_cl_rt->CreateCLIaura3D(m_cl_param.m_param.iaura3d.cl_flags, fmt, iaura_width,
                                                       iaura_height, depth);
    if (cl_iaura3d != MI_NULL)
    {
        m_data        = cl_iaura3d;
        m_cl_sync_method = CLMemSyncMethod::AUTO;
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateCLIaura3D failed");
        return Status::ERROR;
    }
}

Status CLMem::EnqueuedData(CLMemSyncType cl_sync_type)
{
    if (!m_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer is invalid");
        return Status::ERROR;
    }

    if (CLMemSyncType::INVALID == cl_sync_type)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_sync_type is invalid");
        return Status::ERROR;
    }

    cl_int cl_err    = CL_SUCCESS;
    MI_U8 *data      = (MI_U8 *)(m_buffer.m_data);
    MI_S32 elem_size = ElemTypeSize(m_elem_type);

    switch (m_cl_param.m_type)
    {
        case CLMemType::BUFFER:
        {
            AURA_VOID *host_ptr = m_cl_cmd->enqueueMapBuffer(GetCLMemRef<cl::Buffer>(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                                            0, m_total_bytes, NULL, NULL, &cl_err);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueMapBuffer failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            for (MI_S32 i = 0; i < m_sizes.m_height; i++)
            {
                MI_U8 *src = (MI_U8 *)host_ptr + i * m_strides.m_width;
                MI_U8 *dst = data + i * m_strides.m_width;

                if (CLMemSyncType::WRITE == cl_sync_type)
                {
                    Swap(src, dst);
                }
                memcpy(dst, src, m_sizes.m_width * m_sizes.m_channel * elem_size);
            }

            cl::Event cl_event;
            cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Buffer>(), host_ptr, NULL, &cl_event);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueUnmapMemObject failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
            cl_event.wait();

            break;
        }

        case CLMemType::IAURA2D:
        {
            cl_iaura_format fmt;
            fmt.iaura_channel_order     = m_cl_param.m_param.iaura2d.cl_ch_order;
            fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura2d.is_norm);

            size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
            if (0 == iaura_width)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
                return Status::ERROR;
            }

            std::array<size_t, 3> origin = {0, 0, 0};
            std::array<size_t, 3> region = {iaura_width, static_cast<size_t>(m_strides.m_height), 1};
            size_t row_pitch             = 0;

            AURA_VOID *host_ptr = m_cl_cmd->enqueueMapIaura(GetCLMemRef<cl::Iaura2D>(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
                                                            &row_pitch, 0, NULL, NULL);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueMapIaura failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            for (MI_S32 i = 0; i < m_sizes.m_height; i++)
            {
                MI_U8 *src = (MI_U8 *)host_ptr + i * row_pitch;
                MI_U8 *dst = data + i * m_strides.m_width;

                if (CLMemSyncType::WRITE == cl_sync_type)
                {
                    Swap(src, dst);
                }
                memcpy(dst, src, m_sizes.m_width * m_sizes.m_channel * elem_size);
            }

            cl::Event cl_event;
            cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Iaura2D>(), host_ptr, NULL, &cl_event);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueUnmapMemObject failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
            cl_event.wait();

            break;
        }

        case CLMemType::IAURA3D:
        {
            cl_iaura_format fmt;
            fmt.iaura_channel_order     = m_cl_param.m_param.iaura2d.cl_ch_order;
            fmt.iaura_channel_data_type = GetCLIauraChannelDataType(m_elem_type, m_cl_param.m_param.iaura2d.is_norm);

            size_t iaura_width = GetCLIauraWidth(m_sizes.m_width * m_sizes.m_channel, fmt.iaura_channel_order);
            if (0 == iaura_width)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "GetCLIauraWidth failed");
                return Status::ERROR;
            }

            size_t depth                 = m_cl_param.m_param.iaura3d.depth;
            size_t iaura_height          = m_strides.m_height / depth;
            std::array<size_t, 3> origin = {0, 0, 0};
            std::array<size_t, 3> region = {iaura_width, iaura_height, depth};
            size_t row_pitch             = 0;
            size_t slice_pitch           = 0;

            AURA_VOID *host_ptr = m_cl_cmd->enqueueMapIaura(GetCLMemRef<cl::Iaura3D>(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
                                                          &row_pitch, &slice_pitch, NULL, NULL);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueMapIaura failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            for (MI_S32 i = 0; i < static_cast<MI_S32>(depth); i++)
            {
                MI_U8 *m_dst = data + i * m_total_bytes / depth;
                for (MI_S32 j = 0; j < static_cast<MI_S32>(iaura_height); j++)
                {
                    MI_U8 *src = (MI_U8 *)host_ptr + i * slice_pitch + j * row_pitch;
                    MI_U8 *dst = m_dst + j * m_strides.m_width;

                    if (CLMemSyncType::WRITE == cl_sync_type)
                    {
                        Swap(src, dst);
                    }
                    memcpy(dst, src, m_sizes.m_width * m_sizes.m_channel * elem_size);
                }
            }

            cl::Event cl_event;
            cl_err = m_cl_cmd->enqueueUnmapMemObject(GetCLMemRef<cl::Iaura3D>(), host_ptr, NULL, &cl_event);
            if (cl_err != CL_SUCCESS)
            {
                std::string info = "enqueueUnmapMemObject failed, Error: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
            cl_event.wait();

            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_type error, onlys suppose BUFFER/IAURA2D/IAURA3D");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

/**
 * 往服务端同步写入数据
 * @param cl_sync_type
 * @return
 */
Status CLMem::Sync(CLMemSyncType cl_sync_type)
{
    Status ret = Status::OK;

    switch (cl_sync_type)
    {
        case CLMemSyncType::READ:
        {
            if (CLMemSyncMethod::ENQUEUE == m_cl_sync_method)
            {
                ret = EnqueuedData(cl_sync_type);
                if (m_buffer.IsValid() && (AURA_MEM_SVM == m_buffer.m_type))
                {
                    ret = m_ctx->GetMemPool()->Map(m_buffer);
                }
            }
            else if (CLMemSyncMethod::FLUSH == m_cl_sync_method)
            {
                ret = m_ctx->GetMemPool()->Map(m_buffer);
            }
            break;
        }
        case CLMemSyncType::WRITE:
        {
            if (CLMemSyncMethod::ENQUEUE == m_cl_sync_method)
            {
                ret = EnqueuedData(cl_sync_type);

                if (m_buffer.IsValid() && (AURA_MEM_SVM == m_buffer.m_type))
                {
                    ret = m_ctx->GetMemPool()->Unmap(m_buffer);
                }
            }
            else if (CLMemSyncMethod::FLUSH == m_cl_sync_method)
            {
                ret = m_ctx->GetMemPool()->Unmap(m_buffer);
            }
            break;
        }
        default:
        {
            break;
        }
    }

    return ret;
}

Status CLMem::Load(const std::string &fname)
{
    Status ret = Status::ERROR;

    if (fname.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fname is empty");
        return ret;
    }

    MI_S32 row_bytes         = m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
    MI_S32 buffer_valid_size = m_sizes.m_height * row_bytes;
    MI_BOOL is_alloc_mem     = m_buffer.IsValid();
    FILE *fp                 = MI_NULL;
    MI_S32 file_length       = 0;

    if (MI_FALSE == is_alloc_mem)
    {
        m_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DMA_BUF_HEAP, m_total_bytes, 0));
        if (!m_buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AURA_ALLOC_PARAM failed");
            return ret;
        }
    }

    fp = fopen(fname.c_str(), "rb");
    if (MI_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    fseek(fp, 0, SEEK_END);
    file_length = ftell(fp);

    if (file_length < buffer_valid_size)
    {
        std::string info = "file size(" + std::to_string(file_length) + ") must greater equal buffer size(" + std::to_string(buffer_valid_size) + ")";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    fseek(fp, 0, SEEK_SET);
    for (MI_S32 i = 0; i < m_sizes.m_height; i++)
    {
        MI_S32 len = fread(static_cast<MI_U8*>(m_buffer.m_data) + i * m_strides.m_width, 1, row_bytes, fp);
        if (len != row_bytes)
        {
            std::string info = "file fread size(" + std::to_string(len) + "," + std::to_string(row_bytes) + ") not match";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }
    }

    ret = EnqueuedData(CLMemSyncType::WRITE);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "EnqueuedData failed");
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    if (fp)
    {
        fclose(fp);
    }

    if (MI_FALSE == is_alloc_mem)
    {
        AURA_FREE(m_ctx, m_buffer.m_origin);
        m_buffer.Clear();
    }

    return ret;
}

Status CLMem::BindBuffer(const Buffer &buffer)
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_mem is invalid");
        return Status::ERROR;
    }

    if (m_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer is valid");
        return Status::ERROR;
    }

    m_buffer = buffer;
    m_cl_sync_method = CLMemSyncMethod::ENQUEUE;

    return Status::OK;
}

cl_channel_type CLMem::GetCLIauraChannelDataType(ElemType elem_type, MI_BOOL is_norm)
{
    cl_channel_type cl_ch_type = 0;

    switch (elem_type)
    {
        case ElemType::U8:
        {
            cl_ch_type = is_norm ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
            break;
        }
        case ElemType::S8:
        {
            cl_ch_type = is_norm ? CL_SNORM_INT8 : CL_SIGNED_INT8;
            break;
        }
        case ElemType::U16:
        {
            cl_ch_type = is_norm ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
            break;
        }
        case ElemType::S16:
        {
            cl_ch_type = is_norm ? CL_SNORM_INT16 : CL_SIGNED_INT16;
            break;
        }
        case ElemType::F16:
        {
            cl_ch_type = CL_HALF_FLOAT;
            break;
        }
        case ElemType::U32:
        {
            cl_ch_type = CL_UNSIGNED_INT32;
            break;
        }
        case ElemType::S32:
        {
            cl_ch_type = CL_SIGNED_INT32;
            break;
        }
        case ElemType::F32:
        {
            cl_ch_type = CL_FLOAT;
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_ch_type;
}

MI_S32 CLMem::GetCLIauraChannelNum(cl_channel_order cl_ch_order)
{
    MI_S32 channel_num = 0;

    switch (cl_ch_order)
    {
        case CL_R:
        case CL_A:
        {
            channel_num = 1;
            break;
        }

        case CL_RG:
        case CL_RA:
        {
            channel_num = 2;
            break;
        }

        case CL_RGB:
        {
            channel_num = 3;
            break;
        }

        case CL_RGBA:
        case CL_BGRA:
        case CL_ARGB:
        {
            channel_num = 4;
            break;
        }
    }

    return channel_num;
}

size_t CLMem::GetCLIauraWidth(MI_S32 elem_count, cl_channel_order cl_ch_order)
{
    MI_S32 channel_num = GetCLIauraChannelNum(cl_ch_order);

    if (0 == channel_num)
    {
        return 0;
    }

    if (elem_count < channel_num || (elem_count % channel_num != 0))
    {
        return 0;
    }

    return elem_count / channel_num;
}

} // namespace aura