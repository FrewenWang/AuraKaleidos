#include "resize_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

ResizeCL::ResizeCL(Context *ctx, const OpTarget &target) : ResizeImpl(ctx, target), m_profiling_string()
{}

Status ResizeCL::SetArgs(const Array *src, Array *dst, InterpType type)
{
    if (ResizeImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel > 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL current supported channel: 1 2 3.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ResizeCL::Initialize()
{
    if (ResizeImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeImpl::Initialize() failed");
        return Status::ERROR;
    }

    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ResizeCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();

    if (ResizeImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::string ResizeCL::ToString() const
{
    return ResizeImpl::ToString() + m_profiling_string;
}

} // namespace aura