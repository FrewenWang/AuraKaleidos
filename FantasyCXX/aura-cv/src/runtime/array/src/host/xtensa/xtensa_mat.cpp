#include "aura/runtime/array/host/xtensa_mat.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

XtensaMat::XtensaMat() : Array(), m_xtensa_engine(DT_NULL), m_is_external_buffer(DT_TRUE)
{
    m_array_type = ArrayType::XTENSA_MAT;
}

XtensaMat::XtensaMat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides) 
                     : Array(ctx, elem_type, sizes, strides), m_xtensa_engine(DT_NULL), m_is_external_buffer(DT_FALSE)
{
    Status ret = Status::ERROR;

    if (DT_NULL == m_ctx)
    {
        goto EXIT;
    }

    m_xtensa_engine = m_ctx->GetXtensaEngine();
    if (DT_NULL == m_xtensa_engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetXtensaEngine failed, m_xtensa_engine is null ptr");
        goto EXIT;
    }

    if (InitRefCount() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitRefCount failed");
        goto EXIT;
    }

    m_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DEFAULT, m_total_bytes, 0));
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer is invalid");
        goto EXIT;
    }

    if (m_xtensa_engine->MapBuffer(m_buffer) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MapBuffer failed");
        goto EXIT;
    }

    m_array_type = ArrayType::XTENSA_MAT;
    ret = Status::OK;

EXIT:
    if (ret != Status::OK)
    {
        Release();
    }
}

XtensaMat::XtensaMat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides)
                     : Array(ctx, elem_type, sizes, strides, buffer), m_xtensa_engine(DT_NULL), m_is_external_buffer(DT_TRUE)
{
    Status ret = Status::ERROR;

    if (DT_NULL == m_ctx)
    {
        goto EXIT;
    }

    m_xtensa_engine = m_ctx->GetXtensaEngine();
    if (DT_NULL == m_xtensa_engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetXtensaEngine failed, m_xtensa_engine is null ptr");
        goto EXIT;
    }

#if !defined(AURA_BUILD_XPLORER)
    if (m_buffer.m_type != AURA_MEM_DMA_BUF_HEAP)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer mem type must be DMA Buffer");
        goto EXIT;
    }
#endif

    if (!m_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer is invalid");
        goto EXIT;
    }

    if (m_total_bytes > m_buffer.m_size)
    {
        std::string info = "m_total_bytes(" + std::to_string(m_total_bytes) + ") must less equal buffer size(" + std::to_string(m_buffer.m_size) + ")";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    if (InitRefCount() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitRefCount failed");
        goto EXIT;
    }

    if (m_xtensa_engine->MapBuffer(m_buffer) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MapBuffer failed");
        goto EXIT;
    }

    m_array_type = ArrayType::XTENSA_MAT;
    m_buffer.m_size = m_total_bytes;
    ret = Status::OK;

EXIT:
    if (ret != Status::OK)
    {
        Release();
    }
}

XtensaMat::XtensaMat(const XtensaMat &xtensa_mat) : Array(xtensa_mat), m_xtensa_engine(xtensa_mat.m_xtensa_engine), 
                                                    m_is_external_buffer(xtensa_mat.m_is_external_buffer)
{
    if (!xtensa_mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "xtensa_mat is invalid");
        Clear();
    }
    else
    {
        AddRefCount(1);
    }
}

XtensaMat::~XtensaMat()
{
    Release();
}

DT_VOID XtensaMat::Release()
{
    if (m_refcount != DT_NULL)
    {
        if (AddRefCount(-1) == 0)
        {
            if (m_xtensa_engine && m_buffer.IsValid())
            {
                m_xtensa_engine->UnmapBuffer(m_buffer);
            }

            AURA_FREE(m_ctx, m_refcount);
            if (DT_FALSE == m_is_external_buffer)
            {
                AURA_FREE(m_ctx, m_buffer.m_origin);
            }
        }
    }

    Clear();
}

XtensaMat& XtensaMat::operator=(const XtensaMat &xtensa_mat)
{
    if (this == &xtensa_mat)
    {
        return *this;
    }

    Release();
    if (!xtensa_mat.IsValid())
    {
        return *this;
    }

    Array::operator=(xtensa_mat);
    AddRefCount(1);

    m_xtensa_engine = xtensa_mat.m_xtensa_engine;
    m_is_external_buffer = xtensa_mat.m_is_external_buffer;

    return *this;
}

XtensaMat XtensaMat::FromArray(Context *ctx, const Array &array)
{
    if (DT_NULL == ctx)
    {
        return XtensaMat();
    }

    if (array.GetArrayType() == ArrayType::MAT)
    {
        const Mat *mat = dynamic_cast<const Mat*>(&array);
        if (!(mat && mat->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "mat is invalid");
            return XtensaMat();
        }

        return XtensaMat(ctx, mat->GetElemType(), mat->GetSizes(), mat->GetBuffer(), mat->GetStrides());
    }
    else if(array.GetArrayType() == ArrayType::XTENSA_MAT)
    {
        const XtensaMat *xtensa_mat = dynamic_cast<const XtensaMat*>(&array);
        if (!(xtensa_mat && xtensa_mat->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "xtensa_mat is invalid");
            return XtensaMat();
        }
        
        return *xtensa_mat;
    }
    else
    {
        std::string info = "arrary type only suppose MAT/XTENSA_MAT, current is " + ArrayTypesToString(array.GetArrayType());
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return XtensaMat();
    }
}

DT_BOOL XtensaMat::IsValid() const
{
    return (Array::IsValid() && m_buffer.IsValid() && ArrayType::XTENSA_MAT == m_array_type);
}

DT_VOID XtensaMat::Show() const
{
    if (m_ctx)
    {
        std::string info = Array::ToString();
        info += "================= XtensaMat Info End =================\n\n";
        AURA_LOGD(m_ctx, AURA_TAG, "%s\n", info.c_str());
    }
}

DT_VOID XtensaMat::Dump(const std::string &fname) const
{
    if (!IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "XtensaMat is invalid");
        return;
    }

    if (fname.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fname is empty");
        return;
    }

    FILE *fp = fopen(fname.c_str(), "wb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return;
    }

    DT_S32 row_bytes = m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
    for (DT_S32 i = 0; i < m_sizes.m_height; i++)
    {
        size_t bytes = fwrite(static_cast<DT_U8*>(m_buffer.m_data) + i * m_strides.m_width, 1, row_bytes, fp);
        if (static_cast<DT_S32>(bytes) != row_bytes)
        {
            std::string info = "fwrite size(" + std::to_string(bytes) + "," + std::to_string(row_bytes) + ") not match";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }
    }

EXIT:
    if (fp)
    {
        fclose(fp);
    }
}

Status XtensaMat::Sync(XtensaSyncType xtensa_sync_type)
{
    if (DT_NULL == m_xtensa_engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_engine is null ptr");
        return Status::ERROR;
    }

    if (xtensa_sync_type == XtensaSyncType::WRITE)
    {
        return m_xtensa_engine->CacheEnd(m_buffer.m_property);
    }
    else if (xtensa_sync_type == XtensaSyncType::READ)
    {
        return m_xtensa_engine->CacheStart(m_buffer.m_property);
    }
    else
    {
        return Status::OK;
    }

    return Status::OK;
}

DT_VOID XtensaMat::Clear()
{
    Array::Clear();
    m_array_type = ArrayType::INVALID;
    m_xtensa_engine = DT_NULL;
    m_is_external_buffer = DT_TRUE;
}

} // namespace aura