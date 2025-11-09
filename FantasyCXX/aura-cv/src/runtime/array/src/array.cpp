#include "aura/runtime/array/array.hpp"
#include "aura/runtime/memory.h"
#include "aura/runtime/logger.h"

namespace aura
{

Array::Array() : m_ctx(DT_NULL), m_elem_type(ElemType::INVALID), m_array_type(ArrayType::INVALID),
                 m_total_bytes(0), m_refcount(DT_NULL)
{}

Array::Array(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Sizes &strides, const Buffer &buffer)
             : m_ctx(ctx), m_elem_type(elem_type), m_array_type(ArrayType::INVALID), m_sizes(sizes),
               m_strides(strides), m_total_bytes(0), m_refcount(DT_NULL), m_buffer(buffer)
{
    if (m_ctx)
    {
        /// 这个就是一行数据的字节数
        DT_S32 pitch = m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
        m_strides = m_strides.Max(Sizes(m_sizes.m_height, pitch));
        /// 这个就是行字节数 * 宽字节数
        m_total_bytes = static_cast<DT_S64>(m_strides.m_width) * m_strides.m_height;
        //// TODO 没看懂，这句话有事他妈的什么意思？？
        if (m_buffer.m_size == (m_sizes.m_height - 1) * m_strides.m_width + m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type))
        {
            m_total_bytes = m_buffer.m_size;
        }
    }
}

Array::Array(const Array &array): m_ctx(array.m_ctx), m_elem_type(array.m_elem_type), m_array_type(array.m_array_type),
                                  m_sizes(array.m_sizes), m_strides(array.m_strides), m_total_bytes(array.m_total_bytes),
                                  m_refcount(array.m_refcount), m_buffer(array.m_buffer)
{}

Array& Array::operator=(const Array &array)
{
    if (this == &array)
    {
        return *this;
    }

    m_ctx         = array.m_ctx;
    m_elem_type   = array.m_elem_type;
    m_array_type  = array.m_array_type;
    m_sizes       = array.m_sizes;
    m_strides     = array.m_strides;
    m_total_bytes = array.m_total_bytes;
    m_refcount    = array.m_refcount;
    m_buffer      = array.m_buffer;

    return *this;
}

Array::~Array()
{}

ElemType Array::GetElemType() const
{
    return m_elem_type;
}

ArrayType Array::GetArrayType() const
{
    return m_array_type;
}

Sizes3 Array::GetSizes() const
{
    return m_sizes;
}

Sizes Array::GetStrides() const
{
    return m_strides;
}

DT_S64 Array::GetTotalBytes() const
{
    return m_total_bytes;
}

DT_S32 Array::GetRowPitch() const
{
    return m_strides.m_width;
}

DT_S32 Array::GetRowStep() const
{
    DT_S32 pixels = ElemTypeSize(m_elem_type) * m_sizes.m_channel;

    if (0 == pixels)
    {
        return 0;
    }
    else
    {
        return m_strides.m_width / pixels;
    }
}

DT_S32 Array::GetRefCount() const
{
    return AddRefCount(0);
}

const Buffer& Array::GetBuffer() const
{
    return m_buffer;
}

DT_S32 Array::GetMemType() const
{
    return m_buffer.m_type;
}

DT_BOOL Array::IsValid() const
{
    return (m_ctx && (m_elem_type != ElemType::INVALID) && (m_array_type != ArrayType::INVALID) && (m_total_bytes > 0));
}

DT_BOOL Array::IsEqual(const Array &array) const
{
    return ((array.m_elem_type == m_elem_type) && (array.m_sizes == m_sizes));
}

DT_BOOL Array::IsSizesEqual(const Array &array) const
{
    return (array.m_sizes == m_sizes);
}

DT_BOOL Array::IsChannelEqual(const Array &array) const
{
    return (array.m_sizes.m_channel == m_sizes.m_channel);
}

std::string Array::ToString() const
{
    std::stringstream oss;

    oss << std::endl << "=================== Info Start ===================" << std::endl;
    oss << "array type          : " << m_array_type << std::endl;
    oss << "elem type           : " << m_elem_type << std::endl;
    oss << "sizes               : " << m_sizes << std::endl;
    oss << "strides             : " << m_strides << std::endl;
    oss << "ref count           : " << GetRefCount() << std::endl;
    oss << "Total bytes         : " << m_total_bytes << std::endl;
    oss << "buffer              : " << m_buffer << std::endl;
    return oss.str();
}

Status Array::InitRefCount()
{
    m_refcount = static_cast<DT_S32*>(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, sizeof(DT_S32), 0));
    if (m_refcount != DT_NULL)
    {
        *m_refcount = 1;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }
    return Status::OK;
}

DT_S32 Array::AddRefCount(DT_S32 num) const
{
    if (m_refcount)
    {
        return AURA_XADD(m_refcount, num) + num;
    }
    return 0;
}

DT_VOID Array::Clear()
{
    m_ctx         = DT_NULL;
    m_elem_type   = ElemType::INVALID;
    m_sizes       = Sizes3();
    m_strides     = Sizes();
    m_total_bytes = 0;
    m_refcount    = DT_NULL;
    m_buffer      = Buffer();
}

} // namespace aura