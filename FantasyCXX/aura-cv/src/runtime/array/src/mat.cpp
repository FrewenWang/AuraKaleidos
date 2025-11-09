#include "aura/runtime/array/mat.hpp"
#include "aura/runtime/logger.h"

#include <cstring>
#include <fstream>
#include <sys/stat.h>

namespace aura
{
#pragma pack(push, 1)

/*
 *  BMP file header structure
 */
struct BitMapFileHeader
{
    BitMapFileHeader() : type(0), size(0), reserved1(0), reserved2(0), offset(0)
    {}

    DT_U16 type;      /* File type, must be 'BM' (0x4D42).*/
    DT_U32 size;      /* File size in bytes. */
    DT_U16 reserved1; /* Reserved, must be 0. */
    DT_U16 reserved2; /* Reserved, must be 0. */
    DT_U32 offset;    /* Offset from the beginning of the file to the bitmap data. */
};

/*
 *  BMP information header structure
 */
struct BitMapInfoHeader
{
    BitMapInfoHeader(): header_sz(0), width(0), height(0), planes(0), pixels(0), compression(0),
                        data_sz(0), pixels_x(0), pixels_y(0), colors(0), imp_colors(0)
    {}

    DT_U32 header_sz;   /* Size of this header in bytes (40 bytes). */
    DT_S32 width;       /* Width of the bitmap in pixels. */
    DT_S32 height;      /* Height of the bitmap in pixels. */
    DT_U16 planes;      /* Number of color planes, must be 1. */
    DT_U16 pixels;      /* Number of bits per pixel (1, 4, 8, 16, 24, or 32). */
    DT_U32 compression; /* Compression method (0 = BI_RGB, no compression). */
    DT_U32 data_sz;     /* Size of the raw bitmap data (including padding). */
    DT_S32 pixels_x;    /* Horizontal resolution (pixels per meter). */
    DT_S32 pixels_y;    /* Vertical resolution (pixels per meter). */
    DT_U32 colors;      /* Number of colors in the color palette (0 = all colors). */
    DT_U32 imp_colors;  /* Number of important colors (0 = all colors are important). */
};

struct Pixel32
{
    Pixel32() : blue(0), green(0), red(0), alpha(0)
    {}

    DT_U8 blue;  /* Blue color component (0-255) */
    DT_U8 green; /* Green color component (0-255) */
    DT_U8 red;   /* Red color component (0-255) */
    DT_U8 alpha; /* Alpha (transparency) component (0-255); 0 means fully transparent, 255 means fully opaque */
};

#pragma pack(pop)

Mat::Mat()
{
    m_array_type = ArrayType::MAT;
}

Mat::Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, DT_S32 mem_type, const Sizes &strides)
         : Array(ctx, elem_type, sizes, strides)
{
    m_array_type = ArrayType::MAT;

    if ((m_ctx && mem_type != AURA_MEM_INVALID) || (!Array::IsValid()))
    {
        if (InitRefCount() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "InitRefCount failed");
            Clear();
        }
        else
        {
            m_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, mem_type, m_total_bytes, 0));
            if (!IsValid())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "invalid mat");
                Release();
            }
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mem_type is AURA_MEM_INVALID");
        Clear();
    }
}

Mat::Mat(Context *ctx, ElemType elem_type, const Sizes3 &sizes, const Buffer &buffer, const Sizes &strides)
         : Array(ctx, elem_type, sizes, strides, buffer)
{
    m_array_type = ArrayType::MAT;

    if (m_ctx && m_buffer.IsValid())
    {
        if (m_total_bytes > m_buffer.m_size)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "the memory of buffer does not meet the requirements of Mat");
            Clear();
        }
        else
        {
            m_buffer.m_size = m_total_bytes;
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "buffer is invalid");
        Clear();
    }
}

Mat::Mat(const Mat &mat) : Array(mat)
{
    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid mat");
        Clear();
    }
    else
    {
        AddRefCount(1);
    }
}

Mat::Mat(const Mat &mat, const Rect &roi)
         : Array(mat.m_ctx, mat.m_elem_type, Sizes3(roi.Size(), mat.m_sizes.m_channel), mat.m_strides, mat.m_buffer)
{
    m_array_type = mat.m_array_type;
    m_refcount   = mat.m_refcount;

    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid mat");
        Clear();
    }
    else if((roi.m_x < 0) || (roi.m_y < 0) || ((roi.m_x + roi.m_width) > mat.m_sizes.m_width) || ((roi.m_y + roi.m_height) > mat.m_sizes.m_height))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the ROI size does not match the mat size");
        Clear();
    }
    else
    {
        AddRefCount(1);

        DT_S32 ptr_offset = (roi.m_y * mat.m_strides.m_width) + ElemTypeSize(m_elem_type) * roi.m_x * mat.m_sizes.m_channel;

        m_sizes.m_channel = mat.m_sizes.m_channel;
        m_sizes.m_width   = (0 == m_sizes.m_width ) ? (mat.m_sizes.m_width  - roi.m_x) : m_sizes.m_width;
        m_sizes.m_height  = (0 == m_sizes.m_height) ? (mat.m_sizes.m_height - roi.m_y) : m_sizes.m_height;

        if (1 == mat.m_sizes.m_height)
        {
            m_strides.m_width = m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
        }
        m_strides.m_height = m_sizes.m_height;

        if (roi.m_y + roi.m_height == mat.m_sizes.m_height)
        {
            m_total_bytes = mat.m_strides.m_width * (m_strides.m_height - 1) + m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
        }
        else
        {
            m_total_bytes = mat.m_strides.m_width * m_strides.m_height;
        }

        m_buffer.Resize(m_total_bytes, ptr_offset);
    }
}

Mat::~Mat()
{
    Release();
}

DT_VOID Mat::Release()
{
    if (m_refcount != DT_NULL)
    {
        if (AddRefCount(-1) == 0)
        {
            AURA_FREE(m_ctx, m_refcount);
            AURA_FREE(m_ctx, m_buffer.m_origin);
        }
    }
    Clear();
}

Mat& Mat::operator=(const Mat &mat)
{
    if (this == &mat)
    {
        return *this;
    }

    Release();
    if (!mat.IsValid())
    {
        return *this;
    }

    Array::operator=(mat);
    AddRefCount(1);

    return *this;
}

Mat Mat::Roi(const Rect &roi) const
{
    return Mat(*this, roi);
}

Mat Mat::RowRange(DT_S32 start, DT_S32 end) const
{
    return Mat(*this, Rect(0, start, m_sizes.m_width, end - start));
}

Mat Mat::ColRange(DT_S32 start, DT_S32 end) const
{
    return Mat(*this, Rect(start, 0, end - start, m_sizes.m_height));
}

Mat Mat::Clone() const
{
    if (!IsValid())
    {
        return Mat();
    }

    Mat mat(m_ctx, m_elem_type, m_sizes, m_buffer.m_type, m_strides);
    if (mat.IsValid())
    {
        memcpy(mat.m_buffer.m_data, m_buffer.m_data, m_total_bytes);
    }

    return mat;
}

Mat Mat::Clone(const Rect &roi, const Sizes &strides) const
{
    if (!IsValid())
    {
        return Mat();
    }

    Mat mat(m_ctx, m_elem_type, Sizes3(roi.Size(), m_sizes.m_channel), m_buffer.m_type, strides);
    if (mat.IsValid())
    {
        const DT_S32 copy_bytes = ElemTypeSize(m_elem_type) * roi.m_width * m_sizes.m_channel;
        const DT_S32 col_off    = ElemTypeSize(m_elem_type) * roi.m_x * m_sizes.m_channel;
        for (DT_S32 h = 0; h < roi.m_height; ++h)
        {
            memcpy(mat.Ptr<DT_VOID>(h), &(Ptr<DT_U8>(roi.m_y + h)[col_off]), copy_bytes);
        }
    }

    return mat;
}

Status Mat::CopyTo(Mat &mat) const
{
    if (this == &mat)
    {
        return Status::OK;
    }

    if (!mat.IsValid())
    {
        mat = Clone();
        return Status::OK;
    }

    Status ret = Status::ERROR;

    if (IsEqual(mat))
    {
        const DT_S32 copy_bytes = ElemTypeSize(m_elem_type) * m_sizes.m_width * m_sizes.m_channel;
        for (DT_S32 h = 0; h < m_sizes.m_height; ++h)
        {
            memcpy(mat.Ptr<DT_VOID>(h), Ptr<DT_U8>(h), copy_bytes);
        }
        ret = Status::OK;
    }

    return ret;
}

DT_VOID* Mat::GetData()
{
    return m_buffer.m_data;
}

const DT_VOID* Mat::GetData() const
{
    return m_buffer.m_data;
}

DT_BOOL Mat::IsValid() const
{
    return (Array::IsValid() && m_buffer.IsValid() && ArrayType::MAT == m_array_type);
}

DT_BOOL Mat::IsContinuous() const
{
    return (IsValid() && (m_strides.m_width == m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type)));
}

DT_VOID Mat::Clear()
{
    Array::Clear();
}

DT_VOID Mat::Show() const
{
    if (m_ctx)
    {
        std::string info = Array::ToString();
        info += "================= MAT Info End =================\n\n";
        AURA_LOGD(m_ctx, AURA_TAG, "%s\n", info.c_str());
    }
}

template<typename Tp>
static Status PrintHelper(Context *ctx, const Mat &mat, DT_S32 mode, const Rect &roi, FILE *fp)
{
    Status ret = Status::ERROR;

    if (DT_NULL == ctx)
    {
        return ret;
    }

    if (!mat.IsValid())
    {
        AURA_LOGE(ctx, AURA_TAG, "mat is invalid\n");
        return ret;
    }

    const Sizes3 &msize = mat.GetSizes();

    if ((roi.m_x < 0) || (roi.m_y < 0) ||
        ((roi.m_x + roi.m_width) > msize.m_width) ||
        ((roi.m_y + roi.m_height) > msize.m_height))
    {
        AURA_LOGE(ctx, AURA_TAG, "roi is invalid, roi m_x m_y must >=0 and (roi.m_x + roi.m_width) <= msize.m_width) and roi.m_y + roi.m_height) <= msize.m_height\n");
        return ret;
    }

    DT_S32 start_h = roi.m_y;
    DT_S32 start_w = roi.m_x;
    DT_S32 len_h   = (roi.m_height <= 0) ? msize.m_height : roi.m_height;
    DT_S32 len_w   = (roi.m_width <= 0) ? msize.m_width : roi.m_width;
    DT_S32 cn      = msize.m_channel;

    DT_CHAR fmt[16] = {0};
    if (16 == mode)
    {
        if ((std::is_same<Tp, DT_U8>::value) || (std::is_same<Tp, DT_U16>::value) ||
            (std::is_same<Tp, DT_U32>::value) || (std::is_same<Tp, DT_F32>::value) ||
            (std::is_same<Tp, DT_F64>::value))
        {
#if defined(AURA_BUILD_HOST)
            snprintf(fmt, sizeof(fmt), "0x%%0%dx ", static_cast<DT_S32>(sizeof(Tp) << 1));
#elif defined(AURA_BUILD_HEXAGON)
            snprintf(fmt, sizeof(fmt), "0x%%0%ldx ", static_cast<DT_S32>(sizeof(Tp) << 1));
#endif
        }
        else
        {
            snprintf(fmt, sizeof(fmt), "0x%%08x ");
        }
    }
    else if (10 == mode)
    {
        if (std::is_same<Tp, DT_F32>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%.6f ");
        }
        else if (std::is_same<Tp, DT_F64>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%.8lf ");
        }
        else if (std::is_same<Tp, DT_U8>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%3d ");
        }
        else if (std::is_same<Tp, DT_S8>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%4d ");
        }
        else if (std::is_same<Tp, DT_U16>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%5d ");
        }
        else if (std::is_same<Tp, DT_S16>::value)
        {
            snprintf(fmt, sizeof(fmt), "%%6d ");
        }
        else
        {
            snprintf(fmt, sizeof(fmt), "%%8d ");
        }
    }

    DT_CHAR str[64];
    for (DT_S32 h = start_h; h < (start_h + len_h); ++h)
    {
        std::string show_str = std::string();
        const Tp *src = mat.Ptr<Tp>(h);
        for (DT_S32 w = start_w; w < (start_w + len_w); ++w)
        {
            show_str += "[";
            for (DT_S32 c = 0; c < cn; ++c)
            {
                snprintf(str, sizeof(str), fmt, src[w * cn + c]);
                show_str += str;
            }
            show_str.pop_back(); // remove multi channel last  space
            show_str += "] ";
        }

        if (fp)
        {
            fprintf(fp, "%s\n", show_str.c_str());
        }
        else
        {
            AURA_LOGD(ctx, AURA_TAG, "%s\n", show_str.c_str());
        }
    }

    return Status::OK;
}

DT_VOID Mat::Print(DT_S32 mode, const Rect &roi, const std::string &fname) const
{
    if (DT_NULL == m_ctx)
    {
        return;
    }

    if ((mode != 10) && (mode != 16))
    {
        AURA_LOGE(m_ctx, AURA_TAG, "only suppose 10 or 16 format\n");
        return;
    }

    Status ret = Status::ERROR;

    FILE *fp = DT_NULL;
    if (!fname.empty())
    {
        fp = fopen(fname.c_str(), "wt");
        if (DT_NULL == fp)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "fname %s fopen failed\n", fname.c_str());
            return;
        }
    }

    switch (m_elem_type)
    {
        case ElemType::U8:
        {
            ret = PrintHelper<DT_U8>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::S8:
        {
            ret = PrintHelper<DT_S8>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::U16:
        {
            ret = PrintHelper<DT_U16>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::S16:
        {
            ret = PrintHelper<DT_S16>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::U32:
        {
            ret = PrintHelper<DT_U32>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::S32:
        {
            ret = PrintHelper<DT_S32>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::F32:
        {
            ret = PrintHelper<DT_F32>(m_ctx, *this, mode, roi, fp);
            break;
        }

        case ElemType::F64:
        {
            ret = PrintHelper<DT_F64>(m_ctx, *this, mode, roi, fp);
            break;
        }
        default:
        {
            AURA_LOGE(m_ctx, AURA_TAG, "do not surpport elem_type type F16\n");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "PrintPlane failed\n");
    }

    if (fp)
    {
        fclose(fp);
    }
}

DT_VOID Mat::Dump(const std::string &fname) const
{
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
        size_t bytes = fwrite(Ptr<DT_CHAR>(i), 1, row_bytes, fp);
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

Status Mat::Load(const std::string &fname)
{
    Status ret = Status::ERROR;

    if (fname.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fname is empty");
        return ret;
    }

    DT_S32 row_bytes         = m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type);
    DT_S32 buffer_valid_size = m_sizes.m_height * row_bytes;
    FILE *fp                 = DT_NULL;
    DT_S32 file_length       = 0;

    fp = fopen(fname.c_str(), "rb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return ret;
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

    for (DT_S32 i = 0; i < m_sizes.m_height; i++)
    {
        size_t bytes = fread(Ptr<DT_CHAR>(i), 1, row_bytes, fp);
        if (static_cast<DT_S32>(bytes) != row_bytes)
        {
            std::string info = "file fread size(" + std::to_string(bytes) + "," + std::to_string(row_bytes) + ") not match";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }
    }

    ret = Status::OK;

EXIT:
    if (fp)
    {
        fclose(fp);
    }

    return ret;
}

Status Mat::Reshape(const Sizes3 &sizes)
{
    Status ret = Status::ERROR;

    if (sizes.Empty() || sizes.m_channel <= 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sizes is empty");
        return ret;
    }

    if (m_sizes.Total() * ElemTypeSize(m_elem_type) == m_strides.Total())
    {
        if (m_sizes.Total() == sizes.Total())
        {
            m_sizes = sizes;
            m_strides = Sizes(m_sizes.m_height, m_sizes.m_width * m_sizes.m_channel * ElemTypeSize(m_elem_type));
        }
        else
        {
            std::string info = "sizes total(" + std::to_string(sizes.Total()) + ") must be same with m_sizes total(" + std::to_string(m_sizes.Total()) + ")";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }
    }
    else if (m_sizes.m_height == sizes.m_height && ((sizes.m_width * sizes.m_channel * ElemTypeSize(m_elem_type)) <= m_strides.m_width))
    {
        m_sizes.m_width = sizes.m_width;
        m_sizes.m_channel = sizes.m_channel;
    }
    else
    {
        std::string info = "mat size " + m_sizes.ToString() + "can't reshape to " + sizes.ToString();
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    ret = Status::OK;
EXIT:
    return ret;
}

Status GetIauraFormatSize(IauraFormat fmt, Sizes img_sizes, const Sizes &img_strides,
                          std::vector<Sizes3> &mat_sizes, std::vector<Sizes> &mat_strides)
{
    Status ret = Status::OK;

    if ((img_sizes.m_width < 1) || (img_sizes.m_height < 1))
    {
        return Status::ERROR;
    }

    mat_sizes.clear();
    mat_strides.clear();

    switch (fmt)
    {
        case IauraFormat::None:
        case IauraFormat::Gray:
        {
            mat_sizes.reserve(1);
            mat_strides.reserve(1);

            DT_S32 pitch = img_sizes.m_width * ElemTypeSize(ElemType::U8);
            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, pitch));

            mat_strides.emplace_back(strides);
            mat_sizes.emplace_back(Sizes3(img_sizes, 1));
            break;
        }
        case IauraFormat::RGB:
        case IauraFormat::BGR:
        {
            mat_sizes.reserve(1);
            mat_strides.reserve(1);

            DT_S32 cn = 3;

            DT_S32 pitch = img_sizes.m_width * cn * ElemTypeSize(ElemType::U8);
            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, pitch));

            mat_strides.emplace_back(strides);
            mat_sizes.emplace_back(Sizes3(img_sizes, cn));
            break;
        }
        case IauraFormat::RGBA:
        case IauraFormat::BGRA:
        {
            mat_sizes.reserve(1);
            mat_strides.reserve(1);

            DT_S32 cn = 4;

            DT_S32 pitch = img_sizes.m_width * cn * ElemTypeSize(ElemType::U8);
            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, pitch));

            mat_strides.emplace_back(strides);
            mat_sizes.emplace_back(Sizes3(img_sizes, cn));
            break;
        }
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        {
            mat_sizes.reserve(2);
            mat_strides.reserve(2);

            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, img_sizes.m_width * ElemTypeSize(ElemType::U8)));

            mat_strides.emplace_back(strides);
            mat_strides.emplace_back(Sizes(strides.m_height / 2, strides.m_width));

            mat_sizes.emplace_back(Sizes3(img_sizes, 1));
            mat_sizes.emplace_back(Sizes3(img_sizes / 2, 2));
            break;
        }
        case IauraFormat::YU12:
        case IauraFormat::YV12:
        {
            mat_sizes.reserve(3);
            mat_strides.reserve(3);

            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, img_sizes.m_width * ElemTypeSize(ElemType::U8)));

            mat_strides.emplace_back(strides);
            mat_strides.emplace_back(strides / 2);
            mat_strides.emplace_back(strides / 2);

            mat_sizes.emplace_back(Sizes3(img_sizes, 1));
            mat_sizes.emplace_back(Sizes3(img_sizes / 2, 1));
            mat_sizes.emplace_back(Sizes3(img_sizes / 2, 1));
            break;
        }
        case IauraFormat::I422:
        {
            mat_sizes.reserve(3);
            mat_strides.reserve(3);

            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, img_sizes.m_width * ElemTypeSize(ElemType::U8)));

            mat_strides.emplace_back(strides);
            mat_strides.emplace_back(Sizes(strides.m_height, strides.m_width / 2));
            mat_strides.emplace_back(Sizes(strides.m_height, strides.m_width / 2));

            mat_sizes.emplace_back(Sizes3(img_sizes, 1));
            mat_sizes.emplace_back(Sizes3(Sizes(img_sizes.m_height, img_sizes.m_width / 2), 1));
            mat_sizes.emplace_back(Sizes3(Sizes(img_sizes.m_height, img_sizes.m_width / 2), 1));
            break;
        }
        case IauraFormat::I444:
        {
            mat_sizes.reserve(3);
            mat_strides.reserve(3);

            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, img_sizes.m_width * ElemTypeSize(ElemType::U8)));

            mat_strides = std::vector<Sizes>(3, strides);
            mat_sizes   = std::vector<Sizes3>(3, Sizes3(img_sizes, 1));
            break;
        }
        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            mat_sizes.reserve(2);
            mat_strides.reserve(2);

            Sizes strides = img_strides.Max(Sizes(img_sizes.m_height, img_sizes.m_width * sizeof(DT_U16)));

            mat_strides.emplace_back(strides);
            mat_strides.emplace_back(Sizes(strides.m_height / 2, strides.m_width));

            mat_sizes.emplace_back(Sizes3(img_sizes, 1));
            mat_sizes.emplace_back(Sizes3(img_sizes / 2, 2));
            break;
        }
        default:
        {
            ret = Status::ERROR;
            break;
        }
    }

    return ret;
}

Status GetIauraFormatRoi(IauraFormat fmt, const Rect &img_roi, std::vector<Rect> &mat_rois)
{
    Status ret = Status::OK;

    if (img_roi.m_x < 0 || img_roi.m_y < 0 || img_roi.m_width < 0 || img_roi.m_height < 0)
    {
        return Status::ERROR;
    }

    switch (fmt)
    {
        case IauraFormat::None:
        case IauraFormat::Gray:
        case IauraFormat::RGB:
        case IauraFormat::RGBA:
        case IauraFormat::BGR:
        case IauraFormat::BGRA:
        {
            mat_rois.reserve(1);
            mat_rois.emplace_back(img_roi);
            break;
        }
        case IauraFormat::I444:
        {
            mat_rois.reserve(3);
            mat_rois = std::vector<Rect>(3, img_roi);
            break;
        }
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            mat_rois.reserve(2);
            Rect rect(img_roi.m_x & (~1), img_roi.m_y & (~1), ((img_roi.m_width + 1) & (~1)), ((img_roi.m_height + 1) & (~1)));
            mat_rois.emplace_back(rect);
            mat_rois.emplace_back(Rect(rect.m_x / 2, rect.m_y / 2, rect.m_width / 2, rect.m_height / 2));
            break;
        }
        case IauraFormat::YU12:
        case IauraFormat::YV12:
        {
            mat_rois.reserve(3);
            Rect rect(img_roi.m_x & (~1), img_roi.m_y & (~1), ((img_roi.m_width + 1) & (~1)), ((img_roi.m_height + 1) & (~1)));
            Rect half(rect.m_x / 2, rect.m_y / 2, rect.m_width / 2, rect.m_height / 2);

            mat_rois.emplace_back(rect);
            mat_rois.emplace_back(half);
            mat_rois.emplace_back(half);
            break;
        }
        case IauraFormat::I422:
        {
            mat_rois.reserve(3);
            Rect rect(img_roi.m_x & (~1), img_roi.m_y, ((img_roi.m_width + 1) & (~1)), img_roi.m_height);
            Rect halfw(rect.m_x / 2, rect.m_y, rect.m_width / 2, rect.m_height);

            mat_rois.emplace_back(img_roi);
            mat_rois.emplace_back(halfw);
            mat_rois.emplace_back(halfw);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            break;
        }
    }

    return ret;
}

Mat ReadBmp(Context *ctx, const std::string &fname, DT_S32 mem_type, const Sizes &strides)
{
    if (DT_NULL == ctx)
    {
        return Mat();
    }

    BitMapFileHeader file_header;
    BitMapInfoHeader info_header;

    DT_S32 buffer_valid_size = sizeof(BitMapFileHeader) + sizeof(BitMapInfoHeader);

    FILE *fp = fopen(fname.c_str(), "rb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return Mat();
    }

    fseek(fp, 0, SEEK_END);
    DT_S32 file_length = ftell(fp);

    if (file_length < buffer_valid_size)
    {
        std::string info = "file size(" + std::to_string(file_length) + ") must be >= buffer size(" + std::to_string(buffer_valid_size) + ")";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        fclose(fp);
        return Mat();
    }

    fseek(fp, 0, SEEK_SET);

    size_t bytes = fread(&file_header, 1, sizeof(file_header), fp);
    if (bytes != sizeof(file_header))
    {
        std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(sizeof(file_header)) + ") not match";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        fclose(fp);
        return Mat();
    }

    bytes = fread(&info_header, 1, sizeof(info_header), fp);
    if (bytes != sizeof(info_header))
    {
        std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(sizeof(info_header)) + ") not match";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        fclose(fp);
        return Mat();
    }

    if (file_header.type != 0x4D42)
    {
        AURA_ADD_ERROR_STRING(ctx, "not a bmp file");
        fclose(fp);
        return Mat();
    }

    DT_S32 height  = Abs(info_header.height);
    DT_S32 width   = info_header.width;
    DT_S32 channel = info_header.pixels / 8;

    Mat mat(ctx, ElemType::U8, Sizes3(height, width, channel), mem_type, strides);
    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "create mat failed!");
        fclose(fp);
        return Mat();
    }

    DT_S32 row_bytes = (width * channel + 3) & (-4); // Number of pixel data bytes per row, 4-byte aligned.
    DT_S32 offset    = file_header.offset;

    buffer_valid_size = offset + row_bytes * height;
    if (file_length < buffer_valid_size)
    {
        std::string info = "file size(" + std::to_string(file_length) + ") must greater equal buffer size(" + std::to_string(buffer_valid_size) + ")";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        fclose(fp);
        return Mat();
    }

    if (info_header.height > 0)
    {
        for (DT_S32 h = (height - 1); h >= 0; h--)
        {
            fseek(fp, offset, SEEK_SET);
            bytes = fread(mat.Ptr<DT_CHAR>(h), 1, width * channel, fp);
            if (static_cast<DT_S32>(bytes) != width * channel)
            {
                std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(width * channel) + ") not match";
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                fclose(fp);
                return Mat();
            }
            offset += row_bytes;
        }
    }
    else
    {
        for (DT_S32 h = 0; h < height; h++)
        {
            fseek(fp, offset, SEEK_SET);
            bytes = fread(mat.Ptr<DT_CHAR>(h), 1, width * channel, fp);
            if (static_cast<DT_S32>(bytes) != width * channel)
            {
                std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(width * channel) + ") not match";
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                fclose(fp);
                return Mat();
            }
            offset += row_bytes;
        }
    }

    if (fp)
    {
        fclose(fp);
    }
    return mat;
}

Status WriteBmp(Context *ctx, const Mat &mat, const std::string &fname)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (!(mat.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid mat");
        return Status::ERROR;
    }

    if (mat.GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(ctx, "mat elem type should be u8");
        return Status::ERROR;
    }

    if (mat.GetSizes().m_channel != 1 && mat.GetSizes().m_channel != 3 && mat.GetSizes().m_channel != 4)
    {
        AURA_ADD_ERROR_STRING(ctx, "mat channel should be 1/3/4");
        return Status::ERROR;
    }

    BitMapFileHeader file_header;
    BitMapInfoHeader info_header;

    DT_S32 height  = mat.GetSizes().m_height;
    DT_S32 width   = mat.GetSizes().m_width;
    DT_S32 channel = mat.GetSizes().m_channel;
    DT_S32 pixels  = channel * 8;

    DT_BOOL is_8bit   = (8 == pixels) ? DT_TRUE : DT_FALSE;  // Is pixels equal to 8 bits ?
    DT_S32  row_bytes = (width * channel + 3) & (-4);        // Number of pixel data bytes per row, 4-byte aligned.
    DT_S32  data_size = row_bytes * height;                  // Iaura data size after padding
    DT_S32  pal_size  = is_8bit ? 256 * sizeof(Pixel32) : 0; // Palette vector size

    file_header.type        = 0x4D42; // 'BM'
    file_header.offset      = sizeof(BitMapFileHeader) + sizeof(BitMapInfoHeader) + pal_size;
    file_header.size        = file_header.offset + data_size;
    info_header.header_sz   = sizeof(BitMapInfoHeader);
    info_header.height      = height;
    info_header.width       = width;
    info_header.planes      = 1;
    info_header.pixels      = pixels; // color depth
    info_header.compression = 0;
    info_header.data_sz     = data_size;
    info_header.pixels_x    = 0;
    info_header.pixels_y    = 0;
    info_header.colors      = is_8bit ? 256 : 0;
    info_header.imp_colors  = is_8bit ? 256 : 0;

    FILE *fp = fopen(fname.c_str(), "wb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return Status::ERROR;
    }

    fwrite(&file_header, 1, sizeof(file_header), fp);
    fwrite(&info_header, 1, sizeof(info_header), fp);
    if (is_8bit)
    {
        // Create a color palette (256 colors, simple gray palette)
        std::vector<Pixel32> palette(256);
        for (DT_S32 i = 0; i < 256; ++i)
        {
            palette[i].blue  = i;
            palette[i].green = i;
            palette[i].red   = i;
            palette[i].alpha = 0;
        }
        fwrite(palette.data(), 1, palette.size() * sizeof(Pixel32), fp);
    }

    // Write pixel data, process row padding
    DT_CHAR padding[3] = {0, 0, 0}; // Up to 3 bytes of padding
    for (DT_S32 h = (height - 1); h >= 0; --h)
    {
        fwrite(mat.Ptr<DT_CHAR>(h), 1, width * channel, fp);
        if (row_bytes > width * channel)
        {
            fwrite(padding, 1, row_bytes - width * channel, fp);
        }
    }

    if (fp)
    {
        fclose(fp);
    }
    return Status::OK;
}

static ElemType GetYuvFormatElemType(IauraFormat fmt)
{
    switch (fmt)
    {
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        case IauraFormat::YU12:
        case IauraFormat::YV12:
        case IauraFormat::I422:
        case IauraFormat::I444:
        {
            return ElemType::U8;
        }

        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            return ElemType::U16;
        }

        default:
            break;
    }

    return ElemType::INVALID;
}

std::vector<Mat> ReadYuv(Context *ctx, const std::string &fname, IauraFormat fmt, const Sizes &sizes, DT_S32 mem_type, const Sizes &strides)
{
    if (DT_NULL == ctx)
    {
        return {};
    }

    ElemType elem_type = GetYuvFormatElemType(fmt);
    if (ElemType::INVALID == elem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "get elem type failed");
        return {};
    }

    std::vector<Sizes3> mat_sizes;
    std::vector<Sizes>  mat_strides;
    if (GetIauraFormatSize(fmt, sizes, strides, mat_sizes, mat_strides) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "get iaura format size failed");
        return {};
    }

    std::vector<Mat> mats;
    for (size_t i = 0; i < mat_sizes.size(); i++)
    {
        Mat mat(ctx, elem_type, mat_sizes[i], mem_type, mat_strides[i]);
        if (!(mat.IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid mat");
            return {};
        }
        mats.emplace_back(mat);
    }

    DT_S32 buffer_valid_size = 0;
    for (size_t i = 0; i < mats.size(); i++)
    {
        DT_S32 row_bytes = mats[i].GetSizes().m_width * mats[i].GetSizes().m_channel * ElemTypeSize(mats[i].GetElemType());
        buffer_valid_size += mats[i].GetSizes().m_height * row_bytes;
    }

    FILE *fp = fopen(fname.c_str(), "rb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return {};
    }

    fseek(fp, 0, SEEK_END);
    DT_S32 file_length = ftell(fp);

    if (file_length < buffer_valid_size)
    {
        std::string info = "file size(" + std::to_string(file_length) + ") must be >= buffer size(" + std::to_string(buffer_valid_size) + ")";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        fclose(fp);
        return {};
    }

    fseek(fp, 0, SEEK_SET);

    for (size_t i = 0; i < mats.size(); i++)
    {
        DT_S32 row_bytes = mats[i].GetSizes().m_width * mats[i].GetSizes().m_channel * ElemTypeSize(mats[i].GetElemType());
        for (DT_S32 h = 0; h < mats[i].GetSizes().m_height; h++)
        {
            size_t bytes = fread(mats[i].Ptr<DT_CHAR>(h), 1, row_bytes, fp);
            if (static_cast<DT_S32>(bytes) != row_bytes)
            {
                std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(row_bytes) + ") not match";
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                fclose(fp);
                return {};
            }
        }
    }

    if (fp)
    {
        fclose(fp);
    }
    return mats;
}

static Status CheckMatsNum(Context *ctx, const std::vector<Mat> &mats, IauraFormat format)
{
    Status ret = Status::ERROR;

    DT_S32 len = mats.size();
    switch (format)
    {
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            ret = (2 == len) ? Status::OK : Status::ERROR;
            break;
        }

        case IauraFormat::YU12:
        case IauraFormat::YV12:
        case IauraFormat::I422:
        case IauraFormat::I444:
        {
            ret = (3 == len) ? Status::OK : Status::ERROR;
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported iaura format");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status CheckMatsParam(Context *ctx, const std::vector<Mat> &mats, IauraFormat format)
{
    Status ret = Status::ERROR;

    DT_S32 len = mats.size();
    for (DT_S32 i = 0; i < len; i++)
    {
        if (!(mats[i].IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid mat");
            goto EIXT;
        }
    }

#define CHECK_MATS_ELEM_TYPE(type)                                     \
    for (DT_S32 i = 0; i < len; i++)                                   \
    {                                                                  \
        if (mats[i].GetElemType() != type)                             \
        {                                                              \
            AURA_ADD_ERROR_STRING(ctx, "mats elem type is not match"); \
            goto EIXT;                                                 \
        }                                                              \
    }

    switch (format)
    {
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        {
            CHECK_MATS_ELEM_TYPE(ElemType::U8)
            if (mats[0].GetSizes() * Sizes3(1, 1, 2) != mats[1].GetSizes() * Sizes3(2, 2, 1) ||
                mats[0].GetSizes().m_channel != 1)
            {
                AURA_ADD_ERROR_STRING(ctx, "mats size is not match");
                goto EIXT;
            }
            break;
        }

        case IauraFormat::YU12:
        case IauraFormat::YV12:
        {
            CHECK_MATS_ELEM_TYPE(ElemType::U8)
            if (mats[0].GetSizes() != mats[1].GetSizes() * Sizes3(2, 2, 1) ||
                mats[0].GetSizes() != mats[2].GetSizes() * Sizes3(2, 2, 1) ||
                mats[0].GetSizes().m_channel != 1)
            {
                AURA_ADD_ERROR_STRING(ctx, "mats size is not match");
                goto EIXT;
            }
            break;
        }

        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            CHECK_MATS_ELEM_TYPE(ElemType::U16)
            if (mats[0].GetSizes() * Sizes3(1, 1, 2) != mats[1].GetSizes() * Sizes3(2, 2, 1) ||
                mats[0].GetSizes().m_channel != 1)
            {
                AURA_ADD_ERROR_STRING(ctx, "mats size is not match");
                goto EIXT;
            }
            break;
        }

        case IauraFormat::I422:
        {
            CHECK_MATS_ELEM_TYPE(ElemType::U8)
            if (mats[0].GetSizes() != mats[1].GetSizes() * Sizes3(1, 2, 1) ||
                mats[0].GetSizes() != mats[2].GetSizes() * Sizes3(1, 2, 1) ||
                mats[0].GetSizes().m_channel != 1)
            {
                AURA_ADD_ERROR_STRING(ctx, "mats size is not match");
                goto EIXT;
            }
            break;
        }

        case IauraFormat::I444:
        {
            CHECK_MATS_ELEM_TYPE(ElemType::U8)
            if (mats[0].GetSizes() != mats[1].GetSizes() ||
                mats[0].GetSizes() != mats[2].GetSizes() ||
                mats[0].GetSizes().m_channel != 1)
            {
                AURA_ADD_ERROR_STRING(ctx, "mats size is not match");
                goto EIXT;
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported iaura format");
            goto EIXT;
        }
    }

#undef CHECK_MATS_ELEM_TYPE

    ret = Status::OK;
EIXT:
    AURA_RETURN(ctx, ret);
}

Status WriteYuv(Context *ctx, const std::vector<Mat> &mats, IauraFormat fmt, const std::string &fname)
{
    Status ret = Status::ERROR;

    if (DT_NULL == ctx)
    {
        return ret;
    }

    if (CheckMatsNum(ctx, mats, fmt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "mats number does not match iaura format");
        return ret;
    }

    if (CheckMatsParam(ctx, mats, fmt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "mats param does not match iaura format");
        return ret;
    }

    FILE *fp = fopen(fname.c_str(), "wb");
    if (DT_NULL == fp)
    {
        std::string info = "open " + fname + " failed";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return ret;
    }

    for (size_t i = 0; i < mats.size(); i++)
    {
        DT_S32 row_bytes = mats[i].GetSizes().m_width * mats[i].GetSizes().m_channel * ElemTypeSize(mats[i].GetElemType());
        for (DT_S32 h = 0; h < mats[i].GetSizes().m_height; h++)
        {
            size_t bytes = fwrite(mats[i].Ptr<DT_CHAR>(h), 1, row_bytes, fp);
            if (static_cast<DT_S32>(bytes) != row_bytes)
            {
                std::string info = "fwrite size(" + std::to_string(bytes) + "," + std::to_string(row_bytes) + ") not match";
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }
        }
    }

    ret = Status::OK;
EXIT:
    if (fp)
    {
        fclose(fp);
    }
    return ret;
}

std::vector<Mat> GetYuvIaura(Context *ctx, const Mat &mat, IauraFormat fmt, DT_BOOL is_deep_copy)
{
    if (DT_NULL == ctx)
    {
        return {};
    }

    DT_S32 height  = mat.GetSizes().m_height;
    DT_S32 width   = mat.GetSizes().m_width;
    DT_S32 channel = mat.GetSizes().m_channel;

    if (channel != 1)
    {
        std::string info = "mat channel must is 1, but now is " + std::to_string(channel);
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return {};
    }

    std::vector<Mat> mats;
    switch (fmt)
    {
        case IauraFormat::NV12:
        case IauraFormat::NV21:
        case IauraFormat::P010:
        case IauraFormat::P016:
        {
            Mat y_mat = mat.Roi(Rect(0, 0, width, height * 2 / 3));
            if (!y_mat.IsValid())
            {
                std::string info = "y_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat uv_mat = mat.Roi(Rect(0, height * 2 / 3, width, height / 3));
            if (!uv_mat.IsValid())
            {
                std::string info = "uv_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            if (uv_mat.Reshape(Sizes3(height / 3, width / 2, 2)) != Status::OK)
            {
                std::string info = "uv_mat Reshape failed, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            mats.push_back(y_mat);
            mats.push_back(uv_mat);
            break;
        }
        case IauraFormat::YU12:
        case IauraFormat::YV12:
        {
            Mat y_mat = mat.Roi(Rect(0, 0, width, height * 2 / 3));
            if (!y_mat.IsValid())
            {
                std::string info = "y_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat u_or_v_mat = mat.Roi(Rect(0, height * 2 / 3, width, height / 6));
            if (!u_or_v_mat.IsValid())
            {
                std::string info = "u_or_v_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            if (u_or_v_mat.Reshape(Sizes3(height / 3, width / 2, 1)) != Status::OK)
            {
                std::string info = "u_or_v_mat Reshape failed, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat v_or_u_mat = mat.Roi(Rect(0, height * 5 / 6, width, height / 6));
            if (!v_or_u_mat.IsValid())
            {
                std::string info = "v_or_u_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            if (v_or_u_mat.Reshape(Sizes3(height / 3, width / 2, 1)) != Status::OK)
            {
                std::string info = "v_or_u_mat Reshape failed, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            mats.emplace_back(y_mat);
            mats.emplace_back(u_or_v_mat);
            mats.emplace_back(v_or_u_mat);
            break;
        }
        case IauraFormat::I422:
        {
            Mat y_mat = mat.Roi(Rect(0, 0, width, height / 2));
            if (!y_mat.IsValid())
            {
                std::string info = "y_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat u_mat = mat.Roi(Rect(0, height / 2, width, height / 4));
            if (!u_mat.IsValid())
            {
                std::string info = "u_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            if (u_mat.Reshape(Sizes3(height / 2, width / 2, 1)) != Status::OK)
            {
                std::string info = "u_mat Reshape failed, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat v_mat = mat.Roi(Rect(0, height * 3 / 4, width, height / 4));
            if (!v_mat.IsValid())
            {
                std::string info = "v_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            if (v_mat.Reshape(Sizes3(height / 2, width / 2, 1)) != Status::OK)
            {
                std::string info = "v_mat Reshape failed, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            mats.emplace_back(y_mat);
            mats.emplace_back(u_mat);
            mats.emplace_back(v_mat);
            break;
        }
        case IauraFormat::I444:
        {
            Mat y_mat = mat.Roi(Rect(0, 0, width, height / 3));
            if (!y_mat.IsValid())
            {
                std::string info = "y_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat u_mat = mat.Roi(Rect(0, height / 3, width, height / 3));
            if (!u_mat.IsValid())
            {
                std::string info = "u_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            Mat v_mat = mat.Roi(Rect(0, height * 2 / 3, width, height / 3));
            if (!v_mat.IsValid())
            {
                std::string info = "v_mat is invalid, mat size is " + mat.GetSizes().ToString();
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            mats.emplace_back(y_mat);
            mats.emplace_back(u_mat);
            mats.emplace_back(v_mat);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, IauraFormatToString(fmt).c_str());
            break;
        }
    }

    if (is_deep_copy)
    {
        for (DT_U32 i = 0; i < mats.size(); i++)
        {
            mats[i] = mats[i].Clone();
        }
    }

EXIT:
    return mats;
}

} // namespace aura