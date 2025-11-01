#include "aura/tools/unit_test/mat_factory.hpp"
#include "aura/runtime/logger.h"

#include <sys/stat.h>

#include <fstream>
#include <type_traits>

namespace aura
{

template<typename Tp>
static AURA_VOID FillMatWithRandom_(std::mt19937_64 &engine, Mat &mat, MI_F32 min, MI_F32 max)
{
    using Distribution = typename std::conditional<is_integral<Tp>::value,
                                                   typename std::conditional<is_signed<Tp>::value,
                                                                             std::uniform_int_distribution<MI_S32>,
                                                                             std::uniform_int_distribution<MI_U32>>::type,
                                                   std::uniform_real_distribution<MI_F32>>::type;

    Tp *data = reinterpret_cast<Tp*>(mat.GetData());
    MI_S32 n = mat.GetTotalBytes() / sizeof(Tp);

    MI_F32 min_type = SaturateCast<MI_F32>(std::numeric_limits<Tp>::lowest());
    MI_F32 max_type = SaturateCast<MI_F32>(std::numeric_limits<Tp>::max());

    MI_F32 min_val = Max(min_type, min);
    MI_F32 max_val = Min(max_type, max);

    Distribution distribution(min_val, max_val);

    for (MI_S32 i = 0; i < n; i++)
    {
        data[i] = SaturateCast<Tp>(distribution(engine));
    }
}

template<typename Tp0, typename Tp1>
static AURA_VOID ResizeConvertTo(const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta)
{
    MI_F32 scale_x = static_cast<MI_F32>(src.GetSizes().m_width)  / static_cast<MI_F32>(dst.GetSizes().m_width);
    MI_F32 scale_y = static_cast<MI_F32>(src.GetSizes().m_height) / static_cast<MI_F32>(dst.GetSizes().m_height);

    for (MI_S32 y = 0; y < dst.GetSizes().m_height; y++)
    {
        for (MI_S32 x = 0; x < dst.GetSizes().m_width; x++)
        {
            MI_S32 src_x = Max(Min(Round(x * scale_x), src.GetSizes().m_width  - 1), (MI_S32)0);
            MI_S32 src_y = Max(Min(Round(y * scale_y), src.GetSizes().m_height - 1), (MI_S32)0);

            for (MI_S32 c = 0; c < dst.GetSizes().m_channel; c++)
            {
                Tp0 src_pt = src.At<Tp0>(src_y, src_x, c);
                Tp1 cvt_pt = SaturateCast<Tp1>(src_pt * alpha + beta);
                dst.At<Tp1>(y, x, c) = cvt_pt;
            }
        }
    }
}

struct ResizeConvertToImpl
{
    ElemType src_type;
    ElemType dst_type;
    std::function<AURA_VOID(const Mat&, Mat&, MI_F32, MI_F32)> func;
    std::string info;
};

static const ResizeConvertToImpl g_resize_convert_to_tbl[] =
{
    //U8
    {ElemType::U8, ElemType::U8,  ResizeConvertTo<MI_U8, MI_U8>,  "ConvertToU8ToU8" },
    {ElemType::U8, ElemType::S8,  ResizeConvertTo<MI_U8, MI_S8>,  "ConvertToU8ToS8" },
    {ElemType::U8, ElemType::U16, ResizeConvertTo<MI_U8, MI_U16>, "ConvertToU8ToU16"},
    {ElemType::U8, ElemType::S16, ResizeConvertTo<MI_U8, MI_S16>, "ConvertToU8ToS16"},
    {ElemType::U8, ElemType::U32, ResizeConvertTo<MI_U8, MI_U32>, "ConvertToU8ToU32"},
    {ElemType::U8, ElemType::S32, ResizeConvertTo<MI_U8, MI_S32>, "ConvertToU8ToS32"},
    {ElemType::U8, ElemType::F32, ResizeConvertTo<MI_U8, MI_F32>, "ConvertToU8ToF32"},
    {ElemType::U8, ElemType::F64, ResizeConvertTo<MI_U8, MI_F64>, "ConvertToU8ToF64"},

    //S8
    {ElemType::S8, ElemType::U8,  ResizeConvertTo<MI_S8, MI_U8>,  "ConvertToS8ToU8" },
    {ElemType::S8, ElemType::S8,  ResizeConvertTo<MI_S8, MI_S8>,  "ConvertToS8ToS8" },
    {ElemType::S8, ElemType::U16, ResizeConvertTo<MI_S8, MI_U16>, "ConvertToS8ToU16"},
    {ElemType::S8, ElemType::S16, ResizeConvertTo<MI_S8, MI_S16>, "ConvertToS8ToS16"},
    {ElemType::S8, ElemType::U32, ResizeConvertTo<MI_S8, MI_U32>, "ConvertToS8ToU32"},
    {ElemType::S8, ElemType::S32, ResizeConvertTo<MI_S8, MI_S32>, "ConvertToS8ToS32"},
    {ElemType::S8, ElemType::F32, ResizeConvertTo<MI_S8, MI_F32>, "ConvertToS8ToF32"},
    {ElemType::S8, ElemType::F64, ResizeConvertTo<MI_S8, MI_F64>, "ConvertToS8ToF64"},

    //U16
    {ElemType::U16, ElemType::U8,  ResizeConvertTo<MI_U16, MI_U8>,  "ConvertToU16ToU8" },
    {ElemType::U16, ElemType::S8,  ResizeConvertTo<MI_U16, MI_S8>,  "ConvertToU16ToS8" },
    {ElemType::U16, ElemType::U16, ResizeConvertTo<MI_U16, MI_U16>, "ConvertToU16ToU16"},
    {ElemType::U16, ElemType::S16, ResizeConvertTo<MI_U16, MI_S16>, "ConvertToU16ToS16"},
    {ElemType::U16, ElemType::U32, ResizeConvertTo<MI_U16, MI_U32>, "ConvertToU16ToU32"},
    {ElemType::U16, ElemType::S32, ResizeConvertTo<MI_U16, MI_S32>, "ConvertToU16ToS32"},
    {ElemType::U16, ElemType::F32, ResizeConvertTo<MI_U16, MI_F32>, "ConvertToU16ToF32"},
    {ElemType::U16, ElemType::F64, ResizeConvertTo<MI_U16, MI_F64>, "ConvertToU16ToF64"},

    //S16
    {ElemType::S16, ElemType::U8,  ResizeConvertTo<MI_S16, MI_U8>,  "ConvertToS16ToU8" },
    {ElemType::S16, ElemType::S8,  ResizeConvertTo<MI_S16, MI_S8>,  "ConvertToS16ToS8" },
    {ElemType::S16, ElemType::U16, ResizeConvertTo<MI_S16, MI_U16>, "ConvertToS16ToU16"},
    {ElemType::S16, ElemType::S16, ResizeConvertTo<MI_S16, MI_S16>, "ConvertToS16ToS16"},
    {ElemType::S16, ElemType::U32, ResizeConvertTo<MI_S16, MI_U32>, "ConvertToS16ToU32"},
    {ElemType::S16, ElemType::S32, ResizeConvertTo<MI_S16, MI_S32>, "ConvertToS16ToS32"},
    {ElemType::S16, ElemType::F32, ResizeConvertTo<MI_S16, MI_F32>, "ConvertToS16ToF32"},
    {ElemType::S16, ElemType::F64, ResizeConvertTo<MI_S16, MI_F64>, "ConvertToS16ToF64"},

    //U32
    {ElemType::U32, ElemType::U8,  ResizeConvertTo<MI_U32, MI_U8>,  "ConvertToU32ToU8" },
    {ElemType::U32, ElemType::S8,  ResizeConvertTo<MI_U32, MI_S8>,  "ConvertToU32ToS8" },
    {ElemType::U32, ElemType::U16, ResizeConvertTo<MI_U32, MI_U16>, "ConvertToU32ToU16"},
    {ElemType::U32, ElemType::S16, ResizeConvertTo<MI_U32, MI_S16>, "ConvertToU32ToS16"},
    {ElemType::U32, ElemType::U32, ResizeConvertTo<MI_U32, MI_U32>, "ConvertToU32ToU32"},
    {ElemType::U32, ElemType::S32, ResizeConvertTo<MI_U32, MI_S32>, "ConvertToU32ToS32"},
    {ElemType::U32, ElemType::F32, ResizeConvertTo<MI_U32, MI_F32>, "ConvertToU32ToF32"},
    {ElemType::U32, ElemType::F64, ResizeConvertTo<MI_U32, MI_F64>, "ConvertToU32ToF64"},

    //S32
    {ElemType::S32, ElemType::U8,  ResizeConvertTo<MI_S32, MI_U8>,  "ConvertToS32ToU8" },
    {ElemType::S32, ElemType::S8,  ResizeConvertTo<MI_S32, MI_S8>,  "ConvertToS32ToS8" },
    {ElemType::S32, ElemType::U16, ResizeConvertTo<MI_S32, MI_U16>, "ConvertToS32ToU16"},
    {ElemType::S32, ElemType::S16, ResizeConvertTo<MI_S32, MI_S16>, "ConvertToS32ToS16"},
    {ElemType::S32, ElemType::U32, ResizeConvertTo<MI_S32, MI_U32>, "ConvertToS32ToU32"},
    {ElemType::S32, ElemType::S32, ResizeConvertTo<MI_S32, MI_S32>, "ConvertToS32ToS32"},
    {ElemType::S32, ElemType::F32, ResizeConvertTo<MI_S32, MI_F32>, "ConvertToS32ToF32"},
    {ElemType::S32, ElemType::F64, ResizeConvertTo<MI_S32, MI_F64>, "ConvertToS32ToF64"},

    //F32
    {ElemType::F32, ElemType::U8,  ResizeConvertTo<MI_F32, MI_U8>,  "ConvertToF32ToU8" },
    {ElemType::F32, ElemType::S8,  ResizeConvertTo<MI_F32, MI_S8>,  "ConvertToF32ToS8" },
    {ElemType::F32, ElemType::U16, ResizeConvertTo<MI_F32, MI_U16>, "ConvertToF32ToU16"},
    {ElemType::F32, ElemType::S16, ResizeConvertTo<MI_F32, MI_S16>, "ConvertToF32ToS16"},
    {ElemType::F32, ElemType::U32, ResizeConvertTo<MI_F32, MI_U32>, "ConvertToF32ToU32"},
    {ElemType::F32, ElemType::S32, ResizeConvertTo<MI_F32, MI_S32>, "ConvertToF32ToS32"},
    {ElemType::F32, ElemType::F32, ResizeConvertTo<MI_F32, MI_F32>, "ConvertToF32ToF32"},
    {ElemType::F32, ElemType::F64, ResizeConvertTo<MI_F32, MI_F64>, "ConvertToF32ToF64"},

    //F64
    {ElemType::F64, ElemType::U8,  ResizeConvertTo<MI_F64, MI_U8>,  "ConvertToF64ToU8" },
    {ElemType::F64, ElemType::S8,  ResizeConvertTo<MI_F64, MI_S8>,  "ConvertToF64ToS8" },
    {ElemType::F64, ElemType::U16, ResizeConvertTo<MI_F64, MI_U16>, "ConvertToF64ToU16"},
    {ElemType::F64, ElemType::S16, ResizeConvertTo<MI_F64, MI_S16>, "ConvertToF64ToS16"},
    {ElemType::F64, ElemType::U32, ResizeConvertTo<MI_F64, MI_U32>, "ConvertToF64ToU32"},
    {ElemType::F64, ElemType::S32, ResizeConvertTo<MI_F64, MI_S32>, "ConvertToF64ToS32"},
    {ElemType::F64, ElemType::F32, ResizeConvertTo<MI_F64, MI_F32>, "ConvertToF64ToF32"},
    {ElemType::F64, ElemType::F64, ResizeConvertTo<MI_F64, MI_F64>, "ConvertToF64ToF64"},

#if defined(AURA_BUILD_HOST)
    {ElemType::U8, ElemType::F16, ResizeConvertTo<MI_U8, MI_F16>, "ConvertToU8ToF16"},
    {ElemType::S8, ElemType::F16, ResizeConvertTo<MI_S8, MI_F16>, "ConvertToS8ToF16"},
    {ElemType::U16, ElemType::F16, ResizeConvertTo<MI_U16, MI_F16>, "ConvertToU16ToF16"},
    {ElemType::S16, ElemType::F16, ResizeConvertTo<MI_S16, MI_F16>, "ConvertToS16ToF16"},
    {ElemType::U32, ElemType::F16, ResizeConvertTo<MI_U32, MI_F16>, "ConvertToU32ToF16"},
    {ElemType::S32, ElemType::F16, ResizeConvertTo<MI_S32, MI_F16>, "ConvertToS32ToF16"},
    {ElemType::F32, ElemType::F16, ResizeConvertTo<MI_F32, MI_F16>, "ConvertToF32ToF16"},
    //F16
    {ElemType::F16, ElemType::U8,  ResizeConvertTo<MI_F16, MI_U8>,  "ConvertToF16ToU8" },
    {ElemType::F16, ElemType::S8,  ResizeConvertTo<MI_F16, MI_S8>,  "ConvertToF16ToS8" },
    {ElemType::F16, ElemType::U16, ResizeConvertTo<MI_F16, MI_U16>, "ConvertToF16ToU16"},
    {ElemType::F16, ElemType::S16, ResizeConvertTo<MI_F16, MI_S16>, "ConvertToF16ToS16"},
    {ElemType::F16, ElemType::U32, ResizeConvertTo<MI_F16, MI_U32>, "ConvertToF16ToU32"},
    {ElemType::F16, ElemType::S32, ResizeConvertTo<MI_F16, MI_S32>, "ConvertToF16ToS32"},
    {ElemType::F16, ElemType::F16, ResizeConvertTo<MI_F16, MI_F16>, "ConvertToF16ToF16"},
    {ElemType::F16, ElemType::F32, ResizeConvertTo<MI_F16, MI_F32>, "ConvertToF16ToF32"},
    {ElemType::F16, ElemType::F64, ResizeConvertTo<MI_F16, MI_F64>, "ConvertToF16ToF64"},
    {ElemType::F64, ElemType::F16, ResizeConvertTo<MI_F64, MI_F16>, "ConvertToF64ToF16"},
#endif // AURA_BUILD_HOST

};

static const ResizeConvertToImpl* GetResizeConvertToFunc(const Mat &mat0, const Mat &mat1)
{
    const MI_S32 n_func = sizeof(g_resize_convert_to_tbl) / sizeof(g_resize_convert_to_tbl[0]);
    const ElemType type0 = mat0.GetElemType();
    const ElemType type1 = mat1.GetElemType();

    for (MI_S32 i = 0; i < n_func; i++)
    {
        if ((type0 == g_resize_convert_to_tbl[i].src_type) &&
            (type1 == g_resize_convert_to_tbl[i].dst_type))
        {
            return g_resize_convert_to_tbl + i;
        }
    }

    return MI_NULL;
}

// if mat1 and mat2 point to same memory block, return true.
static MI_BOOL CompareMatData(const Mat &mat1, const Mat &mat2)
{
    if (mat1.GetData() != mat2.GetData())
    {
        return MI_FALSE;
    }
    return MI_TRUE;
}

static MI_BOOL operator==(const MatDesc &desc0, const MatDesc &desc1)
{
    // if is the same mat type
    if (desc0.type != desc1.type)
    {
        return MI_FALSE;
    }
    // if is the same elem type
    if (desc0.elem_type != desc1.elem_type)
    {
        return MI_FALSE;
    }
    // if is the same mem type
    if (desc0.mem_type != desc1.mem_type)
    {
        return MI_FALSE;
    }
    if (desc0.sizes != desc1.sizes)
    {
        return MI_FALSE;
    }
    if (desc0.strides != desc1.strides)
    {
        return MI_FALSE;
    }

    // compare type info
    switch (desc0.type)
    {
        case MatDesc::Type::FILE:
        {
            if (strcmp(desc0.param.file_path, desc1.param.file_path))
            {
                return MI_FALSE;
            }
            break;
        }
        case MatDesc::Type::RAND:
        {
            if ((desc0.param.rand_range.min != desc1.param.rand_range.min) ||
                (desc0.param.rand_range.max != desc1.param.rand_range.max))
            {
                return MI_FALSE;
            }
            break;
        }
        case MatDesc::Type::DERIVED:
        {
            if ((desc0.param.derived_param.alpha != desc1.param.derived_param.alpha) ||
                (desc0.param.derived_param.beta  != desc1.param.derived_param.beta)  ||
                (desc0.param.derived_param.base  != desc1.param.derived_param.base))
            {
                return MI_FALSE;
            }
            break;
        }
        case MatDesc::Type::EMPTY:
        {
            break;
        }
        default:
        {
            return MI_FALSE;
        }
    }

    return MI_TRUE;
}

Status MatFactory::LoadBaseMat(const std::string &file_path, const ElemType &elem_type, const Sizes3 &sizes, MI_S32 mem_type, const Sizes &strides)
{
    MatDesc desc;
    desc.type = MatDesc::Type::FILE;
    strncpy(desc.param.file_path, file_path.c_str(), MAT_FACTORY_MAX_PATH - 1);
    desc.elem_type = elem_type;
    desc.mem_type  = mem_type;
    desc.sizes     = sizes;
    desc.strides   = strides;

    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        for (auto &it : m_base_list)
        {
            // if mat is already in the base list, return
            if (desc == it.desc)
            {
                return Status::OK;
            }
        }
    }

    // if matched mat is not found, create a new one
    Mat mat = CreateMat(desc);
    // if new mat is valid, put it into list
    if (mat.IsValid())
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        m_base_list.push_back({desc, mat, MI_TRUE});
        return Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "create mat from file failed");
        return Status::ERROR;
    }
}

Mat* MatFactory::FindBaseMat(const MI_S32 channel, const std::string &file_path)
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);

    for (auto &it : m_base_list)
    {
        if (!file_path.empty())
        {
            // if file path dismatches
            if (strcmp(it.desc.param.file_path, file_path.c_str()))
            {
                continue;
            }
        }

        if (channel == it.mat.GetSizes().m_channel)
        {
            // mat in base list is always available!
            return &it.mat;
        }
    }

    return MI_NULL;
}

Mat MatFactory::FindDynamicMat(const MatDesc &desc)
{
    // first loop, looking for available matched mat
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end();)
        {
            // find matched one and available
            if ((MI_TRUE == it->available) && desc == it->desc)
            {
                it->available = MI_FALSE;
                MatInfo info;
                std::swap(info, *it);
                this->m_dynamic_list.erase(it);
                this->m_dynamic_list.push_back(info);
                return this->m_dynamic_list.back().mat;
            }
            it++;
        }
    }

    // if RANDOM or EMPTY type, generate a new mat
    if (MatDesc::Type::RAND == desc.type || MatDesc::Type::EMPTY == desc.type)
    {
        Mat mat = CreateMat(desc);
        if (mat.IsValid())
        {
            std::lock_guard<std::mutex> guard(this->m_handle_lock);

            this->m_total_mem += mat.GetTotalBytes();
            this->m_dynamic_list.push_back({desc, mat, MI_FALSE});
        }
        else
        {
            AURA_ADD_ERROR_STRING(this->m_ctx, "CreateMat failed");
        }

        return mat;
    }

    // second loop, looking for matched non-available mat, clone it and return
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end();)
        {
            if (desc == it->desc)
            {
                MatInfo info = {it->desc, it->mat.Clone(), MI_FALSE};
                this->m_dynamic_list.push_back(info);
                this->m_total_mem += it->mat.GetTotalBytes(); // how to add lock in this place

                return this->m_dynamic_list.back().mat;
            }
            it++;
        }
    }

    return Mat(); // no matched mat
}

Mat MatFactory::CreateMat(const MatDesc &desc)
{
    Status ret = Status::ERROR;

    Mat mat = Mat(this->m_ctx, desc.elem_type, desc.sizes, desc.mem_type, desc.strides);
    switch (desc.type)
    {
        case MatDesc::Type::FILE:
        {
            ret = CreateMat(mat, desc.param.file_path);
            break;
        }
        case MatDesc::Type::RAND:
        {
            ret = CreateMat(mat, desc.param.rand_range.min, desc.param.rand_range.max);
            break;
        }
        case MatDesc::Type::DERIVED:
        {
            Mat base_mat = *(desc.param.derived_param.base);
            ret = CreateMat(base_mat, mat, desc.param.derived_param.alpha, desc.param.derived_param.beta);
            break;
        }
        case MatDesc::Type::EMPTY:
        {
            ret = Status::OK;
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(this->m_ctx, "unsupported mat desc type");
            return Mat();
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "create mat failed");
        return Mat();
    }

    return mat;
}

Mat MatFactory::GetFileMat(const std::string &file_path, const ElemType &elem_type, const Sizes3 &sizes, MI_S32 mem_type, const Sizes &strides)
{
    MatDesc desc;
    desc.type = MatDesc::Type::FILE;
    strncpy(desc.param.file_path, file_path.c_str(), MAT_FACTORY_MAX_PATH - 1);
    desc.elem_type = elem_type;
    desc.mem_type  = mem_type;
    desc.sizes     = sizes;
    desc.strides   = strides;

    // check if already in dynamic list
    Mat mat_dynamic = FindDynamicMat(desc);
    if (mat_dynamic.IsValid())
    {
        return mat_dynamic;
    }

    // check if already in base list
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        for (auto &it : this->m_base_list)
        {
            // if find matched mat
            if (desc == it.desc)
            {
                Mat mat_clone = it.mat.Clone();
                this->m_dynamic_list.push_back({desc, mat_clone, MI_FALSE});
                return mat_clone;
            }
        }
    }

    // if not found, create a new one
    Mat mat_new = CreateMat(desc);
    // if new mat is valid, put it into list
    if (mat_new.IsValid())
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        this->m_dynamic_list.push_back({desc, mat_new, MI_FALSE});
        this->m_total_mem += mat_new.GetTotalBytes();
    }
    else
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "GetFileMat failed");
    }
    CheckTotalMemory();

    return mat_new;
}

Mat MatFactory::GetDerivedMat(MI_F32 alpha, MI_F32 beta, const ElemType &elem_type, const Sizes3 &sizes,
                              MI_S32 mem_type, const Sizes &strides, const std::string &file_path)
{
    // if it exists matched base mat
    Mat *base_mat_ptr = MI_NULL;
    base_mat_ptr = FindBaseMat(sizes.m_channel, file_path);

    if (MI_NULL == base_mat_ptr)
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "cannot find matched base mat");
        return Mat();
    }

    // search in dynamic list
    MatDesc desc;
    desc.type                      = MatDesc::Type::DERIVED;
    desc.elem_type                 = elem_type;
    desc.mem_type                  = mem_type;
    desc.sizes                     = sizes;
    desc.strides                   = strides;
    desc.param.derived_param.base  = base_mat_ptr;
    desc.param.derived_param.alpha = alpha;
    desc.param.derived_param.beta  = beta;

    Mat mat = FindDynamicMat(desc);
    if (mat.IsValid())
    {
        return mat;
    }

    // if not found, create a new one
    mat = CreateMat(desc);
    if (mat.IsValid())
    {
        std::lock_guard<std::mutex> guard(this->m_handle_lock);
        this->m_total_mem += mat.GetTotalBytes();
        this->m_dynamic_list.push_back({desc, mat, MI_FALSE});
    }
    else
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "GetDerivedMat failed");
        return Mat();
    }

    CheckTotalMemory();
    return mat;
}

Mat MatFactory::GetRandomMat(MI_F32 min, MI_F32 max, const ElemType &elem_type, const Sizes3 &sizes, MI_S32 mem_type, const Sizes &strides)
{
    MatDesc desc;
    desc.type             = MatDesc::Type::RAND;
    desc.param.rand_range = {min, max};
    desc.elem_type        = elem_type;
    desc.mem_type         = mem_type;
    desc.sizes            = sizes;
    desc.strides          = strides;

    Mat mat = FindDynamicMat(desc);
    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "GetRandomMat failed");
    }
    CheckTotalMemory();

    return mat;
}

Mat MatFactory::GetEmptyMat(const ElemType &elem_type, const Sizes3 &sizes, MI_S32 mem_type, const Sizes &strides)
{
    MatDesc desc;
    desc.type      = MatDesc::Type::EMPTY;
    desc.elem_type = elem_type;
    desc.mem_type  = mem_type;
    desc.sizes     = sizes;
    desc.strides   = strides;

    Mat mat = FindDynamicMat(desc);
    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "GetDerivedMat failed");
    }

    CheckTotalMemory();
    return mat;
}

AURA_VOID MatFactory::PutMats(Mat &mat)
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);

    for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end();)
    {
        if (CompareMatData(it->mat, mat))
        {
            it->available = MI_TRUE;
            return;
        }
        it++;
    }
}

AURA_VOID MatFactory::PutAllMats()
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);
    for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end(); ++it)
    {
        it->available = MI_TRUE;
    }
}

AURA_VOID MatFactory::Clear()
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);
    this->m_dynamic_list.clear();
    this->m_base_list.clear();
}

Status MatFactory::CreateMat(Mat &mat, const std::string &file)
{
    if (file.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "file is empty");
        return Status::ERROR;
    }

    if (!mat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat is invalid");
        return Status::ERROR;
    }

    FILE *fp = fopen(file.c_str(), "rb");
    if (MI_NULL == fp)
    {
        std::string info = "file " + file + " open failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }

    fseek(fp, 0, SEEK_END);

    MI_S32 file_length = ftell(fp);

    if (file_length < mat.GetTotalBytes())
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "file length is smaller than declared mat data length");
        return Status::ERROR;
    }

    fseek(fp, 0, SEEK_SET);

    size_t bytes = fread(mat.GetData(), 1, mat.GetTotalBytes(), fp);

    if (static_cast<MI_S32>(bytes) != mat.GetTotalBytes())
    {
        std::string info = "fread size(" + std::to_string(bytes) + "," + std::to_string(mat.GetTotalBytes()) + ") not match";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

EXIT:
    if (fp)
    {
        fclose(fp);
    }

    return Status::OK;
}

Status MatFactory::CreateMat(Mat &new_mat, MI_F32 min, MI_F32 max)
{
    // fill mat with random data
    if (ElemType::U8 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_U8>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::S8 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_S8>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::U16 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_U16>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::S16 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_S16>(m_rand_engine, new_mat, min, max);
    }
#if defined(AURA_BUILD_HOST)
    else if (ElemType::F16 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_F16>(m_rand_engine, new_mat, min, max);
    }
#endif // AURA_BUILD_HOST
    else if (ElemType::U32 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_U32>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::S32 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_S32>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::F32 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_F32>(m_rand_engine, new_mat, min, max);
    }
    else if (ElemType::F64 == new_mat.GetElemType())
    {
        FillMatWithRandom_<MI_F64>(m_rand_engine, new_mat, min, max);
    }
    else
    {
        AURA_ADD_ERROR_STRING(this->m_ctx, "Unsupported mat format");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MatFactory::CreateMat(const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta)
{
    if (!src.IsValid() || !dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid ");
        return Status::ERROR;
    }
    if (src.GetSizes().m_channel != dst.GetSizes().m_channel)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src channel is not equal to dst channel ");
        return Status::ERROR;
    }

    const ResizeConvertToImpl *impl = GetResizeConvertToFunc(src, dst);
    if (MI_NULL == impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "impl ptr is null ");
        return Status::ERROR;
    }

    impl->func(src, dst, alpha, beta);

    return Status::OK;
}

AURA_VOID MatFactory::CheckTotalMemory()
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);

    while (this->m_total_mem > this->m_max_mem)
    {
        MI_BOOL is_find_available = MI_FALSE;
        for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end(); ++it)
        {
            if ((*it).available)
            {
                this->m_total_mem -= it->mat.GetTotalBytes();
                this->m_dynamic_list.erase(it);
                is_find_available = MI_TRUE;
                break;
            }
        }

        if (is_find_available) // continue until total mem < max_mem
        {
            continue;
        }
        else    // nothing to delete
        {
            break;
        }
    }
}

AURA_VOID MatFactory::PrintInfo()
{
    std::lock_guard<std::mutex> guard(this->m_handle_lock);
    std::ostringstream os;
    os << "mat factory info:" << std::endl;

    MI_S64 base_bytes   = 0; // total bytes of basic mat list
    MI_S64 base_mat_num = 0; // number of base mat
    for (auto it = this->m_base_list.begin(); it != this->m_base_list.end(); it++)
    {
        base_bytes += it->mat.GetTotalBytes();
        base_mat_num++;
    }

    os << "total base bytes: " << base_bytes << "B | base mat number: " << base_mat_num << std::endl;
    for (auto it = this->m_base_list.begin(); it != this->m_base_list.end(); it++)
    {
        os << "file path: " << it->desc.param.file_path << " | bytes: " << it->mat.GetTotalBytes() << "B" << std::endl;
    }

    MI_S64 total_bytes_dynamic  = 0; // total bytes of dynamic mat list
    // Mat info from FILE type
    MI_S64 file_bytes            = 0;
    MI_S64 file_num_total        = 0;
    MI_S64 file_num_available    = 0;
    // Mat info from RANDOM type
    MI_S64 random_bytes          = 0;
    MI_S64 random_num_total      = 0;
    MI_S64 random_num_available  = 0;
    // Mat info from DERIVED type
    MI_S64 mat_bytes             = 0;
    MI_S64 mat_num_total         = 0;
    MI_S64 mat_num_available     = 0;
    // Mat info from EMPTY type
    MI_S64 empty_bytes           = 0;
    MI_S64 empty_num_total       = 0;
    MI_S64 empty_num_available   = 0;

    for (auto it = this->m_dynamic_list.begin(); it != this->m_dynamic_list.end(); it++)
    {
        switch (it->desc.type)
        {
            case MatDesc::Type::FILE:
            {
                file_bytes += it->mat.GetTotalBytes();
                file_num_total++;
                file_num_available += it->available ? 1 : 0;
                break;
            }
            case MatDesc::Type::RAND:
            {
                random_bytes += it->mat.GetTotalBytes();
                random_num_total++;
                random_num_available += it->available ? 1 : 0;
                break;
            }
            case MatDesc::Type::DERIVED:
            {
                mat_bytes += it->mat.GetTotalBytes();
                mat_num_total++;
                mat_num_available += it->available ? 1 : 0;
                break;
            }
            case MatDesc::Type::EMPTY:
            {
                empty_bytes += it->mat.GetTotalBytes();
                empty_num_total++;
                empty_num_available += it->available ? 1 : 0;
                break;
            }
            default:
            {
                AURA_LOGE(this->m_ctx, AURA_TAG, "unsupported matdesc type");
                break;
            }
        }
    }

    total_bytes_dynamic = file_bytes + random_bytes + mat_bytes + empty_bytes;

    if (total_bytes_dynamic != (this->m_total_mem))
    {
        AURA_LOGD(this->m_ctx, AURA_TAG, "total memory dismatch");
    }

    os << "====> total dynamic bytes / max bytes: " << total_bytes_dynamic << "B / " << m_max_mem << "B" << std::endl;
    os << "FILE    type bytes: " << file_bytes   << "B | available: " << file_num_available   << "/" << file_num_total   << std::endl;
    os << "RANDOM  type bytes: " << random_bytes << "B | available: " << random_num_available << "/" << random_num_total << std::endl;
    os << "DERIVED type bytes: " << mat_bytes    << "B | available: " << mat_num_available    << "/" << mat_num_total    << std::endl;
    os << "EMPTY  type bytes: "  << empty_bytes  << "B | available: " << empty_num_available  << "/" << empty_num_total  << std::endl;

    AURA_LOGD(this->m_ctx, AURA_TAG, os.str().c_str());
}

} // namespace aura
