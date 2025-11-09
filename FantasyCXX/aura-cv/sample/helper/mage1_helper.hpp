#ifndef AURA1_HELPER_HPP__
#define AURA1_HELPER_HPP__

// aura2.0 headers
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/context.h"

// aura1.0 headers
#include "mialgo_mat.h"
#include "mialgo_log.h"
#include "mialgo_utils.h"

#include <map>

/**
 * @brief Converts ElemType from Mialgo to aura.
 *
 * @param elem_type The input Mialgo ElemType.
 *
 * @return The converted aura ElemType, or aura::ElemType::INVALID if conversion fails.
 */
AURA_INLINE aura::ElemType MialgoElemType2Aura(DT_S32 elem_type)
{
    static const std::map<DT_S32, aura::ElemType> type_map
    {
        {MIALGO_MAT_U8,  aura::ElemType::U8},
        {MIALGO_MAT_S8,  aura::ElemType::S8},
        {MIALGO_MAT_U16, aura::ElemType::U16},
        {MIALGO_MAT_S16, aura::ElemType::S16},
        {MIALGO_MAT_U32, aura::ElemType::U32},
        {MIALGO_MAT_S32, aura::ElemType::S32},
        {MIALGO_MAT_F32, aura::ElemType::F32},
        {MIALGO_MAT_F64, aura::ElemType::F64},
    };

    return (type_map.count(elem_type) > 0) ? type_map.at(elem_type) : aura::ElemType::INVALID;

}

/**
 * @brief Converts ElemType from aura to Mialgo.
 *
 * @param ctx The aura::Context pointer.
 * @param elem_type The input aura ElemType.
 *
 * @return The converted Mialgo ElemType, or -1 if conversion fails.
 */
AURA_INLINE DT_S32 AuraElemType2Mialgo(aura::ElemType elem_type)
{
    static const std::map<aura::ElemType, DT_S32> type_map
    {
        {aura::ElemType::U8,  MIALGO_MAT_U8},
        {aura::ElemType::S8,  MIALGO_MAT_S8},
        {aura::ElemType::U16, MIALGO_MAT_U16},
        {aura::ElemType::S16, MIALGO_MAT_S16},
        {aura::ElemType::U32, MIALGO_MAT_U32},
        {aura::ElemType::S32, MIALGO_MAT_S32},
        {aura::ElemType::F32, MIALGO_MAT_F32},
        {aura::ElemType::F64, MIALGO_MAT_F64},
    };

    return (type_map.count(elem_type) > 0) ? type_map.at(elem_type) : -1;
}

/**
 * @brief Converts MialgoMemType to aura MemType.
 *
 * @param mem_type The input MialgoMemType.
 *
 * @return The converted aura MemType, or AURA_MEM_INVALID if conversion fails.
 */
AURA_INLINE DT_S32 MialgoMemType2Aura(MialgoMemType mem_type)
{
    static const std::map<MialgoMemType, DT_S32> type_map
    {
        {MialgoMemType::MIALGO_MEM_HEAP, AURA_MEM_HEAP},
        {MialgoMemType::MIALGO_MEM_ION,  AURA_MEM_DMA_BUF_HEAP},
        {MialgoMemType::MIALGO_MEM_CL,   AURA_MEM_SVM},
    };

    return (type_map.count(mem_type) > 0) ? type_map.at(mem_type) : AURA_MEM_INVALID;
}

/**
 * @brief Converts aura MemType to MialgoMemType.
 *
 * @param ctx The aura::Context pointer.
 * @param DT_S32 The input aura MemType.
 *
 * @return The converted MialgoMemType, or MialgoMemType::MIALGO_MEM_NONE if conversion fails.
 */
AURA_INLINE MialgoMemType AuraMemType2Mialgo(DT_S32 mem_type)
{
    static const std::map<DT_S32, MialgoMemType> type_map
    {
        {AURA_MEM_HEAP,         MialgoMemType::MIALGO_MEM_HEAP},
        {AURA_MEM_DMA_BUF_HEAP, MialgoMemType::MIALGO_MEM_ION},
        {AURA_MEM_SVM,          MialgoMemType::MIALGO_MEM_CL},
    };

    return (type_map.count(mem_type) > 0) ? type_map.at(mem_type) : MialgoMemType::MIALGO_MEM_NONE;
}

/**
 * @brief Converts a MialgoMat to a aura::Mat.
 *
 * @param ctx The aura::Context pointer.
 * @param src The input MialgoMat pointer.
 * @param deep_clone Whether to perform a deep clone (default is shallow copy).
 *
 * @return The converted aura::Mat, or an empty aura::Mat if conversion fails.
 */
AURA_INLINE aura::Mat MialgoMat2AuraMat(aura::Context *ctx, MialgoMat *src, DT_BOOL deep_clone = DT_FALSE)
{
    if (NULL == ctx || NULL == src)
    {
        AURA_ADD_ERROR_STRING(ctx, "input nullptr.");
        return aura::Mat();
    }

    // channel first is unsupported in aura::Mat
    if (MIALGO_MAT_FLAG_CH_FIRST == MIALGO_MAT_CH_ORDER(src->flag))
    {
        AURA_ADD_ERROR_STRING(ctx, "unsuported channel order.");
        return aura::Mat();
    }

    DT_S32 aura_mem_type = MialgoMemType2Aura(src->mem_info.type);
    if (AURA_MEM_INVALID == aura_mem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoMemType2Aura failed!");
        return aura::Mat();
    }

    if (aura_mem_type != AURA_MEM_HEAP && aura_mem_type != AURA_MEM_DMA_BUF_HEAP)
    {
        AURA_ADD_ERROR_STRING(ctx, "Only support heap and dma_buf heap mem type.");
        return aura::Mat();
    }

    aura::Buffer buffer = aura::Buffer(aura_mem_type, src->data_bytes, src->data_bytes,
                                       src->data, src->data, src->mem_info.fd);
    if (!buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer invalid.");
        return aura::Mat();
    }

    aura::ElemType aura_elem_type = MialgoElemType2Aura(src->elem_type);
    if (aura::ElemType::INVALID == aura_elem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoElemType2Aura failed");
        return aura::Mat();
    }

    aura::Mat dst_shallow(ctx, aura_elem_type, aura::Sizes3(src->shape.img.h, src->shape.img.w,
                          src->shape.img.c), buffer, aura::Sizes(0, src->shape.img.pitch));
    if (!dst_shallow.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "the mat constructor failed");
        return aura::Mat();
    }

    // deep clone
    if (deep_clone)
    {
        aura::Rect roi_deep(0, 0, dst_shallow.GetSizes().m_width, dst_shallow.GetSizes().m_height);
        DT_S32 pitch_deep = dst_shallow.GetSizes().m_width * dst_shallow.GetSizes().m_channel * ElemTypeSize(dst_shallow.GetElemType());

        // check whether data pointer or pitch need to be aligned to 128
        if (!(src->shape.img.pitch & 127) && !(reinterpret_cast<DT_U64>(src->data) & 127))
        {
            pitch_deep = AURA_ALIGN(pitch_deep, 128);
        }

        aura::Mat dst_deep = dst_shallow.Clone(roi_deep, aura::Sizes(0, pitch_deep));
        if (!dst_deep.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "the mat is invalid.");
            return aura::Mat();
        }

        return dst_deep;
    }

    return dst_shallow;
}

/**
 * @brief Converts a aura::Mat to a MialgoMat.
 *
 * Attention! The MialgoMat returned by this function should be deleted by the user using MialgoDeleteMat to avoid memory leaks.
 *
 * @param ctx The aura::Context pointer.
 * @param src The input aura::Mat.
 * @param deep_clone Whether to perform a deep clone (default is shallow copy).
 *
 * @return A pointer to the converted MialgoMat, or nullptr if conversion fails.
 */
AURA_INLINE MialgoMat* AuraMat2MialgoMat(aura::Context *ctx, const aura::Mat &src, DT_BOOL deep_clone = DT_FALSE)
{
    if (!src.IsValid() || DT_NULL == ctx)
    {
        AURA_ADD_ERROR_STRING(ctx, "src is invalid or ctx is nullptr.");
        return NULL;
    }

    const MialgoMemType mem_type = AuraMemType2Mialgo(src.GetMemType());
    if (MIALGO_MEM_NONE == mem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "AuraMemType2Mialgo failed.");
        return NULL;
    }

    // only support heap and dma_buf heap mem type
    if ((mem_type != MialgoMemType::MIALGO_MEM_HEAP) && (mem_type != MialgoMemType::MIALGO_MEM_ION))
    {
        AURA_ADD_ERROR_STRING(ctx, "Only support heap and dma_buf heap mem type.");
        return NULL;
    }

    const DT_S32 elem_type = AuraElemType2Mialgo(src.GetElemType());
    if (-1 == elem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "AuraElemType2Mialgo failed.");
        return NULL;
    }

    DT_S32 size_dst[] = {src.GetSizes().m_channel, src.GetSizes().m_height, src.GetSizes().m_width};

    DT_S32 flag_dst = MIALGO_MAT_FLAG_IMG_MAT | MIALGO_MAT_FLAG_CH_LAST;
    flag_dst |= (mem_type == MialgoMemType::MIALGO_MEM_HEAP) ? MIALGO_MAT_FLAG_HEAP_MEM : 0;

    MialgoMat *dst_shallow = (MialgoMat *)MialgoAllocateHeap(sizeof(MialgoMat));
    if (NULL == dst_shallow)
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoAllocateHeap failed.");
        return NULL;
    }
    memset(dst_shallow, 0, sizeof(MialgoMat));

    DT_S32 stride_dst_shallow[] = {0, 0, src.GetRowPitch()};
    if (MialgoInitMat(dst_shallow, 3, size_dst, elem_type, stride_dst_shallow, flag_dst, const_cast<DT_VOID *>(src.GetData())) != MIALGO_OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "no mem");
        MialgoDeallocate(dst_shallow);
        return NULL;
    }

    // mem info
    dst_shallow->mem_info.type = mem_type;
    dst_shallow->mem_info.size = src.GetBuffer().m_size;
    dst_shallow->mem_info.phy_addr = reinterpret_cast<DT_U64>(src.GetBuffer().m_origin);
    dst_shallow->mem_info.fd = src.GetBuffer().m_property;

    if (deep_clone)
    {
        DT_S32 pitch_deep = size_dst[0] * size_dst[2] * ElemTypeSize(src.GetElemType());

        // check whether data pointer or pitch need to be aligned to 128
        if (!(src.GetRowPitch() & 127) && !(reinterpret_cast<DT_U64>(src.GetData()) & 127))
        {
            pitch_deep = MIALGO_ALIGN(pitch_deep, 128);
        }

        DT_S32 stride_dst_deep[] = {0, 0, pitch_deep};

        MialgoMat *dst_deep = MialgoCreateMat(3, size_dst, elem_type, stride_dst_deep, flag_dst);
        if (NULL == dst_deep)
        {
            AURA_ADD_ERROR_STRING(ctx, "no mem");
            MialgoDeallocate(&dst_shallow);
            return NULL;
        }

        if (MialgoCopyMat(dst_shallow, dst_deep, {0, 0}) != MIALGO_OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "copy mat failed");
            MialgoDeallocate(dst_shallow);
            MialgoDeleteMat(&dst_deep);
            return NULL;
        }

        MialgoDeallocate(dst_shallow);

        return dst_deep;
    }

    return dst_shallow;
}

/**
 * @brief Convert MialgoImg to aura::Mat.
 *
 * @param ctx The aura::Context pointer.
 * @param src The source MialgoImg to convert.
 * @param deep_clone Whether to perform a deep clone of the MialgoImg data.
 *
 * @return The converted aura::Mat, or an empty aura::Mat if conversion fails.
 */
AURA_INLINE aura::Mat MialgoImg2AuraMat(aura::Context *ctx, MialgoImg *src, DT_BOOL deep_clone = DT_FALSE)
{
    if (NULL == ctx || NULL == src)
    {
        AURA_ADD_ERROR_STRING(ctx, "ctx or src nullptr.");
        return aura::Mat();
    }

    // 1. mialgo_img >> mialgo_mat
    MialgoMat mialgo_mat_buf;
    MialgoMat *mialgo_mat = MialgoGetMat(src, &mialgo_mat_buf);
    if (NULL == mialgo_mat)
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoGetMat failed.");
        return aura::Mat();
    }

    // 2. mialgo_mat >> aura::Mat
    aura::Mat dst = MialgoMat2AuraMat(ctx, mialgo_mat, deep_clone);

    if (!dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoMat2AuraMat failed");
        return aura::Mat();
    }

    return dst;
}

/**
 * @brief Converts aura::Mat to MialgoImg.
 *
 * Attention! The MialgoImg returned by this function should be deleted by the user using MialgoDeleteImg to avoid memory leaks.
 *
 * @param ctx The aura::Context pointer.
 * @param src The source aura::Mat to convert.
 * @param info MialgoImgFormat type, Iaura format.
 * @param deep_clone Whether to perform a deep clone of the aura::Mat data.
 *
 * @return A pointer to the converted MialgoImg, or nullptr if conversion fails.
 */
AURA_INLINE MialgoImg* AuraMat2MialgoImg(aura::Context *ctx, const aura::Mat& src, MialgoImgFormat info, DT_BOOL deep_clone = DT_FALSE)
{
    if (!src.IsValid() || DT_NULL == ctx)
    {
        AURA_ADD_ERROR_STRING(ctx, "src is invalid or ctx is nullptr");
        return NULL;
    }

    // 1. aura::Mat >> mialgo_mat
    MialgoMat *mialgo_mat = AuraMat2MialgoMat(ctx, src, deep_clone);
    if (NULL == mialgo_mat)
    {
        AURA_ADD_ERROR_STRING(ctx, "AuraMat2MialgoMat failed");
        return NULL;
    }

    // 2. mialgo_mat >> mialgo_img
    MialgoImg *mialgo_img = (MialgoImg *)MialgoAllocateHeap(sizeof(MialgoImg));
    if (NULL == mialgo_img)
    {
        AURA_ADD_ERROR_STRING(ctx, "no mem");
        MialgoDeleteMat(&mialgo_mat);
        return NULL;
    }

    // in fact, the right return is mialgo_img
    if (MialgoGetImg(mialgo_mat, mialgo_img, info) == NULL)
    {
        AURA_ADD_ERROR_STRING(ctx, "MialgoGegImg failed");
        MialgoDeleteMat(&mialgo_mat);
        MialgoDeallocate(mialgo_img);
        return NULL;
    }

    if (deep_clone)
    {
        // Increment the reference count by 1, so the memory will ultimately be deleted by mialgo_img
        *(mialgo_mat->ref_count) += 1;
    }

    MialgoDeleteMat(&mialgo_mat);

    return mialgo_img;
}

#endif // AURA1_HELPER_HPP__