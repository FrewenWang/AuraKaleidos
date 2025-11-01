#ifndef AURA_RUNTIME_HEXAGON_RPC_SERIALIZE_HPP__
#define AURA_RUNTIME_HEXAGON_RPC_SERIALIZE_HPP__

#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

#include <unordered_map>

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup hexagon Hexagon
 *    @{
 *       @defgroup comm Hexagon Common
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup comm
 * @{
 */

class HexagonRpcParam;

/**
 * TODO 这个地方要好好的学习一下。
 * 序列化指定的类型并将其附加到 HexagonRpcParam 缓冲区。
 * @brief Serialize a specified type and append it to the HexagonRpcParam buffer.
 *
 * The supported types are arithmetic, enumeration, Buffer, Mat, std::string, and Time.
 * And each type has a corresponding specialized implementation.
 *
 * @tparam Tp The type of the value to be serialized.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the value is appended.
 * @param val The value to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &val)
{
    MI_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

#if defined(AURA_BUILD_HOST)
    /// 将对应的参数内存，拷贝到rpc_param->m_rpc_param.m_data的buffer中
    memcpy(rpc_param->m_rpc_param.m_data, &val, size);
#else // AURA_BUILD_HEXAGON
    AuraMemCopy(rpc_param->m_rpc_param.m_data, &val, size);
#endif
    rpc_param->m_rpc_param.m_data = static_cast<MI_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

/**
 * @brief Deserialize a specified type from the HexagonRpcParam buffer.
 * 
 * The supported types are arithmetic, enumeration, Buffer, Mat, std::string, and Time.
 * And each type has a corresponding specialized implementation.
 *
 * @tparam Tp The type of the value to be deserialized.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the value is deserialized.
 * @param val The variable to store the deserialized value.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &val)
{
    MI_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

#if defined(AURA_BUILD_HOST)
    memcpy(&val, rpc_param->m_rpc_param.m_data, size);
#else // AURA_BUILD_HEXAGON
    AuraMemCopy(&val, rpc_param->m_rpc_param.m_data, size);
#endif
    rpc_param->m_rpc_param.m_data = static_cast<MI_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &val)
{
    MI_S32 tmp = static_cast<MI_S32>(val);
    Status ret = Serialize(ctx, rpc_param, tmp);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &val)
{
    MI_S32 tmp = 0;
    Status ret = Deserialize(ctx, rpc_param, tmp);
    val = static_cast<Tp>(tmp);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sequence of Plain Old Data (POD) types and append it to the HexagonRpcParam buffer.
 *
 * This function serializes a Sequence of Plain Old Data (POD) types and appends it to the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the sequence is appended.
 * @param seq The Sequence of POD types to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Sequence<Tp> &seq)
{
    Status ret = rpc_param->Set(seq.len);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (MI_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    MI_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

#if defined(AURA_BUILD_HOST)
    memcpy(rpc_param->m_rpc_param.m_data, seq.data, size);
#else // AURA_BUILD_HEXAGON
    AuraMemCopy(rpc_param->m_rpc_param.m_data, seq.data, size);
#endif
    rpc_param->m_rpc_param.m_data = static_cast<MI_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

/**
 * @brief Serialize a Sequence of non-POD types and append it to the HexagonRpcParam buffer.
 *
 * This function serializes a Sequence of non-POD types and appends it to the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the non-POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the sequence is appended.
 * @param seq The Sequence of non-POD types to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Sequence<Tp> &seq)
{
    Status ret = rpc_param->Set(seq.len);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (MI_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    for (MI_S32 i = 0; i < seq.len; i++)
    {
        ret |= rpc_param->Set(seq.data[i]);
    }

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sequence of Plain Old Data (POD) types from the HexagonRpcParam buffer.
 *
 * This function deserializes a Sequence of Plain Old Data (POD) types from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the sequence is deserialized.
 * @param seq The Sequence of POD types to be deserialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Sequence<Tp> &seq)
{
    Status ret = rpc_param->Get(seq.len);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (MI_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    MI_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

#if defined(AURA_BUILD_HOST)
    memcpy(seq.data, rpc_param->m_rpc_param.m_data, size);
#else // AURA_BUILD_HEXAGON
    AuraMemCopy(seq.data, rpc_param->m_rpc_param.m_data, size);
#endif
    rpc_param->m_rpc_param.m_data = static_cast<MI_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

/**
 * @brief Deserialize a Sequence of non-POD types from the HexagonRpcParam buffer.
 *
 * This function deserializes a Sequence of non-POD types from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the non-POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the sequence is deserialized.
 * @param seq The Sequence of non-POD types to be deserialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Sequence<Tp> &seq)
{
    Status ret = rpc_param->Get(seq.len);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (MI_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    for (MI_S32 i = 0; i < seq.len; i++)
    {
        ret |= rpc_param->Get(seq.data[i]);
    }

    AURA_RETURN(ctx, ret);
}

#if defined(AURA_BUILD_HOST)
template <typename Tp, typename std::enable_if<std::is_same<Tp, Buffer>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &buffer)
{
    Status ret = Status::ERROR;

    if (!buffer.IsValid())
    {
        ret = rpc_param->Set(buffer.m_type, buffer.m_size, buffer.m_property, static_cast<MI_U64>(0), static_cast<MI_S32>(-1));
    }
    else
    {
        MI_U64 offset = reinterpret_cast<MI_U64>(buffer.m_data) - reinterpret_cast<MI_U64>(buffer.m_origin);
        ret = rpc_param->Set(buffer.m_type, buffer.m_size, buffer.m_property, offset);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "set failed");
            return Status::ERROR;
        }

        MI_BOOL find_flag = MI_FALSE;
        for (size_t i = 0; i < rpc_param->m_rpc_mem.size(); i++)
        {
            if (rpc_param->m_rpc_mem[i].mem == buffer.m_origin)
            {
                ret = rpc_param->Set(static_cast<MI_S32>(i));
                find_flag = MI_TRUE;
                break;
            }
        }
        if (!find_flag)
        {
            ret = rpc_param->Set(static_cast<MI_S32>(rpc_param->m_rpc_mem.size()));
            rpc_param->m_rpc_mem.push_back({static_cast<MI_U8*>(buffer.m_origin), static_cast<MI_S32>(buffer.m_capacity)});
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Mat>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &mat)
{
    Status ret = rpc_param->Set(mat.GetElemType(), mat.GetSizes(), mat.GetStrides(), mat.GetBuffer());
    AURA_RETURN(ctx, ret);
}
#else
template <typename Tp, typename std::enable_if<std::is_same<Tp, Buffer>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &buffer)
{
    MI_S32 type;
    MI_S64 size;
    MI_S32 property;
    MI_U64 offset;
    MI_S32 idx;

    Status ret = rpc_param->Get(type, size, property, offset, idx);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    if (-1 == idx)
    {
        buffer = Buffer();
    }
    else
    {
        AURA_VOID *data   = static_cast<MI_U8*>(rpc_param->m_rpc_mem[idx].mem) + offset;
        AURA_VOID *origin = rpc_param->m_rpc_mem[idx].mem;
        MI_S64 capacity = rpc_param->m_rpc_mem[idx].memLen;

        buffer = Buffer(type, capacity, size, data, origin, property);
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid buffer");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Mat>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &mat)
{
    ElemType elem_type;
    Sizes3 sizes;
    Sizes strides;
    Buffer buffer;
    Status ret = rpc_param->Get(elem_type, sizes, strides, buffer);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    mat = Mat(ctx, elem_type, sizes, buffer, strides);

    return Status::OK;
}
#endif

/**
 * @brief Serialize a KeyPoint_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a KeyPoint_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the KeyPoint_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the KeyPoint_ is serialized.
 * @param key_point The KeyPoint_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Set(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a KeyPoint_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a KeyPoint_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the KeyPoint_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the KeyPoint_ is deserialized.
 * @param key_point The KeyPoint_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Get(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Point2_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Point2_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Point2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Point2_ is serialized.
 * @param point2 The Point2_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Point2_<Tp> &point2)
{
    Status ret = rpc_param->Set(point2.m_x, point2.m_y);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Point2_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Point2_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Point2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Point2_ is deserialized.
 * @param point2 The Point2_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Point2_<Tp> &point2)
{
    Status ret = rpc_param->Get(point2.m_x, point2.m_y);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Point3_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Point3_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Point3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Point3_ is serialized.
 * @param point3 The Point3_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Point3_<Tp> &point3)
{
    Status ret = rpc_param->Set(point3.m_x, point3.m_y, point3.m_z);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Point3_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Point3_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Point3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Point3_ is deserialized.
 * @param point3 The Point3_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Point3_<Tp> &point3)
{
    Status ret = rpc_param->Get(point3.m_x, point3.m_y, point3.m_z);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Rect_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Rect_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Rect_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Rect_ is serialized.
 * @param rect The Rect_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Rect_<Tp> &rect)
{
    Status ret = rpc_param->Set(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Rect_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Rect_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Rect_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Rect_ is deserialized.
 * @param rect The Rect_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Rect_<Tp> &rect)
{
    Status ret = rpc_param->Get(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Scalar_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Scalar_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Scalar_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Scalar_ is serialized.
 * @param scalar The Scalar_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {const_cast<Tp*>(scalar.m_val), 4};
    Status ret = Serialize(ctx, rpc_param, seq);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Scalar_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Scalar_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Scalar_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Scalar_ is deserialized.
 * @param scalar The Scalar_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {scalar.m_val, 4};
    Status ret = Deserialize(ctx, rpc_param, seq);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sizes2_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Sizes2_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Sizes2_ is serialized.
 * @param sizes2 The Sizes2_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Set(sizes2.m_height, sizes2.m_width);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sizes2_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Sizes2_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Sizes2_ is deserialized.
 * @param sizes2 The Sizes2_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Get(sizes2.m_height, sizes2.m_width);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sizes3_ type into the HexagonRpcParam buffer.
 *
 * This function serializes a Sizes3_ type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the Sizes3_ is serialized.
 * @param sizes3 The Sizes3_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Set(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sizes3_ type from the HexagonRpcParam buffer.
 *
 * This function deserializes a Sizes3_ type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the Sizes3_ is deserialized.
 * @param sizes3 The Sizes3_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Get(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, std::string>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &str)
{
    MI_S32 size = str.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        return Status::OK;
    }

    Sequence<MI_CHAR> seq = {const_cast<MI_CHAR*>(str.c_str()), size};
    ret |= rpc_param->Set(seq);

    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, std::string>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &str)
{
    MI_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        str = std::string();
        return Status::OK;
    }

    str.resize(size);
    Sequence<MI_CHAR> seq = {const_cast<MI_CHAR*>(str.c_str()), size};
    ret |= rpc_param->Get(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a std::vector type into the HexagonRpcParam buffer.
 *
 * This function serializes a std::vector type into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the elements in the std::vector.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the std::vector is serialized.
 * @param vec The std::vector to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const std::vector<Tp> &vec)
{
    MI_S32 size = vec.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        return Status::OK;
    }

    Sequence<Tp> seq = {const_cast<Tp*>(vec.data()), size};
    ret |= rpc_param->Set(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a std::vector type from the HexagonRpcParam buffer.
 *
 * This function deserializes a std::vector type from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of the elements in the std::vector.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the std::vector is deserialized.
 * @param vec The std::vector to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, std::vector<Tp> &vec)
{
    MI_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        vec = std::vector<Tp>();
        return Status::OK;
    }

    vec.resize(size);
    Sequence<Tp> seq = {vec.data(), size};
    ret |= rpc_param->Get(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a std::unordered_map into the HexagonRpcParam buffer.
 *
 * This function serializes a std::unordered_map into the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of values in the std::unordered_map.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam to which the std::unordered_map is serialized.
 * @param map The std::unordered_map to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const std::unordered_map<std::string, Tp> &map)
{
    MI_S32 size = map.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    for (const auto &iter : map)
    {
        ret = rpc_param->Set(iter.first, iter.second);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Set failed");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a std::unordered_map from the HexagonRpcParam buffer.
 *
 * This function deserializes a std::unordered_map from the HexagonRpcParam buffer.
 *
 * @tparam Tp The type of values in the std::unordered_map.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The HexagonRpcParam from which the std::unordered_map is deserialized.
 * @param map The std::unordered_map to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, std::unordered_map<std::string, Tp> &map)
{
    MI_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    for (MI_S32 i = 0; i < size; i++)
    {
        std::string key;
        Tp val;
        ret = rpc_param->Get(key, val);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get failed");
            return Status::ERROR;
        }
        map[key] = val;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &time)
{
    Status ret = rpc_param->Set(time.sec, time.ms, time.us);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &time)
{
    Status ret = rpc_param->Get(time.sec, time.ms, time.us);
    AURA_RETURN(ctx, ret);
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_RPC_SERIALIZE_HPP__