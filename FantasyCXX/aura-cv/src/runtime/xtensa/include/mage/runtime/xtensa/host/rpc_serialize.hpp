#ifndef AURA_RUNTIME_XTENSA_HOST_RPC_SERIALIZE_HPP__
#define AURA_RUNTIME_XTENSA_HOST_RPC_SERIALIZE_HPP__

#include "aura/runtime/array/host/xtensa_mat.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup xtensa Xtensa
 *    @{
 *       @defgroup Xtensa_host Xtensa Host
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup Xtensa_host
 * @{
 */

class XtensaRpcParam;

/**
 * @brief Serialize a specified type and append it to the XtensaRpcParam buffer.
 *
 * The supported types are arithmetic, enumeration, Buffer, Mat, std::string, and Time.
 * And each type has a corresponding specialized implementation.
 *
 * @tparam Tp The type of the value to be serialized.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the value is appended.
 * @param val The value to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &val)
{
    DT_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

    memcpy(rpc_param->m_rpc_param.m_data, &val, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

/**
 * @brief Deserialize a specified type from the XtensaRpcParam buffer.
 * 
 * The supported types are arithmetic, enumeration, Buffer, Mat, std::string, and Time.
 * And each type has a corresponding specialized implementation.
 *
 * @tparam Tp The type of the value to be deserialized.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the value is deserialized.
 * @param val The variable to store the deserialized value.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Tp &val)
{
    DT_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

    memcpy(&val, rpc_param->m_rpc_param.m_data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &val)
{
    DT_S32 tmp = static_cast<DT_S32>(val);
    Status ret = Serialize(ctx, rpc_param, tmp);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Tp &val)
{
    DT_S32 tmp = 0;
    Status ret = Deserialize(ctx, rpc_param, tmp);
    val = static_cast<Tp>(tmp);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sequence of Plain Old Data (POD) types and append it to the XtensaRpcParam buffer.
 *
 * This function serializes a Sequence of Plain Old Data (POD) types and appends it to the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the sequence is appended.
 * @param seq The Sequence of POD types to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Sequence<Tp> &seq)
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

    if (DT_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    DT_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

    memcpy(rpc_param->m_rpc_param.m_data, seq.data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

/**
 * @brief Serialize a Sequence of non-POD types and append it to the XtensaRpcParam buffer.
 *
 * This function serializes a Sequence of non-POD types and appends it to the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the non-POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the sequence is appended.
 * @param seq The Sequence of non-POD types to be serialized and appended.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Sequence<Tp> &seq)
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

    if (DT_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < seq.len; i++)
    {
        ret = rpc_param->Set(seq.data[i]);
    }

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sequence of Plain Old Data (POD) types from the XtensaRpcParam buffer.
 *
 * This function deserializes a Sequence of Plain Old Data (POD) types from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the sequence is deserialized.
 * @param seq The Sequence of POD types to be deserialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Sequence<Tp> &seq)
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

    if (DT_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    DT_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer overflow");
        return Status::ERROR;
    }

    memcpy(seq.data, rpc_param->m_rpc_param.m_data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

/**
 * @brief Deserialize a Sequence of non-POD types from the XtensaRpcParam buffer.
 *
 * This function deserializes a Sequence of non-POD types from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the non-POD element in the sequence.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the sequence is deserialized.
 * @param seq The Sequence of non-POD types to be deserialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Sequence<Tp> &seq)
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

    if (DT_NULL == seq.data)
    {
        AURA_ADD_ERROR_STRING(ctx, "bad data");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < seq.len; i++)
    {
        ret = rpc_param->Get(seq.data[i]);
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Buffer>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &buffer)
{
    XtensaEngine *xtensa_engine = ctx->GetXtensaEngine();
    if (DT_NULL == xtensa_engine)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetXtensaEngine failed");
        return Status::ERROR;
    }

    DT_U32 device_origin = xtensa_engine->GetDeviceAddr(buffer);
    if (0 == device_origin)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetDeviceAddr failed");
        return Status::ERROR;
    }

    DT_S32 offset = buffer.GetOffset();
    if (rpc_param->Set(AURA_MEM_DEFAULT, buffer.m_capacity, buffer.m_size, device_origin + offset, device_origin, buffer.m_property) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "set failed");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, XtensaMat>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &xtenas_mat)
{
    Status ret = rpc_param->Set(xtenas_mat.GetElemType(), xtenas_mat.GetSizes(), xtenas_mat.GetStrides(), xtenas_mat.GetBuffer());
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Scalar_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Scalar_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Scalar_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Scalar_ is serialized.
 * @param scalar The Scalar_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {const_cast<Tp*>(scalar.m_val), 4};
    Status ret = Serialize(ctx, rpc_param, seq);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Scalar_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Scalar_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Scalar_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Scalar_ is deserialized.
 * @param scalar The Scalar_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {scalar.m_val, 4};
    Status ret = Deserialize(ctx, rpc_param, seq);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sizes2_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Sizes2_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Sizes2_ is serialized.
 * @param sizes2 The Sizes2_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Set(sizes2.m_height, sizes2.m_width);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sizes2_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Sizes2_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Sizes2_ is deserialized.
 * @param sizes2 The Sizes2_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Get(sizes2.m_height, sizes2.m_width);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Sizes3_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Sizes3_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Sizes3_ is serialized.
 * @param sizes3 The Sizes3_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Set(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Sizes3_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Sizes3_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Sizes3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Sizes3_ is deserialized.
 * @param sizes3 The Sizes3_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Get(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a KeyPoint_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a KeyPoint_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the KeyPoint_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the KeyPoint_ is serialized.
 * @param key_point The KeyPoint_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Set(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a KeyPoint_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a KeyPoint_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the KeyPoint_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the KeyPoint_ is deserialized.
 * @param key_point The KeyPoint_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Get(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Point2_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Point2_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Point2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Point2_ is serialized.
 * @param point2 The Point2_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Point2_<Tp> &point2)
{
    Status ret = rpc_param->Set(point2.m_x, point2.m_y);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Point2_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Point2_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Point2_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Point2_ is deserialized.
 * @param point2 The Point2_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Point2_<Tp> &point2)
{
    Status ret = rpc_param->Get(point2.m_x, point2.m_y);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Point3_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Point3_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Point3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Point3_ is serialized.
 * @param point3 The Point3_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Point3_<Tp> &point3)
{
    Status ret = rpc_param->Set(point3.m_x, point3.m_y, point3.m_z);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Point3_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Point3_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Point3_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Point3_ is deserialized.
 * @param point3 The Point3_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Point3_<Tp> &point3)
{
    Status ret = rpc_param->Get(point3.m_x, point3.m_y, point3.m_z);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a Rect_ type into the XtensaRpcParam buffer.
 *
 * This function serializes a Rect_ type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Rect_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the Rect_ is serialized.
 * @param rect The Rect_ to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Rect_<Tp> &rect)
{
    Status ret = rpc_param->Set(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a Rect_ type from the XtensaRpcParam buffer.
 *
 * This function deserializes a Rect_ type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the Rect_.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the Rect_ is deserialized.
 * @param rect The Rect_ to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Rect_<Tp> &rect)
{
    Status ret = rpc_param->Get(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, std::string>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &str)
{
    DT_S32 size = str.size();
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

    Sequence<DT_CHAR> seq = {const_cast<DT_CHAR*>(str.c_str()), size};
    ret = rpc_param->Set(seq);

    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, std::string>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Tp &str)
{
    DT_S32 size = 0;
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
    Sequence<DT_CHAR> seq = {const_cast<DT_CHAR*>(str.c_str()), size};
    ret = rpc_param->Get(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a std::vector type into the XtensaRpcParam buffer.
 *
 * This function serializes a std::vector type into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the elements in the std::vector.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the std::vector is serialized.
 * @param vec The std::vector to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const std::vector<Tp> &vec)
{
    DT_S32 size = vec.size();
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
    ret = rpc_param->Set(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Deserialize a std::vector type from the XtensaRpcParam buffer.
 *
 * This function deserializes a std::vector type from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of the elements in the std::vector.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the std::vector is deserialized.
 * @param vec The std::vector to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, std::vector<Tp> &vec)
{
    DT_S32 size = 0;
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
    ret = rpc_param->Get(seq);

    AURA_RETURN(ctx, ret);
}

/**
 * @brief Serialize a std::unordered_map into the XtensaRpcParam buffer.
 *
 * This function serializes a std::unordered_map into the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of values in the std::unordered_map.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam to which the std::unordered_map is serialized.
 * @param map The std::unordered_map to be serialized.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const std::unordered_map<std::string, Tp> &map)
{
    DT_S32 size = map.size();
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
 * @brief Deserialize a std::unordered_map from the XtensaRpcParam buffer.
 *
 * This function deserializes a std::unordered_map from the XtensaRpcParam buffer.
 *
 * @tparam Tp The type of values in the std::unordered_map.
 *
 * @param ctx The pointer to the Context object.
 * @param rpc_param The XtensaRpcParam from which the std::unordered_map is deserialized.
 * @param map The std::unordered_map to store the deserialized data.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, std::unordered_map<std::string, Tp> &map)
{
    DT_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < size; i++)
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

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = DT_NULL>
Status Serialize(Context *ctx, XtensaRpcParam *rpc_param, const Tp &time)
{
    Status ret = rpc_param->Set(time.sec, time.ms, time.us);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = DT_NULL>
Status Deserialize(Context *ctx, XtensaRpcParam *rpc_param, Tp &time)
{
    Status ret = rpc_param->Get(time.sec, time.ms, time.us);
    AURA_RETURN(ctx, ret);
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_HOST_RPC_SERIALIZE_HPP__