#ifndef AURA_RUNTIME_XTENSA_DEVICE_RPC_SERIALIZE_HPP__
#define AURA_RUNTIME_XTENSA_DEVICE_RPC_SERIALIZE_HPP__

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup xtensa Xtensa
 *    @{
 *       @defgroup xtensa_device Xtensa Device
 *    @}
 * @}
 */
namespace aura
{
namespace xtensa
{

/**
 * @addtogroup xtensa_device
 * @{
 */

class XtensaRpcParam;

template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value || std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Serialize(XtensaRpcParam *rpc_param, const Tp &val)
{
    DT_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_XTENSA_LOG("buffer overflow\n");
        return Status::ERROR;
    }

    Memcpy(rpc_param->m_rpc_param.m_data, &val, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_arithmetic<Tp>::value || std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Tp &val)
{
    DT_S32 size = sizeof(val);
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_XTENSA_LOG("buffer overflow\n");
        return Status::ERROR;
    }

    Memcpy(&val, rpc_param->m_rpc_param.m_data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;
    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = DT_NULL>
Status Serialize(XtensaRpcParam *rpc_param, const Tp &val)
{
    DT_S32 tmp = static_cast<DT_S32>(val);
    Status ret = Serialize(rpc_param, tmp);
    return ret;
}

template <typename Tp, typename std::enable_if<std::is_enum<Tp>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Tp &val)
{
    DT_S32 tmp = 0;
    Status ret = Deserialize(rpc_param, tmp);
    val = static_cast<Tp>(tmp);
    return ret;
}

template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Serialize(XtensaRpcParam *rpc_param, const Sequence<Tp> &seq)
{
    Status ret = rpc_param->Set(seq.len);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed\n");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (DT_NULL == seq.data)
    {
        AURA_XTENSA_LOG("bad data\n");
        return Status::ERROR;
    }

    DT_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_XTENSA_LOG("buffer overflow\n");
        return Status::ERROR;
    }

    Memcpy(rpc_param->m_rpc_param.m_data, seq.data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_pod<Tp>::value && !std::is_pointer<Tp>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Sequence<Tp> &seq)
{
    Status ret = rpc_param->Get(seq.len);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (DT_NULL == seq.data)
    {
        AURA_XTENSA_LOG("bad data\n");
        return Status::ERROR;
    }

    DT_S32 size = sizeof(Tp) * seq.len;
    if (rpc_param->m_rpc_param.m_size + size > rpc_param->m_rpc_param.m_capacity)
    {
        AURA_XTENSA_LOG("buffer overflow\n");
        return Status::ERROR;
    }

    Memcpy(seq.data, rpc_param->m_rpc_param.m_data, size);
    rpc_param->m_rpc_param.m_data = static_cast<DT_U8*>(rpc_param->m_rpc_param.m_data) + size;
    rpc_param->m_rpc_param.m_size += size;

    return Status::OK;
}

template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = DT_NULL>
Status Serialize(XtensaRpcParam *rpc_param, const Sequence<Tp> &seq)
{
    Status ret = rpc_param->Set(seq.len);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed\n");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (DT_NULL == seq.data)
    {
        AURA_XTENSA_LOG("bad data\n");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < seq.len; i++)
    {
        ret = rpc_param->Set(seq.data[i]);
    }

    return ret;
}

template <typename Tp, typename std::enable_if<!std::is_pod<Tp>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Sequence<Tp> &seq)
{
    Status ret = rpc_param->Get(seq.len);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    if (0 == seq.len)
    {
        return Status::OK;
    }

    if (DT_NULL == seq.data)
    {
        AURA_XTENSA_LOG("bad data\n");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < seq.len; i++)
    {
        ret = rpc_param->Get(seq.data[i]);
    }
    return ret;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Buffer>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Tp &buffer)
{
    DT_S32 type;
    DT_S64 capacity;
    DT_S64 size;
    DT_VOID *data = DT_NULL;
    DT_VOID *origin = DT_NULL;
    DT_S32 property;

    Status ret = rpc_param->Get(type, capacity, size, data, origin, property);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    buffer = Buffer(type, capacity, size, data, origin, property);
    if (!buffer.IsValid())
    {
        AURA_XTENSA_LOG("invalid buffer\n");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Mat>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Tp &mat)
{
    ElemType elem_type;
    Sizes3 sizes;
    Sizes strides;
    Buffer buffer;
    Status ret = rpc_param->Get(elem_type, sizes, strides, buffer);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    strides.m_width = strides.m_width / ElemTypeSize(elem_type);

    mat = Mat(elem_type, sizes, buffer, strides);

    return Status::OK;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Set(sizes2.m_height, sizes2.m_width);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Sizes2_<Tp> &sizes2)
{
    Status ret = rpc_param->Get(sizes2.m_height, sizes2.m_width);
    return ret;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Set(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Sizes3_<Tp> &sizes3)
{
    Status ret = rpc_param->Get(sizes3.m_height, sizes3.m_width, sizes3.m_channel);
    return ret;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {const_cast<Tp*>(scalar.m_val), 4};
    Status ret = Serialize(rpc_param, seq);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Scalar_<Tp> &scalar)
{
    Sequence<Tp> seq = {scalar.m_val, 4};
    Status ret = Deserialize(rpc_param, seq);
    return ret;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Set(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, KeyPoint_<Tp> &key_point)
{
    Status ret = rpc_param->Get(key_point.m_pt, key_point.m_size, key_point.m_angle,
                                key_point.m_response, key_point.m_octave, key_point.m_class_id);
    return ret;
}
template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Point2_<Tp> &point2)
{
    Status ret = rpc_param->Set(point2.m_x, point2.m_y);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Point2_<Tp> &point2)
{
    Status ret = rpc_param->Get(point2.m_x, point2.m_y);
    return ret;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Point3_<Tp> &point3)
{
    Status ret = rpc_param->Set(point3.m_x, point3.m_y, point3.m_z);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Point3_<Tp> &point3)
{
    Status ret = rpc_param->Get(point3.m_x, point3.m_y, point3.m_z);
    return ret;
}

template <typename Tp>
Status Serialize(XtensaRpcParam *rpc_param, const Rect_<Tp> &rect)
{
    Status ret = rpc_param->Set(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    return ret;
}

template <typename Tp>
Status Deserialize(XtensaRpcParam *rpc_param, Rect_<Tp> &rect)
{
    Status ret = rpc_param->Get(rect.m_x, rect.m_y, rect.m_width, rect.m_height);
    return ret;
}

template <DT_S32 STR_MAX_SIZE>
Status Serialize(XtensaRpcParam *rpc_param, const string_<STR_MAX_SIZE> &str)
{
    DT_S32 size = str.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed\n");
        return Status::ERROR;
    }

    if (0 == size)
    {
        return Status::OK;
    }

    Sequence<DT_CHAR> seq = {const_cast<DT_CHAR*>(str.c_str()), size};
    ret = rpc_param->Set(seq);

    return ret;
}

template <DT_S32 STR_MAX_SIZE>
Status Deserialize(XtensaRpcParam *rpc_param, string_<STR_MAX_SIZE> &str)
{
    DT_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    if (0 == size)
    {
        str = string_<STR_MAX_SIZE>();
        return Status::OK;
    }

    Sequence<DT_CHAR> seq = {const_cast<DT_CHAR*>(str.c_str()), size};
    ret = rpc_param->Get(seq);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    return ret;
}

template <typename Tp, DT_S32 VEC_MAX_SIZE>
Status Serialize(XtensaRpcParam *rpc_param, const vector<Tp, VEC_MAX_SIZE> &vec)
{
    DT_S32 size = vec.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed\n");
        return Status::ERROR;
    }

    if (0 == size)
    {
        return Status::OK;
    }

    Sequence<Tp> seq = {const_cast<Tp*>(vec.data()), size};
    ret = rpc_param->Set(seq);

    return ret;
}

template <typename Tp, DT_S32 VEC_MAX_SIZE>
Status Deserialize(XtensaRpcParam *rpc_param, vector<Tp, VEC_MAX_SIZE> &vec)
{
    DT_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    if (0 == size)
    {
        vec = vector<Tp, VEC_MAX_SIZE>();
        return Status::OK;
    }

    vec.resize(size);
    Sequence<Tp> seq = {const_cast<Tp*>(vec.data()), size};
    ret = rpc_param->Get(seq);

    return ret;
}

template <typename Tp, DT_S32 STR_MAX_SIZE, DT_S32 MAP_MAX_SIZE>
Status Serialize(XtensaRpcParam *rpc_param, const map<Tp, STR_MAX_SIZE, MAP_MAX_SIZE> &map)
{
    DT_S32 size = map.size();
    Status ret = rpc_param->Set(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed\n");
        return Status::ERROR;
    }

    for (auto it = map.begin(); it != map.end(); it++)
    {
        string_<STR_MAX_SIZE> key = it->first;
        Tp val = it->second;
        ret = rpc_param->Set(key, val);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Get failed\n");
            return Status::ERROR;
        }
    }
    return ret;
}

template <typename Tp, DT_S32 STR_MAX_SIZE, DT_S32 MAP_MAX_SIZE>
Status Deserialize(XtensaRpcParam *rpc_param, map<Tp, STR_MAX_SIZE, MAP_MAX_SIZE> &map)
{
    DT_S32 size = 0;
    Status ret = rpc_param->Get(size);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed\n");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < size; i++)
    {
        string_<STR_MAX_SIZE> key;
        Tp val;
        ret = rpc_param->Get(key, val);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Get failed\n");
            return Status::ERROR;
        }
        map[key] = val;
    }

    return ret;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = DT_NULL>
Status Serialize(XtensaRpcParam *rpc_param, const Tp &time)
{
    Status ret = rpc_param->Set(time.sec, time.ms, time.us);
    return ret;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, Time>::value>::type* = DT_NULL>
Status Deserialize(XtensaRpcParam *rpc_param, Tp &time)
{
    Status ret = rpc_param->Get(time.sec, time.ms, time.us);
    return ret;
}

/**
 * @}
 */
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_DEVICE_RPC_SERIALIZE_HPP__