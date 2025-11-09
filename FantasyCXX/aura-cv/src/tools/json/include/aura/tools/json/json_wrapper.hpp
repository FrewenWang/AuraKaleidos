#ifndef AURA_TOOLS_JSON_JSON_WRAPPER_HPP__
#define AURA_TOOLS_JSON_JSON_WRAPPER_HPP__

#include "aura/tools/json/json_serialize.h"
#include "aura/runtime/logger.h"

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup json Json
 *    @{
 *       @defgroup json_wrapper_class JsonWrapper Class
 *    @}
 * @}
 */

/**
 * @addtogroup json_wrapper_class
 * @{
 */

/**
 * @brief Macro for serialize params to json and deserialize json to params.
 *
 * This macro simplifies the process of serialize json.
 *
 * @param ctx The pointer to the Context object.
 * @param json_wrapper The object of JsonWrapper.
 * @param args the param list to be serialized.
 *
 */

#define AURA_WRAPPER_JSON_TO(v1)   aura::JsonTo(json_params, #v1, v1);
#define AURA_WRAPPER_JSON_FROM(v1) v1 = json_obj["params"][#v1].template get<std::remove_reference<decltype(v1)>::type>();

#define AURA_JSON_SERIALIZE(ctx, json_wrapper, ...)                                         \
    do {                                                                                    \
        aura_json::json json_obj;                                                           \
        aura_json::json json_params;                                                        \
                                                                                            \
        json_wrapper.Lock();                                                                \
        json_wrapper.UpdateArrayMap();                                                      \
        AURA_JSON_EXPAND(AURA_JSON_PASTE(AURA_WRAPPER_JSON_TO, __VA_ARGS__))                \
        json_obj["params"] = json_params;                                                   \
        json_obj["name"]   = json_wrapper.GetName();                                        \
                                                                                            \
        json_wrapper.ClearArrayMap();                                                       \
        json_wrapper.UnLock();                                                              \
                                                                                            \
        if (json_wrapper.Write(json_obj) != aura::Status::OK)                               \
        {                                                                                   \
            AURA_ADD_ERROR_STRING(ctx, "Write json failed");                                \
        }                                                                                   \
    } while (0)

#define AURA_JSON_DESERIALIZE(ctx, json_wrapper, ...)                                       \
    do {                                                                                    \
        aura_json::json json_obj;                                                           \
                                                                                            \
        if (json_wrapper.Read(json_obj) != aura::Status::OK)                                \
        {                                                                                   \
            AURA_ADD_ERROR_STRING(ctx, "Read json failed");                                 \
        }                                                                                   \
                                                                                            \
        json_wrapper.Lock();                                                                \
        json_wrapper.SetContext(ctx);                                                       \
        AURA_JSON_EXPAND(AURA_JSON_PASTE(AURA_WRAPPER_JSON_FROM, __VA_ARGS__))              \
        json_wrapper.UnLock();                                                              \
    } while (0)

namespace aura
{

/**
 * @brief Serialize a const Array* or Array* param to aura_json::json object.
 *
 * @tparam Tp The type of the param to be serialized.
 *
 * @param json_params The object of aura_json::json.
 * @param name The name of param to be serialized.
 * @param param The param to be serialized.
 *
 * @return DT_VOID.
 */
template <typename Tp, typename std::enable_if<std::is_same<Array*, Tp>::value || std::is_same<const Array*, Tp>::value>::type* = DT_NULL>
AURA_INLINE DT_VOID JsonTo(aura_json::json &json_params, std::string name, const Tp param)
{
    if (DT_NULL == param)
    {
        return;
    }

    if (param->GetArrayType() == ArrayType::MAT)
    {
        const Mat *mat    = dynamic_cast<const Mat*>(param);
        json_params[name] = mat;
    }
    else if (param->GetArrayType() == ArrayType::CL_MEMORY)
    {
#if defined(AURA_ENABLE_OPENCL)
        const CLMem *cl_mem = dynamic_cast<const CLMem*>(param);
        json_params[name]   = cl_mem;
#endif
    }
}

/**
 * @brief Serialize a const Array* or Array* vector param to aura_json::json object.
 *
 * @tparam Tp The type of the param to be serialized.
 *
 * @param json_params The object of aura_json::json.
 * @param name The name of param to be serialized.
 * @param param The param to be serialized.
 *
 * @return DT_VOID.
 */
template <typename Tp, typename std::enable_if<std::is_same<Array*, Tp>::value || std::is_same<const Array*, Tp>::value>::type* = DT_NULL>
AURA_INLINE DT_VOID JsonTo(aura_json::json &json_params, std::string name, const std::vector<Tp> &param)
{
    if (param.size() == 0)
    {
        return;
    }

    if ((param[0] != DT_NULL) && (param[0]->GetArrayType() == ArrayType::MAT))
    {
        std::vector<const Mat*> mats;
        for (auto array : param)
        {
            if (DT_NULL == array)
            {
                break;
            }

            mats.push_back(dynamic_cast<const Mat*>(array));
        }

        json_params[name] = mats;
    }
    else if ((param[0] != DT_NULL) && param[0]->GetArrayType() == ArrayType::CL_MEMORY)
    {
#if defined(AURA_ENABLE_OPENCL)
        std::vector<const CLMem*> cl_mems;
        for (auto array : param)
        {
            if (DT_NULL == array)
            {
                break;
            }

            cl_mems.push_back(dynamic_cast<const CLMem*>(array));
        }

        json_params[name] = cl_mems;
#endif
    }
}

/**
 * @brief Serialize param except type const Array* and Array* to aura_json::json object.
 *
 * @tparam Tp The type of the param to be serialized.
 *
 * @param json_params The object of aura_json::json.
 * @param name The name of param to be serialized.
 * @param param The param to be serialized.
 *
 * @return DT_VOID.
 */
template <typename Tp, typename std::enable_if<!(std::is_same<Array*, Tp>::value || std::is_same<const Array*, Tp>::value)>::type* = DT_NULL>
AURA_INLINE DT_VOID JsonTo(aura_json::json &json_params, std::string name, const Tp &param)
{
    json_params[name] = param;
}

class AURA_EXPORTS JsonWrapper
{
public:
    /**
     * @brief Constructor for JsonWrapper.
     *
     * @param ctx The pointer to the Context object.
     * @param prefix the prefix path of name.
     * @param name the name of op or algo.
     */
    JsonWrapper(Context *ctx, const std::string &prefix, const std::string &name)
                : m_ctx(ctx), m_prefix(prefix), m_name(name), m_json_path(m_prefix + "_" + name + ".json")
    {}

    /**
     * @brief Constructor for JsonWrapper.
     *
     * @param ctx The pointer to the Context object.
     * @param json_path the path of json to be deserialized.
     */
    JsonWrapper(Context *ctx, const std::string &json_path) : m_ctx(ctx), m_json_path(json_path)
    {}

    AURA_DISABLE_COPY_AND_ASSIGN(JsonWrapper);

    /**
     * @brief Lock JsonHelper object to protect the array_map from being overwritten.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Lock();

    /**
     * @brief UnLock JsonHelper object.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status UnLock();

    /**
     * @brief Update JsonHelper object's array_map.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status UpdateArrayMap();

    /**
     * @brief Clear JsonHelper object's array_map.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status ClearArrayMap();

    /**
     * @brief Set JsonHelper object's context.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status SetContext(Context *ctx);

    /**
     * @brief Set array_map with name and array pointer, support Array* or const Array*.
     *
     * @tparam Tp The type of the array to be serialized.
     *
     * @param name The name of array.
     * @param array The array pointer to be set.
     * @param check Whether to check array is null, if the check is set to true and the array is null return error,
     *              If check is set to false, the array is not checked and always return ok.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::string &name, const Tp *array, DT_BOOL check = DT_TRUE);

    /**
     * @brief Set array_map with name and array pointer, support Array* or const Array* vector type.
     *
     * @tparam Tp The type of the arrays to be serialized.
     *
     * @param name The name vector of array.
     * @param arrays The array object to be set.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::vector<std::string> &name, const std::vector<Tp*> &arrays);

    /**
     * @brief Set array_map with name and array pointer, support Array* or const Array* vector type.
     *
     * @tparam Tp The type of the arrays to be serialized.
     *
     * @param name The name of array.
     * @param arrays The array object to be set.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::string &name, const std::vector<Tp*> &arrays);

    /**
     * @brief Set array_map with name and array pointer, support Mat or CLMem type.
     *
     * @tparam Tp The type of the arrays to be serialized.
     *
     * @param name The name of array.
     * @param array The array object to be set.
     * @param check Whether to check array is valid, if the check is set to true and the array is valid return error,
     *              If check is set to false, the array is not checked and always return ok
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::string &name, const Tp &array, DT_BOOL check = DT_TRUE);

    /**
     * @brief Set array_map with name and array pointer, support Mat or CLMem vector type.
     *
     * @tparam Tp The type of the arrays to be serialized.
     *
     * @param name The name vector of array.
     * @param arrays The array object to be set.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::vector<std::string> &name, const std::vector<Tp> &arrays);

    /**
     * @brief Set array_map with name and array pointer, support Mat or CLMem vector type.
     *
     * @tparam Tp The type of the arrays to be serialized.
     *
     * @param name The name of array.
     * @param arrays The array object to be set.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type* = DT_NULL>
    Status SetArray(const std::string &name, const std::vector<Tp> &arrays);

    /**
     * @brief Get op or algo name.
     *
     * @return the name of of op or algo.
     */
    std::string GetName() const;

    /**
     * @brief Write json object to file.
     *
     * @param json_obj The aura_json::json object to be write.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Write(aura_json::json &json_obj);

    /**
     * @brief Read json object from file.
     *
     * @param json_obj The aura_json::json object to be read.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Read(aura_json::json &json_obj);

private:
    Context     *m_ctx;
    std::string m_prefix;
    std::string m_name;
    std::string m_json_path;

    std::unordered_map<const Array*, std::string> m_array_map;
};

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::string &name, const Tp *array, DT_BOOL check)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (DT_NULL == array && DT_TRUE == check)
    {
        std::string str_info = "array:" + name + "is null";
        AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
        return Status::ERROR;
    }
    else if (array != DT_NULL)
    {
        m_array_map[array] = m_prefix + "_" + name + ".bin";
        array->Dump(m_prefix + "_" + name + ".bin");
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::vector<std::string> &name, const std::vector<Tp*> &arrays)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (name.size() != arrays.size())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "name's size should equal to arrays");
        return Status::ERROR;
    }

    Status ret     = Status::OK;
    auto it_name   = name.begin();
    auto it_arrays = arrays.begin();

    for (; it_name != name.end(); ++it_name, ++it_arrays)
    {
        if (DT_NULL == (*it_arrays))
        {
            std::string str_info = "array:" + (*it_name) + "is null";
            AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
            ret = Status::ERROR;
            break;
        }

        m_array_map[*it_arrays] = m_prefix + "_" + (*it_name) + ".bin";
        (*it_arrays)->Dump(m_prefix + "_" + (*it_name) + ".bin");
    }

    return ret;
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::string &name, const std::vector<Tp*> &arrays)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::OK;
    auto it    = arrays.begin();
    DT_S32 idx = 0;

    for (; it != arrays.end(); ++it, ++idx)
    {
        if (DT_NULL == (*it))
        {
            std::string str_info = "array:" + name + "_" + std::to_string(idx) + "is null";
            AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
            ret = Status::ERROR;
            break;
        }

        m_array_map[*it] = m_prefix + "_" + name + "_" + std::to_string(idx) + ".bin";
        (*it)->Dump(m_prefix + "_" + name + "_" + std::to_string(idx) + ".bin");
    }

    return ret;
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::string &name, const Tp &array, DT_BOOL check)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!array.IsValid() && (DT_TRUE == check))
    {
        std::string str_info = "array:" + name + "is invalid";
        AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
        return Status::ERROR;
    }
    else if (array.IsValid())
    {
        m_array_map[&array] = m_prefix + "_" + name + ".bin";
        array.Dump(m_prefix + "_" + name + ".bin");
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::vector<std::string> &name, const std::vector<Tp> &arrays)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (name.size() != arrays.size())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "name's size should equal to arrays");
        return Status::ERROR;
    }

    Status ret     = Status::OK;
    auto it_name   = name.begin();
    auto it_arrays = arrays.begin();

    for (; it_name != name.end(); ++it_name, ++it_arrays)
    {
        if (!(it_arrays->IsValid()))
        {
            std::string str_info = "array:" + (*it_name) + "is invalid";
            AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
            ret = Status::ERROR;
            break;
        }

        m_array_map[&(*it_arrays)] = m_prefix + "_" + (*it_name) + ".bin";
        it_arrays->Dump(m_prefix + "_" + (*it_name) + ".bin");
    }

    return ret;
}

template <typename Tp, typename std::enable_if<std::is_base_of<Array, Tp>::value>::type*>
Status JsonWrapper::SetArray(const std::string &name, const std::vector<Tp> &arrays)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::OK;
    auto it    = arrays.begin();
    DT_S32 idx = 0;

    for (; it != arrays.end(); ++it, ++idx)
    {
        if (!(it->IsValid()))
        {
            std::string str_info = "array:" + name + "_" + std::to_string(idx) + "is invalid";
            AURA_ADD_ERROR_STRING(m_ctx, str_info.c_str());
            ret = Status::ERROR;
            break;
        }

        m_array_map[&(*it)] = m_prefix + "_" + name + "_" + std::to_string(idx) + ".bin";
        it->Dump(m_prefix + "_" + name + "_" + std::to_string(idx) + ".bin");
    }

    return ret;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a numberic index and the returned
 *        value is expected to be a constant json object.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified index does not exist, return the default value.
 * @param idx The index of array elements.
 *
 * @return nlohamnn::json The json value to find
 */
template <typename Tp, typename std::enable_if<std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
const aura_json::json& GetJson(const aura_json::json &json, const Tp &default_value, DT_S32 idx)
{
    if (json.is_array() && (json.size() > static_cast<size_t>(idx)))
    {
        return json[idx];
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a numberic index and the returned
 *        value is expected to convert to a basic data type, such as int, string, vector and so on.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified index does not exist, return the default value.
 * @param idx The index of array elements.
 *
 * @return nlohamnn::json The json value to find
 */
template <typename Tp, typename std::enable_if<!std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
aura_json::json GetJson(const aura_json::json &json, const Tp &default_value, DT_S32 idx)
{
    if (json.is_array() && (json.size() > static_cast<size_t>(idx)))
    {
        return json[idx];
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a string and the returned
 *        value is expected to be a constant json object.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified key does not exist, return the default value.
 * @param key The key of json object.
 *
 * @return nlohamnn::json The json value to find.
 */
template <typename Tp, typename std::enable_if<std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
const aura_json::json& GetJson(const aura_json::json &json, const Tp &default_value, const std::string &key)
{
    if (json.contains(key))
    {
        return json.at(key);
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a string and the returned
 *        value is expected to convert to a basic data type, such as int, string, vector and so on.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified key does not exist, return the default value.
 * @param key The key of json object.
 *
 * @return nlohamnn::json The json value to find.
 */
template <typename Tp, typename std::enable_if<!std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
aura_json::json GetJson(const aura_json::json &json, const Tp &default_value, const std::string &key)
{
    if (json.contains(key))
    {
        return json[key];
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a numberic index and the returned
 *        value is expected to be a constant json object.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified index does not exist, return the default value.
 * @param idx The index of array elements.
 * @param args Variable parameter list.
 *
 * @return nlohamnn::json The json value to find
 */
template <typename Tp, typename ...ArgsType, typename std::enable_if<std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
const aura_json::json& GetJson(const aura_json::json &json, const Tp &default_value, DT_S32 idx, ArgsType &&...args)
{
    if (json.is_array() && (json.size() > static_cast<size_t>(idx)))
    {
        return GetJson(json[idx], default_value, std::forward<ArgsType>(args)...);
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a numberic index and the returned
 *        value is expected to convert to a basic data type, such as int, string, vector and so on.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified index does not exist, return the default value.
 * @param idx The index of array elements.
 * @param args Variable parameter list.
 *
 * @return nlohamnn::json The json value to find
 */
template <typename Tp, typename ...ArgsType, typename std::enable_if<!std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
aura_json::json GetJson(const aura_json::json &json, const Tp &default_value, DT_S32 idx, ArgsType &&...args)
{
    if (json.is_array() && (json.size() > static_cast<size_t>(idx)))
    {
        return GetJson(json[idx], default_value, std::forward<ArgsType>(args)...);
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a string and the returned
 *        value is expected to be a constant json object.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified key does not exist, return the default value.
 * @param key The key of json object.
 * @param args Variable parameter list.
 *
 * @return nlohamnn::json The json value to find.
 */
template <typename Tp, typename ...ArgsType, typename std::enable_if<std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
const aura_json::json& GetJson(const aura_json::json &json, const Tp &default_value, const std::string &key, ArgsType &&...args)
{
    if (json.contains(key))
    {
        return GetJson(json[key], default_value, std::forward<ArgsType>(args)...);
    }

    return default_value;
}

/**
 * @brief Get json object according to the given keys.
 *        This method matches when the key is a string and the returned
 *        value is expected to convert to a basic data type, such as int, string, vector and so on.
 *
 * @param json The aura_json::json object.
 * @param default_value If the specified key does not exist, return the default value.
 * @param key The key of json object.
 * @param args Variable parameter list.
 *
 * @return nlohamnn::json The json value to find.
 */
template <typename Tp, typename ...ArgsType, typename std::enable_if<!std::is_base_of<aura_json::json, Tp>::value>::type* = DT_NULL>
aura_json::json GetJson(const aura_json::json &json, const Tp &default_value, const std::string &key, ArgsType &&...args)
{
    if (json.contains(key))
    {
        return GetJson(json[key], default_value, std::forward<ArgsType>(args)...);
    }

    return default_value;
}

/**
 * @brief Returns whether the specified key exists in the json.
 *
 * @param json The aura_json::json object.
 * @param key  The key to be queried.
 *
 * @return DT_BOOL  Returns DT_TRUE if it exists, DT_FALSE if it does not exist.
 */
AURA_EXPORTS DT_BOOL JsonHasKeys(const aura_json::json &json, const std::string &key);

/**
 * @brief Returns whether the specified keys exists in the json.
 *
 * @param json The aura_json::json object.
 * @param keys The key list to be queried. The keys are filled in in the order of the
 *             hierarchical relationship in json. For example, if the json file is
 *             {"a": {"b": "c"}}, then to query whether "b" exists, you should to write as follows:
 *             JsonHasKeys(json, {"a", "b"}).
 *
 * @return DT_BOOL  Returns DT_TRUE if it exists, DT_FALSE if it does not exist.
 */
AURA_EXPORTS DT_BOOL JsonHasKeys(const aura_json::json &json, const std::initializer_list<std::string> &keys);

/**
 * @brief Returns whether the specified keys exist in the json.
 *
 * @param json The aura_json::json object.
 * @param key  The key to be queried.
 * @param args Variable parameter list, expect all keys to be string type.
 *
 * @return DT_BOOL  Returns DT_TRUE if it exists, DT_FALSE if it does not exist.
 */
template <typename ...ArgsType>
DT_BOOL JsonHasKeys(const aura_json::json &json, const std::string &key, ArgsType &&...args)
{
    if (!json.contains(key))
    {
        return DT_FALSE;
    }

    return JsonHasKeys(json, std::forward<ArgsType>(args)...);
}

/**
 * @brief Returns whether the specified keys exist in the json.
 *
 * @param json The aura_json::json object.
 * @param keys The key list to be queried.
 * @param args Variable parameter list, expect all keys to be std::initializer_list<std::string>.
 *
 * @return DT_BOOL  Returns DT_TRUE if it exists, DT_FALSE if it does not exist.
 */
template <typename ...ArgsType>
DT_BOOL JsonHasKeys(const aura_json::json &json, std::initializer_list<std::string> &keys, ArgsType &&...args)
{
    if (!JsonHasKeys(json, keys))
    {
        return DT_FALSE;
    }

    return JsonHasKeys(json, std::forward<ArgsType>(args)...);
}

/**
 * @}
*/
} // namespace aura

#endif // AURA_TOOLS_JSON_JSON_WRAPPER_HPP__
