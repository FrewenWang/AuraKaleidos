#include "aura/tools/json/json_wrapper.hpp"
#include "aura/tools/json/json_helper.hpp"

#include <fstream>

namespace aura
{

Status JsonWrapper::Lock()
{
    auto &json_helper = JsonHelper::GetInstance();
    json_helper.Lock();

    return Status::OK;
}

Status JsonWrapper::UnLock()
{
    auto &json_helper = JsonHelper::GetInstance();
    json_helper.UnLock();

    return Status::OK;
}

Status JsonWrapper::UpdateArrayMap()
{
    auto &json_helper = JsonHelper::GetInstance();
    json_helper.SetArrayMap(m_array_map);

    return Status::OK;
}

Status JsonWrapper::ClearArrayMap()
{
    auto &json_helper = JsonHelper::GetInstance();
    json_helper.ClearArrayMap();

    return Status::OK;
}

Status JsonWrapper::SetContext(Context *ctx)
{
    auto &json_helper = JsonHelper::GetInstance();
    json_helper.SetContext(ctx);

    return Status::OK;
}

std::string JsonWrapper::GetName() const
{
    return m_name;
}

Status JsonWrapper::Write(aura_json::json &json_obj)
{
    std::ofstream ofstream(m_json_path, std::ofstream::out);
    if (!ofstream.is_open())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "open json file failed");
        return Status::ERROR;
    }

    ofstream << json_obj.dump(4) << std::endl;
    ofstream.close();

    return Status::OK;
}

Status JsonWrapper::Read(aura_json::json &json_obj)
{
    std::ifstream ifstream(m_json_path);
    if (!ifstream.is_open())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "open json file failed");
        return Status::ERROR;
    }

    json_obj = aura_json::json::parse(ifstream);
    ifstream.close();

    return Status::OK;
}

DT_BOOL JsonHasKeys(const aura_json::json &json, const std::string &key)
{
    return json.contains(key);
}

DT_BOOL JsonHasKeys(const aura_json::json &json, const std::initializer_list<std::string> &keys)
{
    const aura_json::json *json_ptr = &json;

    for (const std::string &name : keys)
    {
        if (!json_ptr->contains(name))
        {
            return DT_FALSE;
        }

        json_ptr = &(*json_ptr)[name];
    }

    return DT_TRUE;
}

} // namespace aura