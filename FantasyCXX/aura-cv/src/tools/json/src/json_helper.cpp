#include "aura/tools/json/json_helper.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

std::unique_ptr<JsonHelper> JsonHelper::m_instance;

JsonHelper& JsonHelper::GetInstance()
{
    static std::once_flag flag;
    std::call_once(flag, [&](){m_instance.reset(new JsonHelper);});

    return *m_instance;
}

Status JsonHelper::Lock()
{
    m_mutex.lock();

    return Status::OK;
}

Status JsonHelper::UnLock()
{
    m_mutex.unlock();

    return Status::OK;
}

Status JsonHelper::SetArrayMap(const std::unordered_map<const Array*, std::string> &array_map)
{
    m_array_map = array_map;

    return Status::OK;
}

Status JsonHelper::ClearArrayMap()
{
    m_array_map.clear();

    return Status::OK;
}

std::string JsonHelper::GetArrayPath(const Array *array) const
{
    if (m_array_map.find(array) != m_array_map.end())
    {
        return m_array_map.at(array);
    }

    return std::string();
}

Status JsonHelper::SetContext(Context *ctx)
{
    m_ctx = ctx;

    return Status::OK;
}

Context* JsonHelper::GetContext() const
{
    return m_ctx;
}

} // namespace aura