#include "np/np_executor_impl.hpp"

namespace aura
{

std::unordered_map<std::string, NpExecutorImplCreator>& GetNpExecutorImplCreatorMap()
{
    static std::unordered_map<std::string, NpExecutorImplCreator> creator_map;
    return creator_map;
}

NpExecutorImplRegister::NpExecutorImplRegister(const std::string &name, NpExecutorImplCreator creator)
{
    auto &creator_map = GetNpExecutorImplCreatorMap();

    if (creator_map.find(name) == creator_map.end())
    {
        creator_map[name] = creator;
    }
}

Status NpExecutorImplRegister::Register()
{
    return Status::OK;
}

NpExecutor::NpExecutor(Context *ctx, const std::shared_ptr<NpModel> model, const NNConfig &config) : NNExecutorInterface(ctx)
{
    do
    {
        const std::string framework_version = model->GetFrameWorkVersion();
        // get np majon minor version
        std::string np_major_minor_version = framework_version.substr(0, framework_version.find_last_of("."));
        // get np majon version
        std::string np_major_version = np_major_minor_version.substr(0, np_major_minor_version.find_last_of("."));

        auto &creator_map = GetNpExecutorImplCreatorMap();

        if (creator_map.find(np_major_version) == creator_map.end())
        {
            std::string error_string = "find creator fail. name: " + np_major_version + " not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_string.c_str());
            break;
        }

        m_impl.reset(creator_map[np_major_version](ctx, model, config));
    } while (0);
}

} // namespace aura