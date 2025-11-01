#include "snpe/snpe_executor_impl.hpp"

namespace aura
{

std::unordered_map<std::string, SnpeExecutorImplCreator>& GetSnpeExecutorImplCreatorMap()
{
    static std::unordered_map<std::string, SnpeExecutorImplCreator> creator_map;
    return creator_map;
}

SnpeExecutorImplRegister::SnpeExecutorImplRegister(const std::string &name, SnpeExecutorImplCreator creator)
{
    auto &creator_map = GetSnpeExecutorImplCreatorMap();

    if (creator_map.find(name) == creator_map.end())
    {
        creator_map[name] = creator;
    }
}

Status SnpeExecutorImplRegister::Register()
{
    return Status::OK;
}

SnpeExecutor::SnpeExecutor(Context *ctx, const std::shared_ptr<SnpeModel> model, const NNConfig &config)
                           : NNExecutorInterface(ctx)
{
    do
    {
        const std::string framework_version = model->GetFrameWorkVersion();
        // revmove snpe patch version, onlys reserve MAJOR MINOR version
        std::string snpe_version = framework_version.substr(0, framework_version.find_last_of("."));

        auto &creator_map = GetSnpeExecutorImplCreatorMap();

        if (creator_map.find(snpe_version) == creator_map.end())
        {
            std::string error_string = "find creator fail. name: " + snpe_version + " not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_string.c_str());
            break;
        }

        m_impl.reset(creator_map[snpe_version](ctx, model, config));
    } while (0);
}

} // namespace aura