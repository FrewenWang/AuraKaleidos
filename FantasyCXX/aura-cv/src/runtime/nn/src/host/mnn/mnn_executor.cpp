#include "mnn/mnn_executor_impl.hpp"

namespace aura
{

std::unordered_map<std::string, MnnExecutorImplCreator>& GetMnnExecutorImplCreatorMap()
{
    static std::unordered_map<std::string, MnnExecutorImplCreator> creator_map;
    return creator_map;
}

MnnExecutorImplRegister::MnnExecutorImplRegister(const std::string &name, MnnExecutorImplCreator creator)
{
    auto &creator_map = GetMnnExecutorImplCreatorMap();

    if (creator_map.find(name) == creator_map.end())
    {
        creator_map[name] = creator;
    }
}

Status MnnExecutorImplRegister::Register()
{
    return Status::OK;
}

MnnExecutor::MnnExecutor(Context *ctx, const std::shared_ptr<MnnModel> model, const NNConfig &config) : NNExecutorInterface(ctx)
{
    do
    {
        const std::string framework_version = model->GetFrameWorkVersion();

        auto &creator_map = GetMnnExecutorImplCreatorMap();

        if (creator_map.find(framework_version) == creator_map.end())
        {
            std::string error_string = "find creator fail. name: " + framework_version + " not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_string.c_str());
            break;
        }

        m_impl.reset(creator_map[framework_version](ctx, model, config));
    } while (0);
}

} // namespace aura