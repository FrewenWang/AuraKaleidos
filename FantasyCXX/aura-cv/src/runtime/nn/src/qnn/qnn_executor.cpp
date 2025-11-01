#include "qnn/qnn_executor_impl.hpp"

namespace aura
{

std::unordered_map<std::string, QnnExecutorImplCreator>& GetQnnExecutorImplCreatorMap()
{
    static std::unordered_map<std::string, QnnExecutorImplCreator> creator_map;
    return creator_map;
}

QnnExecutorImplRegister::QnnExecutorImplRegister(const std::string &name, QnnExecutorImplCreator creator)
{
    auto &creator_map = GetQnnExecutorImplCreatorMap();

    if (creator_map.find(name) == creator_map.end())
    {
        creator_map[name] = creator;
    }
}

Status QnnExecutorImplRegister::Register()
{
    return Status::OK;
}

QnnExecutor::QnnExecutor(Context *ctx, const std::shared_ptr<QnnModel> model, const NNConfig &config) : NNExecutorInterface(ctx)
{
    do
    {
        const std::string framework_version = model->GetFrameWorkVersion();
        // revmove qnn patch version, onlys reserve MAJOR MINOR version
        std::string qnn_version = framework_version.substr(0, framework_version.find_last_of("."));

        auto &creator_map = GetQnnExecutorImplCreatorMap();

        if (creator_map.find(qnn_version) == creator_map.end())
        {
            std::string error_string = "find creator fail. name: " + qnn_version + " not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_string.c_str());
            break;
        }

        m_impl.reset(creator_map[qnn_version](ctx, model, config));
    } while (0);
}

} // namespace aura