#include "xnn/xnn_executor_impl.hpp"

namespace aura
{

std::unordered_map<std::string, XnnExecutorImplCreator>& GetXnnExecutorImplCreatorMap()
{
    static std::unordered_map<std::string, XnnExecutorImplCreator> creator_map;
    return creator_map;
}

XnnExecutorImplRegister::XnnExecutorImplRegister(const std::string &name, XnnExecutorImplCreator creator)
{
    auto &creator_map = GetXnnExecutorImplCreatorMap();

    if (creator_map.find(name) == creator_map.end())
    {
        creator_map[name] = creator;
    }
}

Status XnnExecutorImplRegister::Register()
{
    return Status::OK;
}

XnnExecutor::XnnExecutor(Context *ctx, const std::shared_ptr<XnnModel> model, const NNConfig &config) : NNExecutorInterface(ctx)
{
    do
    {
        const std::string framework_version = model->GetFrameWorkVersion();
        // revmove xnn patch version, onlys reserve MAJOR MINOR version
        std::string xnn_version = framework_version.substr(0, framework_version.find_last_of("."));

        auto &creator_map = GetXnnExecutorImplCreatorMap();

        if (creator_map.find(xnn_version) == creator_map.end())
        {
            std::string error_string = "find creator fail. name: " + xnn_version + " not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_string.c_str());
            break;
        }
        m_impl.reset(creator_map[xnn_version](ctx, model, config));
    } while (0);
}

} // namespace aura