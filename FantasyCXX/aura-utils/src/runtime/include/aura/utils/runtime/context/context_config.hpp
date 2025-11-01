
#pragma once
namespace aura::utils
{
class ContextConfig
{
public:
    ContextConfig()  = default;
    ~ContextConfig() = default;

    ContextConfig(const ContextConfig &)            = delete;
    ContextConfig &operator=(const ContextConfig &) = delete;

    ContextConfig(ContextConfig &&) noexcept            = default;
    ContextConfig &operator=(ContextConfig &&) noexcept = default;
};


} // namespace aura::utils
