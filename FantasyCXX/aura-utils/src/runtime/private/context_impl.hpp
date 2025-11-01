#pragma once

#include "aura/utils/runtime/context/context.hpp"
#include "aura/utils/runtime/context/context_config.hpp"

namespace aura::utils
{

class Context::Impl
{
private:
public:
    Impl(const ContextConfig&config);

    ~Impl();
};


} // namespace aura::utils
