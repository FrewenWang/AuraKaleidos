#include "aura/tools/any_params/any_params.hpp"
#include <sstream>

namespace aura
{

AnyParams::AnyParams(Context *ctx) : m_ctx(ctx)
{}

any& AnyParams::operator[](const std::string &key)
{
    return m_any_params[key];
}

const any& AnyParams::operator[](const std::string &key) const
{
    return m_any_params.at(key);
}

DT_BOOL AnyParams::HasKeys(const std::string &key) const
{
    return m_any_params.find(key) != m_any_params.end();
}

DT_S32 AnyParams::Size()
{
    return m_any_params.size();
}

Status AnyParams::Clear()
{
    m_any_params.clear();
    return Status::OK;
}

std::string AnyParams::ToString() const
{
    std::stringstream ss;
    ss << (*this);
    return ss.str();
}

std::ostream& operator<<(std::ostream &os, const AnyParams &params)
{
    for (const auto &param : params.m_any_params)
    {
        os << "(" << param.first << ", " << param.second.type().name() << ")" << std::endl;
    }

    return os;
}

} // namespace aura
