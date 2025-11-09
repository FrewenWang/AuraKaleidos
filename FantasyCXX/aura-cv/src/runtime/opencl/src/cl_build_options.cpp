#include "aura/runtime/opencl/cl_build_options.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

Status CLBuildOptions::AddOption(const std::string &key, const std::string &value)
{
    if (key.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "bad key");
        return Status::ERROR;
    }
    m_options += "-D" + key;
    if (!value.empty())
    {
        m_options += "=" + value;
    }
    m_options += " ";

    return Status::OK;
}

std::string CLBuildOptions::ToString(ElemType type)
{
    if (m_tbl.empty() || ElemType::INVALID == type)
    {
        return m_options;
    }
    if (!(m_tbl.size() & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "bad tbl");
        return std::string();
    }

    auto Strip = [](const std::string &str) -> std::string
    {
        size_t i = 0, j = 0;
        for (; i < str.size(); ++i)
        {
            if (str[i] != ' ')
            {
                break;
            }
        }
        std::string right(str.begin() + i, str.end());

        for (; j < right.size(); ++j)
        {
            if (right[right.size() - j - 1] != ' ')
            {
                break;
            }
        }
        std::string ret(right.begin(), right.end() - j);

        return ret;
    };

    auto IsEmpty = [](const std::string &str) -> DT_BOOL
    {
        for (auto it = str.begin(); it != str.end(); ++it)
        {
            if (*it != ' ')
            {
                return DT_FALSE;
            }
        }
        return DT_TRUE;
    };

    auto Split = [&](const std::string &str) -> std::vector<std::string>
    {
        std::vector<std::string> v_str;
        std::string tmp;
        for (size_t i = 0; i < str.size(); ++i)
        {
            if (',' == str[i] && !IsEmpty(tmp))
            {
                v_str.push_back(Strip(tmp));
                tmp.clear();
            }
            else if (str[i] != ',')
            {
                tmp.push_back(str[i]);
            }
        }
        if (!IsEmpty(tmp))
        {
            v_str.push_back(Strip(tmp));
        }

        return v_str;
    };

    std::string ret;
    std::string type_str = ElemTypesToString(type);

    std::vector<std::string> types = Split(m_tbl[0]);
    size_t idx = 0;
    for (; idx < types.size(); ++idx)
    {
        if (types[idx] == type_str)
        {
            break;
        }
    }
    if (types.size() == idx)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "bad type");
        return std::string();
    }

    for (size_t i = 1; i < m_tbl.size(); i += 2)
    {
        ret += "-D" + Strip(m_tbl[i]);
        std::vector<std::string> defs = Split(m_tbl[i + 1]);
        if (!defs.empty())
        {
            if (defs.size() != types.size())
            {
                if (defs.size() == 1)
                {
                    ret += "=" + defs[0];
                }
                else
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "bad table");
                    return std::string();
                }
            }
            else
            {
                ret += "=" + defs[idx];
            }
        }
        ret += " ";
    }
    ret += m_options;

    return ret;
}

} // namespace aura