#ifndef AURA_SAMPLE_OPS_HPP__
#define AURA_SAMPLE_OPS_HPP__

#include "sample.hpp"
#include "aura/ops.h"

using SampleOpsFunc = std::function<aura::Status(aura::Context*, aura::TargetType)>;

AURA_INLINE aura::Status InputParser(DT_S32 argc, DT_CHAR *argv[], std::string &help_info,
                                     const std::map<std::string, SampleOpsFunc> &func_map,
                                     SampleOpsFunc &func, aura::TargetType &type)
{
    if (argc != 3)
    {
        PrintHelpInfo(help_info);
        return aura::Status::ERROR;
    }

    std::string op     = std::string(argv[1]);
    std::string target = std::string(argv[2]);

    // find sample function
    std::transform(op.begin(), op.end(), op.begin(), ::tolower);
    auto it = func_map.find(op);
    if (it == func_map.end())
    {
        PrintHelpInfo(help_info);
        return aura::Status::ERROR;
    }
    func = it->second;

    // get target type
    if (GetTargetType(target, type) != aura::Status::OK)
    {
        PrintHelpInfo(help_info);
        return aura::Status::ERROR;
    }

    return aura::Status::OK;
}

#endif // AURA_SAMPLE_OPS_HPP__