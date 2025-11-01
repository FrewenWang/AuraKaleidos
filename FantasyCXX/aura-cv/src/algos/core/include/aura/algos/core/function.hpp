#ifndef AURA_ALGOS_CORE_FUNCTION_HPP__
#define AURA_ALGOS_CORE_FUNCTION_HPP__

#include "aura/ops/core.h"

#include <functional>

namespace aura
{

class AURA_EXPORTS FunctionImpl : public OpImpl
{
public:
    FunctionImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Function", target)
    {}

    template <typename FuncType, typename ...ArgsType>
    Status SetArgs(FuncType &&f, ArgsType &&...args)
    {
        m_run_func = std::bind(std::forward<FuncType>(f), std::forward<ArgsType>(args)...);
        return Status::OK;
    }

    Status Run() override
    {
        if (m_run_func)
        {
            return m_run_func();
        }

        AURA_ADD_ERROR_STRING(m_ctx, "run function is invalid");
        return Status::ERROR;
    }

    std::vector<const Array*> GetInputArrays() const override
    {
        return m_input_arrays;
    }

    std::vector<const Array*> GetOutputArrays() const override
    {
        return m_output_arrays;
    }

    AURA_VOID Dump(const std::string &prefix) const override
    {
        if (m_dump_func)
        {
            m_dump_func(prefix);
        }
    }

    template <typename ...ArgsType>
    Status SetInputArrays(ArgsType &&...args)
    {
        return AddArrays(m_input_arrays, std::forward<ArgsType>(args)...);
    }

    template <typename ...ArgsType>
    Status SetOutputArrays(ArgsType &&...args)
    {
        return AddArrays(m_output_arrays, std::forward<ArgsType>(args)...);
    }

    template <typename FuncType, typename ...ArgsType>
    Status BindDump(FuncType &&f, ArgsType &&...args)
    {
        m_dump_func = std::bind(std::forward<FuncType>(f),
                                std::forward<ArgsType>(args)...,
                                std::placeholders::_1,
                                m_name);
        return Status::OK;
    }

private:
    Status AddArrays(std::vector<const Array*> &vec_array)
    {
        AURA_UNUSED(vec_array);
        return Status::OK;
    }

    template <typename Tp, typename ...ArgsTypes>
    Status AddArrays(std::vector<const Array*> &vec_array, Tp *array, ArgsTypes ...args)
    {
        vec_array.emplace_back(array);
        return AddArrays(vec_array, std::forward<ArgsTypes>(args)...);
    }

    template <typename Tp, typename ...ArgsTypes>
    Status AddArrays(std::vector<const Array*> &vec_array, std::vector<Tp*> arrays, ArgsTypes ...args)
    {
        vec_array.insert(vec_array.end(), arrays.begin(), arrays.end());
        return AddArrays(vec_array, args...);
    }

private:
    std::function<Status()> m_run_func;
    std::function<AURA_VOID(const std::string&)> m_dump_func;

    std::vector<const Array*> m_input_arrays;
    std::vector<const Array*> m_output_arrays;
};

class AURA_EXPORTS Function : public Op
{
public:
    Function(Context *ctx, const OpTarget &target) : Op(ctx)
    {
        m_impl.reset(new FunctionImpl(m_ctx, target));
    }

    template <typename FuncType, typename ...ArgsType>
    Status SetArgs(FuncType &&f, ArgsType &&...args)
    {
        using RetType = typename std::result_of<FuncType(ArgsType...)>::type;
        static_assert(std::is_same<RetType, Status>::value, "The return value of run_func must be Status");

        FunctionImpl *func_impl = dynamic_cast<FunctionImpl*>(m_impl.get());

        if (MI_NULL == func_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "func_impl is null");
            return Status::ERROR;
        }

        return func_impl->SetArgs(std::forward<FuncType>(f), std::forward<ArgsType>(args)...);
    }

    template <typename ...ArgsType>
    Status SetInputArrays(ArgsType &&...args)
    {
        FunctionImpl *func_impl = dynamic_cast<FunctionImpl*>(m_impl.get());

        if (MI_NULL == func_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "func_impl is null");
            return Status::ERROR;
        }

        return func_impl->SetInputArrays(std::forward<ArgsType>(args)...);
    }

    template <typename ...ArgsType>
    Status SetOutputArrays(ArgsType &&...args)
    {
        FunctionImpl *func_impl = dynamic_cast<FunctionImpl*>(m_impl.get());

        if (MI_NULL == func_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "func_impl is null");
            return Status::ERROR;
        }

        return func_impl->SetOutputArrays(std::forward<ArgsType>(args)...);
    }

    template <typename FuncType, typename ...ArgsType>
    Status BindDump(FuncType &&f,  ArgsType &&...args)
    {
        FunctionImpl *func_impl = dynamic_cast<FunctionImpl*>(m_impl.get());

        if (MI_NULL == func_impl)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "func_impl is null");
            return Status::ERROR;
        }

        return func_impl->BindDump(std::forward<FuncType>(f), std::forward<ArgsType>(args)...);
    }
};

} // namespace aura

#endif