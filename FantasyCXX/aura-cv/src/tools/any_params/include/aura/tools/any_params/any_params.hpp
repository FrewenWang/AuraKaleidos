#ifndef AURA_TOOLS_ANY_PARAMS_ANY_PARAMS_HPP__
#define AURA_TOOLS_ANY_PARAMS_ANY_PARAMS_HPP__

#include "aura/tools/any_params/any.hpp"
#include "aura/runtime/context.h"
#include "aura/runtime/logger.h"

#include <unordered_map>

namespace aura
{

/**
 * @brief Class that stores values ​​of any type.
 */
class AURA_EXPORTS AnyParams
{
public:
    /**
     * @brief Single parameter constructor, if ctx is empty, no log will be output.
     *
     * @param ctx Pointer to the associated context.
     */
    AnyParams(Context *ctx = MI_NULL);

    /**
     * @brief Get the value of specified key, you can also use this method
     *        to set the value of the specified key.
     *        For example: inputs["num"] = 1.
     *
     * @param key String of specified key.
     *
     * @return Reference to the value of specified key.
     */
    aura::any& operator[](const std::string &key);

    /**
     * @brief Get the value of specified key.
     *
     * @param key String of specified key.
     *
     * @return Const reference to the value of specified key.
     */
    const aura::any& operator[](const std::string &key) const;

    /**
     * @brief Query whether the specified key exists.
     *
     * @param key String of specified key.
     *
     * @return MI_TRUE if the key exists, otherwise MI_FALSE.
     */
    MI_BOOL HasKeys(const std::string &key) const;

    /**
     * @brief Query whether the specified key exists.
     *
     * @param key  String of specified key.
     * @param args Variable parameter list, each parameter is expected to be of type string.
     *
     * @return MI_TRUE if the key exists, otherwise MI_FALSE.
     */
    template <typename ...ArgsType>
    MI_BOOL HasKeys(const std::string &key, const ArgsType &&...args) const
    {
        return HasKeys(key) && HasKeys(std::forward<ArgsType>(args)...);
    }

    /**
     * @brief Get the value of specified key.
     *
     * @param key   String of specified key.
     * @param value Reference passed in from outside. If it is not queried,
     *              the value will not be modified.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp>
    Status Get(const std::string &key, Tp &value) const
    {
        auto iter = m_any_params.find(key);

        if (iter == m_any_params.end())
        {
            std::string error_msg = "the specified key `" + key + "` does not exist";
            AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
            return Status::ERROR;
        }

        if (iter->second.empty())
        {
            std::string error_msg = "the specified key `" + key + "` returns an empty any object";
            AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
            return Status::ERROR;
        }

        if (iter->second.type() != typeid(Tp) && iter->second.type() != typeid(std::reference_wrapper<Tp>))
        {
            std::string error_msg = "the type stored in the specified key `" + key +"` does not match the given type";
            AURA_ADD_ERROR_STRING(m_ctx, error_msg.c_str());
            return Status::ERROR;
        }

        value = aura::any_cast<Tp>(iter->second);
        return Status::OK;
    }

    /**
     * @brief Return the number of stored key-value pairs.
     *
     * @return The number of stored key-value pairs.
     */
    MI_S32 Size();

    /**
     * @brief Clear stored key-value pair information.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Clear();

    /**
     * @brief Returns a string concatenated by all stored keys.
     *
     * @return A string concatenated by all stored keys.
     */
    std::string ToString() const;

    /**
     * @brief Define a friend function to overload the << operator, which allows us to
     *        directly output the contents of an AnyParams object to an ostream.
     *
     * @param os     Reference to the output stream where the object will be printed.
     * @param params Constant reference to the AnyParams object that we want to print.
     *
     * @return Return the ostream object to allow chaining of << operators.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const AnyParams &params);

private:
    Context *m_ctx;  /*!< Pointer to the associated context. */
    std::unordered_map<std::string, any> m_any_params; /*!< Store key and corresponding value
                                                            information of std::any type. >*/
};

} // namespace aura

#endif // AURA_TOOLS_ANY_PARAMS_ANY_PARAMS_HPP__