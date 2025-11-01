//
// Created by Frewen.Wang on 25-9-14.
//
#pragma once

#include "aura/utils/runtime.h"
#include "common.hpp"

namespace aura::cv
{

class AURA_EXPORTS BaseOp {
public:
    /**
     * @brief Constructor.
     *
     * @param ctx The pointer to the Context object.
     * @param name The name of the operator.
     * @param target The target platform for the operator.
     */
    BaseOp(Context *ctx, const std::string &name, const OpTarget &target) : m_ctx(ctx), m_name(name), m_target(target)
    {
    }


private:
protected:
    Context *m_ctx;                     /*!< Pointer to the Context object */
    std::shared_ptr<OpImpl> m_impl;     /*!< Shared pointer to the implementation class. */
    OpTarget m_target;                  /*!< Target platform for the operator. */
    MI_BOOL m_ready;                    /*!< Flag indicating whether the operator is ready for execution. */
};




/**
 * @brief Base implementation class.
 *
 * This class provides a base implementation for operators.
 */
class AURA_EXPORTS OpImpl {
public:
    /**
     * @brief Constructor.
     *
     * @param ctx The pointer to the Context object.
     * @param name The name of the operator.
     * @param target The target platform for the operator.
     */
    OpImpl(Context *ctx, const std::string &name, const OpTarget &target) : m_ctx(ctx), m_name(name), m_target(target) {
    }

    /**
     * @brief Destructor.
    */
    virtual ~OpImpl() {
        m_ctx = MI_NULL;
    }

    /**
     * @brief Initialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Initialize() {
        return Status::OK;
    }

    /**
     * @brief Run operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Run() = 0;

    /**
     * @brief Deinitialize operator implementation.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status DeInitialize() {
        return Status::OK;
    }

    /**
     * @brief Get a string representation of the operator implementation.
     *
     * @return The string representation.
     */
    virtual std::string ToString() const {
        return std::string();
    }

    /**
     * @brief Dump information about the operator implementation.
     *
     * @param prefix The prefix in the dump.
     */
    virtual AURA_VOID Dump(const std::string &prefix) const {
        AURA_UNUSED(prefix);
    }

    /**
     * @brief Get the target platform for the operator implementation.
     *
     * @return The target platform.
     */
    OpTarget GetOpTarget() const {
        return m_target;
    }

    /**
     * @brief Get the name of the operator implementation.
     *
     * @return The name of the operator implementation.
     */
    std::string GetName() const {
        return m_name;
    }

    /**
     * @brief Get the input arrays.
     *
     * @return A vector of input arrays.
     */
    virtual std::vector<const Array *> GetInputArrays() const {
        return {};
    }

    /**
     * @brief Get the output arrays.
     *
     * @return A vector of output arrays.
     */
    virtual std::vector<const Array *> GetOutputArrays() const {
        return {};
    }

protected:
    Context *m_ctx; /*!< Pointer to the Context object. */
    std::string m_name; /*!< Name of the operator implementation. */
    OpTarget m_target; /*!< Target platform for the operator implementation. */
};

}
