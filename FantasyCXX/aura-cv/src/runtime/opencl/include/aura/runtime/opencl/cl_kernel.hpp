#ifndef AURA_RUNTIME_OPENCL_CL_KERNEL_HPP__
#define AURA_RUNTIME_OPENCL_CL_KERNEL_HPP__

#include "aura/runtime/opencl/cl_runtime.hpp"
#include "aura/runtime/logger.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup cl OpenCL
 * @}
 */

namespace aura
{

/**
 * @addtogroup cl
 * @{
 */

/**
 * @brief Represents an OpenCL kernel wrapper.
 *
 * This class encapsulates an OpenCL kernel for execution and provides
 * methods to initialize, run, and retrieve information about the kernel.
 */
class AURA_EXPORTS CLKernel
{
public:

    /**
     * @brief Default constructor for CLKernel.
     */
    CLKernel();

    /**
     * @brief Constructor for CLKernel.
     *
     * @param ctx The Context object.
     * @param program_name Name of the OpenCL program.
     * @param kernel_name Name of the kernel within the program.
     * @param source Optional source code for the kernel.
     * @param build_options Optional build options for the kernel.
     */
    CLKernel(Context *ctx,
             const std::string &program_name,
             const std::string &kernel_name,
             const std::string &source = std::string(),
             const std::string &build_options = std::string());

    /**
     * @brief Destructor for CLKernel.
     *
     * Calls the DeInitialize() method to reset the object's state before destruction.
     */
    ~CLKernel();

    /**
     * @brief Deinitializes the CLKernel object.
     *
     * Resets internal variables and releases resources held by this object.
     */
    DT_VOID DeInitialize();

    /**
     * @brief Gets the OpenCL kernel associated with this CLKernel object.
     *
     * @return Shared pointer to the OpenCL kernel.
     */
    std::shared_ptr<cl::Kernel> GetClKernel() const;

    /**
     * @brief Runs the OpenCL kernel.
     *
     * @tparam Tp Variadic template representing kernel arguments
     *
     * @param tp The type of Kernel arguments.
     * @param cl_global Global work size for the kernel.
     * @param cl_local Local work size for the kernel.
     * @param cl_event Pointer to a cl::Event object for kernel event management (optional).
     * @param cl_events Vector of events to wait for (optional).
     * @param cl_offset Offset for the kernel execution (optional).
     *
     * @return cl_int The OpenCL error code.
     */
    template<typename... Tp>
    cl_int Run(Tp... tp,
               cl::NDRange cl_global,
               cl::NDRange cl_local,
               cl::Event *cl_event = DT_NULL,
               const std::vector<cl::Event> &cl_events = std::vector<cl::Event>(),
               cl::NDRange cl_offset = cl::NDRange())
    {
        cl_int cl_err = CL_INVALID_VALUE;

        if (m_cl_kernel)
        {
            if (CheckKenrelWorkSize(cl_global, cl_local) == Status::OK)
            {
                auto cl_kernel_func = cl::KernelFunctor<Tp...>(*m_cl_kernel);
                cl::EnqueueArgs enqueue_arg(*m_cl_queue, cl_events, cl_offset, cl_global, cl_local);

                if (DT_NULL == cl_event)
                {
                    cl::Event cl_event = cl_kernel_func(enqueue_arg, std::forward<Tp>(tp)..., cl_err);
                    if (CL_SUCCESS == cl_err)
                    {
                        cl_err = cl_event.wait();
                    }
                }
                else
                {
                    *cl_event = cl_kernel_func(enqueue_arg, std::forward<Tp>(tp)..., cl_err);
                }
            }
        }

        return cl_err;
    }

    /**
     * @brief Checks if the CLKernel object is valid.
     *
     * @return True if the kernel is valid; otherwise, false.
     */
    DT_BOOL IsValid() const;

    /**
     * @brief Retrieves the maximum group size of the kernel.
     *
     * @return The maximum group size supported by the kernel.
     */
    DT_U32 GetMaxGroupSize() const;

    /**
     * @brief Retrieves the name of the kernel.
     *
     * @return The name of the kernel.
     */
    std::string GetKernelName() const;

private:
    /**
     * @brief Checks the validity of the kernel work size before execution.
     *
     * @param cl_global The global work size.
     * @param cl_local The local work size.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status CheckKenrelWorkSize(cl::NDRange &cl_global, cl::NDRange &cl_local);

private:
    Context *m_ctx;                               /*!< The pointer to the Context object. */
    std::shared_ptr<cl::CommandQueue> m_cl_queue; /*!< Shared pointer to the OpenCL command queue. */
    std::shared_ptr<cl::Kernel> m_cl_kernel;      /*!< Shared pointer to the OpenCL kernel. */
    DT_S32 m_max_group_size;                      /*!< Maximum group size supported by the kernel. */
    std::string m_kernel_name;                    /*!< The name of the kernel. */
};

/**
 * @}
 */

/**
 * @brief Check validity of multiple CLKernels.
 * 
 * @param ctx The pointer to the Context object.
 * @param cl_kernels CLKernel array vector to be checked.
 * @param num The number of CLKernel used to specify the check, default 0, not need to check
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_INLINE Status CheckCLKernels(Context *ctx, const std::vector<CLKernel> &cl_kernels, DT_U32 num = 0)
{
    if ((num != 0) && (num != cl_kernels.size()))
    {
        std::string info = "num(" + std::to_string(num) + ") must be same with cl kernels size(" + std::to_string(cl_kernels.size()) + ")";
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return Status::ERROR;
    }

    for (DT_U32 i = 0; i < cl_kernels.size(); i++)
    {
        if (!cl_kernels[i].IsValid())
        {
            std::string info = "total " + std::to_string(num) + " cl kernels, but " + std::to_string(i) + "th kernel is invalid";
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }
    }

    return Status::OK;
}

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_KERNEL_HPP__
