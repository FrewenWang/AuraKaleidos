#ifndef AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_HPP__
#define AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_HPP__

#include "aura/runtime/memory.h"

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup xtensa Xtensa
 *    @{
 *       @defgroup Xtensa_host Xtensa Host
 *    @}
 * @}
 */

struct application_symbol_tray;

namespace aura
{
/**
 * @addtogroup Xtensa_host
 * @{
 */

/**
 * @brief Enumeration representing Hexagon power levels.
 */
enum class XtensaPowerLevel
{
    DEFAULT = 0, /*!< Default power level. */
    STANDBY,     /*!< Standby power level. */
    LOW,         /*!< Low power level. */
    NORMAL,      /*!< Normal power level. */
    TURBO        /*!< Turbo power level. */
};

class XtensaRpcParam;

using PilRpcFunc = MI_S32 (*)(const MI_CHAR*, MI_U8*, MI_S32);

/**
 * @brief Implementation class for XtensaEngine.
 *
 * This class encapsulates the private implementation details of XtensaEngine, providing an
 * interface for interacting with Xtensa VDSP hardware acceleration. It allows users to enable
 * Xtensa VDSP acceleration, run specific operations, and obtain real-time information about
 * the Xtensa VDSP device.
 */
class AURA_EXPORTS XtensaEngine
{
public:
    /**
     * @brief Constructor for XtensaEngine.
     *
     * @param ctx The pointer to the Context object.
     * @param enable_xtensa Whether to enable Xtensa acceleration.
     * @param pil_name The pic lib name should be loaded.
     */
    XtensaEngine(Context *ctx, MI_BOOL enable_xtensa, const std::string &pil_name, XtensaPriorityLevel priority);

    /**
     * @brief Destructor for XtensaEngine.
     */
    ~XtensaEngine();

    AURA_DISABLE_COPY_AND_ASSIGN(XtensaEngine);

    /**
     * @brief Run the Xtensa Engine with the specified parameters.
     *
     * @param package The name of the package.
     * @param module The name of the operation module.
     * @param src_list The Mat object containing operation source.
     * @param src_num The number of the source mat.
     * @param dst_list The Mat object containing operation dst.
     * @param dst_num The number of the dst mat.
     * @param param The pointer object containing operation parameters.
     * @param len The length of the operation parameters length.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status Run(const std::string &package, const std::string &module, XtensaRpcParam &rpc_param);

    /**
     * @brief cache ddr data to cpu side
     * 
     * @param fd  data address handle.
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status CacheStart(MI_S32 fd);

    /**
     * @brief cache cpu data to ddr side
     * 
     * @param fd  data address handle.
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status CacheEnd(MI_S32 fd);

    /**
     * @brief Map the input buffer
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status MapBuffer(const Buffer &buffer);

    /**
     * @brief Unmap the input buffer
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status UnmapBuffer(const Buffer &buffer);

    /**
     * @brief Get Mapping buffer dsp side virtual addres
     *
     * @return the dsp side virtual address.
     */
    MI_U32 GetDeviceAddr(const Buffer &buffer);

    /**
     * @brief Set Xtensa power level
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status SetPower(XtensaPowerLevel level);

#if defined(AURA_BUILD_XPLORER)
    /**
     * @brief Get the application symbol tray.
     *
     * @return application symbol tray.
     */
    application_symbol_tray& GetSymbolTray();

    /**
     * @brief Insert pil rpc function to map.
     *
     * @param rpc_func The function pointer of dsp pil.
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status RegisterRpcFunc(PilRpcFunc rpc_func);
#endif

private:
    class Impl;                   /*!< Forward declaration of implementation class. */
    std::shared_ptr<Impl> m_impl; /*!< Pointer to the implementation class instance. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_HPP__