#include "aura/tools/json/json_serialize.h"
#include <mutex>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup json Json
 *    @{
 *       @defgroup json_helper_class JsonHelper Class
 *    @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup json_helper_class
 * @{
 */

/**
 * @brief Help json serialize, use static instance to store vlues used in json serialize.
 */
class AURA_EXPORTS JsonHelper
{
public:
    /**
     * @brief Get instance of JsonHelper class.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    static JsonHelper& GetInstance();

    /**
     * @brief Lock JsonHelper object to protect the array_map from being overwritten.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Lock();

    /**
     * @brief UnLock JsonHelper object.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status UnLock();

    /**
     * @brief Set JsonHelper object's array_map.
     *
     * @param array_map The array_map store array's pointer and array dump path.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status SetArrayMap(const std::unordered_map<const Array*, std::string> &array_map);

    /**
     * @brief Clear JsonHelper object's array_map.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status ClearArrayMap();

    /**
     * @brief Get array's dump path.
     *
     * @param array The array's pointer.
     *
     * @return Array's dump path.
     */
    std::string GetArrayPath(const Array *array) const;

    /**
     * @brief Set JsonHelper object's context.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status SetContext(Context *ctx);

    /**
     * @brief Get JsonHelper object's context.
     *
     * @return The pointer of context, null if false.
     */
    Context* GetContext() const;

private:
    /**
     * @brief Destructor to ensure proper cleanup in derived classes.
     */
    JsonHelper() : m_ctx(DT_NULL)
    {}

    AURA_DISABLE_COPY_AND_ASSIGN(JsonHelper);

    Context *m_ctx;
    std::mutex m_mutex;
    std::unordered_map<const Array*, std::string> m_array_map;

    static std::unique_ptr<JsonHelper> m_instance;
};

/**
 * @}
*/
} // namespace aura