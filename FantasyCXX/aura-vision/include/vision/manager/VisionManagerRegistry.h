#ifndef VISION_MANAGER_REGISTRY_H
#define VISION_MANAGER_REGISTRY_H

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/manager/AbsVisionManager.h"

namespace aura::vision {

class AbsVisionManager;

using ManagerCreator = std::function<std::shared_ptr<AbsVisionManager>()>;
using ManagerCreatorMap = std::unordered_map<int, std::pair<std::string, ManagerCreator>>;

class VisionManagerMap {
public:
    struct ManagerItem {
        std::shared_ptr<AbsVisionManager> manager;
        std::string name;
    };

    /**
     * insert a new manager into managerMap
     * @param index vision ability ID
     * @param ctor  creator of the manager
     */
    void insert(int index, std::string name, const ManagerCreator& ctor);

    /**
     * whether the manager has been registered
     * @param index vision ability ID
     * @return
     */
    bool exists(int index);

    /**
     *  unregister the manager
     * @param index vision ability ID
     */
    void remove(int index);

    /**
     * @brief clear all items
     */
    void clear();

    /**
     * get the number of managers
     * @return number of managers
     */
    int getManagerCount();

    /**
     * get manager from map
     * @param index vision ability ID
     * @return manager pointer
     */
    std::shared_ptr<AbsVisionManager> getManager(int index);

    /**
     * @brief get all managers from map
     * @return vector of managers
     */
    std::vector<std::shared_ptr<AbsVisionManager>> getManagers();

    /**
     * get the manager name by ability ID
     * @param index vision ability ID
     * @return name of the manager
     */
    std::string getName(int index);

private:
    std::unordered_map<int, ManagerItem> _managers;
};

class VisionManagerRegistry {
public:
    /**
    * @brief constructor
    */ 
    VisionManagerRegistry();

    /**
     * @brief create managers from config
     * @return success(0) or not(-1)
     */
    int CreateFromCfg();

    int clear();

    /**
     * @brief register manager
     * @param name manager name
     * @param index vision ability ID
     * @param creator manager creator function
     * @return success(0) or not(-1)
     */
    static int registerVisionManagerCreator(const char* name, int index, const ManagerCreator& creator);

    /**
     * @brief get manager by vision ability ID
     * @param index vision ability ID
     * @return manager pointer
     */
    std::shared_ptr<AbsVisionManager> getManager(int index);

    /**
     * @brief get all registered managers
     * @return vector of managers
     */
    std::vector<std::shared_ptr<AbsVisionManager>> getManagers();

    /**
     * @brief get manager name by vision ability ID
     * @param index vision ability ID
     * @return manager name
     */
    std::string getManagerName(int index);

private:
    static ManagerCreatorMap* getManagerCreatorMap();     

private:
    std::unique_ptr<VisionManagerMap> _manager_map;
};

#define REGISTRY_STATE_VAR(file, line) manager_state_##file_##line

#define REGISTER_VISION_MANAGER(name, index, creator)                                                           \
    static auto REGISTRY_STATE_VAR(__FILE__, __LINE__) =                                                        \
            vision::VisionManagerRegistry::registerVisionManagerCreator(name, static_cast<int>(index), creator)

} // namespace vision

#endif //VISION_MANAGER_REGISTRY_H
