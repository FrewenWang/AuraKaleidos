#include "vision/manager/VisionManagerRegistry.h"
#include "VisionManagerCfg.h.in"
#include "vision/config/runtime_config/RtConfig.h"
#include <iostream>

namespace aura::vision {

static const char* TAG = "VisionManagerRegistry";

void VisionManagerMap::insert(int index, std::string name, const ManagerCreator& ctor) {
    ManagerItem item;
    item.manager = ctor();
    item.name = name;
    item.manager->name = name;
    item.manager->index = index;
    _managers.insert({index, std::move(item)});
}

bool VisionManagerMap::exists(int index) {
    return _managers.find(index) != _managers.end();
}

void VisionManagerMap::remove(int index) {
    auto it = _managers.find(index);
    if (it == _managers.end()) {
        return;
    } else {
        _managers.erase(it);
    }
}

void VisionManagerMap::clear() {
    _managers.clear();
}

int VisionManagerMap::getManagerCount() {
    return _managers.size();
}

std::shared_ptr<AbsVisionManager> VisionManagerMap::getManager(int index) {
    auto it = _managers.find(index);
    if (it == _managers.end()) {
        return nullptr;
    }
    return it->second.manager;
}

std::string VisionManagerMap::getName(int index) {
    auto it = _managers.find(index);
    if (it == _managers.end()) {
        return "";
    }
    return it->second.name;
}

std::vector<std::shared_ptr<AbsVisionManager>> VisionManagerMap::getManagers() {
    std::vector<std::shared_ptr<AbsVisionManager>> managers;
    for (const auto& p : _managers) {
        managers.emplace_back(p.second.manager);
    }
    return managers;
}

VisionManagerRegistry::VisionManagerRegistry() {
    _manager_map.reset(new VisionManagerMap);
}

int VisionManagerRegistry::CreateFromCfg() {
    for (auto id : manager_indices) {
        auto idx = static_cast<int>(id);
        if (getManagerCreatorMap()->count(idx) == 0) {
            VLOGW(TAG, "no creator found for id=%d", idx);
            continue;
        }
        // VLOGD(TAG, "create %d", idx);
        ManagerCreator creator = (*getManagerCreatorMap())[idx].second;
        const auto& name = (*getManagerCreatorMap())[idx].first;
        _manager_map->insert(static_cast<int>(idx), name, creator);
    }
    // VLOGD(TAG, "create %d managers", _manager_map->getManagerCount());
    return _manager_map->getManagerCount() > 0 ? 0 : -1;
}

int VisionManagerRegistry::clear() {
    _manager_map->clear();
    return 0;
}

int VisionManagerRegistry::registerVisionManagerCreator(const char* name, int index, const ManagerCreator& creator) {
    if (getManagerCreatorMap()->count(index) != 0) {
        VLOGW(TAG, "vision manager creator index: %d has already been registered", index);
        return -1;
    } else {
        getManagerCreatorMap()->insert({index, {std::string(name), creator}});
        // VLOGD(TAG, "register vision manager creator (index: %d, name: %s)", index, name);
    }

    return 0;
}

std::shared_ptr<AbsVisionManager> VisionManagerRegistry::getManager(int index) {
    if (_manager_map->exists(index)) {
        return _manager_map->getManager(index);
    } else {
        VLOGW(TAG, "get vision manager failed, index: %d is not registered", index);
        return nullptr;
    }
}

std::vector<std::shared_ptr<AbsVisionManager>> VisionManagerRegistry::getManagers() {
    if (!_manager_map) {
        return std::vector<std::shared_ptr<AbsVisionManager>>();
    }
    return _manager_map->getManagers();
}

std::string VisionManagerRegistry::getManagerName(int index) {
    if (_manager_map->exists(index)) {
        auto name = _manager_map->getName(index);
        return name;
    } else {
        VLOGW(TAG, "get vision manager name failed, index: %d is not registered", index);
        return "";
    }
}

ManagerCreatorMap* VisionManagerRegistry::getManagerCreatorMap() {
    static ManagerCreatorMap* map = new ManagerCreatorMap;
    return map;
}

} // namespace vision
