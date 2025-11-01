//
// Created by Li,Wendong on 2019-05-11.
//

#ifndef VISION_TI_MANAGER_H
#define VISION_TI_MANAGER_H

#include <abs_vision_manager.h>
#include <vision_request.h>
#include <vision_result.h>

namespace vision {

    class TiManager : public VisionManager<VisionRequest, VisionResult> {

    public:
        static TiManager *instance();
        TiManager();
        ~TiManager();
        void detect(VisionRequest *request, VisionResult *result);

    private:
        static TiManager *_s_instance;

    };
} // namespace vision
#endif //VISION_TI_MANAGER_H
