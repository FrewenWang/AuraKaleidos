//
// Created by Li,Wendong on 2019-05-12.
//

#ifndef VISION_TI_INITIALIZER_H
#define VISION_TI_INITIALIZER_H

#include "vision_initializer.h"

namespace vision {

    class TiInitializer : public VisionInitializer {

    public:
        TiInitializer();
        ~TiInitializer();
        void do_config(Json::Value &jroot);
        virtual void do_init(FILE **files, char file_count);
        virtual void do_init(const char **mems, char mem_count);
    };
} // namespace vision

#endif //VISION_TI_INITIALIZER_H
