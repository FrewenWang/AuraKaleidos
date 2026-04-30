//
// Created by wangzhijiang on 22-9-4.
//
#pragma once

#include "../include/vision/VisionAbility.h"

namespace vision {

class InfoPrinter {

public:
    static void print(VisionService *service, VisionResult *result);

private:
    static void printFaceInfo(VisionService *service, VisionResult *result);

    static void printGestureInfo(VisionService *service, VisionResult *result);

    static void printBodyInfo(VisionService *service, VisionResult *result);

    static void printLivingInfo(VisionService *service, VisionResult *result);
};

}
