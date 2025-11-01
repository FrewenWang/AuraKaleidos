//
// Created by Li,Wendong on 2018/12/24.
//

#ifndef VISION_ABS_VISION_REQUEST_H
#define VISION_ABS_VISION_REQUEST_H

#include "vision/core/common/VConstants.h"
#include <list>
#include <unordered_map>

namespace aura::vision {

class AbsVisionRequest {
public:
    AbsVisionRequest();
    explicit AbsVisionRequest(unsigned char *frame);
    AbsVisionRequest(short width, short height, unsigned char *frame);
    AbsVisionRequest(short width, short height, unsigned char *frame, short mgr_id);
    virtual ~AbsVisionRequest();

public:
    /**
     * @brief verify the request, e.g. frame data and size
     * @return whether the request is valid
     */
    virtual bool verify();

    /**
     * @brief clear the frame only
     */
    virtual void clear();

    /**
     * @brief clear all data
     */
    virtual void clearAll();

    /**
     * @brief get the tag of the request
     * @return
     */
    virtual short tag() const;

    virtual void setFrame(unsigned char *f);

    virtual unsigned char * getFrame();

    virtual bool hasFrame();

public:
    short width;
    short height;
    FrameFormat format;
    unsigned char *frame;
    bool isSupportEbd = true;  // 是否支持ebd的帧数据
    short _mgr_id;
    /**
     * 请求时间戳
     */
    uint64_t timestamp;

private:
    const static short WIDTH_UNKNOWN = -1;
    const static short HEIGHT_UNKNOWN = -1;
};

} // namespace vision

#endif // VISION_ABS_VISION_REQUEST_H
