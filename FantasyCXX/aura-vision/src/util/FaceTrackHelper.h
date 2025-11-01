//
// Created by frewen on 2/3/23.
//
#pragma once

#include "vision/core/bean/FaceInfo.h"
#include <map>
#include <algorithm>
#include <cmath>
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/util/ObjectPool.hpp"
#include "vision/core/result/VisionResult.h"

namespace aura::vision {

/**
 * 临时人脸信息数据
 */
class CurFaceInfo {
public:
    /** 人脸距离 ROI position的欧氏距离  */
    float eulerDistance = 0.f;
    /** 模型检测的人脸结果 */
    FaceInfo *faceInfo = nullptr;

    void clear();

    ~CurFaceInfo(){};
};

/**
 * 跟踪的人脸信息
 */
class FaceTrackInfo : public ObjectPool<FaceTrackInfo> {
public:
    FaceTrackInfo() { };

    /**
     * 状态清除
     */
    void clear();

    /**
     * 资源回收
     */
    void recycle();

    ~FaceTrackInfo() { };

    /** 人脸置信度 */
    float rectConfidence = 0.f;
    /** 人脸距离 ROI position的欧氏距离 */
    float eulerDistance = 0.f;
    /** 人脸跟踪索引 Index */
    int64_t faceIndex = 0;
    /** 人脸框的左上、右下角坐标（left, top, right, bottom）*/
    VRect faceRect{};
    /** 人脸框的左上角坐标（left, top）*/
    VPoint rectLT{};
    /** 人脸框的左上角坐标（right, bottom）*/
    VPoint rectRB{};
    /** 记录当前人脸丢失的次数，暂定超过8次则清除当前人脸信息 */
    short faceLostCount = 0;
    /** 标识人脸是属于什么类型：NO_FACE(无效人脸)、DETECT(模型检测的人脸)、TRACK(已经丢失但丢失8帧以内的人脸) */
    FaceDetectType detectType = FaceDetectType::F_TYPE_UNKNOWN;
    /** 模型检出人脸与跟踪列表中人脸成功匹配的第几对 */
    uint8_t matchOrder = 0;
};

class FaceTrackHelper {

public:
    /** 将待输出字符串收集在一起集中输出 */
    std::stringstream stream;
    /** 跟踪列表中landmark检测前就是无效人脸loseCount统一为-1，landmark检测后新出现的无效人脸根据loseCount决定其类型是Unknow还是Track */
    static constexpr int DEFAULT_FACE_LOST_COUNT = 0;
    /** 允许人脸连续丢失的最大帧数，超过则清楚人脸的所有信息, loseCount默认为-1，从[0, 7]为8帧丢失区间范围 */
    static constexpr int MAX_FACE_LOST_COUNT = 8;

    /** 记录跟踪的所有人脸信息 */
    std::vector<FaceTrackInfo *> trackFacesList{};

    /**
     * 用于存储当前帧检测到的所有FaceInfo列表(经人脸框 + Landmark检测后的)
     * 默认按照置信度高度进行排序
     * 开启人脸ROI position之后按照主驾远近距离排序
     */
    std::vector<CurFaceInfo *> curOrderedFaces{};

    FaceTrackHelper(){};

    /**
     * 配置参数、跟踪列表初始化
     * @param cfg 配置参数
     */
    void init(RtConfig *cfg);

    /**
     * 对匹配列表、跟踪列表中的内容进行有条件清除
     */
    void reset();

    /**
     * 计算离主驾的距离
     * @param faceInfo 包含人脸框位置的人脸信息
     */
    float getDriverRoiDistance(const FaceInfo *faceInfo);

    /**
     * 根据距主驾Roi距离对检测结果Rect结果进行过滤排序
     * @param result 模型经 Rect 检测后的结果
     */
     void faceCopyRoiFaceInfo(VisionResult *result);

    /**
     * 将模型检测的人脸框复制到CurFaceInfo集合
     * @param result 模型经 Rect + Landmark检测后的结果
     * @return
     */
    bool faceCopyFaceInfo(const VisionResult *result);

    /**
     * 进行人脸跟踪和匹配
     * @return
     */
    bool faceTrackAndMatch(VisionResult *result);

    /**
     * 将模型检测的人脸一一与跟踪列表中人脸进行匹配
     */
    void faceMatch();

    /**
     * 将模型新检出的人脸添加到跟踪列表中
     */
    void faceAdd();

    /**
     * 将trackFacesList中的人脸复制到FaceInfo
     * @param pInfo FaceInfo
     * @return
     */
    bool faceCopyTrackList(FaceInfo **pInfo);

    ~FaceTrackHelper();

private:
    /** 外部参数配置 */
    RtConfig *mRtConfig = nullptr;
    /** 匹配到的人脸的索引列表 */
    std::vector<int> matchedFaceIndices{};
    /** 未成功匹配的人脸的索引列表 */
    std::vector<int> noMatchedFaceIndices{};
    /** 记录截至上一帧正在跟踪的人脸数量 */
    int preFaceCount = 0;
    /** 记录当前帧检测人脸数量 */
    short faceCountCurFrame = 0;
    /** 记录当前跟踪列表中的人脸数量 */
    short trackFaceCount = 0;
    /** 匹配成功的第几对人脸 */
    short order = 0;

};

} // namespace aura::vision
