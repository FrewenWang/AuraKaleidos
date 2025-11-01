package com.baidu.iov.vision.ability.core.result;

import com.baidu.iov.vision.ability.core.bean.FrameInfo;
import com.baidu.iov.vision.ability.util.ALog;

/**
 * 混合功能检测结果，适用于自由配置视觉能力的场合
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
@Deprecated
public class VisionResultVerfication extends AbsVisionResult {
    private static final String TAG = VisionResultVerfication.class.getSimpleName();
    public FaceResult mFaceResult;
    public GestureResult mGestureResult;
    public BodyPoseResult mBodyPoseResult;
    public FrameInfo mFrameInfo;

    public VisionResultVerfication() {
        super();
        setTag(TAG);
        if (ALog.DEBUG) {
            mPerfUtil.setTag(TAG);
        }
        mFaceResult = new FaceResult();
        mGestureResult = new GestureResult();
        mBodyPoseResult = new BodyPoseResult();
        mFrameInfo = new FrameInfo();
    }

    /**
     * @return 混合检测结果对象
     */
    public static VisionResultVerfication obtain() {
        VisionResultVerfication result = obtain(VisionResultVerfication.class);
        result.clearAllData(true);
        result.recyclable(true);
        return result;
    }

    @Override
    public void recyclable(boolean recyclable) {
        super.recyclable(recyclable);
        mFaceResult.recyclable(recyclable);
        mGestureResult.recyclable(recyclable);
        mBodyPoseResult.recyclable(recyclable);
    }

    @Override
    public void clearAllData(boolean clearAll) {
        super.clearAllData(clearAll);
        mFaceResult.clearAllData(clearAll);
        mGestureResult.clearAllData(clearAll);
        mBodyPoseResult.clearAllData(clearAll);
    }

    @Override
    public void clear() {
        super.clear();
        mFaceResult.clear();
        mGestureResult.clear();
        mBodyPoseResult.clear();
        mFrameInfo.clear();
    }

    public FaceResult faceResult() {
        return mFaceResult;
    }

    public GestureResult gestureResult() {
        return mGestureResult;
    }

    public BodyPoseResult bodyPoseResult() {
        return mBodyPoseResult;
    }

    public FrameInfo frameInfo() {
        return mFrameInfo;
    }
}