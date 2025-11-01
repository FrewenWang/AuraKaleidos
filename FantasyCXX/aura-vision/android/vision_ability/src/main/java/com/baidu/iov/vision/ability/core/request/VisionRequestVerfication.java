package com.baidu.iov.vision.ability.core.request;

import com.baidu.iov.vision.ability.core.manager.AbsVisionManager;
import com.baidu.iov.vision.ability.core.manager.VisionManager;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;
import com.baidu.iov.vision.ability.core.result.VisionResult;
import com.baidu.iov.vision.ability.task.AbsVisionTask;
import com.baidu.iov.vision.ability.task.iov.IovAbsVisionTask;

/**
 * 混合功能检测请求，适用于自由配置视觉能力的场合
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
@Deprecated
public class VisionRequestVerfication extends AbsVisionRequest {
    public static final Class<? extends AbsVisionManager> DEF_MANAGER_CLASS = VisionManager.class;
    public static final Class<? extends AbsVisionTask> DEF_TASK_CLASS = IovAbsVisionTask.class;
    public static final Class<? extends AbsVisionResult> DEF_RESULT_CLASS = VisionResult.class;
    /**
     * 执行检测的功能
     */
    public short mAbilityType;
    private FaceRequest mFaceRequest;

    public VisionRequestVerfication() {
        mManagerClass = DEF_MANAGER_CLASS;
        mTaskClass = DEF_TASK_CLASS;
        mResultClass = DEF_RESULT_CLASS;
        mFaceRequest = new FaceRequest();
    }

    public static VisionRequestVerfication obtain() {
        VisionRequestVerfication request = obtain(VisionRequestVerfication.class);
        request.setClearAllData(true);
        request.setRecyclable(true);
        return request;
    }

    @Override
    public void setRecyclable(boolean recyclable) {
        super.setRecyclable(recyclable);
        mFaceRequest.setRecyclable(recyclable);
        if (mResult != null) {
            mResult.recyclable(recyclable);
        }
    }

    @Override
    public void setClearAllData(boolean clearAll) {
        super.setClearAllData(clearAll);
        mFaceRequest.setClearAllData(clearAll);
        if (mResult != null) {
            mResult.clearAllData(clearAll);
        }
    }

    @Override
    public void clear() {
        super.clear();
        mFaceRequest.clear();
    }

    public FaceRequest faceRequest() {
        return mFaceRequest;
    }

    public void setFaceLandmarkData(float[] landmark) {
        mFaceRequest.mLandmarkData = landmark;
    }
}