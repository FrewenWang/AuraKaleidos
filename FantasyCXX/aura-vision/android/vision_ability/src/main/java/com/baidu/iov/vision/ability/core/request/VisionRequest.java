package com.baidu.iov.vision.ability.core.request;

import com.baidu.iov.vision.ability.core.manager.VisionManager;
import com.baidu.iov.vision.ability.core.manager.AbsVisionManager;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;
import com.baidu.iov.vision.ability.core.result.VisionResult;
import com.baidu.iov.vision.ability.task.AbsVisionTask;
import com.baidu.iov.vision.ability.task.iov.IovAbsVisionTask;

import static com.baidu.iov.vision.ability.config.VisionConfig.ABILITY_UNKNOWN;

/**
 * 混合功能检测请求，适用于自由配置视觉能力的场合
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class VisionRequest extends AbsVisionRequest {

    public static final Class<? extends AbsVisionManager> DEF_MANAGER_CLASS = VisionManager.class;
    public static final Class<? extends AbsVisionTask>       DEF_TASK_CLASS    = IovAbsVisionTask.class;
    public static final Class<? extends AbsVisionResult>  DEF_RESULT_CLASS  = VisionResult.class;

    public short mSingleAbilityType;
    private FaceRequest mFaceRequest;
    private GestureRequest mGestureRequest;
//    private PersonVehicleRequest mPersonVehicleRequest;

    public VisionRequest() {
        mSingleAbilityType = ABILITY_UNKNOWN;
        mManagerClass = DEF_MANAGER_CLASS;
        mTaskClass = DEF_TASK_CLASS;
        mResultClass = DEF_RESULT_CLASS;

        mFaceRequest = new FaceRequest();
        mGestureRequest = new GestureRequest();
//        mPersonVehicleRequest = new PersonVehicleRequest();
    }

    public static VisionRequest obtain() {
        VisionRequest request = obtain(VisionRequest.class);
        request.setClearAllData(true);
        request.setRecyclable(true);
        return request;
    }

    @Override
    public void setRecyclable(boolean recyclable) {
        super.setRecyclable(recyclable);
        mFaceRequest.setRecyclable(recyclable);
        mGestureRequest.setRecyclable(recyclable);
//        mPersonVehicleRequest.setRecyclable(recyclable);
        if (mResult != null) {
            mResult.recyclable(recyclable);
        }
    }

    @Override
    public void setClearAllData(boolean clearAll) {
        super.setClearAllData(clearAll);
        mFaceRequest.setClearAllData(clearAll);
        mGestureRequest.setClearAllData(clearAll);
//        mPersonVehicleRequest.setClearAllData(clearAll);
        if (mResult != null) {
            mResult.clearAllData(clearAll);
        }
    }

    @Override
    public void clear() {
        super.clear();
        mFaceRequest.clear();
        mGestureRequest.clear();
//        mPersonVehicleRequest.clear();
        mSingleAbilityType = ABILITY_UNKNOWN;
    }

    public void setSingleAbilityType(short ability) {
        mSingleAbilityType = ability;
    }

    public short getSingleAbilityType() {
        return mSingleAbilityType;
    }

    public boolean isSingleAbility() {
        return mSingleAbilityType != ABILITY_UNKNOWN;
    }

    public FaceRequest faceRequest() {
        return mFaceRequest;
    }

    public GestureRequest gestureRequest() {
        return mGestureRequest;
    }

//    public PersonVehicleRequest carOutsideRequestRequest() {
//        return mPersonVehicleRequest;
//    }

}