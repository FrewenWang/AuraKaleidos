package com.baidu.iov.vision.ability;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.core.bean.FaceInfo;
import com.baidu.iov.vision.ability.core.manager.VisionManager;
import com.baidu.iov.vision.ability.core.request.AbsVisionRequest;
import com.baidu.iov.vision.ability.core.request.VisionRequest;
import com.baidu.iov.vision.ability.core.request.VisionRequestVerfication;
import com.baidu.iov.vision.ability.core.result.VisionResult;
import com.baidu.iov.vision.ability.core.result.VisionResultVerfication;
import com.baidu.iov.vision.ability.util.ALog;
import com.baidu.iov.vision.ability.util.VisionExecutors;
import com.baidu.iov.vision.ability.util.VisionNativeHelper;

import java.util.concurrent.ConcurrentHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 视觉能力服务接口类，调用此服务类之前需保证 VisionInitializer 初始化完成
 *
 * @see VisionInitializer
 * Created by liwendong on 2018/5.
 */

public class VisionService {
    private static final String TAG = VisionService.class.getSimpleName();

    // private static VisionService sInstance;
    private VisionManager mVisionMgr;
    private static ConcurrentHashMap<Integer, VisionService> mServiceMap = new ConcurrentHashMap<>();
    private static final int DEFAULT_INDEX = 0;
    private int index = DEFAULT_INDEX;
    private VisionResult mVisionResult;

    public VisionService(int index) {
        this.index = index;
        mVisionMgr = new VisionManager();
        if (!VisionInitializer.isInitialized()) {
            if (ALog.DEBUG) {
                ALog.e(TAG, "SDK has not been initialized");
            }
            return;
        }
        VisionNativeHelper.initService(index);

        mVisionResult = VisionResult.obtain();
        mVisionResult.recyclable(false);
        mVisionResult.clearAllData(false);

        VisionNativeHelper.initFaceBuffer(index, mVisionResult.faceResult().mFaceData,
                VisionConfig.configAsInt(VisionConfig.Key.FACE_MAX_COUNT));
        VisionNativeHelper.initGestureBuffer(index, mVisionResult.gestureResult().mGestureData,
                VisionConfig.configAsInt(VisionConfig.Key.GESTURE_MAX_COUNT));
        VisionNativeHelper.initBodyBuffer(index, mVisionResult.bodyPoseResult().mPersonData,
                VisionConfig.configAsInt(VisionConfig.Key.HUMAN_POSE_MAX_COUNT));
        VisionNativeHelper.initFrameBuffer(index, mVisionResult.frameInfo().mNativeBuffer);
    }

    /**
     * 单例接口，已废弃
     * VisionService 支持多实例，需要采用工厂方式创建
     */
    public static VisionService instance() {
        return instance(DEFAULT_INDEX);
    }

    /**
     * 工厂方法创建VisionService实例，需传入index，若已经创建传入id的实例，直接返回该实例
     */
    public static synchronized VisionService instance(int index) {
        VisionService vs = mServiceMap.get(index);
        if (vs == null) {
            vs = new VisionService(index);
            mServiceMap.put(index, vs);
        }
        return vs;
    }

    /**
     * 视觉能力检测唯一接口，已废弃，使用传入request和index两个参数的版本
     *
     * @param request 视觉能力检测请求
     * @see AbsVisionRequest 及其子类
     */
    public void detect(AbsVisionRequest request) {
        detect(request, index);
    }

    /**
     * 视觉能力检测接口
     *
     * @param request 视觉能力检测请求
     * @param index   vision
     * @see AbsVisionRequest 及其子类
     */
    public void detect(AbsVisionRequest request, int index) {
        if (!request.verify()) {
            if (ALog.DEBUG) {
                ALog.e(TAG, "Ignore request because verify fail : " + request);
            }
            return;
        }

        if (!VisionInitializer.isInitialized()) {
            if (ALog.DEBUG) {
                ALog.e(TAG, "SDK has not been initialized");
            }
            return;
        }

        mVisionMgr.detect((VisionRequest) request, mVisionResult, index);
    }

    /**
     * 资源释放，已废弃
     */
    public void release() {
        release(index);
    }

    /**
     * 资源释放
     *
     * @param index visionService实例序号
     */
    public void release(int index) {
        VisionNativeHelper.release(index);
        VisionInitializer.deInit();

        mVisionResult.recyclable(false);
        mVisionResult.clearAllData(false);
        mVisionResult.recycle();

        mServiceMap.remove(index);
    }

    /**
     * 清除触发计数，已废弃
     *
     * @param ability 能力标识
     */
    public void cleanAbilityTriggerAccumulative(short ability) {
        cleanAbilityTriggerAccumulative(ability, this.index);
    }

    /**
     * 清除触发计数
     *
     * @param ability 能力标识
     * @param index   visionService实例序号
     */
    public void cleanAbilityTriggerAccumulative(short ability, int index) {
        VisionNativeHelper.cleanAbilityTriggerAccumulative(index, ability);
    }

    /**
     * 单项能力检测
     * 已废弃，但向前兼容
     *
     * @param request 检测请求
     * @param result  检测结果
     */
    @Deprecated
    public void detectSingleFunction(VisionRequestVerfication request, VisionResultVerfication result) {
        if (ALog.DEBUG) {
            ALog.e(TAG, "detectSingleFunction(VisionRequestVerfication,VisionResultVerfication) has Deprecated !!!!" +
                    " please use detectSingleFunction(VisionRequest,VisionResult)");
        }
        // if (!request.verify()) {
        //     if (ALog.DEBUG) {
        //         ALog.e(TAG, "Ignore request cause verify fail : " + request);
        //     }
        //     return;
        // }
        // result.recycle();
        // VisionNativeHelper.detectSingleFunction(this.index, request.mAbilityType,
        //         request.mFrameData, request.faceRequest().mLandmarkData);
    }

    /**
     * 单项能力检测，request 中需要传入图像帧数据（mFrameData），检测的能力类型（AbilityType），
     * 依赖关键点数据的，要在result的 mFaceInfos 中传入人脸关键点（可传入多个人脸）,
     * 注意，VisionResult 中 FaceInfo 的 buffer 将会传到 native 层使用，所以 VisionResult 的生命周期由调用方控制
     *
     * @param request 请求
     * @param result  结果
     */
    public void detectSingleFunction(VisionRequest request, VisionResult result) {
        detectSingleFunction(request, result, this.index);
    }

    /**
     * 单项能力检测，request 中需要传入图像帧数据（mFrameData），检测的能力类型（AbilityType），
     * 依赖关键点数据的，要在result的 mFaceInfos 中传入人脸关键点（可传入多个人脸）,
     * 注意，VisionResult 中 FaceInfo 的 buffer 将会传到 native 层使用，所以 VisionResult 的生命周期由调用方控制
     *
     * @param request 请求
     * @param result  结果
     * @param index   visionService实例序号
     */
    private void detectSingleFunction(VisionRequest request, VisionResult result, int index) {
        if (!request.verify()) {
            if (ALog.DEBUG) {
                ALog.w(TAG, "Ignore request cause verify fail : " + request);
            }
            return;
        }

        if (!request.isSingleAbility()) {
            if (ALog.DEBUG) {
                ALog.w(TAG, "Vision ability type is empty, ignore");
            }
            return;
        }

        FaceInfo[] faceInfos = result.faceResult().mFaceInfos;
        VisionNativeHelper.detectSingleFunction(index, request.getSingleAbilityType(), request.mFrameData,
                result.faceResult().mFaceData, faceInfos.length, result.frameInfo().mNativeBuffer);
    }

    // -----------------------------------------------------------------------------------------------------------------
    // --------------------------------------------- Swtiches and Configs -----------------------------------------------

    /**
     * 获取能力开关状态，已废弃
     *
     * @param ability 视觉能力 ID
     * @return 开关状态（是否打开）
     */
    public boolean getSwitch(short ability) {
        return getSwitch(ability, this.index);
    }

    /**
     * 获取能力开关状态
     *
     * @param ability 视觉能力 ID
     * @param index   visionService
     */
    public boolean getSwitch(short ability, int index) {
        return VisionNativeHelper.getSwitch(index, ability);
    }

    /**
     * 获取 Native 层当前各能力开关集合，已废弃
     *
     * @return 视觉能力开关集合
     */
    public Map<Short, Boolean> getSwitches() {
        return getSwitches(this.index);
    }

    /**
     * 获取 Native 层当前各能力开关集合
     *
     * @param index visionService实例序号
     * @return 视觉能力开关集合
     */
    public Map<Short, Boolean> getSwitches(int index) {
        HashMap<Short, Boolean> switches = new HashMap<>();
        VisionNativeHelper.getSwitches(index, switches);
        return switches;
    }

    /**
     * 设置视觉能力开关
     *
     * @param ability  能力标识
     * @param switcher 能力开关
     */
    public boolean setSwitch(short ability, boolean switcher) {
        return setSwitch(ability, switcher, this.index);
    }

    /**
     * 设置视觉能力开关
     *
     * @param ability  能力标识
     * @param switcher 能力开关
     * @param index    visionService实例序号
     */
    public boolean setSwitch(short ability, boolean switcher, int index) {
        return VisionNativeHelper.setSwitch(index, ability, switcher);
    }

    /**
     * 异步设置视觉能力开关，已废弃
     *
     * @param ability  能力标识
     * @param switcher 能力开关
     */
    public void setSwitchAsync(final short ability, final boolean switcher) {
        setSwitchAsync(ability, switcher, this.index);
    }

    /**
     * 异步设置视觉能力开关
     *
     * @param ability  能力标识
     * @param switcher 能力开关
     * @param index    visionService实例序号
     */
    public void setSwitchAsync(final short ability, final boolean switcher, final int index) {
        VisionExecutors.instance().executeTask(new Runnable() {
            @Override
            public void run() {
                setSwitch(ability, switcher, index);
            }
        });
    }

    /**
     * 异步设置视觉能力开关，已废弃
     *
     * @param switches 能力标识和开关 map
     */
    public void setSwitchesAsync(final Map<Short, Boolean> switches) {
        setSwitchesAsync(switches, this.index);
    }

    /**
     * 异步设置视觉能力开关
     *
     * @param switches 能力标识和开关 map
     * @param index    visionService实例序号
     */
    public void setSwitchesAsync(final Map<Short, Boolean> switches, final int index) {
        if (switches != null) {
            VisionExecutors.instance().executeTask(new Runnable() {
                @Override
                public void run() {
                    setSwitches(switches, index);
                }
            });
        }
    }

    /**
     * 设置视觉能力开关，已废弃
     *
     * @param switches 能力标识和开关 map
     * @return 是否设置成功
     */
    public boolean setSwitches(final Map<Short, Boolean> switches) {
        return setSwitches(switches, this.index);
    }

    /**
     * 设置视觉能力开关
     *
     * @param switches 能力标识和开关 map
     * @param index    visionService实例序号
     * @return 是否设置成功
     */
    public boolean setSwitches(final Map<Short, Boolean> switches, int index) {
        if (switches != null) {
            for (Map.Entry<Short, Boolean> e : switches.entrySet()) {
                if (!VisionNativeHelper.setSwitch(index, e.getKey(), e.getValue())) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 批量设置视觉能力开关，已废弃
     *
     * @param switches 能力标识列表
     * @param switcher 开关状态
     * @return 是否设置成功
     */
    public boolean setSwitches(final List<Short> switches, boolean switcher) {
        return setSwitches(switches, switcher, this.index);
    }

    /**
     * 批量设置视觉能力开关
     *
     * @param switches 能力标识列表
     * @param switcher 开关状态
     * @param index    visionService实例序号
     * @return 是否设置成功
     */
    public boolean setSwitches(final List<Short> switches, boolean switcher, int index) {
        if (switches != null) {
            for (Short key : switches) {
                if (!VisionNativeHelper.setSwitch(index, key, switcher)) {
                    if (ALog.DEBUG) {
                        ALog.d(TAG, "setSwitch: [" + key + ", " + switcher + "] fail!");
                    }
                }
            }
            return true;
        }
        return false;
    }

    /**
     * 批量设置视觉能力开关，已废弃
     *
     * @param switches 能力标识列表
     * @param switcher 开关状态
     * @return 是否设置成功
     */
    public boolean setSwitches(final Short[] switches, boolean switcher) {
        return setSwitches(switches, switcher, this.index);
    }

    /**
     * 批量设置视觉能力开关
     *
     * @param switches 能力标识列表
     * @param switcher 开关状态
     * @param index    visionService实例序号
     * @return 是否设置成功
     */
    public boolean setSwitches(final Short[] switches, boolean switcher, int index) {
        if (switches != null) {
            for (Short key : switches) {
                if (!VisionNativeHelper.setSwitch(index, key, switcher)) {
                    if (ALog.DEBUG) {
                        ALog.e(TAG, "setSwitch: [" + key + ", " + switcher + "] fail!");
                    }
                }
            }
            return true;
        }
        return false;
    }

    /**
     * 修改native层 配置参数，已废弃
     *
     * @param key   参数标识
     * @param value 参数值
     */
    public boolean setConfig(VisionConfig.Key key, float value) {
        return setConfig(key, value, this.index);
    }

    /**
     * 修改native层 配置参数
     *
     * @param key   参数标识
     * @param value 参数值
     * @param index visionService实例序号
     */
    public boolean setConfig(VisionConfig.Key key, float value, int index) {
        return VisionNativeHelper.setConfig(index, key.ordinal(), value);
    }

    /**
     * 异步修改native层 配置参数，已废弃
     *
     * @param key   参数标识
     * @param value 参数值
     */
    public void setConfigAsync(final VisionConfig.Key key, final float value) {
        setConfigAsync(key, value, this.index);
    }

    /**
     * 异步修改native层 配置参数
     *
     * @param key   参数标识
     * @param value 参数值
     * @param index visionService实例序号
     */
    public void setConfigAsync(final VisionConfig.Key key, final float value, final int index) {
        VisionExecutors.instance().executeTask(new Runnable() {
            @Override
            public void run() {
                setConfig(key, value, index);
            }
        });
    }

    /**
     * 获取配置参数，已废弃
     *
     * @param key 参数标识
     * @return 参数值
     */
    public float getConfig(VisionConfig.Key key) {
        return getConfig(key, this.index);
    }

    /**
     * 获取配置参数
     *
     * @param key   参数标识
     * @param index visionService实例序号
     * @return 参数值
     */
    public float getConfig(VisionConfig.Key key, int index) {
        return VisionNativeHelper.getConfig(index, key.ordinal());
    }
}
