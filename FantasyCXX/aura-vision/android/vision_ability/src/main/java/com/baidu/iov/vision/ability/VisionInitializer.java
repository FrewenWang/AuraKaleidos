package com.baidu.iov.vision.ability;

import android.content.Context;
import android.os.Looper;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.core.result.VisionResult;
import com.baidu.iov.vision.ability.core.result.VisionResultVerfication;
import com.baidu.iov.vision.ability.task.AbsVisionTask;
import com.baidu.iov.vision.ability.util.ALog;
import com.baidu.iov.vision.ability.util.VisionExecutors;
import com.baidu.iov.vision.ability.util.VisionNativeHelper;

/**
 * 视觉能力初始化器：初始化操作在子线程中完成，在主线程中回调结果
 * <p>
 * Created by liwendong on 2018/5/9.
 */

public class VisionInitializer {

    private static final String TAG = VisionInitializer.class.getSimpleName();
    private static boolean sInited = false; // 是否已经初始化
    private static boolean sInitializing = false; // 是否正在初始化

    /**
     * 初始化回调接口
     */
    public interface Callback {
        void onSuccess();

        void onFailure(String cause);
    }

    /**
     * 获取初始计划状态
     *
     * @return 是否已经初始化
     */
    public static boolean isInitialized() {
        return sInited;
    }

    /**
     * 设置初始化状态
     *
     * @param isInit 是否已经初始化
     */
    public static void setInitialized(boolean isInit) {
        sInited = isInit;
    }

    /**
     * SDK 初始化
     *
     * @param context  Application Context
     * @param callback 初始化回调接口实现
     */
    public static void init(final Context context, final Callback callback) {
        if (sInited) {
            if (ALog.DEBUG) {
                ALog.v(TAG, "Vision SDK is already initialized, do not reinit.");
            }
            return;
        }

        if (sInitializing) {
            return;
        }

        sInitializing = true;
        VisionContext.appContext(context);
        VisionExecutors.instance().executeTask(new AbsVisionTask() {
            @Override
            public void execute() {
                try {
                    Looper.prepare();
                    doInit();
                    sInitializing = false;
                    sInited = true;
                    callback.onSuccess();
                } catch (final Exception e) {
                    sInitializing = false;
                    callback.onFailure(e.getMessage());
                    if (ALog.DEBUG) {
                        ALog.e(TAG, e.getMessage(), e);
                    }
                }
            }
        });
    }

    public static void deInit() {
        setInitialized(false);
    }

    private static void doInit() {
        initIovVisionAbility();
    }

    private static void initIovVisionAbility() {
        VisionNativeHelper.initModel(VisionContext.appContext().getAssets());

//        if (VisionConfig.execMode == VisionConfig.EXEC_MODE_SERIAL) {
//            VisionNativeHelper.initModel(VisionContext.appContext().getAssets());
//        } else if (VisionConfig.execMode == VisionConfig.EXEC_MODE_PIPELINE) {
//            if (ALog.DEBUG) {
//                ALog.d(TAG, "EXEC_MODE_PIPELINE is unsupported currently!");
//            }
//        }
    }

    private static void initVisVisionAbility() {
    }

    /**
     * 单项能力检测的初始化工作
     * 已废弃，单项能力检测不再需要单独初始化，直接调用单项能力检测接口即可
     */
    @Deprecated
    private static void initIovVisionResultVerfication() {
    }

    /**
     * 修改初始化参数，必须要在init之前调用才能生效
     * 已废弃接口，请勿调用
     */
    // @Deprecated
    // public static boolean setInitConfig(VisionConfig.Key key, float value, int index) {
    //     boolean ret = VisionNativeHelper.setInitConfig(index, key.ordinal(), value);
    //     if (ret) {
    //         VisionConfig.config(key, value);
    //     }
    //     return ret;
    // }

    /**
     * 已废弃接口，请勿调用
     *
     * @param key
     */
    // @Deprecated
    // public static float getInitConfig(VisionConfig.Key key, int index) {
    //     return VisionNativeHelper.getInitConfig(index, key.ordinal());
    // }
}
