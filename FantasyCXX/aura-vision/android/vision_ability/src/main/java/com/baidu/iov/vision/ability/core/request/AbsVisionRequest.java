package com.baidu.iov.vision.ability.core.request;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.core.VisionCallback;
import com.baidu.iov.vision.ability.core.manager.AbsVisionManager;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;
import com.baidu.iov.vision.ability.task.AbsVisionTask;
import com.baidu.iov.vision.ability.util.ALog;

import java.util.HashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 视觉能力请求
 * <p>
 * 所有参数都封装在请求中，不同功能需要继承此类，实现自己的请求封装
 * Request对象，请求参数的包装者，指定了具体的Task
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
public abstract class AbsVisionRequest {

    private static final String TAG = AbsVisionRequest.class.getSimpleName();

    /**
     * 每个功能请求唯一对应一个管理器，子类实现中需要设置一个默认值，同时也可在构建请求对象时临时修改，便于测试。
     */
    public Class<? extends AbsVisionManager> mManagerClass;
    public Class<? extends AbsVisionTask> mTaskClass;
    public Class<? extends AbsVisionResult> mResultClass;

    /**
     * 每帧的图像数据
     */
    public byte[] mFrameData;

    /**
     * 计算结果回调
     */
    public VisionCallback mCallback;

    public boolean isRecyclable;

    public boolean isClearAllData;

    public AbsVisionResult mResult;
    /**
     * 请求图像帧的宽度
     */
    public int mFrameWidth;
    /**
     * 请求图像帧的高度
     */
    public int mFrameHeight;

    public AbsVisionRequest() {
        this(null, null);
    }

    public AbsVisionRequest(byte[] frameData, VisionCallback callback) {
        mFrameData = frameData;
        mCallback = callback;
        mManagerClass = AbsVisionManager.class;
    }

    public byte[] frameData() {
        return mFrameData;
    }

    /**
     * 设置是否可被回收
     *
     * @param recyclable
     */
    public void setRecyclable(boolean recyclable) {
        isRecyclable = recyclable;
    }

    /**
     * 设置是否清除所有数据
     *
     * @param clearAll
     */
    public void setClearAllData(boolean clearAll) {
        isClearAllData = clearAll;
    }

    /**
     * 每帧数据请求返回给调用端后，会自动执行清理操作，因为请求对象采用对象池管理，子类根据情况重写此方法。
     */
    public void clear() {
        if (isClearAllData) {
            mFrameData = null;
            mCallback = null;
        } else {
            mFrameData = null;
        }
    }

    /**
     * @return 与当前请求对应的管理器 Class
     */
    public Class<? extends AbsVisionManager> getVisionManagerClass() {
        return mManagerClass;
    }

    /**
     * 校验请求参数是否合法，否则不予执行
     *
     * @return 校验结果
     */
    public boolean verify() {
        if (mFrameData == null) {
            return false;
        }
        return true;
    }

    // ----------------------------------------------------------------------------------------------
    // 缓存组件：请求对象采用对象池缓存，避免频繁创建对象
    private static HashMap<Class<? extends AbsVisionRequest>, ConcurrentLinkedQueue<AbsVisionRequest>> sRequestCache =
            new HashMap<>();

    public static AbsVisionRequest obtain() {
        return obtain(AbsVisionRequest.class);
    }

    public static <T extends AbsVisionRequest> T obtain(Class<? extends AbsVisionRequest> requestCls) {
        return obtain(requestCls, null, null);
    }

    public static <T extends AbsVisionRequest> T obtain(Class<? extends AbsVisionRequest> requestCls,
                                                        byte[] frameData,
                                                        VisionCallback callback) {
        ConcurrentLinkedQueue<AbsVisionRequest> reqs = sRequestCache.get(requestCls);
        if (reqs == null) {
            reqs = new ConcurrentLinkedQueue<>();
            sRequestCache.put(requestCls, reqs);
        }
        T obj = null;
        if (reqs.size() > 0) {
            obj = (T) reqs.poll();
        }
        if (obj == null) {
            try {
                obj = (T) requestCls.newInstance();
                obj.mFrameData = frameData;
                obj.mCallback = callback;
                obj.isRecyclable = true;
                obj.isClearAllData = true;
                if (ALog.DEBUG) {
                    ALog.d(TAG, "New request instance - " + obj);
                }
            } catch (Exception e) {
                if (ALog.DEBUG) {
                    ALog.e(TAG, e.getMessage(), e);
                }
            }
        }
        return obj;
    }

    /**
     * 每帧数据请求返回给调用端后，会自动执行清理操作，因为请求对象采用对象池管理
     */
    public void recycle() {
        clear();
        ConcurrentLinkedQueue<AbsVisionRequest> objs = sRequestCache.get(getClass());
        if (objs != null && objs.size() < VisionConfig.REQUEST_CACHE_SIZE) {
            objs.offer(this);
        }
    }
}
