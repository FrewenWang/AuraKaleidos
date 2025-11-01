package com.baidu.iov.vision.ability.core.result;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.util.ALog;
import com.baidu.iov.vision.ability.util.PerformanceUtil;

import java.util.HashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 视觉能力检测结果
 *
 * Created by liwendong on 2018/5/16.
 */

public abstract class AbsVisionResult {

    private static final String TAG = AbsVisionResult.class.getSimpleName();
    protected String mTag = AbsVisionResult.class.getSimpleName();

    public static final int SUCCESS = 0;
    public static final int FAILURE = 1;


    /**
     * 任务运行结果代码，具体结果值由每类任务的 XxxResult.java 定义
     */
    public int resultCode;

    /**
     * 任务运行错误信息，成功运行时为 null
     */
    public String errMsg;

    /**
     * 性能分析工具对象，只在 DEBUG 模式下启用，一律使用 DEBUG 开关控制
     */
    public PerformanceUtil mPerfUtil;

    public boolean isClearAllData;

    public boolean isRecyclable;

    public AbsVisionResult() {
        if (ALog.DEBUG) {
            mPerfUtil = new PerformanceUtil();
        }
    }

    public void setTag(String tag) {
        mTag = tag;
    }

    public int getResultCode() {
        return resultCode;
    }

    public void setResultCode(int resultCode) {
        this.resultCode = resultCode;
    }

    public void setResultCode(boolean result) {
        resultCode = result ? SUCCESS : FAILURE;
    }

    public boolean isResultSuccess() {
        return resultCode == SUCCESS;
    }

    public String getErrMsg() {
        return errMsg;
    }

    public void setErrMsg(String errMsg) {
        this.errMsg = errMsg;
    }

    public PerformanceUtil getPerfUtil() {
        return mPerfUtil;
    }

    /**
     * 设置是否可被回收
     *
     * @param recyclable
     */
    public void recyclable(boolean recyclable) {
        isRecyclable = recyclable;
    }

    /**
     * 设置是否清除所有数据
     *
     * @param clearAll
     */
    public void clearAllData(boolean clearAll) {
        isClearAllData = clearAll;
    }

    /**
     * 每帧数据结果返回给调用端后，会自动执行清理操作，因为结果对象采用对象池管理
     */
    public void clear() {
        if (ALog.DEBUG) {
            mPerfUtil.clear();
        }
        resultCode = SUCCESS;
        errMsg = "";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(mTag);
        if (ALog.DEBUG) {
            sb.append(" | PERF - ").append(mPerfUtil.toString());
        }
        return sb.toString();
    }

    // ----------------------------------------------------------------------------------------------
    // 缓存组件：结果对象采用对象池缓存，避免频繁创建对象
    private static HashMap<Class<? extends AbsVisionResult>, ConcurrentLinkedQueue<AbsVisionResult>> sResultCache =
            new HashMap<>();


    public static AbsVisionResult obtain() {
        return obtain(AbsVisionResult.class);
    }

    public static <T extends AbsVisionResult> T obtain(Class<? extends AbsVisionResult> resultCls) {

        ConcurrentLinkedQueue<AbsVisionResult> ress = sResultCache.get(resultCls);
        if (ress == null) {
            ress = new ConcurrentLinkedQueue<>();
            sResultCache.put(resultCls, ress);
        }
        T obj = (T) ress.poll();
        if (obj == null) {
            try {
                obj = (T) resultCls.newInstance();
                obj.isRecyclable = true;
                obj.isClearAllData = true;
                if (ALog.DEBUG) {
                    ALog.d(TAG, "New result instance - " + obj);
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
     * 每帧数据结果返回给调用端后，会自动执行清理操作，因为结果对象采用对象池管理
     */
    public void recycle() {
        clear();
        ConcurrentLinkedQueue<AbsVisionResult> objs = sResultCache.get(getClass());
        if (objs != null && objs.size() < VisionConfig.RESULT_CACHE_SIZE) {
            objs.offer(this);
        }
    }
}
