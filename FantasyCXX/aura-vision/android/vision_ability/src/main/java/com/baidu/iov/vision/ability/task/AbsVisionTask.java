package com.baidu.iov.vision.ability.task;

import com.baidu.iov.vision.ability.core.request.AbsVisionRequest;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;
import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.util.ALog;
import com.baidu.iov.vision.ability.util.VisionExecutors;

import java.util.HashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 视觉能力任务实现基类，只负责单帧数据计算，不保存任何业务逻辑；所有跨多帧业务逻辑都在管理器中实现
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public abstract class AbsVisionTask implements Runnable {

    private static final String TAG = AbsVisionTask.class.getSimpleName();

    protected AbsVisionRequest mRequest;
    protected AbsVisionResult mResult;
    protected int mIndex;

    public AbsVisionTask setRequest(AbsVisionRequest visionRequest) {
        mRequest = visionRequest;
        return this;
    }

    public AbsVisionTask setResult(AbsVisionResult absVisionResult) {
        mResult = absVisionResult;
        return this;
    }

    public AbsVisionTask setIndex(int index) {
        mIndex = index;
        return this;
    }

    public void onRejected() {
        if (mRequest != null && mRequest.mCallback != null) {
            mRequest.mCallback.onRejected(mRequest);
        }
        recycle();
        if (ALog.DEBUG) {
            ALog.v(TAG, "Task is rejected.");
        }
    }

    /**
     * 各任务实现类不能重写此方法，而是 execute
     *
     * @see AbsVisionTask#execute()
     */
    @Override
    public final void run() {
        execute();
        notifyResult();
        //        releaseImageCache();
        //        recycle();
    }

    protected <T extends AbsVisionResult> T createResult() {
        if (mRequest.mResult != null) {
            return (T) mRequest.mResult;
        }
        T result = null;
        if (mRequest.isRecyclable) {
            result = AbsVisionResult.obtain(mRequest.mResultClass);
        } else {
            result = (T) mRequest.mResult;
            if (result == null) {
                try {
                    result = (T) mRequest.mResultClass.newInstance();
                    mRequest.mResult = result;
                    result.clearAllData(mRequest.isClearAllData);
                    result.recyclable(mRequest.isRecyclable);
                } catch (Exception e) {
                    if (ALog.DEBUG) {
                        ALog.e(TAG, e.getMessage(), e);
                    }
                }
            }
        }
        return result;
    }

    /**
     * 任务执行启动接口，所有任务实现类需要实现此方法编写业务逻辑，不要实现 run 方法
     */
    public abstract void execute();

    private Runnable mCallbackNotifier = new Runnable() {
        @Override
        public void run() {
            mRequest.mCallback.onResult(mRequest, mResult);
            recycle();
        }
    };

    protected void notifyResult() {
        if (mRequest != null && mRequest.mCallback != null) {
            if (VisionConfig.callbackThreadMode == VisionConfig.CALLBACK_THREAD_MAIN) {
                VisionExecutors.instance().executeMain(mCallbackNotifier);
            } else {
                mCallbackNotifier.run();
            }
        }
    }

    protected void clear() {
        if (mRequest != null) {
            if (mRequest.isRecyclable) {
                mRequest.recycle();
            } else {
                mRequest.clear();
            }
            mRequest = null;
        }
        if (mResult != null) {
            if (mResult.isRecyclable) {
                mResult.recycle();
            } else {
                mResult.clear();
            }
            mResult = null;
        }
    }

    // ----------------------------------------------------------------------------------------------
    // 缓存组件 ： 所有 Task 对象采用对象池缓存
    private static HashMap<Class<? extends AbsVisionTask>,
                           ConcurrentLinkedQueue<AbsVisionTask>> sTaskCache = new HashMap<>();

    public static AbsVisionTask obtain() {
        return obtain(AbsVisionTask.class);
    }

    /**
     * 创建对应的Task，在VisionManager里面被调用
     *
     * @param c
     *
     * @return AbsVisionTask
     */
    public static <T extends AbsVisionTask> T obtain(Class<? extends AbsVisionTask> c, AbsVisionRequest r) {
        T t = obtain(c);
        t.setRequest(r);
        return t;
    }

    public static <T extends AbsVisionTask> T obtain(Class<? extends AbsVisionTask> taskCls) {
        ConcurrentLinkedQueue<AbsVisionTask> tasks = sTaskCache.get(taskCls);
        if (tasks == null) {
            tasks = new ConcurrentLinkedQueue<>();
            sTaskCache.put(taskCls, tasks);
        }

        T t = null;
        if (tasks.size() > 0) {
            t = (T) tasks.poll();
        }
        if (t == null) {
            try {
                t = (T) taskCls.newInstance();
                if (ALog.DEBUG) {
                    ALog.d(TAG, "New task instance - " + t);
                }
            } catch (Exception e) {
                if (ALog.DEBUG) {
                    ALog.e(TAG, e.getMessage(), e);
                }
            }
        }
        return t;
    }

    public void recycle() {
        clear();
        ConcurrentLinkedQueue<AbsVisionTask> tasks = sTaskCache.get(getClass());
        if (tasks != null && tasks.size() < VisionConfig.TASK_CACHE_SIZE) {
            tasks.offer(this);
        }
    }
}
