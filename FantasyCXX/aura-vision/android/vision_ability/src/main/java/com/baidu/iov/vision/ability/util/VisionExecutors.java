package com.baidu.iov.vision.ability.util;

import android.os.Handler;
import android.os.Looper;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.task.AbsVisionTask;

import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 线程池
 *
 * Created by liwendong on 2018/5/8.
 */

public class VisionExecutors {

    private static final String TAG = VisionExecutors.class.getSimpleName();

    private Handler  mMainExecutor;
    private Executor mTaskExecutor;
//    private Executor mDiskExecutor;
//    private Executor mNetworkExecutor;

    private LinkedBlockingQueue mTaskQueue;
    private VisionThreadFactory mThreadFactory;

    private static VisionExecutors sInstance;

    private VisionExecutors() {
        RejectedExecutionHandler taskRejectedHandler = new RejectedExecutionHandler() {
            public void rejectedExecution(Runnable r, ThreadPoolExecutor e) {
                if (!e.isShutdown()) {
                    ((AbsVisionTask) r).onRejected();
                }
            }
        };
        mThreadFactory = new VisionThreadFactory();
        mTaskQueue = new LinkedBlockingQueue<Runnable>(VisionConfig.executorTaskQueueSize);
        mTaskExecutor = new ThreadPoolExecutor(VisionConfig.executorTaskThreadSize,
                VisionConfig.executorTaskThreadSize, 0L, TimeUnit.MILLISECONDS,
                mTaskQueue, mThreadFactory, taskRejectedHandler);
//        mDiskExecutor = Executors.newFixedThreadPool(VisionConfig.executorDiskThreadSize);
//        mNetworkExecutor = Executors.newFixedThreadPool(VisionConfig.executorNetworkThreadSize);
        mMainExecutor = new Handler(Looper.getMainLooper());
    }

    public static VisionExecutors instance() {
        if (sInstance == null) {
            synchronized (VisionExecutors.class) {
                if (sInstance == null) {
                    sInstance = new VisionExecutors();
                }
            }
        }
        return sInstance;
    }

    public Handler mainExecutor() {
        return mMainExecutor;
    }

    public void executeMain(Runnable task) {
        mMainExecutor.post(task);
    }

    public void executeTask(Runnable task) {
        mTaskExecutor.execute(task);
    }

//    public void executeDisk(Runnable task) {
//        mDiskExecutor.execute(task);
//    }

//    public void executeNetwork(Runnable task) {
//        mNetworkExecutor.execute(task);
//    }

    public void setPriority(int priority) {
        mThreadFactory.setPriority(priority);
    }

    private static class VisionThreadFactory implements ThreadFactory {
        private static final AtomicInteger poolNumber = new AtomicInteger(1);
        private final AtomicInteger threadNumber = new AtomicInteger(1);
        private int mPriority = Thread.NORM_PRIORITY;
        private final ThreadGroup group;
        private final String namePrefix;

        VisionThreadFactory() {
            SecurityManager s = System.getSecurityManager();
            group = (s != null) ? s.getThreadGroup() :
                    Thread.currentThread().getThreadGroup();
            namePrefix = "vision-" + poolNumber.getAndIncrement() + "-t-";
        }

        public Thread newThread(Runnable r) {
            Thread t = new Thread(group, r,
                    namePrefix + threadNumber.getAndIncrement(),
                    0);
            if (t.isDaemon()) {
                t.setDaemon(false);
            }
            if (t.getPriority() != mPriority) {
                t.setPriority(mPriority);
            }
            return t;
        }

        public void setPriority(int priority) {
            this.mPriority = priority;
        }

    }

}
