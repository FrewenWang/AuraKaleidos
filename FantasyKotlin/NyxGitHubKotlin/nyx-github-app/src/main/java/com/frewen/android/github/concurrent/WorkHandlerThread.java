package com.frewen.android.github.concurrent;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;

import java.util.Random;

/**
 * 执行耗时任务的HandlerThread
 */
public class WorkHandlerThread extends HandlerThread {
    private static final String TAG = "WorkHandlerThread";
    private static final int MSG_DO_BACKGROUND_TASK = 0x001;
    /**
     * 后台任务
     */
    private static final int MSG_REPORT_BACKGROUND_TASK_TIME = 0x002;
    private Handler mWorkerHandler; //与工作线程相关联的Handler
    private Handler mUIHandler; //与UI线程相关联的Handler


    /**
     * 共有两个构造函数：
     *
     * @param name 设置HandlerThread的名字
     */
    public WorkHandlerThread(String name) {
        super(name);
    }

    /**
     * @param name
     * @param priority
     */
    public WorkHandlerThread(String name, int priority) {
        super(name, priority);
    }

    @Override
    protected void onLooperPrepared() {
        super.onLooperPrepared();
        Log.d(TAG, "onLooperPrepared() called");
        // 实例化WorkHandlerThread的Handler
        mWorkerHandler = new Handler(getLooper(), new Handler.Callback() {
            @Override
            public boolean handleMessage(Message msg) {
                switch (msg.what) {
                    case MSG_DO_BACKGROUND_TASK:
                        Random random = new Random();
                        int secondsToSleep = random.nextInt(10000);

                        long startTime = System.currentTimeMillis();
                        Log.i(TAG, "handleMessage: BackgoundTask Begin Will cost " + secondsToSleep);
                        try {
                            sleep(secondsToSleep);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        long endTime = System.currentTimeMillis();
                        Log.i(TAG, "handleMessage: BackgoundTask End  cost " + (endTime - startTime));

                        Message message = Message.obtain(null, MSG_REPORT_BACKGROUND_TASK_TIME);
                        message.obj = endTime - startTime;
                        mUIHandler.sendMessage(message); //通知UI更新
                        break;
                    default:
                        break;
                }
                return true;

            }
        });
    }


    /**
     * 执行后台耗时任务
     */
    public void doBackgroundTask() {
        Message msg = Message.obtain(null, MSG_DO_BACKGROUND_TASK);
        mWorkerHandler.sendMessage(msg);
    }

    /**
     * 设置回调给主线程的消息Handler
     *
     * @param uiHandler
     */
    public void setUIHandler(Handler uiHandler) {
        this.mUIHandler = uiHandler;
    }
}
