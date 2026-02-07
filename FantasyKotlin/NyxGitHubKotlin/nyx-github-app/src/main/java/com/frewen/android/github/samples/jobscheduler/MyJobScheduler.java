package com.frewen.android.github.samples.jobscheduler;

import android.app.job.JobParameters;
import android.app.job.JobService;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.os.Messenger;
import android.os.RemoteException;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import static com.frewen.android.github.samples.jobscheduler.JobSchedulerDemoActivity.MESSENGER_INTENT_KEY;
import static com.frewen.android.github.samples.jobscheduler.JobSchedulerDemoActivity.MSG_COLOR_START;
import static com.frewen.android.github.samples.jobscheduler.JobSchedulerDemoActivity.MSG_COLOR_STOP;
import static com.frewen.android.github.samples.jobscheduler.JobSchedulerDemoActivity.WORK_DURATION_KEY;

/**
 * @filename: MyJobScheduler
 * @introduction: 这是是参照的官方Demo。
 * MyJobScheduler处理JobScheduler回调处理。JobScheduler调度的请求
 * 最终使用该服务的“onStartJob”方法。让它在特定的时间内运行作业 并完成它们。
 * 最后我们通过Messenger保持和Activity之间的通信
 * @author: Frewen.Wong
 * @time: 2019-06-29 07:56
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class MyJobScheduler extends JobService {
    private static final String TAG = "MyJobScheduler";


    private Messenger mActivityMessenger;

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "FMsg:onCreate() called");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        Log.d(TAG, "FMsg:onStartCommand() called with: intent = [" + intent + "], flags = ["
                + flags + "], startId = [" + startId + "]");
        // 获取Service传入mActivityMessenger
        mActivityMessenger = intent.getParcelableExtra(MESSENGER_INTENT_KEY);
        return START_NOT_STICKY;
    }

    /**
     * 当开始一个任务时，onstartjob（jobparameters params） 是必须使用的方法，
     * 因为它是系统用来触发已经安排的工作（job）的。
     *
     * @param params
     * @return
     */
    @Override
    public boolean onStartJob(JobParameters params) {
        Log.d(TAG, "FMsg:onStartJob() called with: params = [" + params + "]");
        sendMessage(MSG_COLOR_START, params.getJobId());

        long duration = params.getExtras().getLong(WORK_DURATION_KEY);

        // Uses a handler to delay the execution of jobFinished().
        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                sendMessage(MSG_COLOR_STOP, params.getJobId());
                // 当任务完成时，需要调用jobFinished(JobParameters params, boolean needsRescheduled)让系统知道完成了哪项任务，
                // 它可以开始排队接下来的操作。如果不这样做，工作将只运行一次，应用程序将不被允许执行额外的工作。
                jobFinished(params, false);
            }
        }, duration);
        Log.i(TAG, "onStartJob on start job: " + params.getJobId());
        // 返回true，表示该工作耗时，同时工作处理完成后需要调用onStopJob销毁（jobFinished）
        // 返回false，任务运行不需要很长时间，到return时已完成任务处理
        return false;
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        // 使用Messenger来通知
        sendMessage(MSG_COLOR_STOP, params.getJobId());
        Log.i(TAG, "on stop job: " + params.getJobId());
        // 返回false来销毁这个工作
        return false;
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "FMsg:onDestroy() called");
    }

    private void sendMessage(int messageID, @Nullable Object params) {
        // If this service is launched by the JobScheduler, there's no callback Messenger. It
        // only exists when the MainActivity calls startService() with the callback in the Intent.
        if (mActivityMessenger == null) {
            Log.d(TAG, "Service is bound, not started. There's no callback to send a message to.");
            return;
        }
        Message m = Message.obtain();
        m.what = messageID;
        m.obj = params;
        try {
            mActivityMessenger.send(m);
        } catch (RemoteException e) {
            Log.e(TAG, "Error passing service object back to activity.");
        }
    }
}
