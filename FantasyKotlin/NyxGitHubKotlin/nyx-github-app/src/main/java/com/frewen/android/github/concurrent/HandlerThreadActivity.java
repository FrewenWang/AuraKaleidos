package com.frewen.android.github.concurrent;

import android.os.Handler;
import android.os.Message;
import android.os.Bundle;
import android.text.Html;
import android.view.View;
import android.widget.TextView;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;

/**
 * HandlerThreadActivity
 */
public class HandlerThreadActivity extends AppCompatActivity {
    /**
     * 后台任务
     */
    private static final int MSG_REPORT_BACKGROUND_TASK_TIME_MAIN = 0x02;

    private WorkHandlerThread mWorkerThread;
    private Handler mUIHandler;
    private TextView mTvTaskInfo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_handler_thread);

        mTvTaskInfo = findViewById(R.id.textView);
        initUIHandler();
        //创建后台线程
        initBackThread();
    }

    private void initUIHandler() {
        mUIHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case MSG_REPORT_BACKGROUND_TASK_TIME_MAIN:
                        long time = (long) msg.obj;
                        String result = "耗时任务执行完毕，共计耗时：<font color='red'>%d</font>";
                        result = String.format(result, time);
                        mTvTaskInfo.setText(Html.fromHtml(result));
                        break;
                    default:
                        break;
                }

            }
        };
    }

    /**
     * 创建后台耗时线程
     */
    private void initBackThread() {
        mWorkerThread = new WorkHandlerThread("task-thread");
        mWorkerThread.setUIHandler(mUIHandler);
        mWorkerThread.start();
    }

    /**
     * 执行耗时任务
     *
     * @param view
     */
    public void doBackgoundTask(View view) {
        if (null != mWorkerThread) {
            mWorkerThread.doBackgroundTask();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        //释放资源
        if (null != mWorkerThread) {
            mUIHandler.removeMessages(MSG_REPORT_BACKGROUND_TASK_TIME_MAIN);
            mWorkerThread.setUIHandler(null);
            mWorkerThread.quit();
            mWorkerThread = null;
        }
    }
}
