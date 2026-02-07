package com.frewen.android.github.samples.eventbus;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import com.frewen.android.github.R;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import androidx.appcompat.app.AppCompatActivity;

/**
 * 文章参考：https://blog.csdn.net/u011240877/article/details/73015939
 */
public class EventBusActivity extends AppCompatActivity {
    private static final String TAG = "EventBusActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_event_bus);

        //注册EventBus
        EventBus.getDefault().register(this);
        /**
         * 在主线程发送一个消息
         */
        findViewById(R.id.hello).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(EventBusActivity.this,
                        "主线程发送消息：" + Thread.currentThread().getName(), Toast.LENGTH_SHORT).show();
                //发送事件消息
                EventBus.getDefault().post("发送一个Hello");
            }
        });

        findViewById(R.id.sendmainMsg).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(EventBusActivity.this,
                        "发送了主线程消息：" + Thread.currentThread().getName(), Toast.LENGTH_SHORT).show();
                //发送事件消息
                EventBus.getDefault().post("发送了主线程消息");
            }
        });

        findViewById(R.id.sendStickyMsg).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().postSticky("发送了粘性事件，请跳转Activity注册接收。");
                Intent intent = new Intent(EventBusActivity.this, StickyMsgActivity.class);
                startActivity(intent);

            }
        });
    }


    /**
     * 子线程发送消息
     */
    public void sendBackgroundThreadMsg(View view) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(EventBusActivity.this, "发送子线程消息："
                        + Thread.currentThread().getName(), Toast.LENGTH_SHORT).show();
                //发送事件消息
                EventBus.getDefault().post("发送了子线程消息");
            }
        }).start();

    }

    @Override
    protected void onStop() {
        super.onStop();
        //解绑，防止内存泄漏
        EventBus.getDefault().unregister(this);
    }

    /**
     * 在接收事件消息的方法中，可以通过注解的方式设置线程模型，
     * EventBus内置了4中线程模型，分别是ThreadMode.POSTING 、ThreadMode.MAIN、ThreadMode.BACKGROUND、ThreadMode.ASYNC
     *
     * @param event
     */
    @Subscribe(threadMode = ThreadMode.MAIN)
    public void hello(String event) {
        Toast.makeText(this, event, Toast.LENGTH_SHORT).show();
    }

    //==========================================================================================================

    /**
     * 如果使用事件处理函数指定了线程模型为PostThread，那么该事件在哪个线程发布出来的，事件处理函数就会在这个线程中运行，
     * 也就是说发布事件和接收事件在同一个线程。
     * 在线程模型为PostThread的事件处理函数中尽量避免执行耗时操作，因为它会阻塞事件的传递，甚至有可能会引起ANR。
     *
     * @param event
     */
    @Subscribe(threadMode = ThreadMode.POSTING)
    public void onMessageEventPostThread(String event) {
        Log.e(TAG, "onMessageEventPostThread: currentThread===" + Thread.currentThread().getName());
        Log.e("event PostThread", "消息： " + event + " thread: " + Thread.currentThread().getName());
    }

    /**
     * MainThread：如果使用事件处理函数指定了线程模型为MainThread，
     * 那么不论事件是在哪个线程中发布出来的，该事件处理函数都会在UI线程中执行。
     * 该方法可以用来更新UI，但是不能处理耗时操作。
     *
     * @param event
     */
    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onMessageEventMainThread(String event) {
        Log.e("event MainThread", "消息： " + event + " thread: " + Thread.currentThread().getName());
    }

    /**
     * BackgroundThread：如果使用事件处理函数指定了线程模型为BackgroundThread，那么
     * 如果事件是在UI线程中发布出来的，那么该事件处理函数就会在新的线程中运行，
     * 如果事件本来就是子线程中发布出来的，那么该事件处理函数直接在发布事件的线程中执行。
     * 在此事件处理函数中禁止进行UI更新操作。
     *
     * @param event
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onMessageEventBackgroundThread(String event) {
        Log.e("event BackgroundThread", "消息： " + event + " thread: " + Thread.currentThread().getName());
    }

    /**
     * Async：如果使用事件处理函数指定了线程模型为Async，那么无论事件在哪个线程发布，该事件处理函数都会在新建的子线程中执行。
     * 同样，此事件处理函数中禁止进行UI更新操作。
     *
     * @param event
     */
    @Subscribe(threadMode = ThreadMode.ASYNC)
    public void onMessageEventAsync(String event) {
        Log.e("event Async", "消息： " + event + " thread: " + Thread.currentThread().getName());
    }

}
