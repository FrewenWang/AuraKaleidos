package com.frewen.android.github.samples.ipc.remote.aidl;

import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

import androidx.annotation.Nullable;

/**
 * @filename: RemoteService
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/9/4 0004 下午7:51
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class RemoteService extends Service {
    private static final String TAG = "T:RemoteService";
    /**
     * 判断Service是否销毁
     */
    private AtomicBoolean mIsServiceDestoryed = new AtomicBoolean(false);

    private RemoteTicketServer mRemoteTicketServer = new RemoteTicketServer(this);

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "FMsg:onCreate() called");
        Map<String, String> place1List = new HashMap<>(2);
        place1List.put("10月16日", "北京");
        place1List.put("10月21日", "济南");
        // 增加一张新的门票
        addNewTicket(new RemoteTicket(10001, "周杰伦巡回演唱会门票", 1100.50, place1List));
        Map<String, String> place2List = new HashMap<>(2);
        place2List.put("10月12日", "南京");
        place2List.put("3月21日", "香港");
        // 增加一张新的门票
        addNewTicket(new RemoteTicket(10002, "周杰伦无与伦比演唱会买票", 2100.50, place2List));

        // 我们启动一个子线程任务，来模拟不断有演唱会票可售卖。
        new Thread(new ServiceWorker()).start();
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        Log.d(TAG, "FMsg:onBind() called with: intent = [" + intent + "]");
        // 这个地方我们返回继承自IRemoteServiceInterface.Stub的IBinder对象
        // 在这个Demo中也就是RemoteTicketServer
        /**
         * 第一种方法：、我们可以在onBind中进行验证，验证不通过就直接返回nul
         * 一个应用来绑定我们的服务时，会验证这个应用的权限，如果它没有使用这个权限，
         * onBind方法就会直接返回null，最终结果是这个应用无法绑定到我们的服务，
         * 这样就达到了权限验证的效果，这种方法同样适用于Messenger中，读者可以自行扩展。
         */
        int check = checkCallingOrSelfPermission(
                "com.frewen.android.demo.permission.ACCESS_TICKET_SERVICE");
        Log.i(TAG, "FMsg:onBind() check : " + check);
        if (check == PackageManager.PERMISSION_DENIED) {
            return null;
        }
        return mRemoteTicketServer;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // 将标志Service已经被销毁的变量为true
        mIsServiceDestoryed.set(true);
    }

    /**
     * 启动一个子线程版本
     */
    private class ServiceWorker implements Runnable {
        @Override
        public void run() {
            while (!mIsServiceDestoryed.get()) {
                // 模拟一个随机出现可以卖的票据
                int random = new Random().nextInt(10000);
                try {
                    Thread.sleep(5000 + random);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                int ticketId = random + 1;
                RemoteTicket newTicket = new RemoteTicket(ticketId, "周杰伦演唱会#" + ticketId, 1000.00 + ticketId, null);
                addNewTicket(newTicket);
            }
        }

    }

    /**
     * 新的演唱会门票到货的逻辑回调
     *
     * @param newTicket
     */
    private void addNewTicket(RemoteTicket newTicket) {
        if (null != mRemoteTicketServer) {
            try {
                mRemoteTicketServer.addTicket(newTicket);
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }
    }
}
