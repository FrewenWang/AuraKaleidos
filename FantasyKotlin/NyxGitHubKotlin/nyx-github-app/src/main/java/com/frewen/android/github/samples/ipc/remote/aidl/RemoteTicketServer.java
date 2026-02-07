package com.frewen.android.github.samples.ipc.remote.aidl;

import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Parcel;
import android.os.RemoteCallbackList;
import android.os.RemoteException;
import android.os.SystemClock;
import android.util.Log;


import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * @filename: RemoteTicketServer
 * @introduction: 服务端关于客户端调用接口的处理逻辑
 * <p>
 * * 实例化Binder的对象。通过AIDL文件生成对应的Binder对象。
 * * 然后通过操作AIDL中的方法，来给客户端返回服务端的数据
 * @author: Frewen.Wong
 * @time: 2019/9/4 0004 下午8:47
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class RemoteTicketServer extends IRemoteServiceInterface.Stub {
    private static final String TAG = "T:RemoteTicketServer";
    private final Context mContext;
    private CopyOnWriteArrayList<RemoteTicket> mTicketList = new CopyOnWriteArrayList<>();
    /**
     * 实例化CallbackListener
     * 使用RemoteCallbackList，有一点需要注意，我们无法像操作List一样去操作它，尽管它的名字中也带个List，但是它并不是一个List。
     */
    private RemoteCallbackList<IOnNewTicketArrivedListener> mListenerList = new RemoteCallbackList<>();

    public RemoteTicketServer(Context mContext) {
        this.mContext = mContext;
    }

    @Override
    public int getPid() throws RemoteException {
        // 模拟一个跨进程的耗时任务
        SystemClock.sleep(6000);
        return android.os.Process.myPid();
    }

    /**
     * 获取演唱会列表
     *
     * @return
     * @throws RemoteException
     */
    @Override
    public List<RemoteTicket> getTicketList() throws RemoteException {
        Log.d(TAG, "FMsg:getTicketList() begin");
        // 获取门票列表，模拟一个耗时任务的实现
        SystemClock.sleep(30 * 1000);
        Log.d(TAG, "FMsg:getTicketList() end currentThread = " + Thread.currentThread().getName());
        return mTicketList;
    }

    @Override
    public void addTicket(RemoteTicket ticket) throws RemoteException {
        Log.d(TAG, "FMsg:addTicket() called with: ticket = [" + ticket + "]");
        onNewTicketArrived(ticket);
    }

    /**
     * 定义当有新的门票添加的回调方法
     *
     * @param ticket
     * @throws RemoteException
     */
    private void onNewTicketArrived(RemoteTicket ticket) throws RemoteException {
        mTicketList.add(ticket);
        /**
         * 遍历RemoteCallbackList，必须要按照下面的方式进行，
         * 其中beginBroadcast和beginBroadcast必须要配对使用，
         * 哪怕我们仅仅是想要获取RemoteCallbackList中的元素个数，这是必须要注意的地方。
         */
        final int N = mListenerList.beginBroadcast();
        for (int i = 0; i < N; i++) {
            IOnNewTicketArrivedListener listener = mListenerList.getBroadcastItem(i);
            if (listener != null) {
                try {
                    /**
                     * 远程服务端需要调用客户端的listener中的方法时，被调用的方法也运行在Binder线程池中，
                     * 只不过是客户端的线程池。所以，我们同样不可以在服务端中调用客户端的耗时方法。
                     * 比如针对RemoteTicketServer的onNewTicketArrived方法，
                     *          如下所示。在它内部调用了客户端的IOnNewBookArrivedListener中的onNewBookArrived方法，
                     * 如果客户端的这个onNewTicketArrived方法比较耗时的话，
                     * 那么请确保BRemoteTicketServer中的onNewTicketArrived运行在非UI线程中，否则将导致服务端无法响应。
                     */
                    // 回调给客户端当有新的门票被添加的逻辑
                    listener.onNewTicketArrived(ticket);
                } catch (RemoteException e) {
                    e.printStackTrace();
                }
            }
        }
        mListenerList.finishBroadcast();
    }

    @Override
    public void registerListener(IOnNewTicketArrivedListener listener) throws RemoteException {
        Log.d(TAG, "FMsg:registerListener() called with: listener = [" + listener.getClass() + "]");
        Log.d(TAG, "FMsg:registerListener() called with: mListenerList = [" + mListenerList + "]");
        // 注册当前的listener
        mListenerList.register(listener);
        final int N = mListenerList.beginBroadcast();
        mListenerList.finishBroadcast();
        Log.d(TAG, "registerListener, current size:" + N);
    }

    @Override
    public void unregisterListener(IOnNewTicketArrivedListener listener) throws RemoteException {
        boolean success = mListenerList.unregister(listener);
        if (success) {
            Log.d(TAG, "unregister success.");
        } else {
            Log.d(TAG, "not found, can not unregister.");
        }
        final int N = mListenerList.beginBroadcast();
        mListenerList.finishBroadcast();
        Log.d(TAG, "unregisterListener, current size:" + N);
    }

    @Override
    public void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat, double aDouble, String aString) throws RemoteException {

    }

    public void addTicketList(CopyOnWriteArrayList<RemoteTicket> ticketList) {
        Log.d(TAG, "FMsg:addTicketList() called with: ticketList = [" + ticketList + "]");
        mTicketList.addAll(ticketList);
    }

    /**
     * 第二种方法，我们可以在服务端的onTransact方法中进行权限验证，如果验证失败就直接返回false，
     * 这样服务端就不会终止执行AIDL中的方法从而达到保护服务端的效果。至于具体的验证方式有很多，
     * 可以采用permission验证，具体实现方式和第一种方法一样。还可以采用Uid和Pid来做验证，
     * 通过getCallingUid和getCallingPid可以拿到客户端所属应用的Uid和Pid，
     * 通过这两个参数我们可以做一些验证工作，比如验证包名。在下面的代码中，既验证了permission，
     * 又验证了包名。一个应用如果想远程调用服务中的方法，
     * 首先要使用我们的自定义权限“com.frewen.android.demo.permission.ACCESS_TICKET_SERVICE”，
     * 其次包名必须以“com.frewen.android.demo.permission.ACCESS_TICKET_SERVICE”开始，否则调用服务端的方法会失败。
     *
     * 注意：这个方法如果不是跨进程继续绑定的话。这个方法是不会调用的。
     * 换句话说：这种方法只能作为跨进程绑定远程服务的权限判断接口
     * @param code
     * @param data
     * @param reply
     * @param flags
     * @return
     * @throws RemoteException
     */
    @Override
    public boolean onTransact(int code, Parcel data, Parcel reply, int flags)
            throws RemoteException {
        Log.d(TAG, "FMsg:onTransact() called with: code = [" + code + "], data = [" + data + "], reply = [" + reply + "], flags = [" + flags + "]");
        int check = mContext.checkCallingOrSelfPermission(
                "com.frewen.android.demo.permission.ACCESS_TICKET_SERVICE");
        if (check == PackageManager.PERMISSION_DENIED) {
            return false;
        }
        String packageName = null;
        // 这个是packageManager里面的方法。通过Uid来拿到进程的包名
        String[] packages = mContext.getPackageManager().getPackagesForUid(getCallingUid());
        if (packages != null && packages.length > 0) {
            packageName = packages[0];
        }
        Log.d(TAG, "FMsg:onTransact  packageName = [" + packageName + "]");
        if (!packageName.startsWith("com.frewen.android.demo")) {
            return false;
        }
        return super.onTransact(code, data, reply, flags);
    }
}
