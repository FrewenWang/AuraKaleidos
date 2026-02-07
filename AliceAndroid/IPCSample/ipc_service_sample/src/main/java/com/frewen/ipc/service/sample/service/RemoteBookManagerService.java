package com.frewen.ipc.service.sample.service;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import android.os.RemoteCallbackList;
import android.os.RemoteException;
import android.util.Log;

import com.frewen.ipc.service.sample.IOnNewBookArrivedListener;
import com.frewen.ipc.service.sample.IOnNewBookArrivedSuccessListener;
import com.frewen.ipc.service.sample.IRemoteBookManager;
import com.frewen.ipc.service.sample.RemoteBook;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 服务端端RemoteBookManagerService
 */
public class RemoteBookManagerService extends Service {
    private static final String TAG = "BookManagerService";

    private CopyOnWriteArrayList<RemoteBook> mBookList = new CopyOnWriteArrayList<>();

    private AtomicBoolean mIsServiceDestoryed = new AtomicBoolean(false);

    private RemoteCallbackList<IOnNewBookArrivedListener> mListenerList = new
            RemoteCallbackList<>();

    /**
     * Binder的注册监听者
     */
    private IOnNewBookArrivedSuccessListener onNewBookArrivedSuccessListener = new IOnNewBookArrivedSuccessListener.Stub() {

        @Override
        public void onSuccess() throws RemoteException {
            Log.d(TAG, "onSuccess() called :" + Thread.currentThread().getName());
        }

        @Override
        public void onFailed() throws RemoteException {
            Log.d(TAG, "onFailed() called :" + Thread.currentThread().getName());
        }
    };

    /**
     * 实例化Binder的对象。通过AIDL文件生成对应的Binder对象。
     * 然后通过操作AIDL中的方法，来给客户端返回服务端的数据
     */
    private Binder mBinder = new IRemoteBookManager.Stub() {
        @Override
        public List<RemoteBook> getBookList() throws RemoteException {
            return mBookList;
        }

        @Override
        public void addBook(RemoteBook book) throws RemoteException {
            mBookList.add(book);
        }

        @Override
        public void registerListener(IOnNewBookArrivedListener listener) throws RemoteException {
            mListenerList.register(listener);
        }

        @Override
        public void unregisterListener(IOnNewBookArrivedListener listener) throws RemoteException {
            mListenerList.unregister(listener);
        }
    };

    public RemoteBookManagerService() {

    }

    @Override
    public void onCreate() {
        super.onCreate();
        mBookList.add(new RemoteBook(1, "Hello Android"));
        mBookList.add(new RemoteBook(2, "Hello Java"));
        new Thread(new ServiceWorker()).start();
    }


    private class ServiceWorker implements Runnable {
        @Override
        public void run() {
            // do background processing here.....
            while (!mIsServiceDestoryed.get()) {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                int bookId = mBookList.size() + 1;
                RemoteBook newBook = new RemoteBook(bookId, "new book#" + bookId);
                try {
                    onNewBookArrived(newBook);
                } catch (RemoteException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void onNewBookArrived(RemoteBook newBook) throws RemoteException {
        mBookList.add(newBook);
        final int N = mListenerList.beginBroadcast();
        for (int i = 0; i < N; i++) {
            IOnNewBookArrivedListener l = mListenerList.getBroadcastItem(i);
            if (l != null) {
                try {
                    l.onNewBookArrived(newBook, onNewBookArrivedSuccessListener);
                } catch (RemoteException e) {
                    e.printStackTrace();
                }
            }
        }
        mListenerList.finishBroadcast();
    }


    @Override
    public IBinder onBind(Intent intent) {
        //其实就是启动一个远程返回。如果是跨进程的服务，那么就要通过Binder进行通信
        //那么Binder返回的对象其实就是在这个地方
        //那Binder对象又是怎么生成的呢？很简单。AIDL文件可以生成
        return mBinder;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mIsServiceDestoryed.set(true);
    }
}
