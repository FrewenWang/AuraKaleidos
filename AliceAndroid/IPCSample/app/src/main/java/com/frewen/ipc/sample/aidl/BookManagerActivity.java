package com.frewen.ipc.sample.aidl;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.RemoteException;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import com.frewen.ipc.sample.R;
import com.frewen.ipc.service.sample.IOnNewBookArrivedListener;
import com.frewen.ipc.service.sample.IOnNewBookArrivedSuccessListener;
import com.frewen.ipc.service.sample.IRemoteBookManager;
import com.frewen.ipc.service.sample.RemoteBook;

import java.util.List;

public class BookManagerActivity extends AppCompatActivity {
    private static final String TAG = "BookManagerActivity";

    private static final String ACTION = "android.intent.service.REMOTE_BOOK_SERVICE";
    private static final int MESSAGE_NEW_BOOK_ARRIVED = 1;
    private IRemoteBookManager bookManager;

    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MESSAGE_NEW_BOOK_ARRIVED:
                    Log.d(TAG, "receive new book :" + msg.obj);
                    break;
                default:
                    super.handleMessage(msg);
            }
        }
    };

    /**
     * Binder的注册监听者
     */
    private IOnNewBookArrivedListener onNewBookArrivedListener = new IOnNewBookArrivedListener.Stub() {
        @Override
        public void onNewBookArrived(RemoteBook newBook, IOnNewBookArrivedSuccessListener listener) throws RemoteException {
            Log.d(TAG, "onNewBookArrived() called with: newBook = [" + newBook + "], listener = [" + listener + "]");
            Log.d(TAG, "onNewBookArrived() called with: thread = [" + Thread.currentThread().getName() + "]");
            mHandler.obtainMessage(MESSAGE_NEW_BOOK_ARRIVED, newBook).sendToTarget();
            listener.onSuccess();
            listener.onFailed();
        }
    };


    private ServiceConnection mRemoteConn = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "onServiceConnected() called with: name = [" + name + "], service = [" + service + "]");
            bookManager = IRemoteBookManager.Stub.asInterface(service);
            try {
                List<RemoteBook> bookList = bookManager.getBookList();
                Log.i(TAG, "onServiceConnected: query book list:" + bookList.getClass().getCanonicalName());
                Log.i(TAG, "query book list:" + bookList.toArray()[0].toString());

                Toast.makeText(BookManagerActivity.this, "onServiceConnected: query book list:" + bookList.getClass().getCanonicalName(), Toast.LENGTH_LONG).show();
                Toast.makeText(BookManagerActivity.this, "query book list:" + bookList.toArray()[0].toString(), Toast.LENGTH_LONG).show();

                RemoteBook newBook = new RemoteBook(3, "Android开发艺术探索");
                bookManager.addBook(newBook);
                Log.i(TAG, "add book:" + newBook);

                Log.i(TAG, "query book list:" + bookList.toArray()[0].toString());

                Log.i(TAG, "registerListener:onNewBookArrivedListener==" + onNewBookArrivedListener);
                bookManager.registerListener(onNewBookArrivedListener);

            } catch (RemoteException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.d(TAG, "onServiceDisconnected() called with: name = [" + name + "]");
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_book_manager);

        //启动远程服务。绑定远程服务，通过ServiceConnection来拿到远程返回的Binder对象。其实就是AIDL生成的方法对象
        Log.i(TAG, "onCreate:mRemoteConn  RemoteBookManagerService");
        Intent intent = new Intent();
        //自定义Service的包名
        intent.setPackage("com.frewen.ipc.service.sample");
        intent.setAction(ACTION);
        bindService(intent, mRemoteConn, Context.BIND_AUTO_CREATE);


    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (bookManager != null
                && bookManager.asBinder().isBinderAlive()) {
            try {
                Log.i(TAG, "unregister listener:" + onNewBookArrivedListener);
                bookManager.unregisterListener(onNewBookArrivedListener);
            } catch (RemoteException exception) {
                exception.printStackTrace();
            }
        }
        unbindService(mRemoteConn);
    }
}
