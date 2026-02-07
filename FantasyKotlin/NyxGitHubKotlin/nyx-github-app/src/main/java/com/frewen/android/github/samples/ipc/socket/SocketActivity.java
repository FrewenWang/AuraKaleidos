package com.frewen.android.github.samples.ipc.socket;

import android.content.Intent;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.os.Bundle;
import android.widget.TextView;

import com.frewen.android.github.R;
import com.frewen.android.github.samples.ipc.remote.socket.RemoteTCPService;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;

import androidx.appcompat.app.AppCompatActivity;

/**
 * 我们通过Socket来实现进程间的通信。Socket也称为“套接字”，是网络通信中的概念，
 * 它分为流式套接字和用户数据报套接字两种，分别对应于网络的传输控制层中的TCP和UDP协议。
 * TCP协议是面向连接的协议，提供稳定的双向通信功能，TCP连接的建立需要经过“三次握手”才能完成，
 * 为了提供稳定的数据传输功能，其本身提供了超时重传机制，因此具有很高的稳定性；
 * 而UDP是无连接的，提供不稳定的单向通信功能，当然UDP也可以实现双向通信功能。
 * 在性能上，UDP具有更好的效率，其缺点是不保证数据一定能够正确传输，尤其是在网络拥塞的情况下。
 * <p>
 * <p>
 * 关于TCP和UDP的介绍就这么多，更详细的资料请查看相关网络资料。
 * 接下来我们演示一个跨进程的聊天程序，两个进程可以通过Socket来实现信息的传输，
 * Socket本身可以支持传输任意字节流，这里为了简单起见，仅仅传输文本信息，很显然，这是一种IPC方式。
 * <p>
 * 使用Socket来进行通信，有两点需要注意，首先需要声明权限：
 */
public class SocketActivity extends AppCompatActivity {
    private static final String TAG = "T:SocketActivity";

    private static final int MESSAGE_SOCKET_CONNECTED = 0x123;
    private static final int MESSAGE_RECEIVE_NEW_MSG = 0x124;
    private TextView mMessageTextView;

    private Socket mClientSocket;
    private PrintWriter mPeintWriter;
    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MESSAGE_RECEIVE_NEW_MSG: {
                    mMessageTextView.setText(mMessageTextView.getText()
                            + (String) msg.obj);
                    break;
                }
                case MESSAGE_SOCKET_CONNECTED: {
                    //mSendButton.setEnabled(true);
                    break;
                }
                default:
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_socket);

        // 这里我们模拟启动一个服务端。来等待客户端进行连接
        Intent service = new Intent(this, RemoteTCPService.class);
        startService(service);

        // 我们创建一个子线程来开始尝试连接远程服务端
        // 至于为什么要使用子线程，我们也进行了介绍。所有网络请求都需要在子线程去完成。
        tryToConnectServer();
    }

    /**
     * 尝试通过Socket与服务端继续连接
     */
    private void tryToConnectServer() {

        // 步骤1：创建客户端 & 服务器的连接
        Socket socket = null;
        while (null == socket) {
            try {
                //步骤2： 创建Socket对象 & 指定服务端的IP及端口号。由于我们服务端也是运行在本机的Service
                // 所以服务端的IP我们也写为localhost
                socket = new Socket("localhost", 8688);
                mClientSocket = socket;
                // 步骤3：
                mPeintWriter = new PrintWriter(new BufferedWriter(
                        new OutputStreamWriter(socket.getOutputStream())), true);
                System.out.println("connect server success.");
                mHandler.sendEmptyMessage(MESSAGE_SOCKET_CONNECTED);
            } catch (IOException e) {
                SystemClock.sleep(1000);
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onDestroy() {
        try {
            if (null != mClientSocket) {
                mClientSocket.shutdownInput();
                mClientSocket.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }
}
