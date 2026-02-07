package com.frewen.android.github.samples.ipc.remote.socket

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log

import com.frewen.android.github.R
import com.frewen.aura.toolkits.common.IOStreamUtils
import com.frewen.aura.toolkits.concurrent.ThreadPoolFactory

import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.net.ServerSocket
import java.net.Socket
import java.util.Random

/**
 * 我们这里是在远程建立一个TCP服务，等待客户端进行连接。
 * 这个TCPService运行在一个单独的进程中:socket_server
 *
 * @author Frewen.Wong
 */
class RemoteTCPService : Service() {
    private val isServiceDestroyed = false
    private val mDefinedAnswer = resources.getStringArray(R.array.responseStrArray)

    override fun onCreate() {
        super.onCreate()
        // 获取线程池，使用线程池类进行处理调度逻辑
        ThreadPoolFactory.getInstance().execute(RemoteTCPServer())
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }


    /**
     * 启动TCP服务
     * 在子线程中的中启动SocketServer的的的服务
     */
    private inner class RemoteTCPServer : Runnable {

        override fun run() {
            Log.d(TAG, "FMsg:RemoteTCPServer run in " + Thread.currentThread().name)
            // 启动 ServerSocket 同时监听本地8688的端口
            var serverSocket: ServerSocket?
            try {
                serverSocket = ServerSocket(8688)
            } catch (e: IOException) {
                System.err.println("establish tcp server failed,port:8688")
                e.printStackTrace()
                return
            }

            /**
             * 如果远程服务没有销毁
             */
            while (!isServiceDestroyed) {
                try {
                    // 通过服务端的ServerSocket接收到客户端的Socket对象。获取对象实例
                    val clientSocket = serverSocket.accept()

                    ThreadPoolFactory.getInstance().execute {
                        Log.d(TAG, "FMsg:responseToClient run in " + Thread.currentThread().name)
                        try {
                            responseToClient(clientSocket)
                        } catch (e: IOException) {
                            e.printStackTrace()
                        }
                    }
                } catch (e: IOException) {
                    e.printStackTrace()
                }

            }
        }

        /**
         * 给客户端进行应答
         *
         * @param clientSocket
         */
        @Throws(IOException::class)
        private fun responseToClient(clientSocket: Socket) {
            // 用于接收客户端的Client传过来的信息.
            // 字节流与字符流之间的桥梁，能将字节流输出为字符流，并且能为字节流指定字符集，可输出一个个的字符；
            val `in` = BufferedReader(InputStreamReader(clientSocket.getInputStream()))

            // 用于向客户端发送消息
            //输出的字节流变为字符流，即将一个字节流的输出对象变为字符流输出对象

            val out = PrintWriter(BufferedWriter(
                    OutputStreamWriter(clientSocket.getOutputStream())), true)
            out.println("欢迎来到Socket聊天室")
            /**
             * 如果服务没有被销毁
             */
            while (!isServiceDestroyed) {
                // 逐行读取缓冲字符读取流
                val msg = `in`.readLine()
                println("Received Msg From Client : " + msg!!)

                if (null != msg) {
                    // 如果已经读取不到响应的数据。说明客户端已经断开连接
                    return
                }

                val index = Random().nextInt(mDefinedAnswer.size)
                val answer = mDefinedAnswer[index]
                out.println(answer)
                println("Reply Msg To Client : $answer")

                println("Client quit !!!")
                IOStreamUtils.closeAll(out, `in`)
                clientSocket.close()
            }
        }
    }

    companion object {
        private const val TAG = "T:RemoteTCPService"
    }
}
