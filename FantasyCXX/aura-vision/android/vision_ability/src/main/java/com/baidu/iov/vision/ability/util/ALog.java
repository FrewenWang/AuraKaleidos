package com.baidu.iov.vision.ability.util;

import android.util.Log;

import com.baidu.iov.vision.ability.BuildConfig;


/**
 * 日志工具
 *
 * Created by liwendong on 2018/5/9.
 */

public class ALog {

    public static final String TAG = BuildConfig.LOG_TAG;

    public static boolean DEBUG = true;

    public static int v(String tag, String msg) {
        return Log.v(TAG, tag + " : " + msg);
    }

    public static int v(String tag, String msg, Throwable tr) {
        return Log.v(TAG, tag + " : " + msg, tr);
    }

    public static int d(String tag, String msg) {
        return Log.d(TAG, tag + " : " + msg);
    }

    public static int d(String tag, String msg, Throwable tr) {
        return Log.d(TAG, tag + " : " + msg, tr);
    }

    public static int i(String tag, String msg) {
        return Log.i(TAG, tag + " : " + msg);
    }

    public static int i(String tag, String msg, Throwable tr) {
        return Log.i(TAG, tag + " : " + msg, tr);
    }

    public static int w(String tag, String msg) {
        return Log.w(TAG, tag + " : " + msg);
    }

    public static int w(String tag, String msg, Throwable tr) {
        return Log.w(TAG, tag + " : " + msg, tr);
    }

    public static int e(String tag, String msg) {
        return Log.e(TAG, tag + " : " + msg);
    }

    public static int e(String tag, String msg, Throwable tr) {
        return Log.e(TAG, tag + " : " + msg, tr);
    }
}
