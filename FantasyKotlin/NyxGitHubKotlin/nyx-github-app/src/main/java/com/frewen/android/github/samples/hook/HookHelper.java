package com.frewen.android.github.samples.hook;

import android.app.Activity;
import android.os.Build;
import android.util.Log;

import com.frewen.android.github.BuildConfig;
import com.frewen.aura.toolkits.reflection.ReflectionInvokeHelper;

import java.lang.reflect.Proxy;

/**
 * @filename: HookHelper
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/11/19 0019 下午9:05
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class HookHelper {
    private static final String TAG = "T:HookHelper";

    public static void hookActivityManager() {
        Log.d(TAG, "FMsg:hookActivityManager() called");
        // 创建一个这个对象的代理对象iActivityManagerInterface, 然后替换这个字段, 让我们的代理对象帮忙干活
        try {
            Object iActivityManagerSingleton;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                //获取ActivityManager的gDefault单例gDefault，对象IActivityManagerSingleton是静态的
                iActivityManagerSingleton = ReflectionInvokeHelper
                        .getStaticFieldObject("android.app.ActivityManager", "IActivityManagerSingleton");
            } else {
                //获取AMN的gDefault单例gDefault，gDefault是静态的
                iActivityManagerSingleton = ReflectionInvokeHelper.getStaticFieldObject("android.app.ActivityManagerNative", "gDefault");
            }

            //通过获取iActivityManager的获取iActivityManager
            Object iActivityManager = ReflectionInvokeHelper.getFieldObject(
                    "android.util.Singleton",
                    iActivityManagerSingleton, "mInstance");
            // 获取IActivityManager的Class文件
            Class<?> iActivityManagerClazz = Class.forName("android.app.IActivityManager");

            Object proxy = Proxy.newProxyInstance(Thread.currentThread().getContextClassLoader(),
                    new Class<?>[]{iActivityManagerClazz}, new HookInvocationHandler(iActivityManager));

            ReflectionInvokeHelper.setFieldObject(iActivityManagerSingleton, "mInstance", proxy);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * 我们来Hook Instrumentation进行启动
     */
    public static void hookInstrumentation() {
        try {

            //
            Object gDefault = ReflectionInvokeHelper.getFieldObject(Activity.class, "gDefault");

            // gDefault是一个 android.util.Singleton对象; 我们取出这个单例里面的mInstance字段，IActivityManager类型
            Object rawIActivityManager = ReflectionInvokeHelper.getFieldObject(
                    "android.util.Singleton",
                    gDefault, "mInstance");

            Class<?> iActivityManagerInterface = Class.forName("android.app.IActivityManager");

            Object proxy = Proxy.newProxyInstance(
                    Thread.currentThread().getContextClassLoader(),
                    new Class<?>[]{iActivityManagerInterface},
                    new HookInvocationHandler(rawIActivityManager));

            //把Singleton的mInstance替换为proxy
            ReflectionInvokeHelper.setFieldObject("android.util.Singleton", gDefault, "mInstance", proxy);

        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
