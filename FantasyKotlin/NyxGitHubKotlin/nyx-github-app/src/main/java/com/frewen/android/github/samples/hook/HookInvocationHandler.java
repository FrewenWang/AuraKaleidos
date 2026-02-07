package com.frewen.android.github.samples.hook;

import android.content.Intent;
import android.util.Log;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * @filename: HookInvocationHandler
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/11/19 0019 下午9:17
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class HookInvocationHandler implements InvocationHandler {
    private static final String TAG = "HookInvocationHandler";
    public static final String TARGET_INTENT = "target_intent";
    private Object mBase;

    public HookInvocationHandler(Object base) {
        mBase = base;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        Log.d(TAG, "HookInvocationHandler invoke called method:" + method.getName() + " with args:" + Arrays.toString(args));
        // 如果我们判断这个方法的名字是startActivity
        if ("startActivity".equals(method.getName())) {
            Intent intent = null;
            int index = 0;
            for (int i = 0; i < args.length; i++) {
                if (args[i] instanceof Intent) {
                    index = i;
                    break;
                }
            }
            intent = (Intent) args[index];
            Intent subIntent = new Intent();
            String packageName = "com.frewen.android.demo";
            subIntent.setClassName(packageName, "com.frewen.android.demo.samples.hook.target.TargetActivity");
            subIntent.putExtra(HookInvocationHandler.TARGET_INTENT, intent);
            args[index] = subIntent;
        }
        return method.invoke(mBase, args);
    }
}
