package com.baidu.iov.vision.ability;

import android.content.Context;

/**
 * 保存视觉能力上下文
 *
 * Created by liwendong on 2018/5/9.
 */

public class VisionContext {

    private static Context sAppContext;

    public static void appContext(Context context) {
        sAppContext = context != null ? context.getApplicationContext() : context;
    }

    public static Context appContext() {
        return sAppContext;
    }
}
