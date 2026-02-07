package com.frewen.android.github.app;

import android.app.Activity;
import android.content.Context;

import com.frewen.android.github.di.AppInjector;
import com.frewen.android.github.samples.hook.HookHelper;
import com.frewen.aura.framework.app.BaseApp;
import com.frewen.aura.toolkits.core.FreeToolKits;

import javax.inject.Inject;

import dagger.android.AndroidInjector;
import dagger.android.DispatchingAndroidInjector;
import dagger.android.HasActivityInjector;

/**
 * MyApp
 */
public class MyApp extends BaseApp implements HasActivityInjector {
    private static final String TAG = "T:MyApp";

    /**
     * 分发Activity的注入
     *
     * 在Activity调用AndroidInjection.inject(this)时
     * 从Application获取一个DispatchingAndroidInjector<Activity>，并将activity传递给inject(activity)
     * DispatchingAndroidInjector通过AndroidInjector.Factory创建AndroidInjector
     */
    @Inject
    DispatchingAndroidInjector<Activity> dispatchingAndroidInjector;

    @Override

    protected void attachBaseContext(Context base) {
        // 在这里调用Context的方法会崩溃
        super.attachBaseContext(base);
        // 在这里可以正常调用Context的方法
        HookHelper.hookActivityManager();
    }

    @Override
    public void onCreate() {
        super.onCreate();

        FreeToolKits.init(this, "NyxGitHub");
        //Application级别注入
        AppInjector.INSTANCE.inject(this);
    }

    @Override
    public AndroidInjector<Activity> activityInjector() {
        return this.dispatchingAndroidInjector;
    }
}
