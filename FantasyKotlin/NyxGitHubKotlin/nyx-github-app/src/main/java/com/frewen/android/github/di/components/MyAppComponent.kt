package com.frewen.android.github.di.components

import android.app.Application
import com.frewen.android.github.app.MyApp
import com.frewen.android.github.di.modules.ActivityBindModule
import com.frewen.android.github.di.modules.MyAppModule
import com.frewen.aura.framework.kotlin.di.scope.AppScope
import dagger.BindsInstance
import dagger.Component
import dagger.android.support.AndroidSupportInjectionModule

/**
 * @filename: MyAppComponent
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/7 12:52
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@AppScope
@Component(modules = [
    // 用于绑定扩展的组件，如v4
    // 我们使用的是支持库（v4库）的 Fragment
    // 接入后可以使用  AndroidInjection.inject(activity) 和  AndroidSupportInjection.inject(f)
    // 组件集成AppModule
    AndroidSupportInjectionModule::class,
    MyAppModule::class,
    ActivityBindModule::class
])
interface MyAppComponent {
    /**
     * 通过 @Component.Builder 增加builder方法，提供Application 注入方法。
     */
    @Component.Builder
    interface Builder {
        //@BindsInstance注解的作用，只能在 Component.Builder 中使用
        //创建 Component 的时候绑定依赖实例
        @BindsInstance
        fun application(application: Application): Builder

        fun build(): MyAppComponent
    }

    fun inject(app: MyApp)

}