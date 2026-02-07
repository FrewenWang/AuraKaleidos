package com.frewen.android.github.di.modules

import com.frewen.android.github.fragments.NewsFragment
import com.frewen.aura.framework.kotlin.di.scope.ActivityScope
import dagger.Module
import dagger.android.ContributesAndroidInjector

/**
 * @filename: FragmentBindModule
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/5/31 15:35
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@Module
abstract class MainPageFragmentBindModule {

    //主要作用就是通过 @ContributesAndroidInjector  标记哪个类需要使用依赖注入功能
    //节省代码
    @ContributesAndroidInjector
    abstract fun contributeHomeFragment(): NewsFragment

}