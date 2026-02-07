package com.frewen.android.github.di.modules

import com.frewen.android.github.HomeActivity
import com.frewen.aura.framework.kotlin.di.scope.ActivityScope
import dagger.Module
import dagger.android.ContributesAndroidInjector

/**
 * @filename: ActivityBindModule
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/9 20:11
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@Module
abstract class ActivityBindModule {
    /**
     *  针对HomeActivity
     */
    @ActivityScope
    @ContributesAndroidInjector(modules = [HomeActivityModule::class, MainPageFragmentBindModule::class])
    abstract fun homeActivityInjector(): HomeActivity


}