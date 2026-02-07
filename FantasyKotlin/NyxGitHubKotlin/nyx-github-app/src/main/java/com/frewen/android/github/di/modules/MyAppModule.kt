package com.frewen.android.github.di.modules

import com.frewen.android.github.network.RetrofitFactory
import com.frewen.aura.framework.kotlin.di.scope.AppScope
import dagger.Module
import dagger.Provides
import retrofit2.Retrofit

/**
 * @filename: MyAppModule
 * @introduction: 我们没有使用AuraFrameWork里面提供的
 * @see AppMudule
 * 而是我们自己来实现的MyAppModule 是否能考虑扩展？？
 * @author: Frewen.Wong
 * @time: 2020/4/7 12:54
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@Module(includes = [ViewModelModule::class])
class MyAppModule {
    @AppScope
    @Provides
    fun providerRetrofit(): Retrofit {
        return RetrofitFactory.instance.retrofit
    }
}
