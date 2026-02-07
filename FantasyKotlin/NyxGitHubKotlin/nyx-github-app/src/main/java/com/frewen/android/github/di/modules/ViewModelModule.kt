package com.frewen.android.github.di.modules

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.frewen.android.github.di.annotations.ViewModelKey
import com.frewen.android.github.viewmodel.NewsViewModel
import com.frewen.android.github.viewmodel.NyxViewModelFactory
import dagger.Binds
import dagger.Module
import dagger.multibindings.IntoMap

/**
 * @filename: ViewModelModule
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/5/31 17:09
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@Suppress("unused")
@Module
abstract class ViewModelModule {

    // @Binds 类似于 @Provides，在使用接口声明时使用，区别是 @Binds 用于修饰抽象类中的抽象方法的
    // 这个方法必须返回接口或抽象类，比如 ViewModel，不能直接返回 LoginViewModel
    // 方法的参数就是这个方法返回的是注入的对象，类似@Provides修饰的方法返回的对象
    // 这里的 LoginViewModel 会通过上述声明的构造器注入自动构建
    @Binds
    @IntoMap
    //@MapKey的封装注解
    @ViewModelKey(NewsViewModel::class)
    abstract fun bindLoginViewModel(loginViewModel: NewsViewModel): ViewModel


    @Binds
    abstract fun bindViewModelFactory(factory: NyxViewModelFactory): ViewModelProvider.Factory
}