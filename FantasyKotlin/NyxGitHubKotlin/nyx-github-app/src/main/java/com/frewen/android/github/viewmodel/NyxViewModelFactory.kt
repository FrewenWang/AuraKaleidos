package com.frewen.android.github.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.frewen.aura.framework.kotlin.di.scope.AppScope
import javax.inject.Inject
import javax.inject.Provider
import javax.inject.Singleton

/**
 * @filename: NyxViewModelFactory
 * @introduction: 这个是由NyxViewModelFactory来进行生成ViewModel对象的方法
 * @author: Frewen.Wong
 * @time: 2020/5/31 20:05
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@AppScope
class NyxViewModelFactory @Inject constructor(private val creators: Map<Class<out ViewModel>, @JvmSuppressWildcards Provider<ViewModel>>) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        val creator = creators[modelClass] ?: creators.entries.firstOrNull {
            modelClass.isAssignableFrom(it.key)
        }?.value ?: throw IllegalArgumentException("unknown model class $modelClass")
        try {
            @Suppress("UNCHECKED_CAST")
            return creator.get() as T
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
    }


}