package com.frewen.github.library.holder

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.frewen.github.library.utils.DataBindingComponentImpl
import androidx.databinding.DataBindingUtil
import androidx.databinding.ViewDataBinding
import androidx.databinding.library.baseAdapters.DataBinderMapperImpl
import com.shuyu.commonrecycler.BindRecyclerBaseHolder
import com.shuyu.commonrecycler.BindSuperAdapterManager
import java.lang.reflect.Constructor

/**
 * @filename: BindingDataRecyclerManager
 * @introduction:BindSuperAdapterManager的扩展，增加对DataBinding的支持
 * @author: Frewen.Wong
 * @time: 2020/4/14 20:19
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class BindingDataRecyclerManager : BindSuperAdapterManager() {
    var constructor: Constructor<*>? = null
    var constructorFirst = true

    @Suppress("UNCHECKED_CAST")
    override fun <T> contructorHolder(context: Context, parent: ViewGroup, classType: Class<out BindRecyclerBaseHolder>?, layoutId: Int): T? {
        val itemTextBinding: ViewDataBinding = DataBindingUtil.inflate(LayoutInflater.from(context), layoutId, parent, false,
                DataBindingComponentImpl())
        try {
            constructor = classType?.getDeclaredConstructor(Context::class.java, View::class.java, ViewDataBinding::class.java)
        } catch (e: NoSuchMethodException) {
            constructorFirst = false
            e.printStackTrace();
        }

        if (!constructorFirst) {
            try {
                constructor = classType?.getDeclaredConstructor(View::class.java)
            } catch (e: NoSuchMethodException) {
                e.printStackTrace();
            }
        }

        if (constructor == null) {
            throw RuntimeException("Holder Constructor Error For : " + classType?.name)
        }

        try {
            constructor?.isAccessible = true
            return if (constructorFirst) {
                constructor?.newInstance(context, itemTextBinding.root, itemTextBinding) as T
            } else {
                constructor?.newInstance(itemTextBinding.root) as T
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

}