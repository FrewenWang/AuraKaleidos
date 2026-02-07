package com.frewen.github.library.holder

import android.content.Context
import android.view.View
import androidx.databinding.ViewDataBinding
import com.shuyu.commonrecycler.BindRecyclerBaseHolder

/**
 * @filename: DataBindingHolder
 * @introduction: 基础库里面的BindRecyclerBaseHolder.继承自BindRecyclerBaseHolder
 * @author: Frewen.Wong
 * @time: 2020/4/16 21:53
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
abstract class DataBindingHolder<Model, DataBinding>(context: Context, view: View, private val dataBing: ViewDataBinding) :
        BindRecyclerBaseHolder(context, view) {

    override fun createView(v: View) {
    }

    override fun onBind(model: Any, position: Int) {
        onBind(model as Model, position, dataBing as DataBinding)
    }

    abstract fun onBind(model: Model, position: Int, dataBinding: DataBinding)

}