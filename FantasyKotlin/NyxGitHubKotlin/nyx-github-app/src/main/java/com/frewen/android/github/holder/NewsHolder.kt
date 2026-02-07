package com.frewen.android.github.holder

import android.content.Context
import android.view.View
import androidx.databinding.ViewDataBinding
import com.frewen.android.github.R
import com.frewen.android.github.databinding.LayoutItemHomeNewsBinding
import com.frewen.android.github.model.EventUIModel
import com.frewen.github.library.holder.DataBindingHolder
import kotlinx.android.synthetic.main.layout_item_home_news.view.*

/**
 * @filename: NewsHolder
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/16 21:41
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class EventHolder(context: Context, private val v: View, dataBing: ViewDataBinding) : DataBindingHolder<EventUIModel, LayoutItemHomeNewsBinding>
(context, v, dataBing) {


    override fun createView(v: View) {
        super.createView(v)
    }

    /**
     * 这个返回的是新闻页面的Item的布局ID
     */
    companion object {
        const val LAYOUT_ID = R.layout.layout_item_home_news
    }

    override fun onBind(model: EventUIModel, position: Int, dataBinding: LayoutItemHomeNewsBinding) {
        dataBinding.eventUIModel = model
        v.event_user_img.setOnClickListener {
            //
        }
    }
}