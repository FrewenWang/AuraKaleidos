package com.frewen.github.library.utils

import android.graphics.Point
import android.widget.ImageView
import androidx.databinding.BindingAdapter
import androidx.databinding.DataBindingComponent


/**
 * @filename: DataBindingExtUtils
 * @introduction:  DataBinding的拓展适配器
 * @author: Frewen.Wong
 * @time: 2020/4/11 22:03
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class AuraDataBindingAdapter {
    /**
     * Kotlin中的没有类的静态方法的概念
     * 所以我们可以使用伴生对象
     */
    companion object {
        /**
         * 高斯模糊图片加载
         */
        @BindingAdapter("image_blur")
        fun loadImageBlur(view: ImageView, url: String?) {
            ImageLoadUtils.loadImageBlur(view, url ?: "")
        }

        /**
         * 圆形用户头像加载
         */
        @BindingAdapter("userHeaderUrl", "userHeaderSize", requireAll = false)
        fun loadImage(view: ImageView, url: String?, size: Int = 50) {
            ImageLoadUtils.loadUserHeaderImage(view, url ?: "", Point(size, size))
        }


    }
}

/**
 * 加载 DataBinding 的拓展适配器
 */
public class DataBindingComponentImpl : DataBindingComponent {
    override fun getCompanion(): AuraDataBindingAdapter.Companion = AuraDataBindingAdapter

}