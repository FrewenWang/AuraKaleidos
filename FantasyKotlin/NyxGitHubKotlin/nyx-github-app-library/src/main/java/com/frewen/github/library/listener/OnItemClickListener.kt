package com.frewen.github.library.listener

import android.content.Context

/**
 * @filename: OnItemClickListener
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/14 19:42
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
interface OnItemClickListener {
    fun onItemClick(context: Context, position: Int)
}