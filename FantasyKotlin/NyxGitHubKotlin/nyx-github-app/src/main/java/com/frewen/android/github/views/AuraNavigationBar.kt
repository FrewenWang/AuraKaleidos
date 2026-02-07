package com.frewen.android.github.views

import android.content.Context
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import devlight.io.library.ntb.NavigationTabBar

/**
 * @filename: AuraNavigationBar
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/2 11:50
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class AuraNavigationBar : NavigationTabBar {

    var isTouchEnable = true

    var doubleTouchListener: TabDoubleClickListener? = null

    private var gestureDetector = GestureDetector(context.applicationContext, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent?): Boolean {
            doubleTouchListener?.onDoubleClick(mIndex)
            return super.onDoubleTap(e)
        }
    })

    constructor(context: Context) : super(context)

    constructor(context: Context, attrs: AttributeSet?) : super(context, attrs)

    constructor(context: Context, attrs: AttributeSet?, defStyleAttr: Int) : super(context, attrs, defStyleAttr)

    /**
     * Touch事件的监听
     */
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (!isTouchEnable) {
            return true
        }
        super.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        return true
    }

    interface TabDoubleClickListener {
        fun onDoubleClick(position: Int)
    }
}