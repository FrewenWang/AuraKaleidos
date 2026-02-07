package com.frewen.android.github.samples.view;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Scroller;

import static android.view.MotionEvent.*;

/**
 * @filename: HorizontalScrollView
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/10/31 0031 上午10:47
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class HorizontalScrollView extends ViewGroup {
    private static final String TAG = "T:HorizontalScrollView";
    /**
     * 弹性滑动对象，用于实现View的弹性滑动。
     * 使用View的scrollTo/scrollBy方法来进行滑动时，其过程是瞬间完成的，这个没有过渡效果的滑动用户体验不好
     * 使用Scroller来实现有过渡效果的滑动，其过程不是瞬间完成的，而是在一定的时间间隔内完成的。
     */
    private Scroller mScroller;
    /**
     * 速度追踪，用于追踪手指在滑动过程中的速度，包括水平和竖直方向的速度。
     */
    private VelocityTracker mVelocityTracker;
    private int mLastXIntercept;
    private int mLastYIntercept;
    private int mLastX;
    private int mLastY;


    private int mChildrenSize;
    /**
     * Child的宽度
     */
    private int mChildWidth;
    private int mChildIndex;

    public HorizontalScrollView(Context context) {
        super(context);
        initView();
    }

    public HorizontalScrollView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initView();
    }

    public HorizontalScrollView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initView();
    }


    private void initView() {
        // 实例化弹性滑动对象
        mScroller = new Scroller(getContext());
        mVelocityTracker = VelocityTracker.obtain();
    }

    /**
     * 重写ViewGroup的自定义View都要实现这个方法
     *
     * @param changed
     * @param l
     * @param t
     * @param r
     * @param b
     */
    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        int childLeft = 0;
        final int childCount = getChildCount();
        mChildrenSize = childCount;

        for (int i = 0; i < childCount; i++) {
            final View childView = getChildAt(i);
            if (childView.getVisibility() != View.GONE) {
                final int childWidth = childView.getMeasuredWidth();
                mChildWidth = childWidth;
                childView.layout(childLeft, 0, childLeft + childWidth,
                        childView.getMeasuredHeight());
                childLeft += childWidth;
            }
        }
    }

    @Override
    public boolean onInterceptTouchEvent(MotionEvent event) {
        /**
         * 是否拦截的标志变量
         */
        boolean intercepted = false;
        // 获取事件据距离ViewGroup的左上角的偏离值
        int x = (int) event.getX();
        int y = (int) event.getY();
        switch (event.getAction()) {
            case ACTION_DOWN: {
                intercepted = false;
                if (!mScroller.isFinished()) {
                    mScroller.abortAnimation();
                    intercepted = true;
                }
            }
            break;
            case ACTION_MOVE: {
                int deltaX = x - mLastXIntercept;
                int deltaY = y - mLastYIntercept;
                // 如果是水平方向上的滑动，则拦截这个事件
                if (Math.abs(deltaX) > Math.abs(deltaY)) {
                    intercepted = true;
                } else {
                    intercepted = false;
                }
            }
            break;
            case ACTION_UP: {
                intercepted = false;
            }
            break;
            default:
                break;

        }
        Log.d(TAG, "intercepted=" + intercepted);
        mLastX = x;
        mLastY = y;
        mLastXIntercept = x;
        mLastYIntercept = y;
        return intercepted;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        //VelocityTracker使用过程很简单，
        // 首先，在View的onTouchEvent方法中追踪当前单击事件的速度：
        mVelocityTracker.addMovement(event);
        // 获取距离当前View的x\y坐标的偏移值
        int x = (int) event.getX();
        int y = (int) event.getY();
        switch (event.getAction()) {
            case ACTION_DOWN:
                if (!mScroller.isFinished()) {
                    mScroller.abortAnimation();
                }
                break;
            case ACTION_MOVE:
                int deltaX = x - mLastX;
                int deltaY = y - mLastY;
                // scrollBy实际上也是调用了scrollTo方法，
                // 它实现了基于当前位置的相对滑动
                // scrollTo则实现了基于所传递参数的绝对滑动
                // 所以，这个方法就是
                scrollBy(-deltaX, 0);
                break;
            case ACTION_UP:
                int scrollX = getScrollX();
                // 根据滑动的距离来来判断已经滑过了多少个Item
                int scrollToChildIndex = scrollX / mChildWidth;
                // 当我们先知道当前的滑动速度时，这个时候可以采用如下方式来获得当前的速度：
                mVelocityTracker.computeCurrentVelocity(1000);
                // 计算x轴方向上的滑动速度
                float xVelocity = mVelocityTracker.getXVelocity();
                // 如果x轴方向上的速度超过50像素每秒
                if (Math.abs(xVelocity) >= 50) {
                    mChildIndex = xVelocity > 0 ? mChildIndex - 1 : mChildIndex + 1;
                } else {
                    mChildIndex = (scrollX + mChildWidth / 2) / mChildWidth;
                }
                // 针对Index的进行边界
                mChildIndex = Math.max(0, Math.min(mChildIndex, mChildrenSize - 1));

                int dx = mChildIndex * mChildWidth - scrollX;
                smoothScrollBy(dx, 0);
                mVelocityTracker.clear();
                break;
            default:
                break;
        }
        mLastX = x;
        mLastY = y;
        return true;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        // 设置默认的测量的宽度和高度
        int measuredWidth = 0;
        int measuredHeight = 0;
        // 计算子View的个数
        final int childCount = getChildCount();

        measureChildren(widthMeasureSpec, heightMeasureSpec);
        // 计算测量宽度和高度的测量模式和尺寸
        int widthSpaceSize = MeasureSpec.getSize(widthMeasureSpec);
        int widthSpecMode = MeasureSpec.getMode(widthMeasureSpec);
        int heightSpaceSize = MeasureSpec.getSize(heightMeasureSpec);
        int heightSpecMode = MeasureSpec.getMode(heightMeasureSpec);

        if (childCount == 0) {
            setMeasuredDimension(0, 0);
        } else if (heightSpecMode == MeasureSpec.AT_MOST) {
            final View childView = getChildAt(0);
            measuredHeight = childView.getMeasuredHeight();
            setMeasuredDimension(widthSpaceSize, childView.getMeasuredHeight());
        } else if (widthSpecMode == MeasureSpec.AT_MOST) {
            final View childView = getChildAt(0);
            measuredWidth = childView.getMeasuredWidth() * childCount;
            setMeasuredDimension(measuredWidth, heightSpaceSize);
        } else {
            final View childView = getChildAt(0);
            measuredWidth = childView.getMeasuredWidth() * childCount;
            measuredHeight = childView.getMeasuredHeight();
            setMeasuredDimension(measuredWidth, measuredHeight);
        }
    }


    /**
     * 光滑转动
     *
     * @param dx
     * @param dy
     */
    private void smoothScrollBy(int dx, int dy) {
        mScroller.startScroll(getScrollX(), 0, dx, 0, 500);
        invalidate();
    }

    @Override
    public void computeScroll() {
        if (mScroller.computeScrollOffset()) {
            scrollTo(mScroller.getCurrX(), mScroller.getCurrY());
            postInvalidate();
        }
    }

    @Override
    protected void onDetachedFromWindow() {
        mVelocityTracker.recycle();
        super.onDetachedFromWindow();
    }
}
