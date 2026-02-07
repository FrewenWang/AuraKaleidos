package com.frewen.android.github.samples.view;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.frewen.android.github.R;
import com.frewen.aura.framework.kotlin.activity.BaseActivity;

/**
 * @filename: ViewDemoActivity
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019-05-15 08:17
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class ViewDemoActivity extends BaseActivity {
    private static final String TAG = "T:ViewDemoActivity";
    LinearLayout mLinearLayout;
    TextView mTextView;

    @Override
    protected void initData(Bundle savedInstanceState) {

    }

    @Override
    protected void bindContentView() {
        setContentView(R.layout.activity_view_demo);
    }

    @Override
    protected void initView() {
        //在onCreate()、onStrart()、onResume()方法中会返回0，
        // 这是因为当前activity所代表的界面还没显示出来没有添加到WindowPhone的DecorView上
        // 或要获取的view没有被添加到DecorView上或者该View的visibility属性为gone
        // 或者该view的width或height真的为0
        // 所以只有上述条件都不成立时才能得到非0的width和height。

//        getViewInfo("从Resume()方法获取View信息", mLinearLayout);
//        getViewInfo("从Resume()方法获取View信息", mTextView);
//
//        mLinearLayout.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
//            @Override
//            public void onGlobalLayout() {
//                mLinearLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
//                getViewInfo("从OnGlobalLayoutListener()方法获取View信息", mLinearLayout);
//                getViewInfo("从OnGlobalLayoutListener()方法获取View信息", mTextView);
//
//            }
//        });

    }

    /**
     * 获取View的信息
     *
     * @param view
     */
    private void getViewInfo(String testInfo, View view) {
        Log.d(TAG, "FMsg:getViewInfo() called with: testInfo = [" + testInfo + "], view = [" + view + "]");
        int left = view.getLeft();
        int right = view.getRight();
        int top = view.getTop();
        int bottom = view.getBottom();

        int width = view.getWidth();
        int height = view.getHeight();
        int measuredWidth = view.getMeasuredWidth();
        int measuredHeight = view.getMeasuredHeight();
        Log.i(TAG, "FMsg:getViewInfo() :width =  " + width + ",height = " + height
                + ",measuredWidth = " + measuredWidth + ",measuredHeight = " + measuredHeight);
        Log.i(TAG, "FMsg:getViewInfo() :left =  " + left + ",right = " + right
                + ",top=" + top + ",bottom= " + bottom);
    }

    @Override
    protected void destroyView() {

    }
}
