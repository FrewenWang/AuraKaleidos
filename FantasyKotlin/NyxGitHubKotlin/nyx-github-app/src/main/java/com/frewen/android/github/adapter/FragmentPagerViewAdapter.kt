package com.frewen.android.github.adapter

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentPagerAdapter

/**
 * @filename: FragmentPagerViewAdapter
 * @introduction: Fragment的ViewPager的适配器
 * @author: Frewen.Wong
 * @time: 2020/4/3 17:35
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class FragmentPagerViewAdapter(private val fragmentList: List<Fragment>, fragmentManager: FragmentManager) : FragmentPagerAdapter(fragmentManager) {

    override fun getItem(position: Int): Fragment {
        return fragmentList[position]
    }

    override fun getCount(): Int {
        return fragmentList.size
    }

}