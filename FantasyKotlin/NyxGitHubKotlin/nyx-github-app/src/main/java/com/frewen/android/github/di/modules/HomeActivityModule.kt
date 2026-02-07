package com.frewen.android.github.di.modules

import android.app.Application
import android.graphics.Color
import androidx.fragment.app.Fragment
import com.frewen.android.github.R
import com.frewen.android.github.fragments.NewsFragment
import com.mikepenz.iconics.IconicsColor
import com.mikepenz.iconics.IconicsDrawable
import dagger.Module
import dagger.Provides
import devlight.io.library.ntb.NavigationTabBar

/**
 * @filename: MainActivityModule
 * @introduction:
 * @author: Frewen.Wong
 * @time: 200/4/9 20:16
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
@Module
class HomeActivityModule {

    @Provides
    fun providerMainFragmentList(): List<Fragment> {
        return listOf(NewsFragment(), NewsFragment(), NewsFragment(), NewsFragment())
    }

    @Provides
    fun providerMainTabModel(application: Application): List<NavigationTabBar.Model> {
        return listOf()

    }
}