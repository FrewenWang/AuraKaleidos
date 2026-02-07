package com.frewen.android.github.viewmodel

import android.app.Application
import android.content.Context
import com.frewen.github.library.viewmodel.BaseListViewModel
import javax.inject.Inject

/**
 * @filename: NewsViewModel
 * @introduction: 新闻页面的ViewModel
 * @author: Frewen.Wong
 * @time: 2020/4/14 19:20
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class NewsViewModel @Inject constructor(app: Application) : BaseListViewModel(app) {

    override fun refresh() {
        if (isLoading()) {
            return
        }
        super.refresh()
    }

    override fun loadDataByRefresh() {
        loadData()
    }

    override fun loadDataByLoadMore() {
        loadData()
    }

    private fun loadData() {
        clearWhenRefresh()
        //TODO 开始加载数据
        // userRepository.getReceivedEvent(this, page)
    }

}