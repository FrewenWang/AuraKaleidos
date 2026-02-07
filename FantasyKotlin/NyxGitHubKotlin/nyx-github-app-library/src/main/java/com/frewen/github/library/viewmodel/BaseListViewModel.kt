package com.frewen.github.library.viewmodel

import android.content.Context
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.frewen.github.library.common.LoadState
import com.frewen.github.library.network.ResultCallBack

/**
 * @filename: BaseListViewModel
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/11 23:13
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
abstract class BaseListViewModel(private val context: Context) : ViewModel(), ResultCallBack<ArrayList<Any>> {
    // 定义易变性的LiveData
    val dataList = MutableLiveData<ArrayList<Any>>()

    /**
     * 当前数据刷新的状态，包括：
     * 1、刷新
     * 2、加载更多
     *
     */
    private val loadingState = MutableLiveData<LoadState>()

    val needMore = MutableLiveData<Boolean>()

    var lastPage: Int = -1

    var page = 1

    init {
        needMore.value = true
        loadingState.value = LoadState.NONE
        dataList.value = arrayListOf()
    }

    override fun onSuccess(result: ArrayList<Any>?) {

    }

    /**
     * 刷新逻辑的业务逻辑
     */
    open fun refresh() {
        if (isLoading()) {
            return
        }
        page = 1
        loadingState.value = LoadState.Refresh
        loadDataByRefresh()
    }

    /**
     * 加载更多的业务逻辑判断
     */
    open fun loadMore() {
        if (isLoading()) {
            return
        }
        page++
        loadingState.value = LoadState.LoadMore
        loadDataByLoadMore()
    }

    /**
     *TODO 这个方法是什么意思？？
     */
    open fun clearWhenRefresh() {
        if (page <= 1) {
            dataList.value = arrayListOf()
            needMore.value = true
        }
    }

    /**
     * 判断是够正在加载中
     */
    open fun isLoading(): Boolean =
            loadingState.value == LoadState.Refresh && loadingState.value == LoadState.LoadMore

    /**
     * 下拉刷新的业务逻辑
     */
    abstract fun loadDataByRefresh()

    /**
     * 加载更多的业务逻辑处理
     */
    abstract fun loadDataByLoadMore()

}