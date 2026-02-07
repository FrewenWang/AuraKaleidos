package com.frewen.github.library.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import androidx.databinding.ViewDataBinding
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProviders
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.frewen.aura.toolkits.kotlin.ext.autoCleared
import com.frewen.github.library.holder.BindingDataRecyclerManager
import com.frewen.github.library.viewmodel.BaseListViewModel
import com.shuyu.commonrecycler.BindSuperAdapter
import com.shuyu.commonrecycler.BindSuperAdapterManager
import com.shuyu.commonrecycler.listener.OnItemClickListener
import com.shuyu.commonrecycler.listener.OnLoadingListener
import javax.inject.Inject

/**
 * @filename: BaseListFragment
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/14 19:32
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
abstract class BaseListMVVMFragment<VB : ViewDataBinding, VM : BaseListViewModel>
    : BaseMVVMFragment<VB>(), OnItemClickListener, OnLoadingListener {
    private lateinit var adapter: BindSuperAdapter

    // 声明适配器管理类
    private var normalAdapterManager by autoCleared<BindingDataRecyclerManager>()

    /**
     * 基础ViewModel
     */
    private lateinit var baseListViewModel: VM

    /**
     * TODO 需要查一下这个对象是怎么注入的
     */
    @Inject
    lateinit var viewModelFactory: ViewModelProvider.Factory

    override fun onCreateView(mainView: View?) {
        // 实例化BindingDataRecyclerManager
        normalAdapterManager = BindingDataRecyclerManager()
        baseListViewModel = ViewModelProviders.of(this, viewModelFactory)
                .get(getViewModelClass())

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        initListData()
    }

    private fun initListData() {
        if (activity != null && getRecyclerView() != null) {
            normalAdapterManager?.setPullRefreshEnabled(enableRefresh())
                    ?.setLoadingMoreEnabled(enableLoadMore())
                    ?.setOnItemClickListener(this)
                    ?.setLoadingListener(this)

            normalAdapterManager?.apply {
                bindHolder(this)
                adapter = BindSuperAdapter(activity as Context, this, arrayListOf())
                getRecyclerView()?.layoutManager = LinearLayoutManager(activity!!)
                getRecyclerView()?.adapter = adapter
            }
        }
    }

    /**
     * 刷新
     */
    override fun onRefresh() {
        getViewModel().refresh()
    }

    /**
     * 加载更多
     */
    override fun onLoadMore() {
        getViewModel().loadMore()
    }

    /**
     * 绑定Item
     */
    abstract fun bindHolder(manager: BindSuperAdapterManager)

    /**
     * 是否需要下拉刷新.默认不支持下来刷新
     * 方法为open允许重写，
     */
    open fun enableRefresh(): Boolean = false

    /**
     * 是否需要下拉刷新
     */
    open fun enableLoadMore(): Boolean = false

    /**
     * 子类实现，返回ViewModel的ViewModel Class对象
     */
    abstract fun getViewModelClass(): Class<VM>

    /**
     * 当前 recyclerView，为空即不走 @link[initListData] 的初始化
     */
    abstract fun getRecyclerView(): RecyclerView?

    /**
     * 获取当前页面的ViewModel
     */
    open fun getViewModel(): VM = baseListViewModel
}