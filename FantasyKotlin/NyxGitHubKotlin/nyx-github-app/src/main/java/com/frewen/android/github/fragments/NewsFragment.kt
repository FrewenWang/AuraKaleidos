package com.frewen.android.github.fragments

import android.content.Context
import androidx.recyclerview.widget.RecyclerView
import com.frewen.android.github.R
import com.frewen.android.github.databinding.FragmentGithubNewsBinding
import com.frewen.android.github.holder.EventHolder
import com.frewen.android.github.model.EventUIModel
import com.frewen.android.github.viewmodel.NewsViewModel
import com.frewen.github.library.fragments.BaseListMVVMFragment
import com.shuyu.commonrecycler.BindSuperAdapterManager
import kotlinx.android.synthetic.main.fragment_github_news.*

/**
 * @filename: NewFragment
 * @introduction: Github的最新新闻页面的Fragment.
 * 继承自BaseListMVVMFragment。也就是列表相关的基于MVVM框架的封装Fragment
 * 需要传入两个泛型类型：FragmentGithubNewsBinding(fragment_github_new.xml生成)、NewsViewModel
 * @author: Frewen.Wong
 * @time: 2020/4/14 19:30
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class NewsFragment : BaseListMVVMFragment<FragmentGithubNewsBinding, NewsViewModel>() {

    override fun getLayoutId(): Int = R.layout.fragment_github_news

    override fun onItemClick(context: Context, position: Int) {
        print("NewsFragment ,position =  $position")
    }

    override fun getViewModelClass(): Class<NewsViewModel> = NewsViewModel::class.java

    override fun enableRefresh(): Boolean = true

    override fun enableLoadMore(): Boolean = true

    override fun bindHolder(manager: BindSuperAdapterManager) {
        manager.bind(EventUIModel::class.java, EventHolder.LAYOUT_ID, EventHolder::class.java)
    }

    override fun getRecyclerView(): RecyclerView? = news_recyclerview
}