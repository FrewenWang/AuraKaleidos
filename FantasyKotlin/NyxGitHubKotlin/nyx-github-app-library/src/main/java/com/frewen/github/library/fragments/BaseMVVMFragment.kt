package com.frewen.github.library.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.databinding.DataBindingUtil
import androidx.databinding.ViewDataBinding
import androidx.fragment.app.Fragment
import com.frewen.aura.toolkits.kotlin.ext.autoCleared
import com.frewen.github.library.di.injector.Injectable
import com.frewen.github.library.utils.DataBindingComponentImpl

/**
 * @filename: BaseFragment
 * @introduction:
 *  使用 Databinding必须添加
 *   dataBinding {
 *       enabled = true
 *   }
 * @author: Frewen.Wong
 * @time: 2020/4/9 20:35
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
abstract class BaseMVVMFragment<T : ViewDataBinding> : Fragment(), Injectable {
    /**
     * 根据Fragment动态清理和获取binding对象
     */
    private var binding by autoCleared<T>()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        // 执行DataBindingUtil绑定布局的页面
        binding = DataBindingUtil.inflate(
                inflater,
                getLayoutId(),
                container,
                false,
                DataBindingComponentImpl())
        onCreateView(binding?.root)
        return binding?.root
    }


    abstract fun onCreateView(mainView: View?)

    abstract fun getLayoutId(): Int
}