package com.frewen.github.library.network

/**
 * @filename: ResultCallBack
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/11 22:15
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
interface ResultCallBack<T> {

    fun onPage(first: Int, current: Int, last: Int) {

    }

    fun onSuccess(result: T?)

    fun onCacheSuccess(result: T?) {

    }

    fun onFailure() {

    }

}