package com.frewen.github.library.common

/**
 * @filename: LoadState
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/14 19:16
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
enum class LoadState {
    Refresh,
    LoadMore,
    RefreshDone,
    LoadMoreDone,
    NONE
}