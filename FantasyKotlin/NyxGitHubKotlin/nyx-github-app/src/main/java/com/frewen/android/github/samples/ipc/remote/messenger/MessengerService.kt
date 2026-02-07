package com.frewen.android.github.samples.ipc.remote.messenger

import android.app.Service
import android.content.Intent
import android.os.IBinder
import com.frewen.aura.ui.progress.ProgressLoading

/**
 * @filename: MessengerService
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020-02-18 10:58
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class MessengerService : Service() {
    companion object {
        private lateinit var mDialog: ProgressLoading
    }


    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

}
