package com.frewen.ipc.service;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.support.annotation.Nullable;

/**
 * 远程服务
 */
public class RemoteBookManagerService extends Service {

    private static final String TAG = "BookManagerService";

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
