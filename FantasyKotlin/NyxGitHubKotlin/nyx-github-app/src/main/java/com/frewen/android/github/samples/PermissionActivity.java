package com.frewen.android.github.samples;

import android.graphics.Camera;
import android.os.Bundle;
import android.view.SurfaceView;

import com.frewen.android.github.R;
import com.frewen.permission.FreeRxPermissions;

import androidx.appcompat.app.AppCompatActivity;
import io.reactivex.disposables.Disposable;

/**
 * PermissionActivity
 */
public class PermissionActivity extends AppCompatActivity {
    private static final String TAG = "T:PermissionActivity";

    private Camera mCamera;
    private SurfaceView mSurfaceView;
    private Disposable mDisposable;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FreeRxPermissions rxPermissions = new FreeRxPermissions(this);
        rxPermissions.setLogging(true);

        setContentView(R.layout.activity_permission);


    }
}
