package com.baidu.admips;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.TextView;

import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Admips";
    private static final int REQUEST_EXTERNAL_STORAGE_PERMISSION = 1;
    private Timer timer = new Timer();
    private TextView tv;
    private String mTestMode; // dmips or cpumem
    private String mTestInfo;
    private final String mDefaultTestMode = "dmips";

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    TimerTask proc_task = new TimerTask() {
        @Override
        public void run() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    String progress = getProgress();
                    if (mTestMode.equals("dmips")) {
                        tv.setText("Testing DMIPS of " + progress + "%");
                    } else if (mTestMode.equals("cpumem")) {
                        tv.setText("Testing CPUMEM of " + progress + "%");
                    }
                }
            });
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Intent intent = getIntent();
        Bundle bundle = intent.getExtras();
        if (bundle != null) {
            mTestMode = bundle.getString("testMode");
        }
        Log.d(TAG, "testMode is " + mTestMode);

        final String envName = "ADSP_LIBRARY_PATH";
        final String nativeDir = this.getApplicationInfo().nativeLibraryDir + ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";

        setContentView(R.layout.activity_main);

        requestPermissions(new String[]{
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
        }, REQUEST_EXTERNAL_STORAGE_PERMISSION);

        tv = findViewById(R.id.sample_text);
        if (mTestMode.equals("dmips")) {
            tv.setText("Dmips testing...");
        } else {
            tv.setText("CpuMem testing...");
        }
        new Thread(new Runnable() {
            @Override
            public void run() {
                android.os.Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY);
                timer.schedule(proc_task, 100, 500);
                boolean ret = setEnv(envName, nativeDir);
                if (!ret) {
                    Log.e(TAG, "set dsp env failed!");
                    return;
                }
                Log.i(TAG, "read image");
                readImage(getAssets());
                Log.i(TAG, "begin to test dmips");
                if (mTestMode.equals("dmips")) {
                    mTestInfo = testDmips();
                } else if (mTestMode.equals("cpumem")) {
                    mTestInfo = testCpuMem();
                }
                timer.cancel();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tv.setText(mTestInfo);
                    }
                });
            }
        }).start();
    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String testDmips();
    public native String testCpuMem();
    public native String getProgress();
    public native boolean setEnv(String envName, String envPath);
    public native void readImage(Object assetManager);
}
