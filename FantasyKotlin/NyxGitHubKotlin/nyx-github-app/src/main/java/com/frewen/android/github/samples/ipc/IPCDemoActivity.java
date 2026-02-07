package com.frewen.android.github.samples.ipc;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import com.frewen.android.github.R;
import com.frewen.android.github.samples.ipc.client.AIDLDemoActivity;

import androidx.appcompat.app.AppCompatActivity;

/**
 * IPCDemoActivity
 *
 * @author frewen
 */
public class IPCDemoActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ipcdemo);
    }

    /**
     * 测试AIDL的相关逻辑
     *
     * @param view
     */
    public void enterAIDLDemo(View view) {
        startActivity(new Intent(this, AIDLDemoActivity.class));
    }
}
