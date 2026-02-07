package com.frewen.android.github.samples.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.view.Window;
import android.widget.Button;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;

/**
 * FragmentDemoActivity
 */
public class FragmentDemoActivity extends AppCompatActivity {

//    @BindView(R.id.button6)
    Button mBtnWechat;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_fragment_demo);

        initView();
    }

    private void initView() {
        mBtnWechat = findViewById(R.id.button6);
        mBtnWechat.setOnClickListener(v ->
                startActivity(new Intent(FragmentDemoActivity.this, FragmentStackActivity.class)));
    }
}
