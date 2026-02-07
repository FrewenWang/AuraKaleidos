package com.frewen.android.github.samples.fragment;

import android.os.Bundle;
import android.view.Window;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

/**
 * FragmentDemoActivity
 *
 * @author Frewen.Wong
 */
public class FragmentStackActivity extends AppCompatActivity {

//    @BindView(R.id.button6)
//    Button mBtnWechat;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_fragment_stack);

        FragmentManager fm = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fm.beginTransaction();
        fragmentTransaction.add(R.id.id_content, FragmentStackOne.newInstance("one", "FragmentStackOne"), "ONE");
        fragmentTransaction.commit();
    }

    private void initView() {

    }
}
