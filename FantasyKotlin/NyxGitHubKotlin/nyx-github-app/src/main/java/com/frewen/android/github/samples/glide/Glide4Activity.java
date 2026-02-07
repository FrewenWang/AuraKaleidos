package com.frewen.android.github.samples.glide;

import android.os.Bundle;

import com.bumptech.glide.Glide;
import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;

/**
 * Glide4Activity
 */
public class Glide4Activity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_glide4);

        Glide.with(this).load("");


    }
}
