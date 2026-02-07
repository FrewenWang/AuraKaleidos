package com.frewen.android.github.samples.bitmap;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;

public class BitmapDemoActivity extends AppCompatActivity {
    private static final String TAG = "BitmapDemoActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bitmap_demo);


        initBitmap();
    }

    private void initBitmap() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        // 图片复用，这个属性必须设置；
        options.inMutable = true;
        // 手动设置缩放比例，使其取整数，方便计算、观察数据；
        options.inDensity = 320;
        options.inTargetDensity = 320;

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.ic_bitmap, options);

        // 对象内存地址；
        Log.i(TAG, "bitmap = " + bitmap);
        Log.i(TAG, "ByteCount = " + bitmap.getByteCount() + ":::bitmap：AllocationByteCount = " + bitmap.getAllocationByteCount());

        // 使用inBitmap属性，这个属性必须设置；
        options.inBitmap = bitmap;
        options.inDensity = 320;

        // 设置缩放宽高为原始宽高一半；
        options.inTargetDensity = 160;
        options.inMutable = true;
        Bitmap bitmapReuse = BitmapFactory.decodeResource(getResources(), R.drawable.ic_bitmap_reuse, options);

        // 复用对象的内存地址；
        Log.i(TAG, "bitmapReuse = " + bitmapReuse);
        Log.i(TAG, "bitmap：ByteCount = " + bitmap.getByteCount() + ":::bitmap：AllocationByteCount = " + bitmap.getAllocationByteCount());
        Log.i(TAG, "bitmapReuse：ByteCount = " + bitmapReuse.getByteCount() + ":::bitmapReuse：AllocationByteCount = " + bitmapReuse.getAllocationByteCount());


    }
}
