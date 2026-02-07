package com.frewen.android.github.samples.ipc.client;

import android.net.Uri;
import android.os.Bundle;
import android.view.View;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;

/**
 * ContentProvider是Android中提供的专门用于不同应用间进行数据共享的方式
 * <p>
 * 有很多我们需要注意的点：
 * 后续我们会慢慢讲到：
 * 比如CRUD操作、防止SQL注入和权限控制等
 */
public class ContentProviderActivity extends AppCompatActivity {

    private Uri uri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_content_provider);
        uri = Uri.parse("content://com.frewen.ipc.content.provider");
    }

    /**
     * query
     * @param view
     */
    public void query(View view) {
        getContentResolver().query(uri, null, null, null, null);
        getContentResolver().query(uri, null, null, null, null);
        getContentResolver().query(uri, null, null, null, null);
    }

    /**
     * delete
     * @param view
     */
    public void delete(View view) {
        getContentResolver().delete(uri, null, null);
    }

    /**
     * insert
     * @param view
     */
    public void insert(View view) {
        getContentResolver().insert(uri, null);
    }

    /**
     * update
     * @param view
     */
    public void update(View view) {
        getContentResolver().update(uri, null, null, null);
    }
}
