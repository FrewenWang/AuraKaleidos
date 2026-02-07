package com.frewen.android.github.samples.view;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.frewen.android.github.R;
import com.frewen.aura.toolkits.utils.PhoneInfoUtils;

import java.util.ArrayList;

/**
 * @author Frewen.Wong
 */
public class ViewTouchConflictActivity extends AppCompatActivity {
    private static final String TAG = "ViewTouchConflict";
    private HorizontalScrollView mListContainer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_touch_conflict);

        Log.d(TAG, "onCreate");
        initView();
    }

    private void initView() {
        LayoutInflater inflater = getLayoutInflater();
        mListContainer = findViewById(R.id.container);
        final int screenWidth = PhoneInfoUtils.getScreenMetrics(this).widthPixels;
        final int screenHeight = PhoneInfoUtils.getScreenMetrics(this).heightPixels;
        for (int i = 0; i < 3; i++) {
            ViewGroup layout = (ViewGroup) inflater.inflate(
                    R.layout.layout_content, mListContainer, false);
            // 将这个View的宽度设置为屏幕的宽度
            layout.getLayoutParams().width = screenWidth;
            TextView textView = layout.findViewById(R.id.title);
            textView.setText("page " + (i + 1));
            layout.setBackgroundColor(Color.rgb(255 / (i + 1), 255 / (i + 1), 0));
            createList(layout);
            // 将layout的布局设置到容器中
            mListContainer.addView(layout);
        }
    }


    private void createList(ViewGroup layout) {
        ListView listView = (ListView) layout.findViewById(R.id.list);
        ArrayList<String> datas = new ArrayList<String>();
        for (int i = 0; i < 50; i++) {
            datas.add("name " + i);
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                R.layout.item_list, R.id.name, datas);
        listView.setAdapter(adapter);
        listView.setOnItemClickListener((parent, view, position, id) -> Toast.makeText(ViewTouchConflictActivity.this, "click item",
                Toast.LENGTH_SHORT).show());
    }
}
