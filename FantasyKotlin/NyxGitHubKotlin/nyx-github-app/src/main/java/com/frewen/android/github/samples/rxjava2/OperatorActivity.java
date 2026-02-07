package com.frewen.android.github.samples.rxjava2;

import android.os.Bundle;
import android.util.Log;

import com.frewen.android.github.R;

import androidx.appcompat.app.AppCompatActivity;
import io.reactivex.Flowable;

/**
 * RxJava2的操作符测试
 */
public class OperatorActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_operator);

        Flowable.just("test", "test2")
                .subscribe(str -> Log.i("tag", str));

        /**
         * fromArray 会将不定参数转化成为一个数组来进行解析
         *
         */
        Flowable.fromArray(1, 2, 3, 4, 5)
                .subscribe(integer -> Log.i("tag", String.valueOf(integer)));

        /**
         * empty操作符不会发送任何数据，而是直接发送onComplete事件，我们写一个例子：
         * 只会打印complete，其他回调并不会触发。
         */
        Flowable.empty().subscribe(
                obj -> Log.i("tag", "next" + obj.toString()),
                e -> Log.i("tag", "error"),
                () -> Log.i("tag", "complete"));

        Flowable.error(new RuntimeException("test")).subscribe(
                obj -> Log.i("tag", "next" + obj.toString()),
                e -> Log.i("tag", "error"),
                () -> Log.i("tag", "complete"));


    }
}
