package com.frewen.android.github.samples.retrofit;

import androidx.appcompat.app.AppCompatActivity;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import android.os.Bundle;

import com.frewen.android.github.R;

/**
 * @author Frewen.Wong
 */
public class RetrofitActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_retrofit);
        /**
         * 这是建造者模式下，我们来创建Retrofit对象
         */
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://apis.juhe.cn/simpleWeather/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        WeatherService weatherService = retrofit.create(WeatherService.class);
    }
}
