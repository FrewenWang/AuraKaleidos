package com.frewen.android.github.samples.retrofit;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;

/**
 * @filename: WeatherService
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/12/3 0003 上午11:32
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public interface WeatherService {

    @GET("query/{city}")
    Call<String> getWeather(@Path("city") String newsId);

}
