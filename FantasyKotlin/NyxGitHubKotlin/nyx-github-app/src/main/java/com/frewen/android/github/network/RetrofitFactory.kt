package com.frewen.android.github.network

import com.frewen.github.library.common.Config
import com.frewen.aura.framework.net.LoggerInterceptor
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * @filename: RetrofitFactory
 * @introduction:  创建Retrofit的单例工厂模式
 * @author: Frewen.Wong
 * @time: 2019/1/30 23:14
 * Copyright ©2018 Frewen.Wong. All Rights Reserved.
 */
class RetrofitFactory private constructor() {

    val retrofit: Retrofit

    private val interceptor: Interceptor

    /**
     * kotlin中没有静态方法。所以我们使用伴生对象达到类似的效果
     * by lazy 本来就是线程安全的
     */
//    companion object {
//        // 延迟加载。直接调用他的构造方法
//        val instance: RetrofitFactory by lazy {
//            // 调用RetrofitFactory工厂类的私有构造方法
//            RetrofitFactory()
//        }
//    }
    companion object {
        @Volatile
        private var mRetrofitFactory: RetrofitFactory? = null

        val instance: RetrofitFactory
            get() {
                if (null == mRetrofitFactory) {
                    synchronized(RetrofitFactory::class.java) {
                        if (mRetrofitFactory == null)
                            mRetrofitFactory = RetrofitFactory()
                    }
                }
                return mRetrofitFactory!!
            }

    }

    /**
     * init代码块是类在构造函数初始化的时候调用的代码掳爱
     */
    init {
        // 我们来自己合成一个拦截器
        interceptor = Interceptor { chain: Interceptor.Chain ->
            val request = chain.request()
                    .newBuilder()
                    .addHeader("Content-Type", "application/json")
                    .addHeader("charset", "utf-8")
                    .build()
            chain.proceed(request)
        }

        retrofit = Retrofit.Builder()
                .baseUrl(Config.GITHUB_API_BASE_URL)
                //数据转换的工厂，我们一般用的是Gson的数据转换的工厂
                .addConverterFactory(GsonConverterFactory.create())
                .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
                .client(getHttpClient())
                .build()
    }

    /**
     * 实例化OKHttpClient
     *
     * 在方法的名字后面加一个？ 表示可以为空。
     */
    private fun getHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
                .addInterceptor(initHeaderInterceptor())
                .addInterceptor(initLoggerInterceptor())
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .build()
    }

    /**
     * 实例化HeaderInterceptor
     */
    private fun initHeaderInterceptor(): Interceptor {
        // 可以使用我们上面实例化的interceptor
        // return interceptor
        // 也可以可以直接使用FreeFrame里面的的LoggerInterceptor
        //return HeadersInterceptor()
        return com.frewen.github.library.network.HeadInterceptor()
    }

    private fun initLoggerInterceptor(): Interceptor {
        // 我们可以直接使用FreeFrame里面的的LoggerInterceptor
        return LoggerInterceptor()
        // 或者我们也可以依赖OKHttps官方提供的日志拦截器
//        val interceptor = HttpLoggingInterceptor()
//        interceptor.level = HttpLoggingInterceptor.Level.BODY
//        return interceptor
    }

    /**
     * 声明一个create的泛型方法。需要返回的一个Service对象
     */
    fun <T> create(service: Class<T>): T {
        return retrofit.create(service)
    }

}