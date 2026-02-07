package com.frewen.github.library.network

import com.frewen.aura.toolkits.kotlin.common.SharedPreferencesUtils
import com.frewen.github.library.common.Constant
import okhttp3.Interceptor
import okhttp3.Response

/**
 * @filename: HeadIntercepter
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/8 10:42
 * Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
class HeadInterceptor : Interceptor {

    private var accessTokenStorage: String by SharedPreferencesUtils(Constant.ACCESS_TOKEN, "")

    private var userBasicCodeStorage: String by SharedPreferencesUtils(Constant.USER_BASIC_CODE, "")


    override fun intercept(chain: Interceptor.Chain): Response {
        var request = chain.request()
        //add access token
        val accessToken = getAuthorization()

        if (accessToken.isNotEmpty()) {
            val url = request.url().toString()
            request = request.newBuilder()
                    .addHeader("Authorization", accessToken)
                    .url(url)
                    .build()
        }
        return chain.proceed(request)
    }


    /**
     * 获取token
     */
    fun getAuthorization(): String {
        if (accessTokenStorage.isBlank()) {
            val basic = userBasicCodeStorage
            return if (basic.isBlank()) {
                //提示输入账号密码
                ""
            } else {
                //通过 basic 去获取token，获取到设置，返回token
                "Basic $basic"
            }
        }
        return "token $accessTokenStorage"

    }
}