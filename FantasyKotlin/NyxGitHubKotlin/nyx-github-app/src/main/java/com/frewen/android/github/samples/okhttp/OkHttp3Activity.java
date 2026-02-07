package com.frewen.android.github.samples.okhttp;

import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;

import com.frewen.android.github.R;
import com.frewen.aura.toolkits.common.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import androidx.appcompat.app.AppCompatActivity;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * OkHttp3Activity
 */
public class OkHttp3Activity extends AppCompatActivity {
    private static final String TAG = "T:OkHttp3Activity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ok_http3);


    }

    /**
     * 同步的GET网络请求
     *
     * @param view
     */
    public void getStringFromServer(View view) {
        //创建 OkHttpClient 对象
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url("http://www.baidu.com")
                .build();
        //excute()是同步的网络请求，不能直接在UI线程进行网络请求
        // 否则会报 android.os.NetworkOnMainThreadException
        new Thread(() -> {
            try {
                Response response = client.newCall(request).execute();
                Log.i(TAG, "FMsg:getStringFromServer() called : " + response.body().string());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();
    }

    /**
     * asyncGetStringFromServer
     * 异步获取从服务端获取String
     * @param view
     */
    public void asyncGetStringFromServer(View view) {
        //创建 OkHttpClient 对象
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url("http://www.baidu.com")
                .build();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {

            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Log.i(TAG, "FMsg:asyncGetStringFromServer() called : " + response.body().string());
            }
        });
    }

    /**
     * @param view
     */
    public void postRequest(View view) {
        //创建 OkHttpClient 对象
        OkHttpClient client = new OkHttpClient();
        RequestBody formBody = new FormBody.Builder()
                .add("ip", "59.108.54.37")
                .build();

        Request postRequest = new Request.Builder()
                .url("http://ip.taobao.com/service/getIpInfo.php")
                .post(formBody)
                .build();

        client.newCall(postRequest).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.d(TAG, "FMsg:onFailure() called with: call = [" + call + "], e = [" + e + "]");
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Log.i(TAG, "FMsg:asyncGetStringFromServer() called : " + response.body().string());
            }
        });
    }

    /**
     * 异步上传文件
     *
     * @param view
     */
    public void asyncUploadFile(View view) {
        String filePath = FileUtils.getExternalStorageAbsolutePath();
        if (TextUtils.isEmpty(filePath)) {
            return;
        }
        Log.i(TAG, "FMsg:asyncUploadFile() called : filePath == " + filePath);
        File file = new File(filePath, "test.txt");

        if (!file.exists()) {
            return;
        }

        //创建 OkHttpClient 对象
        OkHttpClient client = new OkHttpClient();
        MediaType mediaType = MediaType.parse("text/plain");

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "test.txt",
                        RequestBody.create(MediaType.parse("multipart/form-data"), new File(filePath)))
                .build();

        // 异步上传文件的功能实现就是来post一个文件的RequestBody
        Request request = new Request.Builder()
                .url("http://api.github.com/markdown/raw")
                .post(RequestBody.create(mediaType, file))
                .build();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.d(TAG, "FMsg:onFailure() called with: call = [" + call + "], e = [" + e + "]");
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Log.i(TAG, "FMsg:asyncGetStringFromServer() onResponse called : " + response.body().string());
            }
        });
    }

    /**
     * 上传表单数据以及多个文件。
     *
     * @param formFields 表单信息
     * @param files      文件信息
     */
    public void postMultipart(String url, Map<String, String> formFields, File[] files) throws Exception {
        // 实例化MultipartBody
        MultipartBody.Builder bodyBuilder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM);
        // 添加表单数据
        for (Map.Entry<String, String> field : formFields.entrySet()) { // 添加表单信息
            bodyBuilder.addFormDataPart(field.getKey(), field.getValue());
        }
        MediaType mediaType = MediaType.parse("text/plain");
        // 添加文件上传
        for (File file : files) { // 添加文件
            bodyBuilder.addFormDataPart("files", file.getName(),
                    RequestBody.create(mediaType, file));
        }
        //创建 OkHttpClient 对象
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .url(url)
                .post(bodyBuilder.build())
                .build();
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
            } else {
                throw new java.lang.IllegalStateException("Not [200..300) range code response ["
                        + response.toString() + "]");
            }
        } catch (IOException e) {
            Log.i(TAG, "FMsg:postMultipart() called Http error [" + request.toString() + "]", e);
            throw e;
        }
    }


}
