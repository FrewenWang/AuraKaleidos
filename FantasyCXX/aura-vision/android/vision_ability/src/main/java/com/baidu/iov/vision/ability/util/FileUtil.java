package com.baidu.iov.vision.ability.util;

import android.content.res.AssetManager;
import android.util.Log;

import com.baidu.iov.vision.ability.VisionContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class FileUtil {

    private static final String TAG = FileUtil.class.getSimpleName();

    public static String readFileFromAsset(String fileName) {
        BufferedReader br = null;
        try {
            AssetManager am = VisionContext.appContext().getResources().getAssets();
            br = new BufferedReader(new InputStreamReader(am.open(fileName)));
            String str = null;
            StringBuilder content = new StringBuilder();
            while ((str = br.readLine()) != null) {
                content.append(str);
            }
            str = content.toString();
            if (ALog.DEBUG) {
                ALog.v(TAG, str);
            }
            return str;
        } catch (Exception e) {
            if (ALog.DEBUG) {
                Log.e(TAG, e.getMessage(), e);
            }
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    // ignore
                }
            }
        }
        return null;
    }
}
