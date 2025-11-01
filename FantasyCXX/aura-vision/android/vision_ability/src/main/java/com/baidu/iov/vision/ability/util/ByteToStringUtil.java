package com.baidu.iov.vision.ability.util;

import android.text.TextUtils;

import java.io.UnsupportedEncodingException;

/**
 * bytetostring 工具
 *
 * Created by v_shaodafu on 2018/3/19.
 */

public class ByteToStringUtil {

    public static String byteToString(byte[] data) {
        String facedata = null;
        if (data.length > 0) {
            try {
                facedata = new String(data, "ISO_8859_1");
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
            ;
        }
        return facedata;
    }

    public static byte[] stringToByte(String strdata) {
        if (!TextUtils.isEmpty(strdata)) {
            byte[] facedata = new byte[0];
            try {
                facedata = strdata.getBytes("ISO_8859_1");
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
            return facedata;
        }
        return null;
    }

}
