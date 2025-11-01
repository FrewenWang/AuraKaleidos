package com.baidu.iov.vision.ability.util;

import android.graphics.Bitmap;

import java.io.ByteArrayOutputStream;

/**
 * Bitmap工具类
 *
 * Created by v_lichaoqun on 2018/4/23.
 */

public class IovBitmapUtils {

    /**
     * Bitmap转byte[]
     *
     * @param bitmap
     * @return
     */
    public static byte[] getByteFromBitmap(Bitmap bitmap) {
        if (bitmap != null) {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
            return outputStream.toByteArray();
        }
        return null;
    }
}
