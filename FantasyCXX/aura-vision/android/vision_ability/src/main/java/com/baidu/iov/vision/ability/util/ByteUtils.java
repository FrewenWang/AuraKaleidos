package com.baidu.iov.vision.ability.util;

import android.graphics.Bitmap;
import android.os.Environment;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * byte工具类
 *
 * Created by liwendong on 2018/5/8.
 */
public class ByteUtils {

    /**
     * desc: 存储 argb 图片
     * @param mArgbData
     * @param imageWidth
     * @param imageHeight
     */
    public static void saveArgbFile(int[] mArgbData, int imageWidth, int imageHeight) {
        Bitmap canvasImg = null;
        if (canvasImg == null) {
            canvasImg = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        }
        canvasImg.setPixels(mArgbData, 0, imageWidth, 0, 0, imageWidth, imageHeight);

        FileOutputStream out = null;
        File file = new File(Environment.getExternalStorageDirectory(), System.currentTimeMillis() + "argb.jpg");
        try {
            out = new FileOutputStream(file);
            canvasImg.compress(Bitmap.CompressFormat.JPEG, 90, out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            if (out != null) {
                out.flush();
                out.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * desc:  拼接byte数组
     * @param left
     * @param top
     * @param right
     * @param bottom
     * @param data
     */
    public static byte[] combineBytes(int left, int top, int right, int bottom, byte[] data) {
        byte[] bytesLeft   = intToByteArray(left);
        byte[] bytesTop    = intToByteArray(top);
        byte[] bytesRight  = intToByteArray(right);
        byte[] bytesBottom = intToByteArray(bottom);

        byte[] bytes1 = addBytes(bytesLeft,  bytesTop);
        byte[] bytes2 = addBytes(bytesRight, bytesBottom);
        byte[] total  = addBytes(bytes1, bytes2);

        return addBytes(total, data);
    }


    /**
     * 将int转为低字节在后，高字节在前的byte数组
     *  b[0] = 11111111(0xff) & 01100001
     *  b[1] = 11111111(0xff) & 00000000
     *  b[2] = 11111111(0xff) & 00000000
     *  b[3] = 11111111(0xff) & 00000000
     * @param value
     * @return
     */
    public static byte[] intToByteArray(int value) {
        byte[] src = new byte[4];
        src[0] = (byte) ((value >> 24) & 0xFF);
        src[1] = (byte) ((value >> 16) & 0xFF);
        src[2] = (byte) ((value >> 8)  & 0xFF);
        src[3] = (byte) (value         & 0xFF);
        return src;
    }

    /**
     *
     * @param data1
     * @param data2
     * @return data1 与 data2拼接的结果
     */
    private static byte[] addBytes(byte[] data1, byte[] data2) {
        byte[] data3 = new byte[data1.length + data2.length];
        System.arraycopy(data1, 0, data3, 0, data1.length);
        System.arraycopy(data2, 0, data3, data1.length, data2.length);
        return data3;

    }

    /**
     * 解析检测人脸信息数据
     * @param builder
     * @return
     */
    public static synchronized int getNumber(StringBuilder builder) {
        String s = builder.toString();

        int l1 = s.length() - s.replace("1", "").length();
        int l2 = s.length() - s.replace("2", "").length();

//        int max = Math.max(l1, l2);
//        return max == 0 ? 0 : (l1 == max ? 1 : 2);
        return l2 != 0 ? 2 : (l1 != 0 ? 1 : 0);
    }


}
