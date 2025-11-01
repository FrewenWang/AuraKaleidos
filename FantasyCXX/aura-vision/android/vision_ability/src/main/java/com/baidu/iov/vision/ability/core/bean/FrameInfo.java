package com.baidu.iov.vision.ability.core.bean;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class FrameInfo {

    static {
        System.loadLibrary("vision_jni");
    }

    // ---------------------------------------------------------------------------------------------
    // 状态值

    // 图片攻击状态
    public static final byte SPOOF_UNKNOWN = 0;
    public static final byte SPOOF_PHOTO_ATTACK = 1;
    public static final byte SPOOF_SCREEN_ATTACK = 2;

    // 图片遮挡状态
    public static final byte BLOCKAGE_UNKNOWN = 0;
    public static final byte BLOCKAGE_NEAR = 1;
    public static final byte BLOCKAGE_DISTANCE = 2;

    // 图片光照状态
    public static final byte IMAGE_STATE_UNKNOWN = 0;
    public static final byte IMAGE_STATE_BRIGHT = 1;
    public static final byte IMAGE_STATE_DARK = 2;
    public static final byte IMAGE_STATE_NORMAL = 3;


    public FloatBuffer mNativeBuffer;

    public FrameInfo() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(getBufferSize());
        buffer.order(ByteOrder.nativeOrder());
        mNativeBuffer = buffer.asFloatBuffer();
        init(mNativeBuffer);
    }

    public void copy(FrameInfo src) {
        if (src != null) {
            mNativeBuffer.position(0);
            src.mNativeBuffer.position(0);
            mNativeBuffer.put(src.mNativeBuffer);
        }
    }

    private native void init(FloatBuffer buffer);

    private native int getBufferSize();

    public void clear() {
        mNativeBuffer.clear();
    }

    /**
     * @return 图像是否被攻击， 0:UNKNOW, 1:PHOTO_ATTACK, 2:SCREEN_ATTACK
     * */
    public native int spoofed();

    /**
     * @return 亮度值
     */
    public native float brightness();

    /**
     * @return 画面光照状态 0:UNKNOW, 1:bright, 2:dark, 3:normal
     */
    public native float imageState();

    /**
     * @return 画面遮挡状态， 0:UNKNOW, 1:near摄像头遮挡, 2:distance人脸遮挡
     */
    public native float blockage();

    /**
     * @return 当前帧时间戳
     * */
    public native long timestamp();

//    /**
//     * @return 图片是否模糊
//     */
//    public native int isBlur();
//
//    /**
//     * @return 图片是否有遮挡
//     */
//    public native int isBlocked();
//
//    /**
//     * @return 图片是否有噪声
//     */
//    public native int hasNoise();

}
