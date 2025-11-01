package com.baidu.iov.vision.ability.core.bean;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * 手势相关检测结果
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class GestureInfo {

    static {
        System.loadLibrary("vision_jni");
    }

    /** 没有检测 */
    public static final short STATIC_NO_DETECT = -1;
    /** 检测但没有识别到手势 */
    public static final short STATIC_NONE = 0;
    /** 点赞 */
    public static final short STATIC_LIKE = 1;
    /** OK */
    public static final short STATIC_OK = 2;
    /** 不喜欢 */
    public static final short STATIC_DISLIKE = 3;
    /** 零 */
    public static final short STATIC_FIST = 4;
    /** 一 */
    public static final short STATIC_1 = 5;
    /** 二 */
    public static final short STATIC_2 = 6;
    /** 三 */
    public static final short STATIC_3 = 7;
    /** 四 */
    public static final short STATIC_4 = 8;
    /** 五 */
    public static final short STATIC_5 = 9;
    /** 上一首(左点赞) */
    public static final short STATIC_PREVIOUS = 10;
    /** 下一首(右点赞) */
    public static final short STATIC_NEXT = 11;
    /** 左五 */
    public static final short STATIC_LEFT_5 = 12;
    /** 右五 */
    public static final short STATIC_RIGHT_5 = 13;
    /** 比心 */
    public static final short STATIC_FINGER_HEART = 14;
    /** ROCK */
    public static final short STATIC_ROCK = 15;

    // ----------------------------------------------------
    // 动态手势 : 挥手、弹、捏
    /** 没有检测 */
    public static final short DYNAMIC_NO_DETECT = -1;
    /** 没有检测到 */
    public static final short DYNAMIC_NONE = 0;
    /** 捏 */
    public static final short DYNAMIC_PINCH = 1;
    /** 握拳 */
    public static final short DYNAMIC_GRASP = 2;
    /** 左挥手 */
    public static final short DYNAMIC_WAVE_LEFT = 3;
    /** 右挥手 */
    public static final short DYNAMIC_WAVE_RIGHT = 4;

    public FloatBuffer mNativeBuffer;

    public GestureInfo() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(getBufferSize());
        buffer.order(ByteOrder.nativeOrder());
        mNativeBuffer = buffer.asFloatBuffer();
        init(mNativeBuffer);
    }

    public void copy(GestureInfo src) {
        if (src != null) {
            mNativeBuffer.position(0);
            src.mNativeBuffer.position(0);
            mNativeBuffer.put(src.mNativeBuffer);
        }
    }

    private native void init(FloatBuffer buffer);

    private native int getBufferSize();

    /**
     * 数据清理
     */
    public void clear() {
        mNativeBuffer.clear();
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 手势基础信息相关
    // -----------------------------------------------------------------------------------------------------------------
    /**
     * 是否检测到了手势
     * @return true-检测到了、false-没有检测到
     */
    public boolean hasGesture() {
        return id() > 0;
    }

    /**
     * 是否没有手势
     * @return ture-没有手势、false-有手势
     */
    public boolean noGesture() {
        return id() == 0;
    }

    /**
     * 手势 ID
     * @return 为每帧图中检测的手势分配的 ID
     */
    public native int id();

    /**
     * 手势框检测结果置信度
     * @return 0.0~1.0 值越大表示越可信
     */
    public float rectConfidence() {
        return 0;
    }

    /**
     * 手势框的位置
     * @return 手势框的左上、右下两个坐标点;
     * (float[0], float[1]) : 左上点;
     * (float[2], float[3]) : 右下点;
     */
    public float[] rect() {
        return null;
    }

    /**
     * 手势关键点检测结果置信度
     * @return  0.0~1.0 值越大表示越可信
     */
    public float landmarkConfidence() {
        return 0;
    }

    /**
     * 手势关键点坐标
     * @return 21个手势关键点的2D坐标位置, 如 (x1=float[0], y1=float[1]) ...
     */
    public float[] landmark() {
        return null;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 手势检测结果类型相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 窗口过滤-手势静态状态
     * @return
     * @see #STATIC_NO_DETECT
     * @see #STATIC_NONE
     * @see #STATIC_LIKE
     * @see #STATIC_OK
     * @see #STATIC_DISLIKE
     * @see #STATIC_FIST
     * @see #STATIC_1
     * @see #STATIC_2
     * @see #STATIC_3
     * @see #STATIC_4
     * @see #STATIC_5
     * @see #STATIC_PREVIOUS
     * @see #STATIC_NEXT
     * @see #STATIC_LEFT_5
     * @see #STATIC_RIGHT_5
     * @see #STATIC_FINGER_HEART
     * @see #STATIC_ROCK
     */
    public native int staticType();

    /**
     * 动态手势状态
     * @return
     * @sse #DYNAMIC_NO_DETECT
     * @sse #DYNAMIC_NONE
     * @sse #DYNAMIC_PINCH
     * @sse #DYNAMIC_GRASP
     * @sse #DYNAMIC_LEFT_WAVE
     * @sse #DYNAMIC_RIGHT_WAVE
     */
    public native int dynamicType();


}
