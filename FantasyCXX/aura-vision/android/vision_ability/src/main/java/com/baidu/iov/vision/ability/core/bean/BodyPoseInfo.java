package com.baidu.iov.vision.ability.core.bean;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * 手势检测结果
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
public class BodyPoseInfo {

    private static final short LEN_RECT_CENTER = 2;
    private static final short LEN_RECT_LT = 2;
    private static final short LEN_RECT_RB = 2;
    private static final short LEN_RECT = LEN_RECT_LT + LEN_RECT_RB;
    private static final short LEN_PERSON_LM_18 = 2 * 18;

    private static final short IND_PERSON_RECT_ID = 0;
    private static final short IND_PERSON_LANDMARK_ID = IND_PERSON_RECT_ID + 1;
    private static final short IND_PERSON_RECT_CENTER = IND_PERSON_LANDMARK_ID + 1;
    private static final short IND_PERSON_LT = IND_PERSON_RECT_CENTER + LEN_RECT_CENTER;
    private static final short IND_PERSON_RB = IND_PERSON_LT + LEN_RECT_LT;
    private static final short IND_PERSON_LANDMARK_2D_18 = IND_PERSON_RB + LEN_RECT_RB;

    private static final short NATIVE_DATA_SIZE = IND_PERSON_LANDMARK_2D_18 + LEN_PERSON_LM_18;

    public FloatBuffer mNativeBuffer;

    private float[] personRectCenter;
    private float[] personRect;
    private float[] personLandmark;

    public BodyPoseInfo() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(NATIVE_DATA_SIZE * 4);
        buffer.order(ByteOrder.nativeOrder());
        mNativeBuffer = buffer.asFloatBuffer();
    }

    /**
     * 数据清理
     */
    public void clear() {
//        Arrays.fill(mNativeBuffer, 0);
        mNativeBuffer.clear();
    }

    public int id() {
        return (int) mNativeBuffer.get(IND_PERSON_RECT_ID);
    }

    public void id(int id) {
        mNativeBuffer.put(IND_PERSON_RECT_ID, id);
    }

    /**
     * 姿态框
     *
     * @return
     */
    public float[] personRectCenter() {
        if (personRectCenter == null) {
            personRectCenter = new float[LEN_RECT_CENTER];
        }
        mNativeBuffer.position(IND_PERSON_RECT_CENTER);
        mNativeBuffer.get(personRectCenter, 0, LEN_RECT_CENTER);
        return personRectCenter;
    }


    /**
     * 姿态框
     *
     * @return
     */
    public float[] personRect() {
        if (personRect == null) {
            personRect = new float[LEN_RECT];
        }
        mNativeBuffer.position(IND_PERSON_LT);
        mNativeBuffer.get(personRect, 0, LEN_RECT);
        return personRect;
    }

    /**
     * 姿态关键点
     *
     * @return
     */
    public float[] personLandmark() {
        if (personLandmark == null) {
            personLandmark = new float[LEN_PERSON_LM_18];
        }
        mNativeBuffer.position(IND_PERSON_LANDMARK_2D_18);
        mNativeBuffer.get(personLandmark, 0, LEN_PERSON_LM_18);
        return personLandmark;
    }


    /**
     * @return
     */
    public boolean hasPersonRect() {
        return mNativeBuffer.get(IND_PERSON_RECT_ID) > 0;
    }

    /**
     * @return
     */
    public boolean hasPersonLandmark() {
        return mNativeBuffer.get(IND_PERSON_LANDMARK_ID) > 0;
    }
}
