package com.baidu.iov.vision.ability.util;

import java.util.HashMap;
import java.util.Map;

/**
 * jni 接口
 * <p>
 * Created by liwendong on 2018/5/9.
 */

public class VisionNativeHelper {

    static {
        System.loadLibrary("vision_jni");
    }

    private static native void registerNatives();

    @Deprecated
    public static native void init(int index, String configJson, Object am,
                                   Object[] faceInfos, int faceMaxCount,
                                   Object[] gestureInfos, int gestureMaxCount,
                                   Object[] postureInfos, int postureMaxCount,
                                   Object frameInfo);

    public static native void initModel(Object assetManager);

    public static native void initService(int index);

    public static native void initFaceBuffer(int index, Object[] buffers, int count);
    public static native void initGestureBuffer(int index, Object[] buffers, int count);
    public static native void initBodyBuffer(int index, Object[] buffers, int count);
    public static native void initFrameBuffer(int index, Object buffer);

    @Deprecated
    public static native void initSingleFunctionResultData(int index, Object[] faceInfos,
                                                           int faceMaxCount, Object frameInfo);

    @Deprecated
    public static native boolean detect(int index, byte[] frame, PerformanceUtil perfUtil);

    @Deprecated
    public static native boolean detect(int index, byte[] frame,
                                        float[] faceInfos, float[][] gestureInfos, float[] frameInfo,
                                        PerformanceUtil perfUtil);

    public static native boolean detectWithIndex(int index, byte[] frame, PerformanceUtil perfUtil);

    public static native boolean detectWithIndex(int index, byte[] frame,
                                                 int frameWidth, int frameHeight, PerformanceUtil perfUtil);


    public static native boolean getSwitch(int index, short ability);

    public static native void getSwitches(int index, HashMap<Short, Boolean> switches);

    public static native boolean setSwitch(int index, short ability, boolean switcher);

    @Deprecated
    public static native void setSwitches(Map<Short, Boolean> switches);

    public static native float compareFaceFeatures(float[] first, float[] second);

    public static native int getFrameResize(float[] landmark, byte[] data, short frameWidth,
                                            short frameHeight, short desireWidth,
                                            short desireHeight, byte[] resizeFrame);

    public static native boolean setConfig(int index, int key, float value);

    public static native float getConfig(int index, int key);

    @Deprecated
    public static native boolean setInitConfig(int index, int key, float value);

    @Deprecated
    public static native float getInitConfig(int index, int key);

    public static native void release(int index);

    public static native void cleanAbilityTriggerAccumulative(int index, short ability);

    public static native void toBytes(float[] srcData, byte[] destData);

    public static native void toFloats(byte[] srcData, float[] destData);

    public static native void convertYuvToRgb(byte[] srcData, int[] destData, int width, int height);

    public static native void saveImage(String path, byte[] frame, int width, int height);

    public static native void computeCosDifference(float[] angleA, float[] angleB, float[] output);

    public static native boolean weightedAverage(float[] src, float[] cur, float[] output);

    @Deprecated
    public static native boolean detectSingleFunction(int index, int type, byte[] frame, float[] landmarks);

    public static native boolean detectSingleFunction(int index, int type, byte[] frame, 
                                                      Object[] faceInfos, int faceMaxCount, Object frameInfo);

    public static native boolean setEnv(String envName, String envPath);
}
