package com.baidu.iov.vision.ability.core.request;

/**
 * 人脸相关请求基类
 * <p>
 * Created by liwendong on 2018/6/12.
 */
public class FaceRequest extends AbsVisionRequest {
    /**
     * 每帧的图像数据
     */
    public float[] mLandmarkData;

    @Override
    public void clear() {
        super.clear();
        mLandmarkData = null;
    }
}
