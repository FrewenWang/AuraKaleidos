package com.baidu.iov.vision.ability.core.result;

import com.baidu.iov.vision.ability.core.bean.FaceInfo;
import com.baidu.iov.vision.ability.config.VisionConfig;

import java.nio.FloatBuffer;

/**
 * 人脸相关结果
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class FaceResult extends AbsVisionResult {

    public FaceInfo[] mFaceInfos;
    public FloatBuffer[] mFaceData;

    public FaceResult() {
        short count = VisionConfig.configAsShort(VisionConfig.Key.FACE_MAX_COUNT);
        mFaceInfos = new FaceInfo[count];
        mFaceData = new FloatBuffer[count];
        FaceInfo fi;
        for (byte i = 0; i < mFaceInfos.length; ++i) {
            fi = new FaceInfo();
            mFaceInfos[i] = fi;
            mFaceData[i] = fi.mNativeBuffer;
        }
    }

    public FaceInfo[] faceInfos() {
        return mFaceInfos;
    }

    public void copy(FaceResult src) {
        if (src != null) {
            int len = mFaceInfos.length;
            for (int i = 0; i < len; i++) {
                mFaceInfos[i].copy(src.mFaceInfos[i]);
            }
        }
    }

    @Override
    public void clear() {
        super.clear();
        if (mFaceInfos != null) {
            if (isClearAllData) {
                for (FaceInfo i : mFaceInfos) {
                    i.clear();
                }
//            } else {
//                for (FaceInfo i : mFaceInfos) {
//                    i.id(0);
//                }
            }
        }
    }

    @Override
    public String toString() {
        return super.toString() + " | id = " + mFaceInfos[0].id();
    }

}
