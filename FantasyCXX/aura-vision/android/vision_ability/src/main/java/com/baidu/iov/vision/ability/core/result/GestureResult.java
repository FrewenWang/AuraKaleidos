package com.baidu.iov.vision.ability.core.result;

import com.baidu.iov.vision.ability.core.bean.GestureInfo;
import com.baidu.iov.vision.ability.config.VisionConfig;

/**
 * 手势检测相关结果
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class GestureResult extends AbsVisionResult {

    public GestureInfo[] mGestureInfos;
    public Object[] mGestureData;

    public GestureResult() {
        short count = VisionConfig.configAsShort(VisionConfig.Key.GESTURE_MAX_COUNT);
        mGestureInfos = new GestureInfo[count];
        mGestureData = new Object[count];
        GestureInfo gi;
        for (byte i = 0; i < mGestureInfos.length; ++i) {
            gi = new GestureInfo();
            mGestureInfos[i] = gi;
            mGestureData[i] = gi.mNativeBuffer;
        }
    }

    public GestureInfo[] gestureInfos() {
        return mGestureInfos;
    }

    public void copy(GestureResult src) {
        if (src != null) {
            int len = mGestureInfos.length;
            for (int i = 0; i < len; i++) {
                mGestureInfos[i].copy(src.mGestureInfos[i]);
            }
        }
    }

    @Override
    public void clear() {
        super.clear();
        if (mGestureInfos != null) {
            if (isClearAllData) {
                for (GestureInfo i : mGestureInfos) {
                    i.clear();
                }
//            } else {
//                for (GestureInfo i : mGestureInfos) {
//                    i.id(0);
//                }
            }
        }
    }
}