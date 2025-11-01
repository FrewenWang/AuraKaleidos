package com.baidu.iov.vision.ability.core.result;

import com.baidu.iov.vision.ability.config.VisionConfig;
import com.baidu.iov.vision.ability.core.bean.BodyPoseInfo;

/**
 * 身体姿态相关结果
 *
 */
public class BodyPoseResult extends AbsVisionResult {

    public BodyPoseInfo[] mBodyPoseInfos;
    public Object[] mPersonData;

    public BodyPoseResult() {
        short count = VisionConfig.configAsShort(VisionConfig.Key.BODY_MAX_COUNT);
        mBodyPoseInfos = new BodyPoseInfo[count];
        mPersonData = new Object[count];
        BodyPoseInfo pi;
        for (byte i = 0; i < mBodyPoseInfos.length; ++i) {
            pi = new BodyPoseInfo();
            mBodyPoseInfos[i] = pi;
            mPersonData[i] = pi.mNativeBuffer;
        }
    }

    @Override
    public void clear() {
        super.clear();
        if (mBodyPoseInfos != null) {
            if (isClearAllData) {
                for (BodyPoseInfo i : mBodyPoseInfos) {
                    i.clear();
                }
            }
        }
    }
}