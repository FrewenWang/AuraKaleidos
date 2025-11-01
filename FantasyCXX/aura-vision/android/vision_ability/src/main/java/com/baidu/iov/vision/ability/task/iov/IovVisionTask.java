package com.baidu.iov.vision.ability.task.iov;

import android.util.Log;

import com.baidu.iov.vision.ability.core.request.VisionRequest;
import com.baidu.iov.vision.ability.util.VisionNativeHelper;

/**
 * 混合功能任务
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
public class IovVisionTask extends IovAbsVisionTask {

    private static final String TAG = IovVisionTask.class.getSimpleName();

    public static IovVisionTask obtain() {
        return obtain(IovVisionTask.class);
    }

    public static IovVisionTask obtain(VisionRequest request) {
        return obtain(IovVisionTask.class, request);
    }

    @Override
    public void execute() {
        // VisionResult result = VisionResult.obtain();
        mResult.clear();
        boolean ret;
        if (mRequest.mFrameWidth > 0 && mRequest.mFrameHeight > 0) {
            ret = VisionNativeHelper.detectWithIndex(mIndex, mRequest.mFrameData,
                    mRequest.mFrameWidth, mRequest.mFrameHeight, mResult.mPerfUtil);
        } else {
            ret = VisionNativeHelper.detectWithIndex(mIndex, mRequest.mFrameData, mResult.mPerfUtil);
        }
        if (!ret) {
            Log.w(TAG, "detect with index failed");
        }
        mResult.setResultCode(ret);
        // setResult(result);
    }
}
