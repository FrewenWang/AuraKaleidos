package com.baidu.iov.vision.ability.core;

import com.baidu.iov.vision.ability.core.request.AbsVisionRequest;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;

/**
 * 获取视觉能力结果的回调接口适配器
 *
 * Created by liwendong on 2018/5/16.
 */

public class VisionCallbackAdapter<R extends AbsVisionRequest,
                                   T extends AbsVisionResult> implements VisionCallback<R, T> {

    public void onResult(R request, T result) {
    }

    public void onRejected(R request) {
    }
}
