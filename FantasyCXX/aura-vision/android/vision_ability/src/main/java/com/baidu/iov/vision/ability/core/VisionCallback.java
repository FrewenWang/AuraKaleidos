package com.baidu.iov.vision.ability.core;

import com.baidu.iov.vision.ability.core.request.AbsVisionRequest;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;

/**
 * 获取视觉能力结果的回调接口
 *
 * Created by liwendong on 2018/5/16.
 */

public interface VisionCallback<R extends AbsVisionRequest, T extends AbsVisionResult> {

    /**
     * AbsVisionResult 结果回调函数
     *
     * @param request 原始请求对象
     * @param result  结果对象
     */
    void onResult(R request, T result);

    /**
     * 请求被拒绝时的回调函数
     * <p>
     * 通常情况下，一帧 Camera 数据处理完成后才会执行下一帧，所以此函数一般不会被调用
     *
     * @param request 原始请求对象
     */
    void onRejected(R request);
}
