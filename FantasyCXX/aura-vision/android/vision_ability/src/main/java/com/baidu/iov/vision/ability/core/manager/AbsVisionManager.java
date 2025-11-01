package com.baidu.iov.vision.ability.core.manager;

import com.baidu.iov.vision.ability.core.request.AbsVisionRequest;
import com.baidu.iov.vision.ability.core.result.AbsVisionResult;
import com.baidu.iov.vision.ability.task.AbsVisionTask;
import com.baidu.iov.vision.ability.util.VisionExecutors;

/**
 * 视觉能力管理器
 *
 * 管理器系列组件的设计目的，是为了处理具备"生命周期"信息的业务逻辑，即处理跨越多帧数据、多个任务（AbsVisionTask）的场合
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class AbsVisionManager<T1 extends AbsVisionRequest, T2 extends AbsVisionResult> {

    /**
     * 唯一检测功能接口
     *
     * @param request 一帧数据请求，正常情况下只需要设置像素数据即可
     */
    @Deprecated 
    public void detect(T1 request, T2 result) {
        VisionExecutors.instance().executeTask(AbsVisionTask.obtain()
                .setRequest(request)
                .setResult(result));
    }
    
    /**
     * 多实例检测功能接口
     */
    public void detect(T1 request, T2 result, int index) {
        VisionExecutors.instance().executeTask(AbsVisionTask.obtain()
                .setRequest(request)
                .setResult(result)
                .setIndex(index));
    }
}
