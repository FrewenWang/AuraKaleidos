package com.baidu.iov.vision.ability.core.manager;

import com.baidu.iov.vision.ability.core.request.VisionRequest;
import com.baidu.iov.vision.ability.core.result.VisionResult;
import com.baidu.iov.vision.ability.task.iov.IovVisionTask;
import com.baidu.iov.vision.ability.util.VisionExecutors;

/**
 * 混合功能检测管理器，适用于自由配置视觉能力的场合
 *
 * create by v_liuyong01 on 2019/2/20.
 */
public class VisionManager extends AbsVisionManager<VisionRequest, VisionResult> {

    public VisionManager() {
    }

    @Override
    public void detect(VisionRequest request, VisionResult result) {
        IovVisionTask task = IovVisionTask.obtain();
        task.setRequest(request);
        task.setResult(result);
        task.setIndex(0);
        VisionExecutors.instance().executeTask(task);
    }

    @Override
    public void detect(VisionRequest request, VisionResult result, int index) {
        IovVisionTask task = IovVisionTask.obtain();
        task.setRequest(request);
        task.setResult(result);
        task.setIndex(index);
        VisionExecutors.instance().executeTask(task);
    }

}
