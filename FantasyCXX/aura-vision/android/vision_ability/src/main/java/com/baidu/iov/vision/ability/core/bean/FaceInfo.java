package com.baidu.iov.vision.ability.core.bean;

import android.graphics.Rect;

import com.baidu.iov.vision.ability.config.VisionConfig;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * 人脸相关检测结果
 */
public class FaceInfo {

    static {
        System.loadLibrary("vision_jni");
    }

    // ---------------------------------------------------------------------------------------------
    // 各个能力结果状态值

    // 头部方向状态值
    /** 头部方向-没有检测到 */
    public static final byte HEAD_DIRECTION_NONE = 0;   // 头部方向-没有检测到
    /** 头部方向-向左 */
    public static final byte HEAD_DIRECTION_LEFT = 1;   // 头部方向-向左
    /** 头部方向-向右 */
    public static final byte HEAD_DIRECTION_RIGHT = 2;  // 头部方向-向右
    /** 头部方向-向上 */
    public static final byte HEAD_DIRECTION_UP = 3;     // 头部方向-向上
    /** 头部方向-向下 */
    public static final byte HEAD_DIRECTION_DOWN = 4;   // 头部方向-向下

    /** 七分类状态值 */
    /** 七分类-没有检测到 */
    public static final byte DANGEROUS_NONE = 0;
    /** 七分类-抽烟 */
    public static final byte DANGEROUS_SMOKE = 1;
    /** 七分类-比虚 */
    public static final byte DANGEROUS_SILENCE = 2;
    /** 七分类-喝水 */
    public static final byte DANGEROUS_DRINK = 3;
    /** 七分类-张嘴 */
    public static final byte DANGEROUS_OPEN_MOUTH = 4;
    /** 七分类-口罩遮挡 */
    public static final byte DANGEROUS_MASK_COVER = 5;
    /** 七分类-捂嘴遮挡 */
    public static final byte DANGEROUS_COVER_MOUTH = 6;

    // 疲劳状态值
    /** 疲劳-没有检测到 */
    public static final byte FATIGUE_NONE = 0;          // 疲劳-没有检测到
    /** 疲劳-打哈欠 */
    public static final byte FATIGUE_YAWN = 1;          // 疲劳-打哈欠
    /** 疲劳-长闭眼 */
    public static final byte FATIGUE_EYECLOSE = 2;      // 疲劳-长闭眼
    /** 疲劳-打哈欠+长闭眼 */
    public static final byte FATIGUE_YAWN_EYECLOSE = 3; // 疲劳-打哈欠+长闭眼

    // 睁闭眼状态值
    /** 睁闭眼-没有检测到 */
    public static final byte EYE_STATE_UNKNOWN = 0; // 睁闭眼-没有检测到
    /** 睁闭眼-睁眼 */
    public static final byte EYE_STATE_OPEN = 1;    // 睁闭眼-睁眼
    /** 睁闭眼-闭眼 */
    public static final byte EYE_STATE_CLOSE = 2;   // 睁闭眼-闭眼

    // 眼睛检测状态值
    /**  */
    public static final byte EYE_DETECT_STATE_UNKNOWN = 0;
    /**  */
    public static final byte EYE_DETECT_STATE_SUCCESS = 1;
    /**  */
    public static final byte EYE_DETECT_STATE_AVAILABLE = 2;
    /**  */
    public static final byte EYE_DETECT_STATE_VISIBILITY = 3;
    /**  */
    public static final byte EYE_DETECT_STATE_GLASSES = 4;

    // 眨眼状态值
    /** 眨眼-没有检测到 */
    public static final byte EYE_BLINK_UNKNOWN = 0; // 眨眼-没有检测到
    /** 眨眼-眨眼了 */
    public static final byte EYE_BLINK_TRUE = 1;    // 眨眼-眨眼了
    /** 眨眼-没有眨眼 */
    public static final byte EYE_BLINK_FALSE = 2;   // 眨眼-没有眨眼

    // 注意力状态值
    /** 注意力-注意力不集中 */
    public static final byte ATTENTION_NOT_FOCUS = 1;       // 注意力-注意力不集中
    /** 注意力-动作不协调 */
    public static final byte ATTENTION_UNCOORDINATED = 2;   // 注意力-动作不协调
    /** 注意力-看前方 */
    public static final byte ATTENTION_LOOK_FORWARD = 3;    // 注意力-看前方
    /** 注意力-看左后视镜 */
    public static final byte ATTENTION_LOOK_LEFT = 4;       // 注意力-看左后视镜
    /** 注意力-看右后视镜 */
    public static final byte ATTENTION_LOOK_RIGHT = 5;      // 注意力-看右后视镜
    /** 注意力-向上看 */
    public static final byte ATTENTION_LOOK_UP = 6;         // 注意力-向上看
    /** 注意力-向下看 */
    public static final byte ATTENTION_LOOK_DOWN = 7;       // 注意力-向下看

    // 多模(点头、摇头)状态值
    /** 多模-未知的 */
    public static final byte HEAD_BEHAVIOR_UNKNOWN = 0; // 多模-未知的
    /** 多模-摇头 */
    public static final byte HEAD_BEHAVIOR_SHAKE = 1;   // 多模-摇头
    /** 多模-点头 */
    public static final byte HEAD_BEHAVIOR_NOD = 2;     // 多模-点头
    /** 多模-检测中 */
    public static final byte HEAD_BEHAVIOR_GOON = 3;    // 多模-检测中

    // 头部静止状态值
    /** 头部静止检测-没有检测到 */
    public static final byte NO_MOVING_UNKNOWN = 0; // 头部静止检测-没有检测到
    /** 头部静止检测-没有移动 */
    public static final byte NO_MOVING_TRUE = 1;    // 头部静止检测-没有移动
    /** 头部静止检测-移动了 */
    public static final byte NO_MOVING_FALSE = 2;   // 头部静止检测-移动了

    // 无感活体状态值
    /** 无感活体-没有检测到 */
    public static final byte NO_INTERACT_LIVING_NONE = 0;   // 无感活体-没有检测到
    /** 无感活体-攻击 */
    public static final byte NO_INTERACT_LIVING_ATTACK = 1; // 无感活体-攻击
    /** 无感活体-活体 */
    public static final byte NO_INTERACT_LIVING_LIVING = 2; // 无感活体-活体

    // 有感活体状态值
    /** 有感活体-指定了错误的动作 */
    public static final byte INTERACT_LIVING_DETECT_TYPE_ERROR = 0; // 有感活体-指定了错误的动作
    /** 有感活体-人脸没有看向指定方向 */
    public static final byte INTERACT_LIVING_FACE_INCORRECT = 1;    // 有感活体-人脸没有看向指定方向
    /** 有感活体-动作异常 */
    public static final byte INTERACT_LIVING_ACTION_INCORRECT = 2;  // 有感活体-动作异常
    /** 有感活体-检测超时 */
    public static final byte INTERACT_LIVING_TIME_OUT = 3;          // 有感活体-检测超时
    /** 有感活体-开始检测中 */
    public static final byte INTERACT_LIVING_START = 4;             // 有感活体-开始检测中
    /** 有感活体-动作检测成功 */
    public static final byte INTERACT_LIVING_SUCCESS = 5;           // 有感活体-动作检测成功
    /** 有感活体-人脸朝向检测成功 */
    public static final byte INTERACT_LIVING_FACE_OPTIMAL = 10;     // 有感活体-人脸朝向检测成功

    // 人脸属性状态值
    /** 属性-未知状态 */
    public static final byte ATTR_UNKNOWN = -1;         // 属性-未知状态

    // 属性-表情状态值
    /** 属性-表情-没有检测到 */
    public static final byte ATTR_EMOTION_NORMAL = 0;   // 属性-表情-没有检测到
    /** 属性-表情-开心 */
    public static final byte ATTR_EMOTION_LIKE = 1;     // 属性-表情-开心
    /** 属性-表情-不开心 */
    public static final byte ATTR_EMOTION_DISLIKE = 2;  // 属性-表情-不开心
    /** 属性-表情-惊讶 */
    public static final byte ATTR_EMOTION_SURPRISE = 3; // 属性-表情-惊讶

    // 属性-眼镜状态值
    /** 属性-眼镜-太阳镜 */
    public static final byte ATTR_GLASS_SUN = 0;    // 属性-眼镜-太阳镜
    /** 属性-眼镜-没有戴 */
    public static final byte ATTR_GLASS_NONE = 1;   // 属性-眼镜-没有戴
    /** 属性-眼镜-普通眼镜 */
    public static final byte ATTR_GLASS_NORMAL = 2; // 属性-眼镜-普通眼镜

    // 属性-性别状态值
    /** 属性-性别-男性 */
    public static final byte ATTR_GENDER_MALE = 0;   // 属性-性别-男性
    /** 属性-性别-女性 */
    public static final byte ATTR_GENDER_FEMALE = 1; // 属性-性别-女性

    // 属性-种族状态值
    /** 属性-种族-黑 */
    public static final byte ATTR_RACE_BLACK = 0;   // 属性-种族-黑
    /** 属性-种族-白 */
    public static final byte ATTR_RACE_WHITE = 1;   // 属性-种族-白
    /** 属性-种族-黄 */
    public static final byte ATTR_RACE_YELLOW = 2;  // 属性-宗族-黄

    // 属性-年龄状态值
    /** 属性-年龄-婴儿 */
    public static final byte ATTR_AGE_BABY = 0;     // 属性-年龄-婴儿
    /** 属性-年龄-青少年 */
    public static final byte ATTR_AGE_TEENAGER = 1; // 属性-年龄-青少年
    /** 属性-年龄-青年 */
    public static final byte ATTR_AGE_YOUTH = 2;    // 属性-年龄-青年
    /** 属性-年龄-中年 */
    public static final byte ATTR_AGE_MIDDLE = 3;   // 属性-年龄-中年
    /** 属性-年龄-老年 */
    public static final byte ATTR_AGE_ELDERLY = 4;  // 属性-年龄-老年

    // 眼神唤醒状态值
    /** 眼神唤醒-失败 */
    public static final byte EYEWAKING_FAIL = 0;    // 眼神唤醒-失败
    /** 眼神唤醒-成功 */
    public static final byte EYEWAKING_SUCCESS = 1; // 眼神唤醒-成功

    // 打电话状态值
    /** 打电话-没有检测到 */
    public static final byte CALL_NONE = 0;     // 打电话-没有检测到
    /** 打电话-正在打电话 */
    public static final byte CALL_CALLING = 1;  // 打电话-正在打电话

    // 图片质量状态值
    /** 图像质量-模糊-不模糊 */
    public static final byte QUALITY_BLUR_GOOD = 0;  // 图像质量-模糊-不模糊
    /** 图像质量-模糊-模糊的 */
    public static final byte QUALITY_BLUR_BAD = 1;   // 图像质量-模糊-模糊的
    /** 图像质量-遮挡-没有遮挡 */
    public static final byte QUALITY_COVER_GOOD = 0; // 图像质量-遮挡-没有遮挡
    /** 图像质量-遮挡-有遮挡 */
    public static final byte QUALITY_COVER_BAD = 1;  // 图像质量-遮挡-有遮挡
    /** 图像质量-噪声-没有噪声 */
    public static final byte QUALITY_NOISE_GOOD = 0; // 图像质量-噪声-没有噪声
    /** 图像质量-噪声-有噪声 */
    public static final byte QUALITY_NOISE_BAD = 1;  // 图像质量-噪声-有噪声

    // 人脸追踪状态
    public static final byte TRACK_STATE_UNKNOWN = 0;
    public static final byte TRACK_STATE_INIT = 1;
    public static final byte TRACK_STATE_TRACKING = 2;
    public static final byte TRACK_STATE_MISS = 3;

    // 人脸检测状态
    public static final byte SEARCH_STATE_UNKNOWN = 0;
    public static final byte SEARCH_STATE_NOTHING = 1;
    public static final byte SEARCH_STATE_FACE_FEATURE = 2;
    public static final byte SEARCH_STATE_FACE = 3;

    // 滑窗结果状态值
    /** 滑窗结果-没有执行检测 */
    public static final short ABILITY_STATE_DEFAULT = -1;   // 滑窗结果-没有执行检测
    /** 滑窗结果-没有检测到 */
    public static final short ABILITY_STATE_NONE = 0;       // 滑窗结果-没有检测到
    /** 滑窗结果-开始检测到结果了 */
    public static final short ABILITY_STATE_START = 1;      // 滑窗结果-开始检测到结果了
    /** 滑窗结果-持续检测到了结果 */
    public static final short ABILITY_STATE_CONTINUE = 2;   // 滑窗结果-持续检测到了结果
    /** 滑窗结果-检测结束了 */
    public static final short ABILITY_STATE_INTERRUPT = 3;  // 滑窗结果-检测结束了

    /** 人脸特征数据大小 */
    public static final int FEATURE_LENGTH = 1024;

    /** 左右转头的敏感度 */
    private static final float YAW_ANGEL_THRESHOLD = 20;

    /** 上下转头的敏感度 */
    private static final float PITCH_ANGEL_THRESHOLD = 10;

    public FloatBuffer mNativeBuffer;

    public FaceInfo() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(getBufferSize());
        buffer.order(ByteOrder.nativeOrder());
        mNativeBuffer = buffer.asFloatBuffer();
        init(mNativeBuffer);
    }

    /**
     * 拷贝 FaceInfo
     *
     * @param src 拷贝的对象
     */
    public void copy(FaceInfo src) {
        if (src != null) {
            mNativeBuffer.position(0);
            src.mNativeBuffer.position(0);
            mNativeBuffer.put(src.mNativeBuffer);
        }
    }

    /**
     * 初始化
     * @param buffer 申请的内存数据
     */
    private native void init(FloatBuffer buffer);

    /**
     * 获取 FaceInfo 需要占用的内存大小
     */
    private native int getBufferSize();

    /**
     * 数据清理
     */
    public void clear() {
        mNativeBuffer.clear();
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 面部基础信息相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 检测是否有人脸
     * @return true-有 false-没有
     */
    public boolean hasFace() {
        return id() > 0;
    }

    /**
     * 检测是否没人脸
     * @return true-没有 false-有
     */
    public boolean noFace() {
        return id() == 0;
    }

    /**
     * 人脸检测状态
     * @return 人脸检测的状态
     * @see #SEARCH_STATE_UNKNOWN
     * @see #SEARCH_STATE_NOTHING
     * @see #SEARCH_STATE_FACE_FEATURE
     * @see #SEARCH_STATE_FACE
     */
    public native int searchState();

    /**
     * 人脸的 ID
     * @return 每帧数据中，每张人脸检测时临时分配的人脸 ID
     */
    public native int id();

    /**
     * 人脸框检测结果的置信度
     * @return 0.0~1.0 值越高表示可信度越高
     */
    public native float rectConfidence();

    /**
     * 人脸框的坐标
     * @return 返回左上、右下两个点的坐标值 [x1, y1, x2, y2]
     */
    public native float[] rect();

    /**
     * 人脸关键点检测结果的置信度
     * @return 0.0~1.0 值越高表示可信度越高
     */
    public native float landmarkConfidence();

    /**
     * 人脸关键点2D106的坐标
     * @return 106个关键点坐标值(x, y)，float[] size = 212
     */
    public native float[] landmark2D106();

    /**
     * 人脸关键点2D68的坐标
     * @return 68个关键点坐标值(x, y)，float[] size = 136
     */
    public native float[] landmark2D68();

    /**
     * 人脸关键点3D68的坐标
     * @return 68个关键点坐标值(x, y, z)，float[] size = 204
     */
    public native float[] landmark3D68();

    /**
     * 人脸特征数据
     * @return float[] size 1024 的特征向量
     */
    public native float[] feature();

    /**
     * 经过滑窗过滤的用户无感知（交互）活体检测状态
     * @return int
     * @see #NO_INTERACT_LIVING_NONE
     * @see #NO_INTERACT_LIVING_ATTACK
     * @see #NO_INTERACT_LIVING_LIVING
     */
    public native int noInteractLivingState();

    /**
     * 单帧的用户无感知（交互）活体检测状态
     * @return int
     * @see #NO_INTERACT_LIVING_NONE
     * @see #NO_INTERACT_LIVING_ATTACK
     * @see #NO_INTERACT_LIVING_LIVING
     */
    public native int noInteractLivingSingleState();

    /**
     * 用户有感知（交互）活体检测状态
     * @return 用户有感知（交互）活体检测状态
     * @see #INTERACT_LIVING_DETECT_TYPE_ERROR
     * @see #INTERACT_LIVING_FACE_INCORRECT
     * @see #INTERACT_LIVING_ACTION_INCORRECT
     * @see #INTERACT_LIVING_TIME_OUT
     * @see #INTERACT_LIVING_START
     * @see #INTERACT_LIVING_SUCCESS
     * @see #INTERACT_LIVING_FACE_OPTIMAL
     */
    public native int interactLivingState();

    /**
     * 人脸是否在指定区域中
     * @param detectRect 指定的区域
     * @param threshold  阈值
     * @return ture-在区域中 false-不在区域中
     */
    public boolean isFaceInRect(Rect detectRect, int threshold) {
        detectRect.set(detectRect.left - detectRect.width() / 2, 0, detectRect.right + detectRect.width() / 2, 720);
        int outCount = 0;
        float[] landmark = landmark2D106();
        if (landmark == null) {
            return false;
        }
        int len = landmark.length / 2;
        for (int i = 0; i < len; i++) {
            if (!detectRect.contains((int) landmark[i * 2], (int) landmark[i * 2 + 1])) {
                ++outCount;
            }
        }
        if (outCount < threshold && isFaceUpFront()) {
            return true;
        }
        return false;
    }

    /**
     * 人脸是否在区域中，人脸是否满足角度要求
     * @param detectRect 指定的区域
     * @param angels 指定角度区域
     * @param threshold  阈值
     * @return true-符合要求 false-不符合要求
     */
    @Deprecated
    public boolean isFaceInRect(Rect detectRect, float[] angels, int threshold) {
        detectRect.set(detectRect.left - detectRect.width() / 2, 0, detectRect.right + detectRect.width() / 2, 720);
        int outCount = 0;
        float[] landmark = landmark2D106();
        if (landmark == null) {
            return false;
        }
        int len = landmark.length / 2;
        for (int i = 0; i < len; i++) {
            if (!detectRect.contains((int) landmark[i * 2], (int) landmark[i * 2 + 1])) {
                ++outCount;
            }
        }
//        Log.d("faceinfo_isFaceInRect","outCount:"+outCount+" threshold:"+threshold);
        // 先判断所有点都在框中，然后判断角度是否在你们传进来的范围
        if (outCount < threshold && isFaceUpFront(angels)) {
            return true;
        }
        return false;
    }

    /**
     * 针对奇瑞t1c摄像头定制的人脸激活头部角度
     * @return ture-符合 false-不符合
     */
    private boolean isFaceUpFront() {
        float[] hd = optimizedHeadDeflection();
        // 左右
        float yaw = hd[0];
        // 上下的判断
        float pitch = hd[1];

        if (yaw < 0 && (Math.abs(yaw) - VisionConfig.faceUpFrontThreshold[2]) > 0) {
            return false;
        }
        if (yaw > 0 && (Math.abs(yaw) - VisionConfig.faceUpFrontThreshold[0]) > 0) {
            return false;
        }

        if (pitch > 0 && (Math.abs(pitch) - VisionConfig.faceUpFrontThreshold[3]) > 0) {
            return false;
        }
        if (pitch < 0 && (Math.abs(pitch) - VisionConfig.faceUpFrontThreshold[1]) > 0) {
            return false;
        }

        return true;
    }

    /**
     * 针对奇瑞t1c摄像头定制的人脸激活头部角度
     * @param angels 指定角度范围
     * @return ture-符合 false-不符合
     */
    private boolean isFaceUpFront(float[] angels) {
        float[] hd = optimizedHeadDeflection();
        // 左右
        float yaw = hd[0];
        // 上下的判断
        float pitch = hd[1];
        // angels 是你们传进来的值，[左上右下] yaw,pitch是摄像头获取到的角度
        // yaw>0左  yaw<0右 pitch>0下，pitch<0下
        if (yaw < 0 && (Math.abs(yaw) - angels[2]) > 0) {
            return false;
        }
        if (yaw > 0 && (Math.abs(yaw) - angels[0]) > 0) {
            return false;
        }

        if (pitch > 0 && (Math.abs(pitch) - angels[3]) > 0) {
            return false;
        }
        if (pitch < 0 && (Math.abs(pitch) - angels[1]) > 0) {
            return false;
        }

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // 危险驾驶行为相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 单帧-危险驾驶的五分类的状态
     * @return 检测结果
     * @see #DANGEROUS_NONE
     * @see #DANGEROUS_SMOKE
     * @see #DANGEROUS_SILENCE
     * @see #DANGEROUS_DRINK
     * @see #DANGEROUS_OPEN_MOUTH
     */
    public native int singleDangerDriveState();

    /**
     * 窗口过滤-危险驾驶的五分类的状态
     * @return 检测结果
     * @see #DANGEROUS_NONE
     * @see #DANGEROUS_SMOKE
     * @see #DANGEROUS_SILENCE
     * @see #DANGEROUS_DRINK
     * @see #DANGEROUS_OPEN_MOUTH
     */
    public native int dangerousState();

    /**
     * 窗口数据-危险驾驶-抽烟的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] dangerousSmokeState();

    /**
     * 窗口数据-危险驾驶-喝水的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] dangerousDrinkState();

    /**
     * 窗口数据-危险驾驶-张嘴的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] dangerousMouthState();

    /**
     * 单帧-打电话的状体
     * @return true-检测到了打电话 false-没有检测到打电话
     */
    public native boolean singleCallState();

    /**
     * 窗口过滤-打电话的状态
     * @return 打电话检测结果
     * @see #CALL_NONE
     * @see #CALL_CALLING
     */
    public native int dangerousCallState();

    /**
     * 窗口数据-打电话的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] dangerousPhoneCallState();

    // -----------------------------------------------------------------------------------------------------------------
    // 疲劳状态相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 窗口过滤-疲劳检测
     * @return 没有结果、打哈欠、闭眼、打哈欠+闭眼
     * @see #FATIGUE_NONE
     * @see #FATIGUE_YAWN
     * @see #FATIGUE_EYECLOSE
     * @see #FATIGUE_YAWN_EYECLOSE
     */
    public native int fatigueState();

    /**
     * 窗口数据-闭眼的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] fatigueEyeCloseState();

    /**
     * 窗口数据-打哈欠的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] fatigueYawnState();

    // -----------------------------------------------------------------------------------------------------------------
    // 注意力不集中相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 滑窗过滤-注意力检测
     * @return 向前看、向左看、向右看、向上看、向下看
     * @see #ATTENTION_LOOK_FORWARD
     * @see #ATTENTION_LOOK_LEFT
     * @see #ATTENTION_LOOK_RIGHT
     * @see #ATTENTION_LOOK_UP
     * @see #ATTENTION_LOOK_DOWN
     */
    public native int attentionState();

    /**
     * 窗口数据-向左看检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] attentionLookLeftState();

    /**
     * 窗口数据-向右看检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] attentionLookRightState();

    /**
     * 窗口数据-向上看检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] attentionLookUpState();

    /**
     * 窗口数据-向下看检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] attentionLookDownState();

    // -----------------------------------------------------------------------------------------------------------------
    // 面部属性相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 窗口数据-小眼睛检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] smallEyeState();

    /**
     * 用户表情的状态
     * @return 正常、开心、不开心、惊讶
     * @see #ATTR_EMOTION_NORMAL
     * @see #ATTR_EMOTION_LIKE
     * @see #ATTR_EMOTION_DISLIKE
     * @see #ATTR_EMOTION_SURPRISE
     */
    public native int attributeEmotionState();

    /**
     * 用户戴眼镜的状态
     * @return 没有戴眼镜、太阳镜、普通眼镜
     * @see #ATTR_GLASS_NONE
     * @see #ATTR_GLASS_SUN
     * @see #ATTR_GLASS_NORMAL
     */
    public native int attributeGlassState();

    /**
     * 用户性别
     * @return 男、女
     * @see #ATTR_GENDER_MALE
     * @see #ATTR_GENDER_FEMALE
     */
    public native int attributeGenderState();

    /**
     * 种族检测状态
     * @return 黑、白、黄
     * @see #ATTR_RACE_BLACK
     * @see #ATTR_RACE_WHITE
     * @see #ATTR_RACE_YELLOW
     */
    public native int attributeRaceState();

    /**
     * 年龄检测状态
     * @return 婴儿、青少年、青年、中年、老年
     * @see #ATTR_AGE_BABY
     * @see #ATTR_AGE_TEENAGER
     * @see #ATTR_AGE_YOUTH
     * @see #ATTR_AGE_MIDDLE
     * @see #ATTR_AGE_ELDERLY
     */
    public native int attributeAgeState();

    // -----------------------------------------------------------------------------------------------------------------
    // 眼睛检测相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 睁闭眼的置信度
     * @return 0.0~1.0 值越小表示其闭眼的概率越高
     */
    public native float eyeCloseConfidence();

    /**
     * 双眼瞳孔质心
     * @return
     * float[0] float[1] : 左眼瞳孔质点在图像上的坐标;
     * float[2] float[3] : 右眼瞳孔质点在图像上的坐标;
     */
    public native float[] eyeCentroid();

    /**
     * 眼球的 4 个关键点
     * @return
     * (float[0], float[1]) : 左眼中心点 (x, y);
     * (float[2], float[3]) : 左眼凝视点 (x, y);
     * (float[4], float[5]) : 右眼中心点 (x, y);
     * (float[6], float[7]) : 右眼凝视点 (x, y);
     */
    public native float[] eyeTracking();

    /**
     * 眼神唤醒状态
     * @return 0-否 1-是
     * @see #EYEWAKING_FAIL
     * @see #EYEWAKING_SUCCESS
     */
    public native int eyeWakingState();

    /**
     * 左眼中心点三维坐标
     * @return
     * float[0] : x;
     * float[1] : y;
     * float[2] : z;
     */
    public native float[] leftEyeGazeOrigin();

    /**
     * 左眼凝视点三维坐标
     * @return
     * float[0] : x;
     * float[1] : y;
     * float[2] : z;
     */
    public native float[] leftEyeGazeDirection();

    /**
     * 右眼中心点三维坐标
     * @return
     * float[0] : x;
     * float[1] : y;
     * float[2] : z;
     */
    public native float[] rightEyeGazeOrigin();

    /**
     * 右眼凝视点三维坐标
     * @return
     * float[0] : x;
     * float[1] : y;
     * float[2] : z;
     */
    public native float[] rightEyeGazeDirection();

    /**
     * 左眼上下眼睑距离
     * @return 左眼上下眼睑距离
     */
    public native float leftEyelidOpening();

    /**
     * 右眼上下眼睑距离
     * @return 右眼上下眼睑距离
     */
    public native float rightEyelidOpening();

    /**
     * 左眼眼睑状态
     * @return 0:unknown, 1:open, 2:close
     * @see #EYE_STATE_UNKNOWN
     * @see #EYE_STATE_OPEN
     * @see #EYE_STATE_CLOSE
     */
    public native int leftEyelidState();

    /**
     * 右眼眼睑状态
     * @return 0:unknown, 1:open, 2:close
     * @see #EYE_STATE_UNKNOWN
     * @see #EYE_STATE_OPEN
     * @see #EYE_STATE_CLOSE
     */
    public native int rightEyelidState();

    /**
     * 眼睑状态
     * @return 0:unknown, 1:open, 2:close
     * @see #EYE_STATE_UNKNOWN
     * @see #EYE_STATE_OPEN
     * @see #EYE_STATE_CLOSE
     */
    public native int eyeState();

    /**
     * 左眼眼睛检测状态
     * @return 0:UNKNOWN, 1:SUCCESS, 2:AVAILABLE, 3:VISIBILITY, 4:GLASSES
     * @see #EYE_DETECT_STATE_UNKNOWN
     * @see #EYE_DETECT_STATE_SUCCESS
     * @see #EYE_DETECT_STATE_AVAILABLE
     * @see #EYE_DETECT_STATE_VISIBILITY
     * @see #EYE_DETECT_STATE_GLASSES
     */
    public native int leftEyeState();

    /**
     * 右眼眼睛检测状态
     * @return 0:UNKNOWN, 1:SUCCESS, 2:AVAILABLE, 3:VISIBILITY, 4:GLASSES
     * @see #EYE_DETECT_STATE_UNKNOWN
     * @see #EYE_DETECT_STATE_SUCCESS
     * @see #EYE_DETECT_STATE_AVAILABLE
     * @see #EYE_DETECT_STATE_VISIBILITY
     * @see #EYE_DETECT_STATE_GLASSES
     */
    public native int rightEyeState();

    /**
     * 闭眼频率
     * @return 闭眼频率
     */
    public native float eyeBlinkCloseSpeed();

    /**
     * 闭眼开始时间
     * @return 闭眼开始时间
     */
    public native long eyeBlinkCloseStartTime();

    /**
     * 闭眼结束时间
     * @return 闭眼结束时间
     */
    public native long eyeBlinkCloseStopTime();

    /**
     * 眨眼状态
     * @return 0:no blink, 1:blink
     * @see #EYE_BLINK_UNKNOWN
     * @see #EYE_BLINK_TRUE
     * @see #EYE_BLINK_FALSE
     */
    public native int eyeBlinkState();

    /**
     * 眨眼时长
     * @return 眨眼时长
     */
    public native float eyeBlinkDuration();

    // -----------------------------------------------------------------------------------------------------------------
    // 头部检测相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 头部坐标，相对摄像机
     * @return
     * float[0] : x;
     * float[1] : y;
     * float[2] : z;
     */
    public native float[] headPosition();

    /**
     * 头部行为动作
     * @return 摇头、点头、检测中
     * @see #HEAD_BEHAVIOR_SHAKE
     * @see #HEAD_BEHAVIOR_NOD
     * @see #HEAD_BEHAVIOR_GOON
     */
    public native int headBehaviorState();

    /**
     * 头部头部三向偏转角
     * @return
     * float[0] : Yaw;
     * float[1] : Pitch;
     * float[2] : Roll;
     */
    public native float[] optimizedHeadDeflection();

    /**
     * 获取头部偏转方向，相对于摄像头
     *
     * @return 没有检测到、头部向左、头部向右、头部向上、头部向下
     * @see #HEAD_DIRECTION_NONE
     * @see #HEAD_DIRECTION_LEFT
     * @see #HEAD_DIRECTION_RIGHT
     * @see #HEAD_DIRECTION_UP
     * @see #HEAD_DIRECTION_DOWN
     */
    public int headDirection() {
        float[] hd = optimizedHeadDeflection();
        // 左右
        float yaw = hd[0];
        // 上下的判断
        float pitch = hd[1];
        boolean horizontal = (Math.abs(yaw) - YAW_ANGEL_THRESHOLD) > 0;
        boolean vertical = (Math.abs(pitch) - PITCH_ANGEL_THRESHOLD) > 0;
        if (horizontal && vertical) {
            return HEAD_DIRECTION_UP;
        } else if (horizontal) {
            if (yaw > 0) {
                return HEAD_DIRECTION_RIGHT;
            } else {
                return HEAD_DIRECTION_LEFT;
            }
        } else if (vertical) {
            if (pitch > 0) {
                return HEAD_DIRECTION_DOWN;
            } else {
                return HEAD_DIRECTION_UP;
            }
        } else {
            return HEAD_DIRECTION_NONE;
        }
    }

    /**
     * 窗口数据-比虚动作检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native float[] beQuiteState();

    // -----------------------------------------------------------------------------------------------------------------
    // 人脸追踪相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 人脸追踪状态
     * @return 0-未知、1-INIT、2-TRACKING、3-MISS
     */
    public native int trackState();

    // -----------------------------------------------------------------------------------------------------------------
    // 成像质量相关
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * 窗口过滤-人脸是否被遮挡
     *
     * @return 0-没有遮挡、1-遮挡了
     * @see #QUALITY_COVER_GOOD
     * @see #QUALITY_COVER_BAD
     */
    public native int isFaceBlocked();

    /**
     * 摄像头是否被遮挡
     *
     * 【针对多人脸、多摄像头方案的重构，后续将转移到 FrameInfo 中】
     *
     * @return 0-没有被遮挡、1-被遮挡了
     * */
    public native int isCameraBlocked();

    /**
     * 窗口数据-无人脸检测的状态
     * @return int[] [0]=state [1]=continue time [2]=trigger num
     * @see #ABILITY_STATE_DEFAULT
     * @see #ABILITY_STATE_NONE
     * @see #ABILITY_STATE_START
     * @see #ABILITY_STATE_CONTINUE
     * @see #ABILITY_STATE_INTERRUPT
     */
    public native int[] noFaceState();

    /**
     * 人脸图像是否模糊
     *
     * @return 0-不模糊、1-模糊
     * @see #QUALITY_BLUR_GOOD
     * @see #QUALITY_BLUR_BAD
     */
    public native int isBlur();

    /**
     * 单帧-人脸是否被遮挡
     *
     * @return 0-没有遮挡、1-遮挡了
     * @see #QUALITY_COVER_GOOD
     * @see #QUALITY_COVER_BAD
     */
    public native int isSingleBlocked();

    /**
     * 人脸图像是否有噪声
     *
     * @return 0-没有噪声、1-有噪声
     * @see #QUALITY_NOISE_GOOD
     * @see #QUALITY_NOISE_BAD
     */
    public native int isNoisy();

}