package com.frewen.android.github.samples.soundpool;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.Build;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.HashMap;
import java.util.Map;

import androidx.annotation.IntDef;
import androidx.annotation.NonNull;
import androidx.annotation.RawRes;

/**
 * @filename: SoundPoolHelper
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/11/7 0007 下午3:27
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class SoundPoolHelper {

    public final static int TYPE_MUSIC = AudioManager.STREAM_MUSIC;
    public final static int TYPE_ALARM = AudioManager.STREAM_ALARM;
    public final static int TYPE_RING = AudioManager.STREAM_RING;
    private SoundPool.OnLoadCompleteListener mListener;

    public void setOnLoadCompleteListener(SoundPool.OnLoadCompleteListener listener) {
        this.mListener = listener;
    }

    @IntDef({TYPE_MUSIC, TYPE_ALARM, TYPE_RING})
    @Retention(RetentionPolicy.SOURCE)
    public @interface TYPE {
    }

    /**
     * SoundPool对象
     */
    private SoundPool soundPool;
    private int maxStream;
    private Map<String, Integer> ringtoneIds;

    public SoundPoolHelper() {
        this(1, TYPE_MUSIC);
    }

    public SoundPoolHelper(int maxStream) {
        this(maxStream, TYPE_MUSIC);
    }

    public SoundPoolHelper(int maxStream, @TYPE int streamType) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            SoundPool.Builder builder = new SoundPool.Builder();
            //可同时播放的音频流
            builder.setMaxStreams(maxStream);
            //音频属性的Builder
            AudioAttributes.Builder attrBuild = new AudioAttributes.Builder();
            //音频类型
            attrBuild.setLegacyStreamType(streamType);
            builder.setAudioAttributes(attrBuild.build());
            soundPool = builder.build();
        } else {
            soundPool = new SoundPool(maxStream, streamType, 1);
        }
        if (null != mListener) {
            soundPool.setOnLoadCompleteListener(mListener);
        }
        this.maxStream = maxStream;
        ringtoneIds = new HashMap<>();
    }

    /**
     * 加载音频资源
     *
     * @param context 上下文
     * @param resId   资源ID
     * @return a sound ID. This value can be used to play or unload the sound.
     */
    public int load(Context context, @NonNull String ringtoneName, @RawRes int resId) {
        if (maxStream == 0) {
            return 0;
        }
        maxStream--;
        int soundId = soundPool.load(context, resId, 1);
        ringtoneIds.put(ringtoneName, soundId);
        return maxStream;
    }

    public void unload(String name) {
        if (null != soundPool) {
            soundPool.unload(ringtoneIds.get(name));
        }
    }

    /**
     * int play(int soundID, float leftVolume, float rightVolume, int priority, int loop, float rate) ：
     * 1)该方法的第一个参数指定播放哪个声音；
     * 2) leftVolume 、
     * 3) rightVolume 指定左、右的音量：
     * 4) priority 指定播放声音的优先级，数值越大，优先级越高；
     * 5) loop 指定是否循环， 0 为不循环， -1 为循环；
     * 6) rate 指定播放的比率，数值可从 0.5 到 2 ， 1 为正常比率。
     */
    public int play(@NonNull String ringtoneName, boolean isLoop) {
        if (ringtoneIds.containsKey(ringtoneName)) {
            int soundId = ringtoneIds.get(ringtoneName);
            return soundPool.play(soundId, 1, 1, 1, isLoop ? -1 : 0, 1.0f);
        }
        return 0;
    }

    /**
     * 自定义播放参数的属性
     *
     * @param ringtoneName
     * @param leftVol      指定左、右的音量
     * @param rightVol
     * @param priority     指定播放声音的优先级，数值越大，优先级越高
     * @param isLoop
     * @param rate         指定播放的比率，数值可从 0.5 到 2 ， 1 为正常比率。
     * @return non-zero streamID if successful, zero if failed
     */
    public int play(@NonNull String ringtoneName, int leftVol, int rightVol, int priority, boolean isLoop, float rate) {
        if (ringtoneIds.containsKey(ringtoneName)) {
            int soundId = ringtoneIds.get(ringtoneName);
            return soundPool.play(soundId, leftVol, rightVol, priority, isLoop ? -1 : 0, rate);
        }
        return 0;
    }

    /**
     * @param streamID
     */
    public void pause(int streamID) {
        if (null != soundPool) {
            soundPool.pause(streamID);
        }
    }

    /**
     * @param streamID
     */
    public void resume(int streamID) {
        if (null != soundPool) {
            soundPool.resume(streamID);
        }
    }

    public void autoPause() {
        if (null != soundPool) {
            soundPool.autoPause();
        }
    }

    public void autoResume() {
        if (null != soundPool) {
            soundPool.autoResume();
        }
    }

    public void release() {
        if (null != ringtoneIds) {
            ringtoneIds.clear();
        }
        if (soundPool != null) {
            soundPool.release();
        }
    }
}
