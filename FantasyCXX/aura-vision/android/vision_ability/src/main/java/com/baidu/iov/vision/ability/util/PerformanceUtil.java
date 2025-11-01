package com.baidu.iov.vision.ability.util;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * 时间工具类
 *
 * Created by liwendong on 2018/5/23.
 */

public class PerformanceUtil {

    private static final String TAG = "PerfUtil";
    public static final String TAG_PREFIX = "PERF_";

    private String mTag = TAG;
    private long mItemStartTime;

    private TreeMap<String, Long> mItemsLong;
    private TreeMap<String, Float> mItemsFloat;

    public PerformanceUtil() {
        mItemsLong = new TreeMap<>();
        mItemsFloat = new TreeMap<>();
    }

    public Map<String, Long> getPerfLong() {
        return mItemsLong;
    }

    public Map<String, Float> getPerfFloat() {
        return mItemsFloat;
    }

    public void addItem(String key, long value) {
        mItemsLong.put(key, value);
    }

    public void addItem(String key, float value) {
        mItemsFloat.put(key, value);
    }

    public void copy(Map<String, Long> src) {
        mItemsLong.putAll(src);
    }

    public void clear() {
        mItemsLong.clear();
        mItemsFloat.clear();
    }

    public void setTag(String tag) {
        mTag = TAG_PREFIX + tag;
    }

    public void tick() {
        mItemStartTime = System.currentTimeMillis();
    }

    public void tick(String key) {
        mItemsLong.put(key, System.currentTimeMillis());
    }

    public void tock(String key) {
        Long val = mItemsLong.get(key);
        if (val == null) {
            val = mItemStartTime;
            mItemStartTime = 0;
        }
        mItemsLong.put(key, System.currentTimeMillis() - val);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Set<Map.Entry<String, Long>> entries = mItemsLong.entrySet();
        for (Map.Entry<String, Long> e : entries) {
            sb.append(e.getKey()).append("=").append(e.getValue()).append("; ");
        }
        Set<Map.Entry<String, Float>> entries2 = mItemsFloat.entrySet();
        for (Map.Entry<String, Float> e : entries2) {
            sb.append(e.getKey()).append("=").append(e.getValue()).append("; ");
        }
        return sb.toString();
    }

    public void show() {
        ALog.d(mTag, toString());
    }
}
