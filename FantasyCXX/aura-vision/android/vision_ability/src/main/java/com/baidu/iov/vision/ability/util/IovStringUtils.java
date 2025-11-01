package com.baidu.iov.vision.ability.util;

/**
 * 字符串管理类
 *
 * Created by v_lichaoqun on 2018/4/24.
 */

public class IovStringUtils {
    /**
     * 判断字符串是否是空串
     *
     * @param str
     * @return
     */
    public static boolean isEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
}
